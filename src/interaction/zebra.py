"""Prepare selected or complete Zebra-CoT Jigsaw / Visual Search Parquet shards."""
import argparse
import ast
from collections import Counter, defaultdict
from functools import lru_cache
import glob
import hashlib
import io
import json
from pathlib import Path
import random
import re

from .data import file_hash, json_write, resolve_image


DATASET = "multimodal-reasoning-lab/Zebra-CoT"
DATASET_REVISION = "f6d1defc169d19180a69f517925ed6cbb06a1c97"
ILVR_REVISION = "13a67ed2ee975d6ca56b4e9eba18d68b352b166b"
OFFICIAL_RULE_AST_SHA256 = "3acc5e092202300afa7be0f407e7ee0df1ea892fd0a3ebc138f0caca5611e237"
TASKS = {
    "jigsaw": {"directory": "2D Visual Reasoning - Visual Jigsaw", "rows": 21899, "shards": 26},
    "visual_search": {"directory": "2D Visual Reasoning - Visual Search", "rows": 30000, "shards": 27},
}
PROTOCOL = "zebra_question_image_first_v1"
COLUMNS = ["Question", "Final Answer", "problem_image_1"]
IMAGE_MARKER = re.compile(r"<image_start>\s*\[problem_image_1\]\s*<image_end>")


def build_query(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty text")
    # The official ILVR evaluator takes images first, followed by text_input.
    # Its raw-Zebra conversion was not released; this adapter removes only the
    # known image placeholder, preserving the question and its instructions.
    text = IMAGE_MARKER.sub("", question).strip()
    if not text or re.search(r"<image_|\[(?:problem|reasoning)_image_\d+\]", text):
        raise ValueError("Unexpected/missing Zebra question or image marker")
    return text


def discover_shards(raw_dir=None, parquet=None, subset="all"):
    if (raw_dir is None) == (not parquet):
        raise ValueError("Provide exactly one of --raw_dir and --parquet")
    if subset not in ("all", *TASKS):
        raise ValueError(f"Unknown subset: {subset}")
    if raw_dir is not None:
        root = Path(raw_dir)
        if not root.is_dir():
            raise FileNotFoundError(root)
        files = sorted(root.rglob("*.parquet"))
    else:
        files = []
        for pattern in parquet:
            matches = sorted(glob.glob(str(pattern), recursive=True))
            if not matches:
                raise FileNotFoundError(pattern)
            files.extend(Path(p) for p in matches)
    result, seen = defaultdict(list), set()
    for path in files:
        # Inspect the immediate parent: a full snapshot also has unrelated tasks.
        task = next((t for t, spec in TASKS.items() if path.parent.name in (t, spec["directory"])), None)
        if task is None:
            if raw_dir is not None and path.parent.resolve() != Path(raw_dir).resolve():
                continue
            if subset == "all":
                raise ValueError(f"Cannot infer task for {path}; keep the official folder or specify --subset")
            task = subset  # Renamed files / flat browser downloads need an explicit task.
        if subset != "all" and task != subset:
            continue
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"Duplicate Parquet input: {path}")
        seen.add(resolved)
        result[task].append(resolved)
    if not result or subset != "all" and subset not in result:
        raise FileNotFoundError("No requested Zebra task shards found")
    return dict(result)


def prepare(output_dir, raw_dir=None, parquet=None, subset="all", max_samples_per_task=None, seed=42):
    import pyarrow.parquet as pq
    from PIL import Image
    if max_samples_per_task is not None and max_samples_per_task <= 0:
        raise ValueError("max_samples_per_task must be positive")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new prepared output directory: {output}")
    sources, seen_hashes = {}, set()
    for task, paths in discover_shards(raw_dir, parquet, subset).items():
        sources[task] = []
        for path in paths:
            print(f"Inspecting and hashing {task}: {path.name}", flush=True)
            pf = pq.ParquetFile(path)
            if not set(COLUMNS).issubset(pf.schema_arrow.names) or pf.metadata.num_rows == 0:
                raise ValueError(f"Empty or invalid Zebra schema: {path}; need {COLUMNS}")
            digest = file_hash(path)
            if digest in seen_hashes:
                raise ValueError(f"Duplicate Parquet contents: {path}")
            seen_hashes.add(digest)
            sources[task].append({"path": str(path), "sha256": digest, "rows": pf.metadata.num_rows})
        # IDs and random selection remain stable across directory moves/renames.
        sources[task].sort(key=lambda s: s["sha256"])
    output.mkdir(parents=True)
    test = output / "TEST.jsonl"
    counts, available = {}, {}
    with test.open("w", encoding="utf-8") as handle:
        for task in TASKS:
            if task not in sources:
                continue
            total = available[task] = sum(s["rows"] for s in sources[task])
            count = min(total, max_samples_per_task) if max_samples_per_task is not None else total
            selected = set(random.Random(f"{seed}:{task}").sample(range(total), count)) if count < total else None
            offset, written = 0, 0
            for source in sources[task]:
                wanted = (set(range(source["rows"])) if selected is None else
                          {n - offset for n in selected if offset <= n < offset + source["rows"]})
                pf, row_start = pq.ParquetFile(source["path"]), 0
                print(f"Extracting {task}: {Path(source['path']).name}, selected={len(wanted)}", flush=True)
                for group in range(pf.num_row_groups):
                    group_end = row_start + pf.metadata.row_group(group).num_rows
                    if any(row_start <= n < group_end for n in wanted):
                        cursor = row_start
                        # Never materialize Text Reasoning Trace or reasoning_image_*.
                        for batch in pf.iter_batches(batch_size=8, row_groups=[group], columns=COLUMNS):
                            for raw in batch.to_pylist():
                                row_index = cursor
                                cursor += 1
                                if row_index not in wanted:
                                    continue
                                pid = f"zebra_{task}_{source['sha256'][:24]}_{row_index:06d}"
                                answer = raw["Final Answer"]
                                if not isinstance(answer, str) or not answer.strip():
                                    raise ValueError(f"{pid}: missing Final Answer")
                                query = build_query(raw["Question"])
                                cell = raw["problem_image_1"]
                                payload = cell.get("bytes") if isinstance(cell, dict) else None
                                if not payload:
                                    raise ValueError(f"{pid}: problem_image_1 has no embedded bytes; use original HF Parquet")
                                with Image.open(io.BytesIO(payload)) as img:
                                    suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "GIF": ".gif",
                                              "BMP": ".bmp", "TIFF": ".tiff"}.get(img.format)
                                    if suffix is None:
                                        raise ValueError(f"{pid}: unsupported image format {img.format}")
                                    img.verify()
                                relative = Path("images") / task / (pid + suffix)
                                target = output / relative
                                target.parent.mkdir(parents=True, exist_ok=True)
                                target.write_bytes(payload)  # Preserve original pixels/encoding, no resize.
                                sample = {"pid": pid, "zebra_task": task, "question": raw["Question"],
                                          "text_input": query, "image_input": [relative.as_posix()],
                                          "image_sha256": hashlib.sha256(payload).hexdigest(),
                                          "original_final_answer": answer,
                                          "source": {"shard": Path(source["path"]).name,
                                                     "sha256": source["sha256"], "row": row_index}}
                                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                                written += 1
                    row_start = group_end
                offset += source["rows"]
            if written != count:
                raise ValueError(f"{task}: expected {count} selected rows, wrote {written}")
            counts[task] = written
    report = {"dataset": DATASET, "suggested_source_revision": DATASET_REVISION,
              "source_split": "train", "protocol": PROTOCOL, "ilvr_source_revision": ILVR_REVISION,
              "samples": sum(counts.values()), "by_task": counts, "available_by_task": available,
              "published_full_counts": {t: spec["rows"] for t, spec in TASKS.items()}, "sources": sources,
              "selection": {"max_samples_per_task": max_samples_per_task, "seed": seed,
                            "method": "all_available" if max_samples_per_task is None else "seeded_random_from_available_shards"},
              "test_sha256": file_hash(test), "ilvr_exact_subset_verified": False}
    json_write(output / "manifest.json", report)
    return report


def image_paths(sample, image_root):
    return [str(resolve_image(image_root, p)) for p in sample["image_input"]]


def load_samples(path, image_root=None, check_images=True, subset="all", verify_images=False):
    from PIL import Image
    if subset not in ("all", *TASKS):
        raise ValueError(f"Unknown subset: {subset}")
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != DATASET or manifest.get("protocol") != PROTOCOL:
        raise ValueError("Zebra preparation protocol mismatch; prepare the data again")
    if file_hash(path) != manifest["test_sha256"]:
        raise ValueError("Zebra TEST hash differs from preparation manifest")
    rows, seen, counts = [], set(), Counter()
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        try:
            sample = json.loads(text)
            task, pid = sample["zebra_task"], sample["pid"]
            if task not in TASKS or not isinstance(pid, str) or pid in seen:
                raise ValueError("Unknown task or duplicate/invalid pid")
            source = sample["source"]
            expected = f"zebra_{task}_{source['sha256'][:24]}_{source['row']:06d}"
            if pid != expected or sample["text_input"] != build_query(sample["question"]):
                raise ValueError("Prepared identity/question mismatch")
            if not isinstance(sample["original_final_answer"], str) or not sample["original_final_answer"].strip():
                raise ValueError("Missing gold answer")
            if not isinstance(sample["image_input"], list) or len(sample["image_input"]) != 1:
                raise ValueError("Expected exactly one problem image")
            seen.add(pid)
            counts[task] += 1
            if subset != "all" and task != subset:
                continue
            if check_images:
                for image_path in image_paths(sample, image_root or path.parent):
                    if verify_images:
                        if file_hash(image_path) != sample["image_sha256"]:
                            raise ValueError("Prepared image hash mismatch")
                        with Image.open(image_path) as image:
                            image.verify()
            rows.append(sample)
        except Exception as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    if dict(counts) != manifest["by_task"] or sum(counts.values()) != manifest["samples"]:
        raise ValueError("Prepared Zebra counts do not match the manifest")
    if not rows:
        raise ValueError(f"No samples for subset={subset}")
    return rows, {"dataset": DATASET, "samples": len(rows), "by_task": dict(Counter(s["zebra_task"] for s in rows)),
                  "subset": subset, "source_split": "train", "source_sha256": manifest["test_sha256"],
                  "manifest_sha256": file_hash(path.parent / "manifest.json"), "protocol": PROTOCOL,
                  "selection": manifest["selection"], "available_by_task": manifest["available_by_task"],
                  "ilvr_exact_subset_verified": False}


@lru_cache(maxsize=1)
def ilvr_rules():
    """Load only the actual repository's pure scoring helpers, without CUDA imports."""
    path = Path(__file__).resolve().parents[2] / "eval.py"
    functions = {"_strip_special_tokens", "_clean_answer_span", "extract_final_answer", "normalize_for_match"}
    constants = {"_yes_set", "_no_set"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)]
    module = ast.Module(body=nodes, type_ignores=[])
    digest = hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
    if digest != OFFICIAL_RULE_AST_SHA256:
        raise ValueError("Repository eval.py scoring rules differ from the audited ILVR source; review before scoring Zebra")
    namespace = {"re": re}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def score(sample, raw):
    rules = ilvr_rules()
    prediction = rules["extract_final_answer"](raw)
    gold = sample["original_final_answer"]
    normalized = rules["normalize_for_match"](prediction)
    normalized_gold = rules["normalize_for_match"](gold)
    return {"pid": sample["pid"], "zebra_task": sample["zebra_task"], "gold": gold,
            "prediction": prediction, "prediction_normalized": normalized, "gold_normalized": normalized_gold,
            "correct": normalized == normalized_gold, "scoring_status": "scored", "scorer": "ilvr_rule_exact_match"}


def summarize(records):
    def stats(items):
        known = [r for r in items if type(r.get("correct")) is bool]
        correct = sum(r["correct"] for r in known)
        return {"samples": len(items), "scored": len(known), "pending": len(items) - len(known), "correct": correct,
                "accuracy": correct / len(items) if items and len(known) == len(items) else None,
                "accuracy_on_scored": correct / len(known) if known else None}
    grouped = defaultdict(list)
    for row in records:
        grouped[row["zebra_task"]].append(row)
    by_task = {task: stats(rows) for task, rows in sorted(grouped.items())}
    return {"task": "zebra", **stats(records), "by_task": by_task,
            "macro_task_accuracy": (sum(v["accuracy"] for v in by_task.values()) / len(TASKS)
                                    if set(by_task) == set(TASKS) and all(v["accuracy"] is not None for v in by_task.values()) else None),
            "token_limit_count": sum(r.get("hit_token_limit", False) for r in records),
            "ilvr_exact_reproduction": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw_dir")
    source.add_argument("--parquet", nargs="+", help="Files or quoted glob patterns; retain task folders or specify --subset")
    prep.add_argument("--output_dir", required=True)
    prep.add_argument("--max_samples_per_task", type=int)
    prep.add_argument("--seed", type=int, default=42)
    check = commands.add_parser("validate")
    check.add_argument("--test_data_path", required=True)
    check.add_argument("--image_root", required=True)
    for command in (prep, check):
        command.add_argument("--subset", choices=["all", *TASKS], default="all")
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare(args.output_dir, args.raw_dir, args.parquet, args.subset, args.max_samples_per_task, args.seed)
    else:
        _, report = load_samples(args.test_data_path, args.image_root, subset=args.subset, verify_images=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
