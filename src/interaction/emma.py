"""Prepare official EMMA Parquet data and multimodal prompts; no CUDA required."""
import argparse
import ast
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import re

from .data import file_hash, json_write, resolve_image


EMMA_REVISION = "47259952dd8e95bc1e3855a4164ccd3634f8eb1e"
SUBJECTS = ("Chemistry", "Coding", "Math", "Physics")
DATASETS = {
    "mini": {"repo": "luckychao/EMMA-mini", "revision": "6b9ae9a74733bb57f0b741213f8d0ec9ebae067a",
             "counts": dict.fromkeys(SUBJECTS, 100)},
    "full": {"repo": "luckychao/EMMA", "revision": "6c87aec9048c489108170088e50b768646b5bcf9",
             "counts": {"Chemistry": 1176, "Coding": 564, "Math": 892, "Physics": 156}},
}
PROTOCOL_FILES = {
    "configs/gpt.yaml": "77197b693fab464bc392bd604857f99c5ec6130149e1e5a38f864e10bf3435d4",
    "evaluation/evaluate.py": "2b6fc041c91955a116f8a17e0954e4e701c178a83b0c1ac6a0031ea35020c6ff",
    "evaluation/utils.py": "dbf584bb47a568bfed8df2194ee5690e3c76c0daf5dc3b670672d6794eb94028",
}
IMAGE_MARKER = re.compile(r"<image_(\d+)>")
METADATA = ("pid", "question", "context", "options", "answer", "subject", "type", "task", "category", "source")


def protocol(repo):
    """Read pinned benchmark prompts without importing its optional grading stack."""
    import yaml
    root = Path(repo)
    for name, expected in PROTOCOL_FILES.items():
        path = root / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"EMMA official source mismatch: {path}; checkout {EMMA_REVISION}")
    config = yaml.safe_load((root / "configs/gpt.yaml").read_text(encoding="utf-8"))
    tree = ast.parse((root / "evaluation/utils.py").read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "score_demo_prompt" for t in node.targets))
    return config, ast.literal_eval(assignment.value)


def build_query(sample, config, strategy="CoT"):
    """Same formatting as EMMA data_utils.build_query; never read answer/solution."""
    if strategy not in ("CoT", "Direct"):
        raise ValueError(f"Unknown strategy: {strategy}")
    if sample["type"].lower() == "multiple choice":
        options = "".join(f"{chr(65 + i)}: {option}\n" for i, option in enumerate(sample["options"]))
        text = config["multi_choice_format"].format(context=sample["context"],
                                                     question=sample["question"], options=options)
    else:
        text = config["open_ended_format"].format(context=sample["context"], question=sample["question"])
    return text + config["Strategy_Instruction"]["CoT" if strategy == "CoT" else "Directly"]


def validate_record(sample):
    for key in METADATA:
        if key not in sample:
            raise ValueError(f"Missing EMMA field: {key}")
    if not isinstance(sample["pid"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sample["pid"]):
        raise ValueError("Invalid pid")
    if sample["subject"] not in SUBJECTS:
        raise ValueError(f"Unknown subject: {sample['subject']}")
    if not isinstance(sample["question"], str) or not sample["question"].strip():
        raise ValueError("Empty question")
    if not isinstance(sample["answer"], str) or not sample["answer"].strip():
        raise ValueError("Missing gold answer")
    if sample["type"].lower() not in ("multiple choice", "open-ended"):
        raise ValueError(f"Unknown question type: {sample['type']}")
    if sample["type"].lower() == "multiple choice":
        options = sample["options"]
        answer = sample["answer"].upper()
        if not isinstance(options, list) or not 2 <= len(options) <= 26 or not all(isinstance(x, str) for x in options):
            raise ValueError("Multiple-choice options must be a list of strings")
        if len(answer) != 1 or not 0 <= ord(answer) - 65 < len(options):
            raise ValueError("Gold option letter is outside the choices")


def gt_content(sample):
    return (sample["options"][ord(sample["answer"].upper()) - 65]
            if sample["type"].lower() == "multiple choice" else sample["answer"])


def referenced_images(sample):
    """Return fields in occurrence order, including repeats and option images."""
    fields = [f"image_{n}" for n in IMAGE_MARKER.findall(sample["text_input"])]
    for field in fields:
        if field not in sample.get("images", {}) or not sample["images"][field]:
            raise ValueError(f"{sample['pid']}: missing image referenced by <{field}>")
    return fields


def image_paths(sample, image_root):
    return [str(resolve_image(image_root, sample["images"][field])) for field in referenced_images(sample)]


def user_message(sample, paths):
    """Match official Qwen create_message; images occur at <image_n> markers."""
    fragments = IMAGE_MARKER.split(sample["text_input"])
    if len(paths) != (len(fragments) - 1) // 2:
        raise ValueError(f"{sample['pid']}: prompt/image count mismatch")
    content = []
    for i, fragment in enumerate(fragments[::2]):
        if fragment.strip():
            content.append({"type": "text", "text": fragment})
        if i < len(paths):
            content.append({"type": "image", "image": paths[i]})
    return {"role": "user", "content": content}


def vision_inputs(message):
    from qwen_vl_utils import process_vision_info
    images, videos = process_vision_info([message])
    if videos:
        raise ValueError("EMMA expects still images, not videos")
    return images or []


def prepare(raw_dir, output_dir, emma_repo, variant="mini", strategy="CoT"):
    import pyarrow.parquet as pq
    from PIL import Image, ImageOps
    config, _ = protocol(emma_repo)
    source, output = Path(raw_dir), Path(output_dir)
    shards, counts = {}, {}
    # Check expected counts before creating output or decoding any image.
    for subject in SUBJECTS:
        files = sorted((source / subject).glob("test-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No {subject}/test-*.parquet in {source}")
        counts[subject] = sum(pq.ParquetFile(p).metadata.num_rows for p in files)
        shards[subject] = files
    if counts != DATASETS[variant]["counts"]:
        raise ValueError(f"Wrong {variant} dataset counts: {counts}; expected {DATASETS[variant]['counts']}")
    output.mkdir(parents=True, exist_ok=False)
    records, identities = [], set()
    for subject, files in shards.items():
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
                for raw in batch.to_pylist():
                    sample = {key: raw.get(key) for key in METADATA}
                    validate_record(sample)
                    if sample["pid"] in identities or sample["subject"] != subject:
                        raise ValueError(f"Duplicate/mismatched pid: {sample['pid']}")
                    identities.add(sample["pid"])
                    sample.update(text_input=build_query(sample, config, strategy), images={},
                                  emma_variant=variant, prompt_strategy=strategy)
                    for number in range(1, 6):
                        field = f"image_{number}"
                        value = raw.get(field)
                        if value is None:
                            continue
                        if not isinstance(value, dict) or not value.get("bytes"):
                            raise ValueError(f"{sample['pid']}/{field}: expected image bytes in official Parquet")
                        relative = f"images/{subject}/{sample['pid']}/{field}.png"
                        target = output / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with Image.open(io.BytesIO(value["bytes"])) as image:
                            # HF Image decoding applies EXIF orientation. Keep alpha
                            # for qwen_vl_utils' official white-background conversion.
                            image = ImageOps.exif_transpose(image)
                            if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"):
                                image = image.convert("RGB")
                            image.save(target, format="PNG")
                        sample["images"][field] = relative
                    referenced_images(sample)  # Fail on missing option images, never omit them.
                    sample["gt_content"] = gt_content(sample)
                    records.append(sample)
        print(f"Prepared {subject}: {counts[subject]}", flush=True)
    test = output / "TEST.jsonl"
    test.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    report = {"schema_version": 1, "variant": variant, "dataset": DATASETS[variant], "split": "test",
              "samples": len(records), "by_subject": counts, "strategy": strategy,
              "emma_source_revision": EMMA_REVISION, "protocol_files": PROTOCOL_FILES,
              "prompt_config": config, "test_sha256": file_hash(test),
              "raw_files": {str(p.relative_to(source)): file_hash(p) for paths in shards.values() for p in paths},
              "ilvr_exact_subset_verified": False}
    json_write(output / "manifest.json", report)
    return report


def load_samples(path, image_root=None, check_images=True):
    from PIL import Image
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if file_hash(path) != manifest["test_sha256"]:
        raise ValueError("EMMA TEST hash differs from preparation manifest")
    if manifest.get("emma_source_revision") != EMMA_REVISION:
        raise ValueError("Prepared EMMA prompt source revision mismatch")
    rows, seen = [], set()
    for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        try:
            sample = json.loads(text)
            validate_record(sample)
            if sample["pid"] in seen:
                raise ValueError(f"Duplicate pid {sample['pid']}")
            seen.add(sample["pid"])
            if sample["text_input"] != build_query(sample, manifest["prompt_config"], manifest["strategy"]):
                raise ValueError("Prepared query differs from official prompt template")
            referenced_images(sample)
            if check_images:
                for image_path in set(image_paths(sample, image_root)):
                    with Image.open(image_path) as image:
                        image.verify()
            rows.append(sample)
        except Exception as exc:
            raise ValueError(f"{path}:{line}: {exc}") from exc
    counts = dict(Counter(s["subject"] for s in rows))
    if counts != DATASETS[manifest["variant"]]["counts"] or len(rows) != manifest["samples"]:
        raise ValueError(f"Prepared dataset has incorrect subject counts: {counts}")
    return rows, {"samples": len(rows), "by_subject": counts, "source_sha256": manifest["test_sha256"],
                  "variant": manifest["variant"], "dataset": manifest["dataset"],
                  "prompt_strategy": manifest["strategy"], "emma_source_revision": EMMA_REVISION,
                  "ilvr_exact_subset_verified": False}


def clean_response(text):
    return re.sub(r"<\|[^>]*\|>", "", text).strip()


def pending_record(sample, raw):
    return {"pid": sample["pid"], "subject": sample["subject"], "type": sample["type"],
            "category": sample["category"], "task": sample["task"], "response": clean_response(raw),
            "correct": None, "scoring_status": "pending"}


def official_results(samples, records):
    by_pid = {s["pid"]: s for s in samples}
    return {r["pid"]: {**{k: by_pid[r["pid"]][k] for k in METADATA},
                        "query": by_pid[r["pid"]]["text_input"], "gt_content": gt_content(by_pid[r["pid"]]),
                        "response": r["response"], **({"true_false": r["correct"]} if r["correct"] is not None else {})}
            for r in records}


def summarize(records):
    def stats(items):
        known = [r for r in items if type(r.get("correct")) is bool]
        correct = sum(r["correct"] for r in known)
        return {"samples": len(items), "scored": len(known), "pending": len(items) - len(known),
                "correct": correct, "accuracy": correct / len(items) if items and len(known) == len(items) else None,
                "accuracy_on_scored": correct / len(known) if known else None}
    grouped = {field: defaultdict(list) for field in ("subject", "type", "task", "category")}
    for row in records:
        for field, buckets in grouped.items():
            value = row.get(field) or "unknown"
            if field == "type":
                value = value.lower()
            values = value.split(";") if field == "category" and row["subject"] == "Coding" else [value]
            for name in values:
                key = f"{row['subject']}/{name.strip()}" if field in ("task", "category") else name
                buckets[key].append(row)
    result = {"task": "emma", **stats(records),
              **{f"by_{field}": {k: stats(v) for k, v in sorted(buckets.items())} for field, buckets in grouped.items()},
              "token_limit_count": sum(r.get("hit_token_limit", False) for r in records)}
    subjects = result["by_subject"]
    result["macro_subject_accuracy"] = (sum(v["accuracy"] for v in subjects.values()) / len(SUBJECTS)
        if set(subjects) == set(SUBJECTS) and all(v["accuracy"] is not None for v in subjects.values()) else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--raw_dir", required=True)
    prep.add_argument("--output_dir", required=True)
    prep.add_argument("--emma_repo", default="data/emma/official")
    prep.add_argument("--variant", choices=DATASETS, default="mini")
    prep.add_argument("--strategy", choices=["CoT", "Direct"], default="CoT")
    check = commands.add_parser("validate")
    check.add_argument("--test_data_path", required=True)
    check.add_argument("--image_root", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare(args.raw_dir, args.output_dir, args.emma_repo, args.variant, args.strategy)
    else:
        _, report = load_samples(args.test_data_path, args.image_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
