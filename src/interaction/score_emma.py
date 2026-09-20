"""Offline EMMA scoring: official fast rules or an OpenAI-compatible local judge."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import emma
from .data import json_write, stable_hash


def load_fast_evaluator(repo):
    """Use the pinned official implementation, including symbolic equivalence."""
    emma.protocol(repo)
    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, Path(repo) / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    try:
        utils = load("_emma_official_utils", "evaluation/utils.py")
        previous = sys.modules.get("utils")
        sys.modules["utils"] = utils
        try:
            return load("_emma_official_evaluate", "evaluation/evaluate.py")
        finally:
            if previous is None:
                sys.modules.pop("utils", None)
            else:
                sys.modules["utils"] = previous
    except ImportError as exc:
        raise RuntimeError("Official fast scoring needs requirements-emma.txt; install it in the evaluation environment") from exc


def fast_score(sample, response, evaluator):
    if not response:
        return {"correct": False, "scoring_status": "scored", "extraction": "", "scorer": "official_fast"}
    answer = evaluator.fast_extract_answer(response)
    correct = evaluator.is_equal(answer, sample["answer"]) or evaluator.is_equal(answer, emma.gt_content(sample))
    return {"correct": bool(correct), "scoring_status": "scored", "extraction": answer, "scorer": "official_fast"}


def judge_prompt(template, sample, response):
    # Exact EMMA create_test_prompt contract. Only the scorer receives the gold.
    return f"{template.strip()}\nResponse: {response}\nAnswer: {sample['answer']}\nCorrect_or_not:"


def judge_score(sample, response, template, base_url, model, api_key=None,
                timeout=120, attempts=3, max_tokens=32):
    if not response:
        return {"correct": False, "scoring_status": "scored", "scorer": "qwen_judge", "judge_response": "",
                "reason": "empty_model_response"}
    prompt = judge_prompt(template, sample, response)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "top_p": 1, "max_tokens": max_tokens, "seed": 42, "stream": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    error, verdict = "No judge attempt completed", None
    for attempt in range(attempts):
        try:
            request = Request(base_url.rstrip("/") + "/chat/completions",
                              data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
            with urlopen(request, timeout=timeout) as handle:
                payload = json.load(handle)
            choice = payload["choices"][0]
            verdict = choice["message"]["content"]
            if choice.get("finish_reason") == "length":
                raise ValueError("Judge output hit its token limit")
            normalized = verdict.strip().lower()
            if normalized not in ("correct", "incorrect"):
                raise ValueError("Judge must return exactly Correct or Incorrect")
            return {"correct": normalized == "correct", "scoring_status": "scored", "scorer": "qwen_judge",
                    "judge_response": verdict, "judge_prompt_sha256": stable_hash(prompt),
                    "judge_returned_model": payload.get("model"), "judge_usage": payload.get("usage")}
        except HTTPError as exc:
            error = f"Judge HTTP {exc.code}"
            if exc.code not in (408, 429) and exc.code < 500:
                break
        except (URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < attempts:
            time.sleep(min(2 ** attempt, 4))
    # An API/format failure is NOT evidence that the evaluated model is wrong.
    return {"correct": None, "scoring_status": "judge_error", "scorer": "qwen_judge",
            "judge_response": verdict, "error": error}


def load_predictions(paths, samples, allow_partial=False):
    by_pid = {s["pid"]: (i, s) for i, s in enumerate(samples)}
    rows, seen, inputs = [], set(), []
    for path in paths:
        payload = Path(path).read_bytes()
        inputs.append({"path": str(Path(path).resolve()), "sha256": hashlib.sha256(payload).hexdigest()})
        for line, text in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not text.strip():
                continue
            row = json.loads(text)
            pid = row.get("pid")
            if pid not in by_pid or pid in seen:
                raise ValueError(f"{path}:{line}: unknown or duplicate pid {pid}")
            index, sample = by_pid[pid]
            if type(row.get("index")) is not int or row["index"] != index or row.get("subject") != sample["subject"]:
                raise ValueError(f"{path}:{line}: sample identity mismatch: {pid}")
            if not isinstance(row.get("raw_output"), str):
                raise ValueError(f"{path}:{line}: missing raw_output")
            # Derive the judged response and metadata from original raw output and
            # the authoritative dataset, never trust a hand-edited cached gold.
            rows.append({**row, **emma.pending_record(sample, row["raw_output"])})
            seen.add(pid)
    if not rows or (len(rows) != len(samples) and not allow_partial):
        raise ValueError(f"Found {len(rows)}/{len(samples)} predictions; use --allow_partial only for diagnostics")
    # Do not accidentally aggregate outputs of different inference configurations.
    sizes = {n for r in rows for n in r.get("segment_steps", [])}
    styles = {r.get("prompt_style") for r in rows}
    if len(sizes) > 1 or len(styles) > 1:
        raise ValueError("Mixed latent sizes or prompt styles; score each run separately")
    return sorted(rows, key=lambda r: r["index"]), inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--emma_repo", default="data/emma/official")
    parser.add_argument("--backend", choices=["judge", "fast"], default="judge")
    parser.add_argument("--judge_base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--judge_model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--api_key_env", default="EMMA_JUDGE_API_KEY")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--judge_max_tokens", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.timeout, args.attempts, args.judge_max_tokens) <= 0:
        parser.error("workers, timeout, attempts and judge_max_tokens must be positive")
    samples, data_report = emma.load_samples(args.test_data_path, check_images=False)
    rows, inputs = load_predictions(args.predictions, samples, args.allow_partial)
    _, template = emma.protocol(args.emma_repo)
    evaluator = load_fast_evaluator(args.emma_repo) if args.backend == "fast" else None
    settings = {"backend": args.backend, "inputs": inputs, "data": data_report,
                "emma_source_revision": emma.EMMA_REVISION, "protocol_files": emma.PROTOCOL_FILES,
                "judge_model": args.judge_model if args.backend == "judge" else None,
                "judge_base_url": args.judge_base_url if args.backend == "judge" else None,
                "judge_max_tokens": args.judge_max_tokens, "judge_temperature": 0, "judge_seed": 42}
    signature = stable_hash(settings)
    output, cached = Path(args.output_dir), {}
    if args.resume:
        previous = json.loads((output / "scoring_config.json").read_text(encoding="utf-8"))
        if previous["run_signature"] != signature:
            raise ValueError("Scoring inputs/settings changed; use a new output directory")
        cache_path = output / "scored.jsonl"
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    cached[record["pid"]] = record
        if set(cached) - {r["pid"] for r in rows}:
            raise ValueError("Resume output contains unknown prediction IDs")
    else:
        output.mkdir(parents=True, exist_ok=False)
        json_write(output / "scoring_config.json", {**settings, "run_signature": signature})
    by_pid = {s["pid"]: s for s in samples}
    pending = [r for r in rows if type(cached.get(r["pid"], {}).get("correct")) is not bool]
    def score_one(row):
        sample = by_pid[row["pid"]]
        if evaluator is not None:
            scored = fast_score(sample, row["response"], evaluator)
        else:
            scored = judge_score(sample, row["response"], template, args.judge_base_url, args.judge_model,
                                 os.environ.get(args.api_key_env), args.timeout, args.attempts, args.judge_max_tokens)
        return {**row, **scored}
    print(f"Scoring {len(pending)} remaining / {len(rows)} predictions; backend={args.backend}", flush=True)
    with (output / "scored.jsonl").open("a", encoding="utf-8") as handle:
        # Symbolic fast scoring uses global parser state in third-party libraries;
        # keep it sequential. Judge requests are independent and can be concurrent.
        with ThreadPoolExecutor(max_workers=args.workers if args.backend == "judge" else 1) as pool:
            futures = [pool.submit(score_one, row) for row in pending]
            for count, future in enumerate(as_completed(futures), 1):
                record = future.result()
                cached[record["pid"]] = record
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{count}/{len(pending)}] {record['pid']} {record['scoring_status']} correct={record['correct']}", flush=True)
    final = [cached[r["pid"]] for r in rows]
    (output / "predictions_scored.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final), encoding="utf-8")
    report = {**emma.summarize(final), "scoring": args.backend, "data": data_report,
              "complete_test_coverage": len(final) == len(samples), "run_signature": signature,
              "ilvr_exact_reproduction": False}
    json_write(output / "metrics.json", report)
    json_write(output / "emma_results.json", emma.official_results(samples, final))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["pending"]:
        raise SystemExit("Some judge calls failed; accuracy is null. Fix the service and rerun with --resume.")


if __name__ == "__main__":
    main()
