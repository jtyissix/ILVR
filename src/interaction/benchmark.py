"""Compare matched training runs after warmup using measured metrics.jsonl files."""
import argparse
import json
from pathlib import Path
import statistics
from .data import json_write


def summarize_run(path, warmup_steps):
    path = Path(path)
    with (path / "metrics.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    # Resume may append a replay of already recorded steps; the newest measurement wins.
    rows = sorted({r["step"]: r for r in rows if r["step"] > warmup_steps}.values(), key=lambda r: r["step"])
    if not rows:
        raise ValueError(f"No measurements after step {warmup_steps}: {path}")
    seconds = sum(r["step_seconds"] for r in rows)
    info = json.loads((path / "run_info.json").read_text(encoding="utf-8"))
    config = json.loads((path / "train_config.json").read_text(encoding="utf-8"))
    return {"directory": str(path), "measured_steps": [r["step"] for r in rows],
            "tokens_per_second": sum(r["effective_tokens"] for r in rows)/seconds,
            "samples_per_second": sum(r["samples"] for r in rows)/seconds,
            "median_step_seconds": statistics.median(r["step_seconds"] for r in rows),
            "peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows),
            "mean_reference_ms": statistics.mean(r["reference_ms"] for r in rows),
            "mean_student_ms": statistics.mean(r["student_ms"] for r in rows),
            "mean_backward_ms": statistics.mean(r["backward_ms"] for r in rows),
            "info": info, "config": config}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--output", default="outputs/benchmark_comparison.json")
    args = parser.parse_args()
    reports = [summarize_run(path, args.warmup_steps) for path in args.runs]
    first = reports[0]
    for report in reports[1:]:
        for field in ("effective_batch", "reference_fingerprint", "world_size", "dataset_sha256"):
            if report["info"][field] != first["info"][field]:
                raise ValueError(f"Unmatched benchmark control: {field}")
        for field in ("mode", "data_path", "prepared_dir", "seed", "max_seq_length", "epochs", "backbone_lr", "fusion_lr"):
            if report["config"][field] != first["config"][field]:
                raise ValueError(f"Unmatched benchmark control: {field}")
        if report["measured_steps"] != first["measured_steps"]:
            raise ValueError("Benchmark runs must measure the same optimizer steps")
    for report in reports:
        report["speedup_vs_first"] = report["tokens_per_second"] / first["tokens_per_second"]
        report.pop("config")
    json_write(args.output, reports)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
