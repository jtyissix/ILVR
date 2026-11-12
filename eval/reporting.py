"""Uniform per-question accuracy alongside each benchmark's original metric.

Reporting is separate from prompting/scoring so presentation changes do not
invalidate saved inference or per-question judge results.
"""
import csv
from pathlib import Path

from .common import write_json
from .protocols import aggregate as benchmark_aggregate


SUMMARY_COLUMNS = [
    "benchmark", "accuracy", "accuracy_percent", "correct", "incorrect",
    "scored", "expected", "pending", "coverage", "scoring_complete",
    "partial_accuracy", "partial_accuracy_percent",
    "metric", "score", "score_percent", "partial_score",
    "complete", "predictions", "diagnostic_subset", "full_dataset_count",
    "truncated_predictions", "output",
]


def aggregate(samples, scores):
    summary = benchmark_aggregate(samples, scores)
    values = [scores[s["id"]]["score"] for s in samples
              if s["id"] in scores and scores[s["id"]].get("score") is not None]
    if any(value not in (0, 1) for value in values):
        raise ValueError("Per-question accuracy requires binary correctness scores")
    correct, scored = sum(int(value) for value in values), len(values)
    pending = len(samples) - scored
    observed_accuracy = correct / scored if scored else None
    accuracy = observed_accuracy if pending == 0 else None
    partial_accuracy = observed_accuracy if pending else None
    summary.update(
        accuracy=accuracy,
        accuracy_percent=100 * accuracy if accuracy is not None else None,
        correct=correct, incorrect=scored - correct, pending=pending,
        coverage=scored / len(samples), scoring_complete=pending == 0,
        partial_accuracy=partial_accuracy,
        partial_accuracy_percent=100 * partial_accuracy if partial_accuracy is not None else None,
        # Backwards-compatible name, with an explicit separate partial field.
        sample_accuracy=accuracy,
        score_percent=100 * summary["score"] if summary["score"] is not None else None,
    )
    return summary


def write_summaries(root, summaries):
    root = Path(root)
    write_json(root / "summary.json", summaries)
    temporary = root / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    temporary.replace(root / "summary.csv")


def format_summary(summary):
    label = summary["benchmark"]
    if summary.get("diagnostic_subset"):
        label += " [diagnostic subset]"
    counts = f"correct={summary['correct']}/{summary['scored']}, scored={summary['scored']}/{summary['expected']}"
    if summary["accuracy_percent"] is not None:
        return (f"{label}: accuracy={summary['accuracy_percent']:.2f}% ({counts}); "
                f"{summary['metric']}={summary['score_percent']:.2f}%")
    partial = summary["partial_accuracy_percent"]
    observed = f"{partial:.2f}%" if partial is not None else "N/A"
    return f"{label}: accuracy=PENDING ({counts}, pending={summary['pending']}); partial_accuracy={observed}"
