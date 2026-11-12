import csv
import json

import pytest

from eval.common import BENCHMARKS
from eval.data import record
from eval.evaluate import run_dataset
from eval.protocols import score_one
from eval.reporting import aggregate, format_summary, write_summaries


def rows_for(name):
    rows = [record(name, str(i), "Which?", "A", {"image_1": "one.png"},
                   {"A": "red", "B": "blue"}, "small" if i == 0 else "large",
                   cycle=str(i)) for i in range(4)]
    if name == "mmvp":
        for i, row in enumerate(rows):
            row["group"] = str(i // 2)
    elif name.startswith("hrbench_"):
        for row in rows:
            row["group"] = "0"
    return rows


@pytest.mark.parametrize("name", BENCHMARKS)
@pytest.mark.parametrize("state", ["complete", "partial", "empty", "all_wrong"])
def test_every_benchmark_has_unambiguous_accuracy(name, state):
    scores = {str(i): {"score": int(i != 3)} for i in range(4)}
    if state == "partial":
        scores.pop("1")
        scores["2"] = {"status": "error", "score": None}
    elif state == "empty":
        scores = {}
    elif state == "all_wrong":
        scores = {str(i): {"score": 0} for i in range(4)}
    result = aggregate(rows_for(name), scores)
    if state in ("complete", "all_wrong"):
        expected = .75 if state == "complete" else 0
        assert result["accuracy"] == result["sample_accuracy"] == expected
        assert result["accuracy_percent"] == expected * 100
        assert result["correct"] == expected * 4
        assert result["incorrect"] == 4 - expected * 4
        assert result["pending"] == 0 and result["scoring_complete"]
        assert result["partial_accuracy"] is None
    else:
        assert result["accuracy"] is result["accuracy_percent"] is result["sample_accuracy"] is None
        assert result["score"] is None and not result["scoring_complete"]
        if state == "partial":
            assert result["partial_accuracy"] == .5
            assert result["partial_accuracy_percent"] == 50
            assert result["correct"] == result["incorrect"] == 1
            assert result["pending"] == 2 and result["coverage"] == .5
        else:
            assert result["partial_accuracy"] is None and result["pending"] == 4
            assert result["correct"] == result["incorrect"] == 0
        assert "accuracy=PENDING" in format_summary(result)


def test_original_pair_and_macro_metrics_remain_separate():
    scores = {str(i): {"score": int(i != 3)} for i in range(4)}
    result = aggregate(rows_for("mmvp"), scores)
    assert result["accuracy_percent"] == 75 and result["score_percent"] == 50
    assert result["metric"] == "pair_accuracy"
    assert "accuracy=75.00%" in format_summary(result)
    assert "pair_accuracy=50.00%" in format_summary(result)
    result = aggregate(rows_for("blink"), {str(i): {"score": int(i == 0)} for i in range(4)})
    assert result["accuracy_percent"] == 25 and result["score_percent"] == 50


def test_json_and_csv_export_accuracy_for_every_dataset(tmp_path):
    summaries = [aggregate(rows_for(name), {str(i): {"score": int(i != 3)} for i in range(4)})
                 for name in BENCHMARKS]
    write_summaries(tmp_path, summaries)
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summaries
    with (tmp_path / "summary.csv").open(encoding="utf-8-sig", newline="") as f:
        exported = list(csv.DictReader(f))
    assert [row["benchmark"] for row in exported] == list(BENCHMARKS)
    for row in exported:
        assert float(row["accuracy_percent"]) == 75
        assert row["correct"] == "3" and row["scored"] == row["expected"] == "4"
        assert row["pending"] == "0"
    write_summaries(tmp_path, [aggregate(rows_for("mmvp"), {"0": {"score": 1}})])
    with (tmp_path / "summary.csv").open(encoding="utf-8-sig", newline="") as f:
        partial = next(csv.DictReader(f))
    assert partial["accuracy_percent"] == "" and partial["score"] == ""
    assert float(partial["partial_accuracy_percent"]) == 100


@pytest.mark.parametrize("name", ["vstar", "seedbench"])
def test_likelihood_results_report_correctness_not_nll(name):
    rows = rows_for(name)
    scores = {s["id"]: score_one(s, {"method": "likelihood", "selected": "A" if i < 3 else "B",
                                    "nll": {"A": 0.37, "B": 0.48}}, None)
              for i, s in enumerate(rows)}
    assert aggregate(rows, scores)["accuracy_percent"] == 75


def test_inference_only_has_no_accuracy_even_when_inference_complete(tmp_path):
    class Backend:
        def generate(self, messages):
            return {"text": "A", "method": "generation"}
    result = run_dataset(rows_for("mmstar"), tmp_path, tmp_path / "out", "infer", "score",
                         backend_factory=Backend, judge=None, mode="infer")
    assert result["complete"] and result["status"] == "inference_only"
    assert not result["scoring_complete"] and result["accuracy"] is None
    saved = json.loads((tmp_path / "out/summary_score.json").read_text(encoding="utf-8"))
    assert saved["pending"] == 4 and saved["accuracy_percent"] is None
