import json
import shutil

import pytest

from eval import evaluate
from eval.common import read_json, write_json
from eval.data import record


@pytest.fixture
def inferred(tmp_path, monkeypatch):
    data = tmp_path / "data/mmstar/prepared"
    data.mkdir(parents=True)
    rows = [record("mmstar", str(i), "Which?", "A", {"image_1": "not_copied.png"},
                   {"A": "red", "B": "blue"}) for i in range(2)]
    (data / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in rows), encoding="utf-8")
    write_json(data / "manifest.json", {"count": 2, "images": ["not_copied.png"], "source_url": "fixture"})
    for key, value in {"MODE": "infer", "ONLY_DATASETS": ["mmstar"], "EXCLUDE_DATASETS": [],
                       "DATASET_DIR": str(tmp_path / "data"), "OUTPUT_DIR": str(tmp_path / "original"),
                       "RUN_NAME": "portable", "MAX_SAMPLES": None, "SCORING_PROTOCOL": "robust",
                       "SCORE_INFERENCE_IDS": {}, "RESUME": True, "RULES_ONLY": False}.items():
        monkeypatch.setattr(evaluate, key, value)
    monkeypatch.setattr(evaluate, "model_metadata", lambda _: {"path": "/original/model"})
    calls = []
    class Backend:
        runtime = {"fixture": True}
        def __init__(self, *a, **kw):
            pass
        def generate(self, messages):
            calls.append("inference")
            return {"text": "I select the reddish item."}
    class Judge:
        def __init__(self, *a, **kw):
            pass
        def ask(self, *a, **kw):
            assert evaluate.MODE == "score", "infer must never call an API"
            calls.append("api")
            return json.dumps({"answer": "A", "evidence": "the reddish item"})
    monkeypatch.setattr(evaluate, "QwenBackbone", Backend)
    monkeypatch.setattr(evaluate, "Judge", Judge)
    evaluate.main()
    assert calls == ["inference", "inference"]
    original = tmp_path / "original/portable"
    copied = tmp_path / "other_machine/portable"
    shutil.copytree(original, copied)
    monkeypatch.setattr(evaluate, "MODE", "score")
    monkeypatch.setattr(evaluate, "OUTPUT_DIR", str(copied.parent))
    monkeypatch.setattr(evaluate, "MODEL_DIR", "/not/copied/model")
    monkeypatch.setattr(evaluate, "DATASET_DIR", "/not/copied/data")
    monkeypatch.setattr(evaluate, "MAX_SAMPLES", 1)  # Use the saved denominator, not the new machine's limit.
    def forbidden(*a, **kw):
        pytest.fail("score must not read the model or dataset")
    monkeypatch.setattr(evaluate, "model_metadata", forbidden)
    monkeypatch.setattr(evaluate, "QwenBackbone", forbidden)
    monkeypatch.setattr(evaluate, "load_prepared", forbidden)
    destination = next((copied / "mmstar").glob("*/inference_config.json")).parent
    return copied, destination, calls


def test_copy_then_score_without_model_dataset_or_gpu(inferred, monkeypatch):
    root, destination, calls = inferred
    monkeypatch.setattr(evaluate, "RESUME", False)
    evaluate.main()
    summary = read_json(root / "summary.json")[0]
    assert summary["accuracy"] == 1 and summary["expected"] == 2
    assert calls == ["inference", "inference", "api", "api"]
    assert len(read_json(destination / "samples_snapshot.json")) == 2


def test_missing_prediction_remains_in_denominator(inferred):
    root, destination, _ = inferred
    next((destination / "predictions").glob("*.json")).unlink()
    with pytest.raises(SystemExit, match="Incomplete scoring"):
        evaluate.main()
    summary = read_json(root / "summary.json")[0]
    assert summary["expected"] == 2 and summary["scored"] == 1
    assert summary["pending"] == 1 and summary["accuracy"] is None


def test_complete_legacy_run_uses_embedded_samples(inferred):
    root, destination, _ = inferred
    (destination / "samples_snapshot.json").unlink()
    evaluate.main()
    assert read_json(root / "summary.json")[0]["accuracy"] == 1


def test_incomplete_legacy_run_requires_annotations(inferred):
    root, destination, _ = inferred
    (destination / "samples_snapshot.json").unlink()
    next((destination / "predictions").glob("*.json")).unlink()
    with pytest.raises(ValueError, match="denominator"):
        evaluate.load_scoring_run(root, "mmstar")


def test_multiple_saved_runs_require_explicit_selection(inferred):
    root, destination, _ = inferred
    other = destination.parent / "00000002"
    other.mkdir()
    write_json(other / "inference_config.json", read_json(destination / "inference_config.json"))
    with pytest.raises(ValueError, match="SCORE_INFERENCE_IDS"):
        evaluate.load_scoring_run(root, "mmstar")
    selected, _, _, _ = evaluate.load_scoring_run(root, "mmstar", destination.name)
    assert selected == destination
