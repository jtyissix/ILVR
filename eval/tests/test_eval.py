import base64
import copy
import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from eval.common import (BENCHMARKS, read_json, select_benchmarks,
                         subset_samples, validate_samples, write_json)
from eval.data import ImageStore, extract_archive, normalize, record
from eval.evaluate import load_prepared, run_dataset
from eval.model import LatentPolicy, decode_stream
from eval.protocols import aggregate, messages_for, prompt_for, score_one


def image_bytes(color="red"):
    buf = io.BytesIO()
    Image.new("RGB", (56, 56), color).save(buf, "PNG")
    return buf.getvalue()


def sample(name="mmstar", uid="0", **meta):
    return record(name, uid, "Which?", "A", {"image_1": "one.png"}, {"A": "red", "B": "blue"}, **meta)


class NoJudge:
    def ask(self, *args, **kwargs):
        raise RuntimeError("Judge intentionally unavailable")


class StaticJudge:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def ask(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.answer


@pytest.mark.parametrize("only,exclude", [([], []), (["all"], ["mmvp"]), ("mmvp", [])])
def test_exclusive_selection(only, exclude):
    with pytest.raises(ValueError):
        select_benchmarks(only, exclude)


def test_aliases_and_all():
    assert select_benchmarks(["hrbench", "V*"], []) == ["vstar", "hrbench_4k", "hrbench_8k"]
    assert select_benchmarks([], ["hrbench"]) == [x for x in BENCHMARKS if not x.startswith("hrbench_")]


def test_pair_metric_and_incomplete():
    rows = [sample("mmvp", str(i)) for i in range(4)]
    for i, row in enumerate(rows):
        row["group"] = str(i // 2)
    scores = {str(i): {"score": int(i != 3)} for i in range(4)}
    result = aggregate(rows, scores)
    assert result["score"] == .5 and result["sample_accuracy"] == .75
    assert aggregate(rows, {"0": {"score": 1}})["score"] is None
    assert len(subset_samples(rows, 3)) == 2
    with pytest.raises(ValueError):
        subset_samples(rows, 1)


def test_hrbench_cycle_mean_is_not_all_correct():
    rows = [sample("hrbench_4k", str(i), cycle=str(i % 4)) for i in range(8)]
    for i, row in enumerate(rows):
        row["group"] = str(i // 4)
    scores = {str(i): {"score": int(i != 7)} for i in range(8)}
    assert aggregate(rows, scores)["score"] == .875


def test_blink_macro_not_micro():
    rows = [sample("blink", str(i)) for i in range(4)]
    rows[0]["category"] = "small"
    scores = {str(i): {"score": int(i == 0)} for i in range(4)}
    assert aggregate(rows, scores)["score"] == .5


def test_inline_images_repeat_and_no_gold(tmp_path):
    row = sample("emma", context="<image_2> first")
    row["question"] = "Compare <image_1> with <image_2>."
    row["images"]["image_2"] = "two.png"
    messages = messages_for(row, tmp_path)
    assert [Path(c["image"]).name for c in messages[0]["content"] if c["type"] == "image"] == ["two.png", "one.png", "two.png"]
    other = copy.deepcopy(row)
    other["answer"] = "B"
    assert messages_for(other, tmp_path) == messages
    other["question"] = "<image_9>"
    with pytest.raises(ValueError):
        messages_for(other, tmp_path)


def test_visualpuzzles_official_parser_and_seed():
    row = sample("visualpuzzles", image_options=True)
    row["choices"] = dict(zip("ABCD", "ABCD"))
    good = score_one(row, {"text": "Reasoning. Answer: (A)"}, NoJudge())
    assert good["score"] == 1 and not good["random_fallback"]
    a = score_one(row, {"text": "I cannot solve this puzzle"}, NoJudge())
    b = score_one(row, {"text": "I cannot solve this puzzle"}, NoJudge())
    assert a == b and a["random_fallback"]


def test_official_judges_and_blink_match():
    row = sample("mmvp", original_options="(a) red (b) blue")
    judge = StaticJudge("yes")
    assert score_one(row, {"text": "red"}, judge)["score"] == 1
    assert "correct answer is (a)" in judge.calls[0][0][1]
    with pytest.raises(RuntimeError):
        score_one(row, {"text": "red"}, NoJudge())
    row = sample("blink", prompt="Which? (A) red (B) blue")
    assert score_one(row, {"text": "The answer is (A) because red"}, NoJudge())["score"] == 1
    assert score_one(row, {"text": "Red seems right."}, StaticJudge("(A)"))["score"] == 1
    assert score_one(row, {"text": "No option."}, StaticJudge("(Z)"))["score"] == 0


def test_mmstar_preserves_original_prefix_rule():
    row = sample("mmstar")
    for text in ["A", "(A)", "Option A", "The answer is A."]:
        assert score_one(row, {"text": text}, NoJudge())["score"] == 1
    assert score_one(row, {"text": "Reasoning first. The answer is A."}, NoJudge())["score"] == 0


def test_scienceqa_official_extraction_and_random_fallback():
    row = sample("scienceqa")
    result = score_one(row, {"text": "The answer is A."}, NoJudge())
    assert result["score"] == 1 and not result["random_fallback"]
    a = score_one(row, {"text": "invalid response"}, NoJudge())
    b = score_one(row, {"text": "invalid response"}, NoJudge())
    assert a == b and a["random_fallback"]


def test_visulogic_official_extraction_and_fallback():
    row = sample("visulogic")
    assert score_one(row, {"text": r"Answer: \boxed{A}"}, NoJudge())["score"] == 1
    assert score_one(row, {"text": "the final answer is A"}, NoJudge())["score"] == 1
    assert score_one(row, {"text": "I choose the red item"}, StaticJudge("A"))["score"] == 1


def test_offline_download_module_produces_verified_artifact(tmp_path, monkeypatch):
    from eval.download import prepare_one
    from eval.registry import EXPECTED_COUNTS
    raw = tmp_path / "mmvp/raw"
    (raw / "MMVP Images").mkdir(parents=True)
    for i in (1, 2):
        (raw / f"MMVP Images/{i}.jpg").write_bytes(image_bytes())
    (raw / "Questions.csv").write_text('Index,Question,Options,Correct Answer\n1,Which?,(a) red (b) blue,(a)\n2,Which?,(a) red (b) blue,(b)\n')
    monkeypatch.setitem(EXPECTED_COUNTS, "mmvp", 2)
    manifest = prepare_one("mmvp", tmp_path, prepare_only=True)
    assert manifest["count"] == 2 and manifest["unique_images"] == 2
    assert prepare_one("mmvp", tmp_path, prepare_only=True) == manifest
    assert len(load_prepared(tmp_path, "mmvp")[2]) == 2


def test_mathvista_official_normalization():
    pytest.importorskip("Levenshtein")
    row = record("mathvista", "0", "Number?", "2.3", {"image_1": "one.png"},
                 query="Number?", options=[], question_type="free_form", answer_type="float", precision=1)
    assert score_one(row, {"text": "2.34"}, NoJudge())["score"] == 1
    judge = StaticJudge("2.34")
    assert score_one(row, {"text": "I calculate two and a bit."}, judge)["score"] == 1
    assert "Answer: 2.3" not in judge.calls[0][0][1]
    row["meta"].update(options=["red", "blue"], question_type="multi_choice", answer_type="text")
    row["answer"] = "blue"
    assert score_one(row, {"text": "The result is B"}, StaticJudge("B"))["score"] == 1
    empty = score_one(row, {"text": ""}, NoJudge())
    assert empty["extracted"] == ""  # Official empty-string handling must not hallucinate via an API.


def test_mm_math_symbolic():
    pytest.importorskip("antlr4")
    row = record("mm_math", "0", "Half?", r"\boxed{\frac{1}{2}}", {"image_1": "one.png"})
    assert score_one(row, {"text": r"Thus \boxed{0.5}"}, NoJudge())["score"] == 1
    assert score_one(row, {"text": r"Thus \boxed{0.8}"}, NoJudge())["score"] == 0
    assert score_one(row, {"text": "no answer"}, NoJudge())["score"] == 0


class StreamBackend:
    eos_ids = {0}

    def __init__(self, predictions):
        self.predictions = iter(predictions)
        self.feeds = []

    def next_token(self, state, banned):
        token = next(self.predictions)
        assert token not in banned
        return token

    def advance(self, state, token):
        self.feeds.append((state, token))
        return state + 1


def test_latents_use_hidden_recurrence_and_learned_end():
    backend = StreamBackend([5, 10, 77, 88, 11, 6, 10, 11, 11, 0])
    result = decode_stream(backend, 100, LatentPolicy(10, 11, 1, 64, True), 20)
    assert result["token_ids"] == [5, 10, 11, 6, 10, 11, 0]
    assert [event["steps"] for event in result["latent_segments"]] == [2, 1]
    assert backend.feeds[2:4] == [(102, None), (103, None)]
    assert result["stopped_on_eos"] and not result["forced_latent_ends"]


def test_forced_end_and_truncation_are_visible():
    backend = StreamBackend([10, 99, 99, 99, 0])
    result = decode_stream(backend, 0, LatentPolicy(10, 11, 1, 2, True), 10)
    assert result["token_ids"] == [10, 11, 0]
    assert result["forced_latent_ends"] == 1
    result = decode_stream(StreamBackend([10, 4]), 0, LatentPolicy(10, 11, 1, 2, True), 2)
    assert result["truncated"] and not result["latent_segments"][0]["closed"]


@pytest.mark.parametrize("entry", ["../escape", "/escape", "C:/escape", "foo:bar"])
def test_archive_traversal_rejected(tmp_path, entry):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(entry, "bad")
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "out")


def test_image_storage_converts_png_without_dedup(tmp_path):
    store = ImageStore(tmp_path, tmp_path / "prepared")
    buf = io.BytesIO()
    Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(buf, "PNG")
    a = store.save(buf.getvalue())
    b = store.save({"bytes": buf.getvalue(), "path": None})
    assert a != b
    assert a.endswith(".png") and b.endswith(".png")
    with Image.open(tmp_path / "prepared" / a) as image:
        assert image.format == "PNG"
        assert image.getpixel((0, 0)) == (255, 255, 255)


def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.mark.parametrize("name", BENCHMARKS)
def test_every_release_adapter(tmp_path, name):
    raw, prepared = tmp_path / "raw", tmp_path / "prepared"
    raw.mkdir()
    im = {"bytes": image_bytes(), "path": None}
    if name == "mmvp":
        (raw / "MMVP Images").mkdir()
        for i in (1, 2):
            (raw / f"MMVP Images/{i}.jpg").write_bytes(image_bytes())
        (raw / "Questions.csv").write_text('Index,Question,Options,Correct Answer\n1,Which?,(a) red (b) blue,(a)\n2,Which?,(a) red (b) blue,(b)\n')
    elif name == "vstar":
        (raw / "direct_attributes").mkdir()
        write_json(raw / "direct_attributes/one.json", {"question": "Which?", "options": ["red", "blue"]})
        (raw / "direct_attributes/one.jpg").write_bytes(image_bytes())
    elif name in ("mm_math", "visulogic", "seedbench"):
        (raw / "one.png").write_bytes(image_bytes())
        if name == "mm_math":
            (raw / "MM_Math").mkdir()
            write_json(raw / "MM_Math/MM_Math.jsonl", {"question": "Which?", "solution": r"\boxed{2}", "file_name": "one.png"})
            # Official JSONL is one complete object per line.
            path = raw / "MM_Math/MM_Math.jsonl"
            path.write_text(json.dumps(read_json(path)))
        elif name == "visulogic":
            (raw / "data.jsonl").write_text(json.dumps({"id": "1", "question": "Which?", "label": "A", "image_path": "one.png", "tag": "Spatial Reasoning"}))
        else:
            row = {"question_id": "1", "question": "Which?", "data_type": "image", "data_id": "one", "answer": "A", "question_type_id": 1,
                   **{f"choice_{k.lower()}": k for k in "ABCD"}}
            write_json(raw / "SEED-Bench.json", {"questions": [row, {**row, "data_type": "video"}]})
    elif name.startswith("hrbench_"):
        rows = []
        for i in range(4):
            choices = ["red", "blue", "white", "black"]
            rotated = choices[i:] + choices[:i]
            rows.append({"index": i, "question": "Which?", "answer": "ABCD"[rotated.index("red")], "category": "shape",
                         "cycle_category": str(i), **dict(zip("ABCD", rotated)),
                         "image": base64.b64encode(image_bytes()).decode() if i == 0 else "0"})
        write_parquet(raw / f"hr_bench_{name[-2:]}.parquet", rows)
    else:
        row = {"question": "Which?", "image": im, "answer": "A"}
        path = "data/train-test.parquet"
        if name == "mmstar":
            row.update(question="Which? Options: A: red, B: blue", index=0, category="math")
            path = "mmstar.parquet"
        elif name == "mathvista":
            row.update(pid="1", image="one.jpg", decoded_image=im, choices=["red", "blue"], query="Which?", question_type="multi_choice", answer_type="text", precision=None, answer="red", metadata={"task": "figure"})
            path = "data/testmini-0.parquet"
        elif name == "scienceqa":
            row.update(choices=["red", "blue"], answer=0, subject="science", hint="visible", lecture="GOLD_LEAK", solution="GOLD_LEAK")
            path = "data/test-0.parquet"
        elif name == "emma":
            row.update(pid="math_0", image_1=im, options=["red", "blue"], type="multiple choice", subject="Math", context="<image_1>")
            path = "Math/test-0.parquet"
        elif name == "blink":
            row.update(idx="val_Counting_0", image_1=im, choices=["red", "blue"], answer="(A)", prompt="Which? (A) red (B) blue")
            path = "Counting/val-0.parquet"
        elif name == "visualpuzzles":
            row.update(id=1, options=None, category="algorithmic", difficulty="hard")
        write_parquet(raw / path, [row])
    samples = normalize(name, raw, prepared)
    validate_samples(samples, name, prepared)
    assert len(samples) == (4 if name.startswith("hrbench_") else 2 if name == "mmvp" else 1)
    for row in samples:
        assert "GOLD_LEAK" not in json.dumps(messages_for(row, prepared))


def test_resume_retries_judge_without_rerunning_model(tmp_path):
    row = sample("mmvp", original_options="(a) red (b) blue")
    row["group"] = "0"
    other = copy.deepcopy(row)
    other["id"] = "1"
    rows = [row, other]
    class Backend:
        calls = 0
        def generate(self, messages):
            self.calls += 1
            return {"text": "A", "method": "generation"}
    backend = Backend()
    result = run_dataset(rows, tmp_path, tmp_path / "out", "infer1", "score1",
                         backend_factory=lambda: backend, judge=NoJudge())
    assert backend.calls == 2 and result["score"] is None and result["predictions"] == 2
    result = run_dataset(rows, tmp_path, tmp_path / "out", "infer1", "score1",
                         backend_factory=lambda: backend, judge=StaticJudge("yes"))
    assert backend.calls == 2 and result["score"] == 1
    result = run_dataset(rows, tmp_path, tmp_path / "out", "infer1", "different_judge",
                         backend_factory=lambda: pytest.fail("score mode must not load model"),
                         judge=StaticJudge("no"), mode="score")
    assert result["score"] == 0


def test_manifest_checks_structure_and_files(tmp_path):
    root = tmp_path / "mmstar/prepared"
    root.mkdir(parents=True)
    (root / "one.png").write_bytes(image_bytes())
    (root / "samples.jsonl").write_text(json.dumps(sample()))
    manifest = {"count": 1, "images": ["one.png"]}
    write_json(root / "manifest.json", manifest)
    assert len(load_prepared(tmp_path, "mmstar")[2]) == 1
    (root / "one.png").write_bytes(image_bytes("blue"))
    assert len(load_prepared(tmp_path, "mmstar")[2]) == 1
    (root / "one.png").unlink()
    assert len(load_prepared(tmp_path, "mmstar")[2]) == 1
    with pytest.raises(FileNotFoundError):
        load_prepared(tmp_path, "mmstar", validate=True)
