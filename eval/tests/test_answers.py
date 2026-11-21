import copy
import json

import pytest

from eval.answers import compare_math, local_extract, reference_answer
from eval.data import record
from eval.evaluate import run_dataset
from eval.protocols import score_one


class Judge:
    def __init__(self, *replies):
        self.replies, self.calls = iter(replies), []

    def ask(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        try:
            return next(self.replies)
        except StopIteration:
            raise RuntimeError("API unavailable")


def row(name="mmstar", gold="A", choices=None, **meta):
    return record(name, "test", "Which result?", gold, {"image_1": "one.png"},
                  choices if choices is not None else {"A": "red", "B": "blue"}, **meta)


def score(sample, text, judge=None):
    return score_one(sample, {"text": text}, judge or Judge(), scoring_protocol="robust")


@pytest.mark.parametrize("text", [r"\boxed{A}", r"\boxed{\text{A}}", "The final answer is: A",
                                 "**Final answer:** (A).", "最终答案是：A", "Reasoning.\nAnswer: red", "A"])
def test_supported_final_formats(text):
    result = score(row(), text)
    assert result["score"] == 1 and result["extracted"] == "A"


@pytest.mark.parametrize("name", ["mmvp", "mmstar", "vstar", "hrbench_4k", "hrbench_8k", "scienceqa",
                                 "emma", "blink", "visualpuzzles", "visulogic", "seedbench"])
def test_same_extraction_on_all_mcq_datasets(name):
    assert score(row(name), "Reasoning. The final answer is: red")["score"] == 1
    assert score(row(name), "The final answer is: B")["score"] == 0


def test_final_wrong_answer_is_not_rescued_from_reasoning():
    judge = Judge()
    result = score(row(), r"An intermediate guess was \boxed{A}. The final answer is: B", judge)
    assert result["score"] == 0 and result["extracted"] == "B" and not judge.calls


def test_ambiguous_multiple_boxes_go_to_api_without_gold():
    text = r"I considered \boxed{A} and \boxed{B}, and am undecided."
    judge = Judge(json.dumps({"answer": None, "evidence": "am undecided"}))
    result = score(row(), text, judge)
    assert result["score"] == 0 and result["unparsed"] and result["api_fallback"]
    assert len(judge.calls) == 1


def test_api_extraction_is_reference_blind_and_quotes_evidence():
    text = "I select the reddish item."
    response = json.dumps({"answer": "A", "evidence": "the reddish item"})
    a, b = Judge(response), Judge(response)
    first, second = score(row(gold="A"), text, a), score(row(gold="B"), text, b)
    assert first["score"] == 1 and second["score"] == 0
    assert a.calls[0][0] == b.calls[0][0]
    assert {k: v for k, v in a.calls[0][1].items() if k != "response_validator"} == {
        k: v for k, v in b.calls[0][1].items() if k != "response_validator"}
    assert "reference" not in json.loads(a.calls[0][0][1].split("\n")[-1])


def test_api_cannot_invent_evidence_or_rewrite_math_answer():
    with pytest.raises(ValueError, match="evidence"):
        score(row(), "unclear", Judge(json.dumps({"answer": "A", "evidence": "not in response"})))
    with pytest.raises(ValueError, match="rewrote"):
        score(row("mm_math", "17", {}), "I think it comes to seventeen.",
              Judge(json.dumps({"answer": "17", "evidence": "seventeen"})))


@pytest.mark.parametrize("gold,text", [
    (r"\boxed{17}", "The final answer is: 17"),
    (r"\boxed{\frac{1}{2}}", "The final answer is: 0.5"),
    (r"\boxed{80^{\circ}}", "The final answer is: 80°"),
    (r"\boxed{2\sqrt{5}}", "The final answer is: 2√5"),
    (r"\boxed{17} and again \boxed{17}", "The final answer is: 17"),
])
def test_math_final_answer_formats(gold, text):
    result = score(row("mm_math", gold, {}), text)
    assert result["score"] == 1 and result["extracted"] is not None


@pytest.mark.parametrize("gold,prediction", [("1", "0"), ("-1", "0"), ("100", "1"), ("17", "3")])
def test_incorrect_math_is_not_promoted_by_percentage_tolerance(gold, prediction):
    judge = Judge()
    result = score(row("mm_math", gold, {}), "The final answer is: " + prediction, judge)
    assert result["score"] == 0 and not judge.calls


def test_multipart_reference_preserved_and_semantic_fallback_does_not_solve():
    gold = r"\boxed{(1,2)} and \boxed{(3,4)}"
    assert reference_answer(gold) == "(1,2), (3,4)"
    judge = Judge("NO")
    result = score(row("mm_math", gold, {}), "The final answer is: (1,2)", judge)
    assert result["score"] == 0 and result["method"] == "final_answer_semantic"
    assert json.loads(judge.calls[0][0][1].split("\n")[-1])["candidate_final_answer"] == "(1,2)"


def test_api_extraction_then_symbolic_comparison():
    judge = Judge(json.dumps({"answer": "17", "evidence": "17"}))
    result = score(row("mm_math", "17", {}), "It comes to a total of 17 altogether.", judge)
    assert result["score"] == 1 and result["extraction_method"] == "api_extract"


def test_empty_response_and_unknown_comparison():
    assert score(row(), "")["score"] == 0
    with pytest.raises(RuntimeError, match="pending"):
        score(row("emma", "a quadrilateral", {}), "Final answer: a four-sided polygon", Judge("UNKNOWN"))


def test_mathvista_preserves_original_numeric_normalization():
    sample = row("mathvista", "2.3", {}, question_type="free_form", answer_type="float", precision=1, options=[])
    result = score(sample, "The final answer is: 2.34")
    assert result["score"] == 1 and result["normalized"] == "2.3"
    sample["answer"] = "blue"
    sample["meta"].update(question_type="multi_choice", answer_type="text", options=["red", "blue"])
    assert score(sample, r"\boxed{B}")["score"] == 1


def test_failed_api_stays_pending_and_resumes_without_model(tmp_path):
    sample = row()
    class Backend:
        calls = 0
        def generate(self, messages):
            self.calls += 1
            return {"text": "I select the reddish item."}
    backend = Backend()
    params = dict(backend_factory=lambda: backend, mode="all", scoring_protocol="robust")
    first = run_dataset([sample], tmp_path, tmp_path / "out", "i", "s", judge=Judge(), **params)
    assert first["accuracy"] is None and first["pending"] == 1
    second = run_dataset([sample], tmp_path, tmp_path / "out", "i", "s",
                         judge=Judge(json.dumps({"answer": "A", "evidence": "the reddish item"})), **params)
    assert backend.calls == 1 and second["accuracy"] == 1
    assert second["api_fallback_count"] == 1 and second["protocol_modified"]


def test_switch_protocol_rescores_same_prediction(tmp_path):
    sample = row()
    class Backend:
        calls = 0
        def generate(self, messages):
            self.calls += 1
            return {"text": "Reasoning. The final answer is: A"}
    backend = Backend()
    params = dict(backend_factory=lambda: backend, judge=Judge())
    original = run_dataset([sample], tmp_path, tmp_path / "out", "i", "s", **params)
    revised = run_dataset([sample], tmp_path, tmp_path / "out", "i", "s", scoring_protocol="robust", **params)
    assert original["accuracy"] == 0 and revised["accuracy"] == 1 and backend.calls == 1
