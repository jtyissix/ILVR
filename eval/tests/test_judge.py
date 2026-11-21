import pytest

from eval.judge import Judge


def test_bad_judge_response_retried_cache_and_key_not_persisted(tmp_path, monkeypatch):
    import requests
    answers = iter(["maybe", "yes"])
    calls = []
    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": next(answers)}}]}
    def post(url, **kwargs):
        calls.append(kwargs)
        return Response()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("eval.judge.time.sleep", lambda _: None)
    judge = Judge("SECRET_API_KEY", "https://example.invalid/v1", "judge", tmp_path)
    assert judge.ask("mmvp", "question", expected_pattern="yes|no") == "yes"
    assert len(calls) == 2
    cached = list(tmp_path.glob("*.json"))
    assert len(cached) == 1 and "SECRET_API_KEY" not in cached[0].read_text()
    judge.api_key = ""
    assert judge.ask("mmvp", "question", expected_pattern="yes|no") == "yes"
    restarted = Judge("", "https://example.invalid/v1", "judge", tmp_path)
    assert restarted.ask("mmvp", "question", expected_pattern="yes|no") == "yes"
    with pytest.raises(RuntimeError, match="requires API_KEY"):
        judge.ask("mmvp", "another question")


def test_structured_extraction_retries_bad_evidence_before_caching(tmp_path, monkeypatch):
    import json
    import requests
    from eval.answers import api_extract
    replies = iter([json.dumps({"answer": "A", "evidence": "invented"}),
                    json.dumps({"answer": "A", "evidence": "reddish item"})])
    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": next(replies)}}]}
    monkeypatch.setattr(requests, "post", lambda *a, **kw: Response())
    monkeypatch.setattr("eval.judge.time.sleep", lambda _: None)
    judge = Judge("test-key", "https://example.invalid", "test-model", tmp_path)
    result = api_extract("Which?", "I select the reddish item.", {"A": "red", "B": "blue"}, judge, "mmstar")
    assert result.answer == "A" and result.evidence == "reddish item"
    cached = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert "invented" not in cached["response"]
