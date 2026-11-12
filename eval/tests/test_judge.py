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
