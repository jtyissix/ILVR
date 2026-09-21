import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.interaction import emma, score_emma_deepseek as deepseek


def sample(identity="Math_1", answer="B"):
    return {"pid": identity, "question": "Which option matches <image_1>?", "context": "Use the diagram.",
            "options": ["wrong", "correct content", "other"], "answer": answer, "subject": "Math",
            "type": "Multiple Choice", "task": "reason", "category": "geometry", "source": "fixture",
            "text_input": "prepared query", "images": {"image_1": "1.png"}}


def prediction(row, index=0):
    raw = "I mention correct content, but my final answer is C.<|im_end|>"
    return {"index": index, **emma.pending_record(row, raw), "raw_output": raw,
            "segment_steps": [9], "prompt_style": "emma_official_CoT", "token_ids": [1]}


def api_response(verdict, finish_reason="stop"):
    return {"id": "request-1", "model": "deepseek-flash",
            "choices": [{"finish_reason": finish_reason,
                         "message": {"content": json.dumps(verdict)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 12, "total_tokens": 112,
                      "prompt_cache_hit_tokens": 80, "prompt_cache_miss_tokens": 20}}


class DeepSeekEMMAScoringTest(unittest.TestCase):
    def test_request_uses_current_model_json_mode_and_lenient_policy(self):
        row = sample()
        response = api_response({"correct": True, "reason": "Reference appears in the rationale",
                                 "matched_reference": "correct content"})
        with mock.patch.object(deepseek, "urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as send:
            result = deepseek.deepseek_score(row, "correct content appears, final C", "secret", attempts=1)
        self.assertTrue(result["correct"])
        self.assertEqual(result["judge_usage"]["total_tokens"], 112)
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer secret")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["temperature"], 0)
        supplied = json.loads(body["messages"][1]["content"])
        self.assertEqual(supplied["reference_answer"], "B")
        self.assertEqual(supplied["reference_answer_content"], "correct content")
        self.assertIn("different conclusion", body["messages"][0]["content"])
        self.assertNotIn("secret", json.dumps(body))

    def test_endpoint_empty_response_and_bad_api_output(self):
        self.assertEqual(deepseek.api_endpoint("https://api.deepseek.com/v1"),
                         "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(deepseek.api_endpoint("https://host/chat/completions"),
                         "https://host/chat/completions")
        with mock.patch.object(deepseek, "urlopen") as send:
            self.assertFalse(deepseek.deepseek_score(sample(), "", "secret")["correct"])
            send.assert_not_called()
        bad = api_response({"correct": "yes", "reason": "bad type", "matched_reference": "B"})
        with mock.patch.object(deepseek, "urlopen", return_value=io.BytesIO(json.dumps(bad).encode())):
            result = deepseek.deepseek_score(sample(), "B", "secret", attempts=1)
        self.assertIsNone(result["correct"])
        self.assertEqual(result["scoring_status"], "judge_error")
        cut = api_response({"correct": True, "reason": "", "matched_reference": "B"}, "length")
        with mock.patch.object(deepseek, "urlopen", return_value=io.BytesIO(json.dumps(cut).encode())):
            self.assertIsNone(deepseek.deepseek_score(sample(), "B", "secret", attempts=1)["correct"])

    def test_api_config_validation_precedence_and_environment_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api.json"
            path.write_text(json.dumps({"api_key": " file-key ", "base_url": "https://configured/v1",
                                        "model": "configured-model"}), encoding="utf-8")
            with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "env-key"}):
                self.assertEqual(deepseek.load_api_config(path),
                                 ("file-key", "https://configured/v1", "configured-model"))
                self.assertEqual(deepseek.load_api_config(path, base_url_override="https://override",
                                                          model_override="override-model"),
                                 ("file-key", "https://override", "override-model"))
                self.assertEqual(deepseek.load_api_config(Path(directory) / "missing.json"),
                                 ("env-key", deepseek.DEFAULT_BASE_URL, deepseek.DEFAULT_MODEL))
            path.write_text(json.dumps({"api_key": "key", "typo": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown.*typo"):
                deepseek.load_api_config(path)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                deepseek.load_api_config(path)

    def test_usage_summary_ignores_missing_and_noninteger_fields(self):
        report = deepseek.usage_summary([
            {"judge_usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
            {"judge_usage": {"prompt_tokens": "unknown", "total_tokens": 7}}, {},
        ])
        self.assertEqual(report["requests_with_usage"], 2)
        self.assertEqual(report["prompt_tokens"], 3)
        self.assertEqual(report["total_tokens"], 12)

    def test_cli_requires_key_and_resume_retries_only_api_failures(self):
        rows = [sample("Math_1"), sample("Math_2")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            old_scored = [{**prediction(row, i), "scorer": "old_judge", "judge_usage": {"total_tokens": 999},
                           "judge_reason": "stale", "correct": False} for i, row in enumerate(rows)]
            predictions.write_text("\n".join(json.dumps(record) for record in old_scored),
                                   encoding="utf-8")
            argv = ["score_emma_deepseek", "--test_data_path", "unused", "--predictions", str(predictions),
                    "--output_dir", str(root / "scores"), "--workers", "1",
                    "--api_config", str(root / "deepseek_api.json")]
            with mock.patch("sys.argv", argv), mock.patch.dict("os.environ", {}, clear=True), \
                 self.assertRaises(SystemExit):
                deepseek.main()
            api_config = root / "deepseek_api.json"
            api_config.write_text(json.dumps({"api_key": "never-save-this", "base_url": "https://api.deepseek.com",
                                              "model": "deepseek-flash"}), encoding="utf-8")
            with mock.patch.object(emma, "load_samples", return_value=(rows, {"samples": 2})), \
                 mock.patch("builtins.print"), mock.patch("sys.argv", argv), \
                 mock.patch.dict("os.environ", {}, clear=True), \
                 mock.patch.object(deepseek, "deepseek_score", side_effect=[
                     {"correct": True, "scoring_status": "scored", "judge_usage": {"total_tokens": 5}},
                     {"correct": None, "scoring_status": "judge_error"},
                 ]) as first_score, self.assertRaises(SystemExit):
                deepseek.main()
            self.assertEqual(first_score.call_args_list[0].args[2], "never-save-this")
            config_text = (root / "scores/scoring_config.json").read_text(encoding="utf-8")
            self.assertNotIn("never-save-this", config_text)
            self.assertNotIn(str(api_config), config_text)
            metrics = json.loads((root / "scores/metrics.json").read_text())
            self.assertIsNone(metrics["accuracy"])
            self.assertEqual(metrics["usage"]["total_tokens"], 5)
            # Rotating a credential does not change the scoring run signature.
            api_config.write_text(json.dumps({"api_key": "different-key", "base_url": "https://api.deepseek.com",
                                              "model": "deepseek-flash"}), encoding="utf-8")
            with mock.patch.object(emma, "load_samples", return_value=(rows, {"samples": 2})), \
                 mock.patch("builtins.print"), mock.patch("sys.argv", argv + ["--resume"]), \
                 mock.patch.dict("os.environ", {}, clear=True), \
                 mock.patch.object(deepseek, "deepseek_score", return_value={
                     "correct": False, "scoring_status": "scored", "judge_usage": {"total_tokens": 6}}) as score:
                deepseek.main()
            self.assertEqual(score.call_count, 1)
            self.assertEqual(score.call_args.args[2], "different-key")
            metrics = json.loads((root / "scores/metrics.json").read_text())
            self.assertEqual(metrics["accuracy"], .5)
            self.assertEqual(metrics["usage"]["total_tokens"], 11)
            scored = [json.loads(line) for line in (root / "scores/predictions_scored.jsonl").read_text().splitlines()]
            self.assertNotIn("stale", json.dumps(scored))
            self.assertNotIn("old_judge", json.dumps(scored))
            self.assertTrue((root / "scores/emma_results.json").is_file())


if __name__ == "__main__":
    unittest.main()
