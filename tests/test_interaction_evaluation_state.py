from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.interaction.evaluation_state import check_resume_config, load_rank_records, EvaluationProgress


class EvaluationStateTest(unittest.TestCase):
    def setUp(self):
        self.samples = [{"pid": f"p{i}", "subject": "Math"} for i in range(8)]

    def row(self, index):
        return {"index": index, **self.samples[index], "raw_output": "answer", "correct": None,
                "prompt_style": "emma_official_CoT", "segment_steps": [9], "token_ids": [90]}

    def load(self, path, rank_id=3, repair=False):
        return load_rank_records(path, self.samples, "emma", rank_id, 4, "emma_official_CoT", 9, repair)

    def test_null_score_is_complete_and_unterminated_tail_is_backed_up(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions_rank3.jsonl"
            good = (json.dumps(self.row(3)) + "\n").encode()
            tail = b'{"index":7,"raw_output":"unfinished'
            path.write_bytes(good + tail)
            with self.assertRaisesRegex(ValueError, "invalid prediction"):
                self.load(path)
            with mock.patch("builtins.print"):
                rows = self.load(path, repair=True)
            self.assertEqual([row["index"] for row in rows], [3])
            self.assertEqual(path.read_bytes(), good)
            self.assertEqual(next(Path(directory).glob("*.partial-*")).read_bytes(), tail)
            # A valid last line without LF is preserved and made safe for append.
            path.write_text(json.dumps(self.row(3)), encoding="utf-8")
            self.assertEqual(len(self.load(path, repair=True)), 1)
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_bad_interior_row_wrong_rank_duplicate_and_wrong_prompt_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions_rank3.jsonl"
            good = json.dumps(self.row(3)) + "\n"
            cases = [('{"unfinished"\n' + good, "invalid prediction"),
                     (good + good, "duplicate"), (json.dumps(self.row(2)), "different sample/rank"),
                     (json.dumps({**self.row(3), "pid": "different"}), "identity mismatch"),
                     (json.dumps({**self.row(3), "segment_steps": [8]}), "configuration mismatch"),
                     (json.dumps({**self.row(3), "prompt_style": "other"}), "configuration mismatch")]
            for payload, message in cases:
                path.write_text(payload, encoding="utf-8")
                before = path.read_bytes()
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    self.load(path, repair=True)
                self.assertEqual(path.read_bytes(), before)

    def test_resume_allows_monitoring_changes_but_rejects_generation_and_data_changes(self):
        old = dict(task="emma", model_path="model", world_size=4, latent_size=9, max_new_tokens=4096,
                   data={"source_sha256": "abc"}, prompt_style="emma_official_CoT")
        current = {**old, "resume": True, "sync_timeout_seconds": 14400, "coordination_backend": "gloo"}
        check_resume_config(old, current)
        for key, value in (("world_size", 2), ("model_path", "other"), ("max_new_tokens", 512),
                           ("data", {"source_sha256": "changed"})):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                check_resume_config(old, {**current, key: value})

    def test_evaluation_uses_cpu_group_with_explicit_timeout(self):
        from src.interaction import evaluate
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "4"}), mock.patch.object(evaluate.dist, "init_process_group") as init:
            evaluate.init_evaluation_group(14400)
            init.assert_called_once_with("gloo", timeout=timedelta(seconds=14400))
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}), mock.patch.object(evaluate.dist, "init_process_group") as init:
            evaluate.init_evaluation_group(14400)
            init.assert_not_called()

    def test_progress_reports_latest_phase_and_completion(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch("builtins.print"):
            progress = EvaluationProgress(directory, 3, 3, 100)
            progress.update("preprocessing", index=7, pid="p7")
            progress.update("decoding", generated_tokens=512, elapsed_seconds=31.0)
            progress.update("waiting_for_other_ranks", completed=100)
            row = json.loads((Path(directory) / "progress_rank3.json").read_text())
            self.assertEqual(row["pid"], "p7")
            self.assertEqual(row["completed"], 100)
            self.assertEqual(row["stage"], "waiting_for_other_ranks")

    def test_generation_monitor_does_not_change_tokens_or_latent_recurrence(self):
        import torch
        from torch import nn
        from src.interaction.generation import generate_continuous
        from tests.interaction_fixtures import TinyDecoder
        torch.set_num_threads(1)
        torch.manual_seed(51)
        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, x):
                self.calls += 1
                logits = x.new_full((1, 96), -100)
                logits[:, 70 if self.calls <= 2 else 90] = 100
                return logits
        decoder = TinyDecoder().eval()
        embeds = decoder.embed_tokens(torch.tensor([[1, 2]]))
        positions = torch.arange(2)[None, None].expand(3, 1, -1)
        for steps, budget in ((8, 64), (9, 64), (9, 4)):
            args = (decoder, Head(), embeds, positions, {"start": 70, "pad": 71, "end": 72}, {90}, steps, budget, "sdpa")
            expected = generate_continuous(*args)
            messages = []
            args = (decoder, Head(), *args[2:])
            with mock.patch("src.interaction.generation.time.perf_counter", side_effect=range(1000)):
                actual = generate_continuous(*args, progress_callback=lambda **row: messages.append(row), progress_interval_seconds=1)
            self.assertEqual(actual, expected)
            self.assertEqual(messages[0]["stage"], "prefill")
            self.assertEqual(messages[-1]["generated_tokens"], len(actual["token_ids"]))
            self.assertGreater(len(messages), 3)


if __name__ == "__main__":
    unittest.main()
