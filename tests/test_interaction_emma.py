import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image
from src.interaction import emma, score_emma


CONFIG = {
    "multi_choice_format": '{context}\n{question}\n{options}Answer in \\boxed{{}}. ',
    "open_ended_format": '{context}\n{question}\nAnswer in \\boxed{{}}. ',
    "Strategy_Instruction": {"CoT": "Solve step by step.", "Directly": "Final answer only."},
}


def sample(subject="Math", identity=None):
    row = dict(pid=identity or f"{subject}_1", question="Compare <image_2> with <image_1>.",
               context=None, options=["<image_2>", "other"], answer="B", subject=subject,
               type="Multiple Choice", task="reason", category="category", source="fixture")
    row.update(text_input=emma.build_query(row, CONFIG), images={"image_1": "1.png", "image_2": "2.png"})
    return row


def prediction(row, index=0):
    return {"index": index, **emma.pending_record(row, r"\boxed{B}<|im_end|>"),
            "raw_output": r"\boxed{B}<|im_end|>", "segment_steps": [9], "prompt_style": "emma_official_CoT"}


class EMMATest(unittest.TestCase):
    def test_inline_images_repeat_order_and_no_gold_in_prompt(self):
        row = sample()
        row["answer"], row["solution"] = "SECRET_GOLD", "SECRET_SOLUTION"
        self.assertEqual(emma.referenced_images(row), ["image_2", "image_1", "image_2"])
        message = emma.user_message(row, ["2.png", "1.png", "2.png"])
        self.assertEqual([x["image"] for x in message["content"] if x["type"] == "image"], ["2.png", "1.png", "2.png"])
        text = json.dumps(message)
        self.assertNotIn("SECRET", text)
        self.assertNotIn("<image_", text)
        self.assertIn("None", emma.build_query(row, CONFIG))  # Official formatting preserves None.
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            emma.user_message(row, ["1.png"])
        del row["images"]["image_2"]
        with self.assertRaisesRegex(ValueError, "missing image"):
            emma.referenced_images(row)

    def test_parquet_roundtrip_counts_hash_and_alpha(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        buffer = io.BytesIO()
        Image.new("RGBA", (8, 8), (10, 20, 30, 0)).save(buffer, format="PNG")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subject in emma.SUBJECTS:
                raw = sample(subject)
                for key in ("text_input", "images"):
                    raw.pop(key)
                raw.update(image_1={"bytes": buffer.getvalue(), "path": None},
                           image_2={"bytes": buffer.getvalue(), "path": None}, solution="SECRET_SOLUTION")
                path = root / subject / "test-00000-of-00001.parquet"
                path.parent.mkdir()
                pq.write_table(pa.Table.from_pylist([raw]), path)
            with mock.patch.object(emma, "protocol", return_value=(CONFIG, "judge")):
                with self.assertRaisesRegex(ValueError, "Wrong mini dataset counts"):
                    emma.prepare(root, root / "wrong", root)
                self.assertFalse((root / "wrong").exists())
                fixture = {**emma.DATASETS["mini"], "counts": dict.fromkeys(emma.SUBJECTS, 1)}
                with mock.patch.dict(emma.DATASETS, {"mini": fixture}), mock.patch("builtins.print"):
                    out = root / "prepared"
                    emma.prepare(root, out, root)
                    rows, report = emma.load_samples(out / "TEST.jsonl", out)
                    self.assertEqual(report["samples"], 4)
                    self.assertNotIn("solution", rows[0])
                    with Image.open(emma.image_paths(rows[0], out)[0]) as picture:
                        self.assertEqual(picture.mode, "RGBA")
                    with self.assertRaises(FileExistsError):
                        emma.prepare(root, out, root)
                    with (out / "TEST.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write("\n")
                    with self.assertRaisesRegex(ValueError, "hash differs"):
                        emma.load_samples(out / "TEST.jsonl", out)

    def test_predictions_identity_coverage_and_raw_response(self):
        rows = [sample(identity="Math_1"), sample(identity="Math_2")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            records = [prediction(row, i) for i, row in enumerate(rows)]
            records[0]["response"] = "STALE_RESPONSE"
            def write(values):
                path.write_text("\n".join(json.dumps(r) for r in values), encoding="utf-8")
            write(records)
            parsed, _ = score_emma.load_predictions([path], rows)
            self.assertEqual(parsed[0]["response"], r"\boxed{B}")
            for values, error in ((records[:1], "1/2"), ([records[0]] * 2, "duplicate"),
                                  ([{**records[0], "index": 1}, records[1]], "identity mismatch"),
                                  ([records[0], {**records[1], "segment_steps": [8]}], "Mixed")):
                write(values)
                with self.assertRaisesRegex(ValueError, error):
                    score_emma.load_predictions([path], rows)
            write(records[:1])
            self.assertEqual(len(score_emma.load_predictions([path], rows, allow_partial=True)[0]), 1)

    def test_judge_contract_strict_verdict_and_failure_not_wrong(self):
        row = sample()
        for verdict, expected in (("Correct", True), ("Incorrect", False), ("probably Correct", None)):
            response = {"choices": [{"message": {"content": verdict}, "finish_reason": "stop"}], "model": "judge"}
            with mock.patch.object(score_emma, "urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as request:
                result = score_emma.judge_score(row, "response", "demo\n", "http://127.0.0.1:8000/v1", "judge", attempts=1)
            self.assertIs(result["correct"], expected)
            call = request.call_args.args[0]
            self.assertEqual(call.full_url, "http://127.0.0.1:8000/v1/chat/completions")
            payload = json.loads(call.data)
            self.assertEqual(payload["messages"], [{"role": "user", "content": "demo\nResponse: response\nAnswer: B\nCorrect_or_not:"}])
            self.assertEqual(payload["temperature"], 0)
        with mock.patch.object(score_emma, "urlopen", side_effect=TimeoutError("timeout")):
            result = score_emma.judge_score(row, "response", "demo", "http://localhost/v1", "judge", attempts=1)
            self.assertIsNone(result["correct"])
        with mock.patch.object(score_emma, "urlopen") as request:
            self.assertFalse(score_emma.judge_score(row, "", "demo", "http://localhost/v1", "judge")["correct"])
            request.assert_not_called()

    def test_metrics_pending_macro_and_multilabel_categories(self):
        records = [{**prediction(sample(s)), "correct": s == "Math"} for s in emma.SUBJECTS]
        records[1]["category"] = "one; two"
        records.append({**prediction(sample("Math", "Math_2")), "correct": True})
        report = emma.summarize(records)
        self.assertEqual(report["accuracy"], 2/5)
        self.assertEqual(report["macro_subject_accuracy"], 1/4)
        self.assertEqual(report["by_category"]["Coding/two"]["samples"], 1)
        records[-1]["correct"] = None
        report = emma.summarize(records)
        self.assertIsNone(report["accuracy"])
        self.assertIsNone(report["macro_subject_accuracy"])
        self.assertEqual(report["pending"], 1)

    def test_judge_resume_retries_only_failed_and_rejects_changed_input(self):
        rows = [sample(identity="Math_1"), sample(identity="Math_2")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "predictions.jsonl"
            path.write_text("\n".join(json.dumps(prediction(row, i)) for i, row in enumerate(rows)), encoding="utf-8")
            argv = ["score_emma", "--test_data_path", "unused", "--predictions", str(path),
                    "--output_dir", str(root / "scores"), "--workers", "1"]
            with mock.patch.object(emma, "load_samples", return_value=(rows, {})), \
                 mock.patch.object(emma, "protocol", return_value=(CONFIG, "demo")), \
                 mock.patch("builtins.print"), mock.patch("sys.argv", argv):
                with mock.patch.object(score_emma, "judge_score", side_effect=[
                    {"correct": True, "scoring_status": "scored"}, {"correct": None, "scoring_status": "judge_error"}
                ]), self.assertRaises(SystemExit):
                    score_emma.main()
                metrics = json.loads((root / "scores/metrics.json").read_text())
                self.assertIsNone(metrics["accuracy"])
                with mock.patch("sys.argv", argv + ["--resume"]), mock.patch.object(score_emma, "judge_score", return_value={"correct": False, "scoring_status": "scored"}) as judge:
                    score_emma.main()
                    self.assertEqual(judge.call_count, 1)
                    metrics = json.loads((root / "scores/metrics.json").read_text())
                    self.assertEqual(metrics["accuracy"], 0.5)
                    with path.open("a") as handle:
                        handle.write("\n")
                    with self.assertRaisesRegex(ValueError, "inputs/settings changed"):
                        score_emma.main()

    def test_emma_generation_wiring_with_cpu_stubs(self):
        from contextlib import ExitStack
        from src.interaction import evaluate
        from transformers import AutoProcessor
        import torch
        row = sample()
        model = mock.MagicMock()
        model.eval.return_value = model.to.return_value = model
        model.config = SimpleNamespace(eos_token_id=90, latent_size=9, image_token_id=80, to_dict=lambda: {})
        model.visual.blocks = []
        processor = mock.MagicMock()
        processor.return_value = dict(input_ids=torch.tensor([[1, 80, 2]]),
                                      image_grid_thw=torch.tensor([[1, 2, 2]]), pixel_values=mock.MagicMock())
        processor.apply_chat_template.return_value = "Picture 1: rendered question"
        processor.tokenizer.decode.return_value = r"\boxed{B}<|im_end|>"
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            for name in ("1.png", "2.png"):
                Image.new("RGB", (4, 4)).save(root / name)
            stack.enter_context(mock.patch("sys.argv", ["evaluate", "--task", "emma", "--model_path", "unused",
                                "--test_data_path", "unused", "--image_root", str(root), "--output_dir", str(root / "out")]))
            stack.enter_context(mock.patch.dict("sys.modules", {"eval": None}))
            stack.enter_context(mock.patch.object(emma, "load_samples", return_value=([row], {"prompt_strategy": "CoT"})))
            stack.enter_context(mock.patch.object(emma, "vision_inputs", return_value=[Image.new("RGB", (4, 4))] * 3))
            for name in ("set_device", "synchronize"):
                stack.enter_context(mock.patch.object(evaluate.torch.cuda, name))
            for name in ("token_ids", "position_ids_for_images", "input_embeddings"):
                stack.enter_context(mock.patch.object(evaluate, name))
            stack.enter_context(mock.patch.object(evaluate, "load_full_model", return_value=model))
            stack.enter_context(mock.patch.object(AutoProcessor, "from_pretrained", return_value=processor))
            generate = stack.enter_context(mock.patch.object(evaluate, "generate_continuous", return_value={
                "token_ids": [1], "segment_steps": [9], "hit_token_limit": False}))
            stack.enter_context(mock.patch("builtins.print"))
            evaluate.main()
            self.assertEqual(generate.call_args.args[7], 4096)
            self.assertTrue(processor.apply_chat_template.call_args.kwargs["add_vision_id"])
            results = json.loads((root / "out/emma_responses.json").read_text())
            self.assertNotIn("true_false", results[row["pid"]])
            self.assertEqual(results[row["pid"]]["response"], r"\boxed{B}")
            self.assertIsNone(json.loads((root / "out/metrics.json").read_text())["accuracy"])


if __name__ == "__main__":
    unittest.main()
