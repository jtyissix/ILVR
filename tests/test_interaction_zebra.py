from contextlib import ExitStack
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from src.interaction import zebra, score_zebra
from src.interaction.data import file_hash, json_write, user_message
from src.interaction.evaluation_state import load_rank_records, check_resume_config


def raw_sample(index=0, task="jigsaw"):
    buffer = io.BytesIO()
    Image.new("RGBA", (4, 4), (index % 255, 30, 40, 0)).save(buffer, "PNG")
    return {"Question": f"Question {index} for {task}? <image_start>[problem_image_1]<image_end>",
            "Final Answer": "C" if task == "jigsaw" else "12/02",
            "problem_image_1": {"bytes": buffer.getvalue(), "path": "upstream.png"},
            "Text Reasoning Trace": "SECRET_REFERENCE_TRACE",
            "reasoning_image_1": {"bytes": b"INVALID_HELPER_IMAGE_DO_NOT_DECODE", "path": "helper.png"}}


def write_shard(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    return path


def fixture(root, count=3, tasks=None):
    for task in tasks or zebra.TASKS:
        write_shard(root / zebra.TASKS[task]["directory"] / "train-00000-of-00026.parquet",
                    [raw_sample(i, task) for i in range(count)])


def prediction(sample, index):
    raw = "The final answer is: " + sample["original_final_answer"] + "<|im_end|>"
    return {"index": index, **zebra.score(sample, raw), "raw_output": raw,
            "token_ids": [1, 2], "segment_steps": [9], "prompt_style": "ilvr_eval", "hit_token_limit": False}


class ZebraTest(unittest.TestCase):
    def prepare(self, output, **kwargs):
        with mock.patch("builtins.print"):
            return zebra.prepare(output, **kwargs)

    def test_full_snapshot_ignores_other_tasks_and_never_reads_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "raw")
            unrelated = root / "raw/Scientific Reasoning/invalid.parquet"
            unrelated.parent.mkdir()
            unrelated.write_bytes(b"NOT EVEN PARQUET")
            original = pq.ParquetFile.iter_batches
            columns = []
            def batches(pf, *args, **kwargs):
                columns.append(kwargs.get("columns"))
                return original(pf, *args, **kwargs)
            with mock.patch.object(pq.ParquetFile, "iter_batches", batches):
                report = self.prepare(root / "prepared", raw_dir=root / "raw")
            self.assertTrue(columns and all(c == zebra.COLUMNS for c in columns))
            self.assertEqual(report["by_task"], {"jigsaw": 3, "visual_search": 3})
            rows, _ = zebra.load_samples(root / "prepared/TEST.jsonl", verify_images=True)
            self.assertEqual(len(rows), 6)
            self.assertEqual(len(list((root / "prepared/images").rglob("*.png"))), 6)
            row = rows[0]
            image = root / "prepared" / row["image_input"][0]
            self.assertEqual(image.read_bytes(), raw_sample()["problem_image_1"]["bytes"])
            row["original_final_answer"] = "SECRET_ANSWER"
            rendered = json.dumps(user_message(row, [str(image)]))
            self.assertNotIn("SECRET", rendered)
            self.assertNotIn("<image_start>", rendered)
            self.assertNotIn("reasoning_image", rendered)
            self.assertEqual([x["type"] for x in user_message(row, [str(image)])["content"]], ["image", "text"])
            self.assertNotIn("SECRET_REFERENCE_TRACE", (root / "prepared/TEST.jsonl").read_text())
            # Metadata-only generation preflight vs explicit deep image validation.
            image.write_bytes(b"CORRUPTED")
            zebra.load_samples(root / "prepared/TEST.jsonl")
            with self.assertRaisesRegex(ValueError, "image hash mismatch"):
                zebra.load_samples(root / "prepared/TEST.jsonl", verify_images=True)

    def test_flat_files_need_task_and_duplicate_contents_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = write_shard(root / "a.parquet", [raw_sample()])
            with self.assertRaisesRegex(ValueError, "Cannot infer task"):
                zebra.discover_shards(parquet=[str(a)])
            report = self.prepare(root / "out", parquet=[str(a)], subset="jigsaw")
            self.assertEqual(report["samples"], 1)
            with self.assertRaises(FileExistsError):
                self.prepare(root / "out", parquet=[str(a)], subset="jigsaw")
            with self.assertRaisesRegex(ValueError, "Duplicate Parquet input"):
                zebra.discover_shards(parquet=[str(a), str(a)], subset="jigsaw")
            shutil.copyfile(a, root / "b.parquet")
            with self.assertRaisesRegex(ValueError, "Duplicate Parquet contents"):
                self.prepare(root / "dup", parquet=[str(root / "*.parquet")], subset="jigsaw")
            self.assertFalse((root / "dup").exists())
            with self.assertRaises(FileNotFoundError):
                zebra.discover_shards(parquet=[str(root / "absent*.parquet")], subset="jigsaw")

    def test_subset_sampling_stable_under_moves_renames_and_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = write_shard(root / "a.parquet", [raw_sample(i) for i in range(10)])
            b = write_shard(root / "b.parquet", [raw_sample(i) for i in range(10, 20)])
            shutil.copyfile(a, root / "renamed.parquet")
            for name, paths, seed in (("one", [a, b], 42), ("two", [b, root / "renamed.parquet"], 42), ("three", [a, b], 7)):
                self.prepare(root / name, parquet=list(map(str, paths)), subset="jigsaw", max_samples_per_task=5, seed=seed)
            datasets = [zebra.load_samples(root / name / "TEST.jsonl")[0] for name in ("one", "two", "three")]
            self.assertEqual([s["pid"] for s in datasets[0]], [s["pid"] for s in datasets[1]])
            self.assertNotEqual([s["pid"] for s in datasets[0]], [s["pid"] for s in datasets[2]])
            self.assertEqual(len(datasets[0]), 5)
            with self.assertRaisesRegex(ValueError, "No samples"):
                zebra.load_samples(root / "one/TEST.jsonl", subset="visual_search")

    def test_corrupt_schema_empty_gold_and_missing_problem_image_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (changes, error) in enumerate([
                ({"Final Answer": ""}, "missing Final Answer"),
                ({"problem_image_1": {"bytes": None, "path": "absent.png"}}, "embedded bytes"),
                ({"Question": "Question <image_start>[reasoning_image_1]<image_end>"}, "image marker"),
            ]):
                shard = write_shard(root / f"{index}.parquet", [{**raw_sample(), **changes}])
                with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                    self.prepare(root / f"out{index}", parquet=[str(shard)], subset="jigsaw")
            path = write_shard(root / "bad.parquet", [{"Question": "no other columns"}])
            with self.assertRaisesRegex(ValueError, "schema"):
                self.prepare(root / "bad_out", parquet=[str(path)], subset="jigsaw")

    def test_manifest_hash_counts_query_and_subset_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "raw")
            self.prepare(root / "out", raw_dir=root / "raw")
            path = root / "out/TEST.jsonl"
            samples, report = zebra.load_samples(path, subset="visual_search")
            self.assertEqual(len(samples), 3)
            self.assertEqual(report["by_task"], {"visual_search": 3})
            self.assertEqual(report["subset"], "visual_search")
            original = path.read_text()
            path.write_text(original + "\n")
            with self.assertRaisesRegex(ValueError, "TEST hash"):
                zebra.load_samples(path)
            records = [json.loads(line) for line in original.splitlines()]
            records[0]["text_input"] += "GOLD_LEAK"
            path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            manifest = json.loads((root / "out/manifest.json").read_text())
            json_write(root / "out/manifest.json", {**manifest, "test_sha256": file_hash(path)})
            with self.assertRaisesRegex(ValueError, "question mismatch"):
                zebra.load_samples(path)

    def test_actual_ilvr_rules_numeric_date_boxes_and_documented_quirks(self):
        self.assertEqual(zebra.ilvr_rules()["normalize_for_match"]("true"), "Yes")
        row = {"pid": "p", "zebra_task": "visual_search", "original_final_answer": "12/02"}
        self.assertTrue(zebra.score(row, "The final answer is: 12/02<|im_end|>")["correct"])
        self.assertFalse(zebra.score(row, "The final answer is: 02/12")["correct"])
        row["original_final_answer"] = "2"
        self.assertTrue(zebra.score(row, r"first \boxed{3}, finally \boxed{2.0}")["correct"])
        row["original_final_answer"] = "A knife."
        self.assertEqual(zebra.score(row, "A knife.")["prediction"], "a")
        self.assertFalse(zebra.score(row, "A knife.")["correct"])
        self.assertTrue(zebra.score(row, "The final answer is: A knife.")["correct"])
        row["original_final_answer"] = "C"
        self.assertTrue(zebra.score(row, "C")["correct"])
        self.assertFalse(zebra.score(row, "")["correct"])

    def test_metrics_macro_micro_and_pending(self):
        rows = [{"zebra_task": "jigsaw", "correct": True},
                {"zebra_task": "visual_search", "correct": False},
                {"zebra_task": "visual_search", "correct": True, "hit_token_limit": True}]
        metrics = zebra.summarize(rows)
        self.assertAlmostEqual(metrics["accuracy"], 2/3)
        self.assertEqual(metrics["macro_task_accuracy"], .75)
        self.assertEqual(metrics["token_limit_count"], 1)
        self.assertIsNone(zebra.summarize(rows[:1])["macro_task_accuracy"])
        rows[0]["correct"] = None
        self.assertIsNone(zebra.summarize(rows)["accuracy"])

    def test_offline_coverage_and_resume_identity_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "raw", count=1)
            self.prepare(root / "out", raw_dir=root / "raw")
            samples, _ = zebra.load_samples(root / "out/TEST.jsonl")
            records = [prediction(s, i) for i, s in enumerate(samples)]
            for i, r in enumerate(records):
                (root / f"rank{i}.jsonl").write_text(json.dumps(r) + "\n")
            paths = [str(root / "rank*.jsonl")]
            self.assertEqual(len(score_zebra.load_predictions(paths, samples)[0]), 2)
            with self.assertRaisesRegex(ValueError, "duplicate pid"):
                (root / "copy.jsonl").write_text(json.dumps(records[0]))
                score_zebra.load_predictions(paths + [str(root / "copy.jsonl")], samples)
            with self.assertRaisesRegex(ValueError, "1/2 predictions"):
                score_zebra.load_predictions([str(root / "rank0.jsonl")], samples)
            self.assertEqual(len(score_zebra.load_predictions([str(root / "rank0.jsonl")], samples, True)[0]), 1)
            load_rank_records(root / "rank0.jsonl", samples, "zebra", 0, 2, "ilvr_eval", 9)
            bad = {**records[0], "zebra_task": "visual_search"}
            (root / "rank0.jsonl").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "Zebra identity"):
                load_rank_records(root / "rank0.jsonl", samples, "zebra", 0, 2, "ilvr_eval", 9)
            with self.assertRaisesRegex(ValueError, "zebra_protocol"):
                check_resume_config({"zebra_protocol": "v1"}, {"zebra_protocol": "v2"})

    def test_offline_fast_cli_and_judge_retry_do_not_regenerate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "raw", count=1)
            self.prepare(root / "prepared", raw_dir=root / "raw")
            path = root / "prepared/TEST.jsonl"
            samples, _ = zebra.load_samples(path)
            predictions = root / "predictions.jsonl"
            predictions.write_text("\n".join(json.dumps(prediction(s, i)) for i, s in enumerate(samples)))
            argv = ["score_zebra", "--test_data_path", str(path), "--predictions", str(predictions)]
            with mock.patch("builtins.print"):
                with mock.patch("sys.argv", argv + ["--output_dir", str(root / "fast")]), \
                     mock.patch.object(score_zebra, "judge_score") as judge:
                    score_zebra.main()
                    judge.assert_not_called()
                self.assertEqual(json.loads((root / "fast/metrics.json").read_text())["accuracy"], 1.0)
                judge_args = argv + ["--output_dir", str(root / "judge"), "--backend", "judge", "--workers", "1"]
                with mock.patch("sys.argv", judge_args), mock.patch.object(score_zebra, "judge_score", side_effect=[
                    {"correct": True, "scoring_status": "scored"}, {"correct": None, "scoring_status": "judge_error"}
                ]) as judge, self.assertRaises(SystemExit):
                    score_zebra.main()
                self.assertEqual(judge.call_count, 2)
                self.assertIn(samples[0]["text_input"], judge.call_args_list[0].args[2])
                self.assertNotIn("SECRET_REFERENCE_TRACE", judge.call_args_list[0].args[2])
                self.assertIsNone(json.loads((root / "judge/metrics.json").read_text())["accuracy"])
                with mock.patch("sys.argv", judge_args + ["--resume"]), \
                     mock.patch.object(score_zebra, "judge_score", return_value={"correct": False, "scoring_status": "scored"}) as judge:
                    score_zebra.main()
                    self.assertEqual(judge.call_count, 1)
                self.assertEqual(json.loads((root / "judge/metrics.json").read_text())["accuracy"], .5)

    def test_generation_entrypoint_resume_and_both_latent_sizes(self):
        from src.interaction import evaluate
        from transformers import AutoProcessor
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "raw", count=1)
            self.prepare(root / "prepared", raw_dir=root / "raw")
            for latent_size in (8, 9):
                with ExitStack() as stack:
                    output = root / f"eval{latent_size}"
                    model = mock.MagicMock()
                    model.eval.return_value = model.to.return_value = model
                    model.config = SimpleNamespace(eos_token_id=90, latent_size=latent_size, image_token_id=80, to_dict=lambda: {})
                    model.visual.blocks = []
                    processor = mock.MagicMock()
                    processor.return_value = dict(input_ids=torch.tensor([[1, 80, 2]]),
                                                  image_grid_thw=torch.tensor([[1, 2, 2]]), pixel_values=mock.MagicMock())
                    processor.apply_chat_template.return_value = "image then question"
                    processor.tokenizer.decode.side_effect = ["The final answer is: C", "The final answer is: 12/02"]
                    argv = ["evaluate", "--task", "zebra", "--model_path", "unused", "--test_data_path",
                            str(root / "prepared/TEST.jsonl"), "--image_root", str(root / "prepared"), "--output_dir", str(output)]
                    stack.enter_context(mock.patch("sys.argv", argv))
                    stack.enter_context(mock.patch.dict("os.environ", {"WORLD_SIZE": "1", "LOCAL_RANK": "0"}))
                    for name in ("set_device", "synchronize"):
                        stack.enter_context(mock.patch.object(evaluate.torch.cuda, name))
                    for name in ("token_ids", "position_ids_for_images", "input_embeddings"):
                        stack.enter_context(mock.patch.object(evaluate, name))
                    stack.enter_context(mock.patch.object(evaluate, "load_full_model", return_value=model))
                    stack.enter_context(mock.patch.object(AutoProcessor, "from_pretrained", return_value=processor))
                    generate = stack.enter_context(mock.patch.object(evaluate, "generate_continuous", return_value={
                        "token_ids": [1], "segment_steps": [latent_size], "hit_token_limit": False}))
                    stack.enter_context(mock.patch("builtins.print"))
                    evaluate.main()
                    self.assertEqual(generate.call_args.args[6:8], (latent_size, 4096))
                    self.assertNotIn("add_vision_id", processor.apply_chat_template.call_args.kwargs)
                    self.assertEqual(json.loads((output / "metrics.json").read_text())["accuracy"], 1.0)
                    prompt = json.loads((output / "prompt_rank0.json").read_text())
                    self.assertEqual(prompt["messages"][0]["content"][0]["type"], "image")
                    # Simulate one completed answer and one interrupted JSONL write.
                    rank_file = output / "predictions_rank0.jsonl"
                    first = rank_file.read_text().splitlines()[0]
                    rank_file.write_text(first + '\n{"index":1,"raw_output":"')
                    processor.tokenizer.decode.side_effect = None
                    processor.tokenizer.decode.return_value = "The final answer is: 12/02"
                    with mock.patch("sys.argv", argv + ["--resume"]):
                        generate.reset_mock()
                        evaluate.main()
                        self.assertEqual(generate.call_count, 1)
                        generate.reset_mock()
                        evaluate.main()
                        generate.assert_not_called()
                    with self.assertRaisesRegex(ValueError, "Existing evaluation output"):
                        evaluate.main()


if __name__ == "__main__":
    unittest.main()
