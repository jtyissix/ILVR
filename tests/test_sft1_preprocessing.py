"""Real CPU image pipeline, spawned preprocessing, and incremental diagnostics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image
import torch

from src.sft1.config import WarmupConfig
from src.sft1.data import (WarmupCollator, _processor_identity, load_training_image,
                           normalize_sample, prepare_sample)
from src.sft1.preprocessing import exclusion_reason, prepare_dataset
from tests.test_sft1_data import real_qwen_processor


class PreprocessingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.processor = real_qwen_processor()
        self.processor.image_processor.max_pixels = 200704
        self.processor.image_processor.size["longest_edge"] = 200704
        self.model = {"model_type": "qwen2_5_vl", "vocab_size": len(self.processor.tokenizer),
                      "image_token_id": 12, "vision_start_token_id": 10, "vision_end_token_id": 11,
                      "vision_config": {"spatial_merge_size": 2}}
        self.source = {"name": "comt", "image_root": str(self.root), "plan_field": "thought_rewritten",
                       "answer_field": "original_final_answer", "allow_append_final_answer": True,
                       "require_structured_actions": True, "skip_missing_plan": True, "skip_empty_answer": True}
        self.row = {"text_input": "question", "image_input": ["question.png"], "original_final_answer": "7",
                    "thought_rewritten": [
                        {"type": "text", "content": [{"sub_type": "analysis", "content": "reason"},
                            {"sub_type": "action", "original_text": "look", "content": "SHORT"}]},
                        {"type": "latent", "helper_image": "helper.png"},
                        {"type": "text", "content": "answer 7"}]}
        Image.new("RGB", (448, 448), "red").save(self.root / "question.png")
        Image.new("RGB", (448, 448), "blue").save(self.root / "helper.png")

    def record(self, row=None, limit=4096, shrink=True):
        return prepare_sample(row or self.row, self.source, self.processor, self.model, limit,
                              shrink_overlong_images=shrink)

    def test_tiny_images_are_upscaled_consistently_without_editing_source(self):
        for size in ((20, 29), (27, 500), (1, 5)):
            with self.subTest(size=size):
                Image.new("RGB", size, "blue").save(self.root / "helper.png")
                fixed = load_training_image(self.root / "helper.png", self.processor.image_processor)
                self.assertGreaterEqual(min(fixed.size), 28)
                with Image.open(self.root / "helper.png") as original:
                    self.assertEqual(original.size, size)
                record = self.record()
                self.assertEqual(record["images"][1]["original_size"], list(size))
                collator = WarmupCollator(self.processor, self.model, {"processor_identity": _processor_identity(self.processor)})
                batch = collator([record])
                self.assertEqual(batch["visual"]["image_grid_thw"].tolist(), record["image_grid_thw"])
                self.assertTrue(torch.isfinite(batch["visual"]["pixel_values"]).all())

    def test_overlong_images_shrink_but_every_text_label_and_final_answer_remains(self):
        original = self.record()
        pixels_before = copy.deepcopy(self.processor.image_processor.size)
        visual = sum(t * h * w // 4 for t, h, w in original["image_grid_thw"])
        limit = original["length"] - visual // 2
        adjusted = self.record(limit=limit)
        self.assertLessEqual(adjusted["length"], limit)
        self.assertTrue(adjusted["image_budget_reduced"])
        self.assertEqual(adjusted["original_length"], original["length"])
        self.assertEqual([v for v in adjusted["labels"] if v != -100],
                         [v for v in original["labels"] if v != -100])
        self.assertEqual(adjusted["final_answer_text"], original["final_answer_text"])
        self.assertEqual(self.processor.image_processor.size, pixels_before)
        # The same image at two budgets is encoded twice and retains both grids.
        collator = WarmupCollator(self.processor, self.model, {"processor_identity": _processor_identity(self.processor)})
        batch = collator([original, adjusted])
        self.assertEqual(batch["visual"]["image_grid_thw"].tolist(),
                         original["image_grid_thw"] + adjusted["image_grid_thw"])
        with self.assertRaisesRegex(ValueError, "Overlength trajectory"):
            self.record(limit=limit, shrink=False)
        with self.assertRaisesRegex(ValueError, "Overlength trajectory"):
            self.record(limit=3)

    def test_comt_field_selection_and_explicit_exclusions(self):
        self.assertIsNone(exclusion_reason(self.row, self.source))
        self.assertEqual(normalize_sample(self.row, self.source)["answer"], "7")
        old_mapping = {**self.source, "answer_field": "answer"}
        with self.assertRaisesRegex(ValueError, "answer_field='answer' is missing"):
            normalize_sample(self.row, old_mapping)
        unrewritten = {k: v for k, v in self.row.items() if k != "thought_rewritten"}
        self.assertIn("missing rewritten field", exclusion_reason(unrewritten, self.source))
        self.assertIn("empty authoritative answer", exclusion_reason({**self.row, "original_final_answer": ""}, self.source))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            normalize_sample({**self.row, "original_final_answer": ""}, self.source)

    def prepare_config(self, rows):
        model_path = self.root / "model"
        model_path.mkdir()
        self.processor.save_pretrained(model_path)
        (model_path / "config.json").write_text(json.dumps(self.model), encoding="utf-8")
        # Preprocessing reads weight inventory only; it never loads model tensors.
        (model_path / "model.safetensors").write_bytes(b"preprocessing inventory fixture")
        source_path = self.root / "input.jsonl"
        source_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return WarmupConfig(model_path=str(model_path), sources=[{**self.source, "path": str(source_path)}],
                            prepared_dir=str(self.root / "serial"), prepare_workers=1, prepare_log_seconds=1,
                            max_image_pixels=200704, max_seq_length=100,
                            shrink_overlong_images=True, skip_overlong_samples=False)

    def test_overlength_skip_preserves_short_samples_and_logs_actual_length(self):
        short = {**self.row, "image_input": [],
                 "thought_rewritten": [{"type": "text", "content": "answer 7"}]}
        config = self.prepare_config([self.row, short])
        config.shrink_overlong_images = False
        config.skip_overlong_samples = True
        manifest = prepare_dataset(config)
        directory = Path(config.prepared_dir)
        self.assertEqual(manifest["samples"], 1)
        self.assertEqual(manifest["skipped_samples"], 1)
        self.assertEqual(manifest["image_budget_reduced"], 0)
        self.assertTrue(manifest["skip_overlong_samples"])
        skipped = json.loads((directory / "skipped_records.jsonl").read_text())
        self.assertEqual(skipped["reason"], "overlength trajectory")
        self.assertEqual(skipped["line"], 1)
        self.assertGreater(skipped["length"], config.max_seq_length)
        self.assertEqual(skipped["max_seq_length"], config.max_seq_length)
        kept = json.loads((directory / "records.jsonl").read_text())
        self.assertEqual(kept["source_line"], 2)
        self.assertEqual(json.loads((directory / "validation_errors.json").read_text()), [])
        # Exclusion does not turn unrelated data errors into silent skips.
        from src.sft1.preprocessing import _process_one
        broken = copy.deepcopy(self.row)
        broken["thought_rewritten"][0]["content"][1].pop("original_text")
        result = _process_one((self.source, 3, broken), self.processor, config, self.model)
        self.assertEqual(result["status"], "error")
        self.assertIn("original_text", result["error"])

    def test_resize_precedes_skip_and_reports_every_image_size_and_token_count(self):
        short = {**self.row, "image_input": [],
                 "thought_rewritten": [{"type": "text", "content": "answer 7"}]}
        text_too_long = {**short, "thought_rewritten": [
            {"type": "text", "content": "reason " * 120 + "answer 7"}]}
        config = self.prepare_config([self.row, text_too_long, short])
        config.skip_overlong_samples = True
        original = self.record()
        manifest = prepare_dataset(config)
        directory = Path(config.prepared_dir)
        self.assertEqual(manifest["samples"], 2)
        self.assertEqual(manifest["skipped_samples"], 1)
        self.assertEqual(manifest["image_budget_reduced"], 1)
        records = [json.loads(s) for s in (directory / "records.jsonl").read_text().splitlines()]
        resized = records[0]
        self.assertEqual([v for v in resized["labels"] if v != -100],
                         [v for v in original["labels"] if v != -100])
        self.assertEqual(resized["final_answer_text"], original["final_answer_text"])
        self.assertLessEqual(resized["length"], config.max_seq_length)
        report = json.loads((directory / "resized_records.jsonl").read_text())
        self.assertEqual(report["line"], 1)
        self.assertEqual(report["source"], "comt")
        self.assertEqual(report["original_length"], original["length"])
        self.assertEqual(report["non_visual_tokens"] + report["visual_tokens"], report["length"])
        self.assertEqual(report["non_visual_tokens"] + report["original_visual_tokens"], report["original_length"])
        self.assertEqual(len(report["images"]), 2)
        for image in report["images"]:
            self.assertEqual(image["original_size"], [448, 448])
            self.assertEqual(image["initial_processed_size"], [448, 448])
            w, h = image["processed_size"]
            self.assertLess(w, 448)
            self.assertLess(h, 448)
            self.assertEqual(w * h // (28 * 28), image["visual_tokens"])
        skipped = json.loads((directory / "skipped_records.jsonl").read_text())
        self.assertEqual(skipped["line"], 2)
        self.assertEqual(skipped["reason"], "overlength trajectory")
        self.assertEqual(json.loads((directory / "validation_errors.json").read_text()), [])

    def test_spawned_workers_match_serial_records_and_report_skips(self):
        unrewritten = {k: v for k, v in self.row.items() if k != "thought_rewritten"}
        config = self.prepare_config([self.row, unrewritten, {**self.row, "original_final_answer": ""}, self.row])
        first = prepare_dataset(config)
        serial = Path(config.prepared_dir)
        config.prepare_workers = 2
        config.prepared_dir = str(self.root / "parallel")
        second = prepare_dataset(config)
        parallel = Path(config.prepared_dir)
        self.assertEqual(first["records_sha256"], second["records_sha256"])
        self.assertEqual((serial / "records.jsonl").read_bytes(), (parallel / "records.jsonl").read_bytes())
        self.assertEqual((serial / "resized_records.jsonl").read_bytes(),
                         (parallel / "resized_records.jsonl").read_bytes())
        self.assertEqual(second["samples"], 2)
        self.assertEqual(second["skipped_samples"], 2)
        self.assertEqual(second["image_budget_reduced"], 2)
        progress = json.loads((parallel / "progress.json").read_text())
        self.assertEqual(progress["status"], "complete")
        self.assertEqual(progress["processed"], 4)
        self.assertEqual(json.loads((parallel / "validation_errors.json").read_text()), [])
        self.assertEqual(len((parallel / "skipped_records.jsonl").read_text().splitlines()), 2)

    def test_errors_and_partial_records_exist_before_scan_finishes(self):
        broken = copy.deepcopy(self.row)
        broken["thought_rewritten"][0]["content"][1].pop("original_text")
        config = self.prepare_config([self.row, broken, self.row])
        from src.sft1 import preprocessing
        original_process = preprocessing._process_one
        visited = []
        def observe(task, *args):
            directory = Path(config.prepared_dir)
            self.assertTrue((directory / "progress.json").is_file())
            self.assertTrue((directory / "records.partial.jsonl").is_file())
            if task[1] == 3:
                self.assertIn("original_text", (directory / "validation_errors.jsonl").read_text())
            visited.append(task[1])
            return original_process(task, *args)
        with mock.patch.object(preprocessing, "_process_one", side_effect=observe):
            with self.assertRaisesRegex(ValueError, "Preparation failed"):
                prepare_dataset(config)
        self.assertEqual(visited, [1, 2, 3])
        directory = Path(config.prepared_dir)
        self.assertEqual(json.loads((directory / "progress.json").read_text())["status"], "failed")
        self.assertEqual(len((directory / "records.partial.jsonl").read_text().splitlines()), 2)
        self.assertFalse((directory / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
