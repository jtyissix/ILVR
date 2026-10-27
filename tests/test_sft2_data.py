"""CPU checks for action compression, causal boundaries and mixed-batch masks."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from src.sft2.data import (Stage2Collator, Stage2Dataset, TOKENIZATION_POLICY, _processor_identity,
                           normalize_sample, prepare_sample)
from src.sft2.config import Stage2Config
from src.sft2.preprocessing import _process_one, prepare_dataset
from src.sft2.tokens import BOUNDARY_TOKENS, ensure_boundary_tokens
from tests import test_sft1_data as stage1_fixtures


class BoundaryTokenizer(stage1_fixtures.FakeTokenizer):
    special = {**stage1_fixtures.FakeTokenizer.special, BOUNDARY_TOKENS[0]: 6, BOUNDARY_TOKENS[1]: 7}
    all_special_ids = list(special.values())
    all_special_tokens = list(special)


class Stage2DataTest(unittest.TestCase):
    def setUp(self):
        fixture = stage1_fixtures.WarmupDataTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.row, self.source, self.processor = fixture.row, fixture.source, fixture.processor
        self.processor.tokenizer = BoundaryTokenizer()
        self.model = {**fixture.model, "hidden_size": 8}
        self.root = fixture.root

    def record(self, row=None):
        return prepare_sample(row or self.row, self.source, self.processor, self.model, 4096)

    def test_action_is_four_masked_slots_and_helper_has_no_image_token_span(self):
        record = self.record()
        action, segment = record["actions"][0], record["segments"][0]
        self.assertEqual(len(action["positions"]), 4)
        self.assertEqual(segment["visual_positions"], list(range(segment["position"] + 3, segment["position"] + 19)))
        slots = action["positions"] + list(range(segment["position"], segment["position"] + 19))
        start, end = action["positions"][0] - 1, segment["visual_positions"][-1] + 1
        self.assertEqual(slots, list(range(start + 1, end)))
        self.assertEqual(end - start + 1, 25)
        self.assertEqual([record["input_ids"][start], record["input_ids"][end]], [6, 7])
        self.assertEqual([record["labels"][start], record["labels"][end]], [6, 7])
        self.assertNotIn(start, segment["text"])
        self.assertNotIn(6, record["teacher_record"]["input_ids"])
        self.assertNotIn(7, record["teacher_record"]["input_ids"])
        self.assertTrue(all(record["labels"][p] == -100 for p in slots))
        self.assertTrue(all(record["input_ids"][p] == 0 for p in slots))
        self.assertTrue(set(action["positions"]).issubset(segment["text"]))
        self.assertEqual(record["input_ids"].count(self.model["image_token_id"]), 4)
        self.assertNotIn("token_start", record["images"][1])
        words = self.processor.tokenizer.decode(action["target_ids"])
        self.assertEqual(words, "SHORT PHRASE<|im_end|>")
        supervised = self.processor.tokenizer.decode([t for t in record["labels"] if t != -100])
        self.assertNotIn("I crop", supervised)
        self.assertNotIn("SHORT PHRASE", supervised)
        self.assertIn("final answer is: red", supervised)

    def test_teacher_boundary_consumes_exact_original_text_before_current_helper(self):
        record = self.record()
        action = record["actions"][0]
        teacher = record["teacher_record"]
        prefix = self.processor.tokenizer.decode(teacher["input_ids"][:action["teacher_position"] + 1])
        self.assertTrue(prefix.endswith("I crop the object."))
        self.assertLess(action["teacher_position"], teacher["segments"][0]["position"])
        self.assertEqual(teacher["input_ids"].count(self.model["image_token_id"]), 8)
        self.assertEqual(record["tokenization_policy"], TOKENIZATION_POLICY)
        self.assertEqual(record["record_id"], self.record()["record_id"])

    def test_multiple_actions_and_consecutive_helpers_preserve_order(self):
        row = copy.deepcopy(self.row)
        row["thought_rewritten"][0]["content"].append(
            {"sub_type": "action", "content": "ZOOM", "original_text": "Then I zoom."})
        row["thought_rewritten"].insert(2, {"type": "latent", "helper_image": "second.png"})
        record = self.record(row)
        self.assertEqual(len(record["actions"]), 2)
        self.assertEqual(len(record["segments"]), 2)
        self.assertEqual(record["segments"][1]["text"], [])
        memory = record["segments"][0]["text"]
        self.assertTrue(all(p in memory for a in record["actions"] for p in a["positions"]))
        self.assertLess(record["actions"][0]["teacher_position"], record["actions"][1]["teacher_position"])
        self.assertEqual([s["image_index"] for s in record["segments"]], [1, 2])
        self.assertEqual(record["segments"][1]["position"], record["segments"][0]["visual_positions"][-1] + 3)
        self.assertEqual(record["input_ids"].count(6), 3)
        self.assertEqual(record["input_ids"].count(7), 3)

    def test_missing_action_words_reserved_words_and_compressed_answer_are_rejected(self):
        for invalid in (None, "", "<|image_pad|>"):
            row = copy.deepcopy(self.row)
            row["thought_rewritten"][0]["content"][1]["content"] = invalid
            with self.assertRaises(ValueError):
                self.record(row)
        row = copy.deepcopy(self.row)
        row["thought_rewritten"][-1]["content"] = [
            {"sub_type": "action", "original_text": "The answer is red", "content": "ANSWER"}]
        with self.assertRaisesRegex(ValueError, "Final answer must remain"):
            normalize_sample(row, self.source)

    def test_mixed_batch_dummy_masks_all_four_losses_without_masking_latent_attention(self):
        record = self.record()
        collator = Stage2Collator(self.processor, self.model,
                                 {"processor_identity": _processor_identity(self.processor)}, "fixture")
        target = torch.arange(8).reshape(1, 8).float()
        teacher = SimpleNamespace(TeacherTargetReader=lambda directory: lambda record_id: target)
        with mock.patch.dict(sys.modules, {"src.sft2.teacher": teacher}):
            batch = collator([record, {**record, "dummy": True}])
        self.assertEqual(batch["sample_mask"].tolist(), [True, False])
        self.assertEqual(batch["teacher_targets"].shape, (2, 1, 8))
        self.assertEqual(batch["teacher_action_mask"].tolist(), [[True], [False]])
        self.assertEqual(batch["loss_counts"]["visual"], 16)
        self.assertEqual(batch["loss_counts"]["action_state"], 1)
        self.assertEqual(batch["loss_counts"]["action_words"], len(record["actions"][0]["target_ids"]))
        self.assertEqual(batch["loss_counts"]["ce"], sum(v != -100 for v in record["labels"][1:]))
        self.assertTrue(batch["attention_mask"].all())
        self.assertTrue(batch["labels"][1].eq(-100).all())
        self.assertEqual(batch["visual"]["image_grid_thw"].shape[0], 2)
        # Both rows retain all question/helper occurrence spans after deduping.
        self.assertEqual(batch["images"][1][-1]["feature_end"], 16)

    def test_no_action_record_and_dummy_only_batch_keep_zero_loss_counts(self):
        row = {**self.row, "image_input": [], "thought_rewritten": [
            {"type": "text", "content": "The final answer is: red", "is_final_answer": True}]}
        record = self.record(row)
        collator = Stage2Collator(self.processor, self.model,
                                 {"processor_identity": _processor_identity(self.processor)}, "fixture")
        teacher = SimpleNamespace(TeacherTargetReader=lambda directory: lambda record_id: torch.zeros(0, 8))
        with mock.patch.dict(sys.modules, {"src.sft2.teacher": teacher}):
            batch = collator([{**record, "dummy": True}])
        self.assertEqual(batch["teacher_targets"].shape, (1, 0, 8))
        self.assertEqual(batch["teacher_action_mask"].shape, (1, 0))
        self.assertEqual(batch["loss_counts"], dict(ce=0, visual=0, action_words=0, action_state=0))

    def test_duplicate_source_lines_have_unique_stable_teacher_cache_ids(self):
        config = Stage2Config(prepare_workers=1)
        one = _process_one((self.source, 1, self.row), self.processor, config, self.model)
        two = _process_one((self.source, 2, self.row), self.processor, config, self.model)
        again = _process_one((self.source, 1, self.row), self.processor, config, self.model)
        self.assertEqual(one["status"], "ok")
        self.assertEqual(two["status"], "ok")
        self.assertNotEqual(one["record"]["record_id"], two["record"]["record_id"])
        self.assertEqual(one["record"]["record_id"], again["record"]["record_id"])

    def test_real_qwen_processor_keeps_action_end_and_helper_feature_grids(self):
        processor = stage1_fixtures.real_qwen_processor()
        model = {"vocab_size": len(processor.tokenizer), "image_token_id": 12,
                 "vision_start_token_id": 10, "vision_end_token_id": 11,
                 "vision_config": {"spatial_merge_size": 2}, "hidden_size": 8}
        row = {"text_input": "question", "image_input": ["question.png"], "original_final_answer": "7",
               "thought_rewritten": [
                   {"type": "text", "content": [{"sub_type": "analysis", "content": "reason"},
                       {"sub_type": "action", "content": "SHORT", "original_text": "look"}]},
                   {"type": "latent", "helper_image": "helper.png"},
                   {"type": "text", "content": "answer 7"}]}
        record = prepare_sample(row, self.source, processor, model, 4096)
        action = record["actions"][0]
        self.assertEqual(record["teacher_record"]["input_ids"][action["teacher_position"]], 13)
        self.assertEqual(action["target_ids"][:-1], [14])
        self.assertEqual(action["target_ids"][-1], processor.tokenizer.eos_token_id)
        self.assertEqual(len(record["image_grid_thw"]), 2)
        self.assertEqual(record["input_ids"].count(12), record["images"][0]["visual_tokens"])

    def test_prepared_manifest_preserves_structure_and_rejects_corruption(self):
        source_path = self.root / "source.jsonl"
        source_path.write_text(json.dumps(self.row) + "\n" + json.dumps(self.row) + "\n", encoding="utf-8")
        config = Stage2Config(prepared_dir=str(self.root / "prepared"), prepare_workers=1,
                              sources=[{**self.source, "path": str(source_path)}])
        inventory = {"config": self.model, "metadata": {"fixture": True}}
        with mock.patch("src.sft2.preprocessing.checkpoint_inventory", return_value=inventory), \
                mock.patch("src.sft2.preprocessing.load_processor", return_value=self.processor):
            manifest = prepare_dataset(config)
        self.assertEqual(manifest["stage"], "sft2_latent")
        self.assertEqual(manifest["actions"], 2)
        self.assertEqual(manifest["helper_segments"], 2)
        self.assertEqual(manifest["tokenization_policy"], TOKENIZATION_POLICY)
        dataset = Stage2Dataset(config)
        self.assertEqual(len(dataset), 2)
        self.assertNotEqual(dataset.records[0]["record_id"], dataset.records[1]["record_id"])
        self.assertTrue(dataset[-1]["dummy"])
        manifest_path = Path(config.prepared_dir) / "manifest.json"
        manifest_path.write_text(json.dumps({**manifest, "version": 1}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "rebuild"):
            Stage2Dataset(config)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with mock.patch("src.sft1.data.checkpoint_inventory", return_value=inventory), \
                mock.patch("src.sft2.data.load_processor", return_value=self.processor):
            dataset.validate_sources(config)
        path = Path(config.prepared_dir) / "records.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            Stage2Dataset(config)


if __name__ == "__main__":
    unittest.main()
