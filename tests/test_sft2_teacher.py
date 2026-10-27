"""Action endpoint causality and resumable safe teacher-cache integrity."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.sft1.model import WarmupModel
from src.sft2.teacher import (TeacherTargetReader, _extract_record, _load_teacher, load_teacher_targets, prepare_teacher_cache,
                             teacher_action_targets, validate_teacher_cache)
from tests.interaction_fixtures import TinyDecoder
from tests.test_sft1_model import example_batch, model_config, tiny_config


class _TeacherTokenizer:
    pad_token_id = 0
    eos_token_id = 90

    def __len__(self):
        return 96


class _TeacherProcessor:
    tokenizer = _TeacherTokenizer()
    image_processor = SimpleNamespace(to_dict=lambda: {"merge_size": 2})
    chat_template = "teacher-loader-test"


class TeacherEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(318)
        self.model = WarmupModel(TinyDecoder(), nn.Linear(24, 96, bias=False), model_config(),
                                 tiny_config(activation_checkpointing=False)).requires_grad_(False).eval()
        self.batch, self.features = example_batch([(1, [[3, 4], "image", [5, 6], "image", [7, 90]])])
        self.actions = [[{"teacher_position": segment["position"] - 1}
                         for segment in self.batch["segments"][0]]]

    def test_target_is_state_after_consuming_original_action_last_token(self):
        states = teacher_action_targets(self.model, self.batch, self.actions, self.features)[0]
        with torch.no_grad():
            full = self.model.trajectory(self.batch, self.features).hidden
        for index, action in enumerate(self.actions[0]):
            position = action["teacher_position"]
            torch.testing.assert_close(states[index], full[0, position])
            self.assertFalse(torch.allclose(states[index], full[0, position - 1]))
        changed = copy.deepcopy(self.batch)
        position = self.actions[0][0]["teacher_position"]
        changed["input_ids"][0, position] = 12
        updated = teacher_action_targets(self.model, changed, self.actions, self.features)[0]
        self.assertFalse(torch.allclose(states[0], updated[0]))
        self.assertFalse(states.requires_grad)
        self.assertFalse(self.model.training)

    def test_current_helper_cannot_change_its_action_but_earlier_helper_affects_next_action(self):
        states = teacher_action_targets(self.model, self.batch, self.actions, self.features)[0]
        changed = self.features.clone()
        helper = next(item for item in self.batch["images"][0] if item["kind"] == "helper")
        changed[helper["feature_start"]:helper["feature_end"]] += torch.randn(24) * 4
        updated = teacher_action_targets(self.model, self.batch, self.actions, changed)[0]
        torch.testing.assert_close(states[0], updated[0], rtol=0, atol=0)
        self.assertFalse(torch.allclose(states[1], updated[1]))

    def test_padding_image_or_duplicate_positions_are_rejected(self):
        duplicate = [[self.actions[0][0], self.actions[0][0]]]
        with self.assertRaises(ValueError):
            teacher_action_targets(self.model, self.batch, duplicate, self.features)
        image_position = self.batch["images"][0][0]["boundary_start"] + 1
        with self.assertRaises(ValueError):
            teacher_action_targets(self.model, self.batch, [[{"teacher_position": image_position}]], self.features)

    def test_loader_restores_frozen_complete_model_and_dict_collator_config(self):
        from safetensors.torch import save_file
        from src.sft1.modules import IMAGE_INDEX_ENCODING, POSITION_ENCODING
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = self.model.recipe
            config_values = {"hidden_size": 24, "image_token_id": 80, "vision_config": {"spatial_merge_size": 2}}
            self.model.model_config.to_dict = lambda: config_values
            full = SimpleNamespace(model=self.model.decoder, lm_head=self.model.lm_head,
                                   config=self.model.model_config, visual=nn.Linear(3, 24))
            (root / "warmup_recipe.json").write_text(json.dumps({
                "format": "sft1-warmup-v1", "bundle_type": "warmup_training_components",
                "config": recipe.to_dict()}), encoding="utf-8")
            (root / "interaction_config.json").write_text(json.dumps({
                "format": "sft1-interaction-v1", "position_encoding": POSITION_ENCODING,
                "image_index_encoding": IMAGE_INDEX_ENCODING,
                "kwargs": {"hidden_size": 24, "dim": recipe.fusion_dim, "heads": recipe.fusion_heads,
                           "ffn_dim": recipe.fusion_ffn_dim, "score_dim": recipe.fusion_score_dim,
                           "image_branch_enabled": recipe.image_branch_enabled, "slots": recipe.adapter_slots,
                           "interaction_tokens": recipe.interaction_tokens}}), encoding="utf-8")
            (root / "warmup_adapter_config.json").write_text(json.dumps({
                "format": "sft1-image-adapter-v1", "position_encoding": POSITION_ENCODING,
                "kwargs": {"hidden_size": 24, "slots": recipe.adapter_slots, "dim": recipe.fusion_dim,
                           "heads": recipe.fusion_heads, "ffn_dim": recipe.fusion_ffn_dim,
                           "layers": recipe.adapter_layers}}), encoding="utf-8")
            for name, module in (("interaction", self.model.interaction), ("warmup_adapter", self.model.warmup_adapter)):
                save_file({k: v.detach().clone() for k, v in module.state_dict().items()}, str(root / (name + ".safetensors")))
            configuration = SimpleNamespace(warmup_path=str(root), attention_backend="sdpa",
                                            min_image_pixels=3136, max_image_pixels=6272)
            with patch("torch.cuda.is_available", return_value=True), \
                    patch("torch.cuda.set_device") as set_device, \
                    patch("src.interaction.model.load_full_model", return_value=full), \
                    patch("src.sft1.data.load_processor", return_value=_TeacherProcessor()), \
                    patch.object(nn.Module, "to", autospec=True, side_effect=lambda module, *args, **kwargs: module):
                teacher, visual, collator, _device = _load_teacher(configuration, device_index=3)
            set_device.assert_called_once_with(3)
            self.assertEqual(_device, torch.device("cuda", 3))
            self.assertIsInstance(collator.config, dict)
            self.assertFalse(teacher.training)
            self.assertFalse(visual.training)
            self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
            self.assertTrue(all(not p.requires_grad for p in visual.parameters()))
            record = {"teacher_record": {"input_ids": [1, 2, 3], "labels": [-100, 2, 3],
                                         "images": [], "segments": [], "image_grid_thw": []},
                      "actions": [{"teacher_position": 2}]}
            result = _extract_record((teacher, visual, collator, torch.device("cpu")), record)
            self.assertEqual(tuple(result.shape), (1, 24))


class TeacherCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        warmup = root / "warmup"
        (warmup / "backbone").mkdir(parents=True)
        for name in ("warmup_recipe.json", "interaction_config.json", "warmup_adapter_config.json"):
            (warmup / name).write_text("{}", encoding="utf-8")
        (warmup / "backbone" / "config.json").write_text(json.dumps({"hidden_size": 4}), encoding="utf-8")
        for name in ("interaction.safetensors", "warmup_adapter.safetensors", "backbone/model.safetensors"):
            (warmup / name).write_bytes(b"dummy-frozen-weights")
        image = root / "image.png"
        image.write_bytes(b"image-content")
        prepared = root / "prepared"
        prepared.mkdir()
        self.config = SimpleNamespace(warmup_path=str(warmup), teacher_cache_dir=str(root / "teacher"),
                                      prepared_dir=str(prepared), attention_backend="sdpa", min_image_pixels=3136,
                                      max_image_pixels=200704, prepare_log_seconds=60)
        records = [{"record_id": f"source/record-{index}", "teacher_record": {
            "input_ids": [1, 2, 3], "labels": [-100, 2, 3], "images": [{"path": str(image)}]},
                    "actions": [{"teacher_position": 2, "original_text": "move right"}]} for index in range(2)]
        self.dataset = SimpleNamespace(records=records, manifest={"stage": "sft2_latent"})
        self.sync_records()
        (prepared / "manifest.json").write_text(json.dumps(self.dataset.manifest), encoding="utf-8")
        self.extract = patch("src.sft2.teacher._extract_record", side_effect=lambda _components, record:
                             torch.full((len(record["actions"]), 4), float(record["record_id"][-1]) + 1))
        self.loader = patch("src.sft2.teacher._load_teacher", return_value=object())
        self.mock_extract = self.extract.start()
        self.mock_loader = self.loader.start()
        self.addCleanup(self.extract.stop)
        self.addCleanup(self.loader.stop)

    def sync_records(self):
        path = Path(self.config.prepared_dir) / "records.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in self.dataset.records), encoding="utf-8")

    def test_roundtrip_reuses_complete_cache_without_loading_teacher_and_safe_paths(self):
        manifest = prepare_teacher_cache(self.config, self.dataset)
        self.assertEqual(self.mock_extract.call_count, 2)
        reader = TeacherTargetReader(self.config.teacher_cache_dir)
        torch.testing.assert_close(reader("source/record-0"), torch.ones(1, 4))
        self.assertNotIn("source/record", manifest["records"]["source/record-0"]["file"])
        validate_teacher_cache(self.config, self.dataset)
        self.mock_loader.reset_mock()
        self.mock_extract.reset_mock()
        repeated = prepare_teacher_cache(self.config, self.dataset)
        self.assertEqual(manifest, repeated)
        self.mock_loader.assert_not_called()
        self.mock_extract.assert_not_called()

    def test_interrupted_cache_resumes_only_verified_records(self):
        original = self.mock_extract.side_effect
        def fail_second(components, record):
            if record["record_id"].endswith("1"):
                raise RuntimeError("simulated interruption")
            return original(components, record)
        self.mock_extract.side_effect = fail_second
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            prepare_teacher_cache(self.config, self.dataset)
        self.assertFalse((Path(self.config.teacher_cache_dir) / "manifest.json").exists())
        self.mock_extract.side_effect = original
        self.mock_extract.reset_mock()
        prepare_teacher_cache(self.config, self.dataset)
        self.assertEqual(self.mock_extract.call_count, 1)
        self.assertTrue(self.mock_extract.call_args.args[1]["record_id"].endswith("1"))

    def test_corrupt_target_is_rejected_then_recomputed(self):
        manifest = prepare_teacher_cache(self.config, self.dataset)
        target = Path(self.config.teacher_cache_dir) / manifest["records"]["source/record-0"]["file"]
        value = bytearray(target.read_bytes())
        value[-1] ^= 1
        target.write_bytes(value)
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_teacher_targets(self.config.teacher_cache_dir, "source/record-0")
        self.mock_extract.reset_mock()
        prepare_teacher_cache(self.config, self.dataset)
        self.assertEqual(self.mock_extract.call_count, 1)
        torch.testing.assert_close(load_teacher_targets(self.config.teacher_cache_dir, "source/record-0"), torch.ones(1, 4))

    def test_image_content_change_invalidates_even_when_size_and_mtime_are_preserved(self):
        first = prepare_teacher_cache(self.config, self.dataset)
        image = Path(self.dataset.records[0]["teacher_record"]["images"][0]["path"])
        stat = image.stat()
        image.write_bytes(b"IMAGE-content")
        os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_teacher_cache(self.config, self.dataset)
        second = prepare_teacher_cache(self.config, self.dataset)
        self.assertNotEqual(first["identity"], second["identity"])

    def test_action_endpoint_change_invalidates_record_and_zero_action_needs_no_model(self):
        prepare_teacher_cache(self.config, self.dataset)
        self.dataset.records[0]["actions"][0]["teacher_position"] = 1
        with self.assertRaisesRegex(ValueError, "positions"):
            validate_teacher_cache(self.config, self.dataset, verify_hashes=False)
        for record in self.dataset.records:
            record["actions"] = []
        self.sync_records()
        self.mock_loader.reset_mock()
        self.mock_extract.reset_mock()
        prepare_teacher_cache(self.config, self.dataset)
        self.mock_loader.assert_not_called()
        self.mock_extract.assert_not_called()
        self.assertEqual(tuple(load_teacher_targets(self.config.teacher_cache_dir, "source/record-0").shape), (0, 4))


if __name__ == "__main__":
    unittest.main()
