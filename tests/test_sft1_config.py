"""CPU checks for config, loss plumbing, resume state and separable export."""
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from src.sft1.checkpoint import export_warmup, load_training, resume_signature, save_training
from src.sft1.config import DEFAULT_CONFIG, PROJECT_ROOT, WarmupConfig, validate_local_inputs
from src.sft1.train import configure_deepspeed, encode_images, optimizer_groups


class LocalEngine:
    """Tests orchestration and RNG continuation, not DeepSpeed's implementation."""
    def __init__(self):
        self.module = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.AdamW(self.module.parameters(), lr=.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=.9)
        self.global_steps = 0

    def step(self):
        self.optimizer.zero_grad()
        self.module(torch.randn(3, 2)).square().sum().backward()
        self.optimizer.step()
        self.scheduler.step()
        self.global_steps += 1

    def save_checkpoint(self, root, tag, client_state, save_latest):
        path = Path(root) / tag
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.module.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(), "steps": self.global_steps,
                    "client": client_state}, path / "test.pt")

    def load_checkpoint(self, root, tag, **kwargs):
        state = torch.load(Path(root) / tag / "test.pt", weights_only=False)
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_steps = state["steps"]
        return str(Path(root) / tag), state["client"]


class ConfigTest(unittest.TestCase):
    def test_default_config_has_four_explicit_rewritten_sources_and_resolved_paths(self):
        config = WarmupConfig.load(DEFAULT_CONFIG)
        self.assertEqual(config.adapter_slots, 16)
        self.assertEqual(config.interaction_tokens, 3)
        self.assertTrue(config.skip_overlong_samples)
        self.assertTrue(config.shrink_overlong_images)
        self.assertEqual(config.max_seq_length, 32768)
        self.assertEqual(configure_deepspeed(config)["zero_optimization"]["stage"], 2)
        self.assertEqual(len(config.sources), 4)
        self.assertEqual(config.sources[0]["plan_field"], "thought_rewritten")
        self.assertEqual(config.sources[0]["answer_field"], "original_final_answer")
        # The shipped JSON is user-editable, including the server checkpoint path.
        self.assertTrue(Path(config.model_path).is_absolute())
        self.assertTrue(all(Path(item["path"]).is_absolute() for item in config.sources))
        self.assertFalse(any("reference" in key for key in config.to_dict()))

    def test_unknown_reference_config_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"reference_path": "old-model"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown SFT-I config keys"):
                WarmupConfig.load(path)

    def test_slots_gpus_and_sources_are_checked(self):
        for overrides in ({"adapter_slots": 0}, {"adapter_slots": 16.0}, {"gpu_ids": [0, 0]}, {"gpu_ids": [0], "num_gpus": 2},
                          {"max_steps": 0}, {"sources": []}, {"fusion_dim": 7}, {"interaction_tokens": 0},
                          {"interaction_tokens": 3.0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                WarmupConfig(**overrides).validate()

    def test_smoke_stop_does_not_change_epoch_schedule(self):
        config = WarmupConfig(mode="smoke", smoke_steps=2, max_steps=30).validate()
        self.assertEqual(config.step_limit, 2)
        config.mode = "train"
        self.assertEqual(config.step_limit, 30)

    def test_preflight_reports_all_missing_source_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            config = WarmupConfig().resolve_paths(directory)
            with self.assertRaises(FileNotFoundError) as error:
                validate_local_inputs(config)
            message = str(error.exception)
            for source in config.sources:
                self.assertIn(source["name"], message)
            self.assertIn("nothing is downloaded", message)

    def test_zero3_explicitly_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ds.json"
            path.write_text(json.dumps({"zero_optimization": {"stage": 3}, "bf16": {"enabled": True}}))
            with self.assertRaisesRegex(ValueError, "call schedules"):
                configure_deepspeed(WarmupConfig(deepspeed=str(path)))

    def test_zero1_and_zero2_lock_their_supported_reduction_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zero.json"
            for stage in (1, 2):
                ds = configure_deepspeed(WarmupConfig(deepspeed=str(PROJECT_ROOT / f"configs/sft1/zero{stage}.json")))
                self.assertEqual(ds["zero_optimization"]["stage"], stage)
                self.assertIs(ds["zero_optimization"]["overlap_comm"], False)
                for change in ({"overlap_comm": True}, {"reduce_scatter": stage != 2},
                               {"contiguous_gradients": False}, {"round_robin_gradients": True},
                               {"offload_optimizer": {"device": "cpu"}}):
                    altered = json.loads(json.dumps(ds))
                    altered["zero_optimization"].update(change)
                    path.write_text(json.dumps(altered))
                    with self.subTest(stage=stage, change=change), self.assertRaises(ValueError):
                        configure_deepspeed(WarmupConfig(deepspeed=str(path)))

    def test_zero2_rejects_mismatched_communication_dtype_and_autocast(self):
        ds = configure_deepspeed(WarmupConfig.load(DEFAULT_CONFIG))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zero.json"
            for change in ({"communication_data_type": "fp32"}, {"torch_autocast": {"enabled": True}},
                           {"fp16": {"enabled": True}}):
                path.write_text(json.dumps({**ds, **change}))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    configure_deepspeed(WarmupConfig(deepspeed=str(path)))

    def test_workers_require_this_launch_preflight_before_distributed_startup(self):
        from src.sft1.prepare import read_preflight
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "python run_sft1.py"):
                read_preflight(WarmupConfig())

    def test_preflight_receipt_rejects_changed_inputs_and_other_launch(self):
        from src.interaction.data import stable_hash
        from src.sft1.prepare import file_stat, read_preflight
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            records = output / "records.jsonl"
            records.write_text("{}\n")
            config = WarmupConfig(output_dir=directory, prepared_dir=directory)
            receipt = {"format": "sft1-preflight-v1", "launch_id": "this-launch",
                       "config_hash": stable_hash(config.to_dict()), "records_stat": file_stat(records),
                       "checkpoint_stat_fingerprint": "checkpoint", "source_stats": {}}
            (output / "preflight.json").write_text(json.dumps(receipt))
            with mock.patch.dict("os.environ", {"SFT1_PREFLIGHT_ID": "this-launch"}), \
                    mock.patch("src.sft1.prepare.checkpoint_inventory", return_value={"fingerprint": "checkpoint"}):
                self.assertEqual(read_preflight(config), receipt)
                records.write_text("changed contents\n")
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    read_preflight(config)
            with mock.patch.dict("os.environ", {"SFT1_PREFLIGHT_ID": "different-launch"}):
                with self.assertRaisesRegex(ValueError, "another launch"):
                    read_preflight(config)

    def test_optimizer_assigns_backbone_and_new_module_rates(self):
        model = torch.nn.Module()
        model.decoder = torch.nn.Linear(2, 2)
        model.lm_head = torch.nn.Linear(2, 3)
        model.warmup_adapter = torch.nn.Linear(2, 2)
        model.interaction = torch.nn.Linear(2, 2)
        model.visual = torch.nn.Linear(2, 2).requires_grad_(False)
        config = WarmupConfig(adapter_lr=7e-5)
        groups = optimizer_groups(model, config)
        by_id = {id(p): group for group in groups for p in group["params"]}
        self.assertEqual(by_id[id(model.decoder.weight)]["lr"], config.backbone_lr)
        self.assertEqual(by_id[id(model.interaction.weight)]["lr"], config.fusion_lr)
        self.assertEqual(by_id[id(model.warmup_adapter.weight)]["lr"], config.adapter_lr)
        self.assertEqual(by_id[id(model.decoder.bias)]["weight_decay"], 0)
        self.assertNotIn(id(model.visual.weight), by_id)

    def test_vision_is_called_once_without_grad_and_occurrences_reused(self):
        class Vision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.calls = 0
            def forward(self, pixels, grid_thw):
                self.calls += 1
                return pixels * self.weight
        visual = Vision()
        pixels = torch.arange(6.).reshape(3, 2).requires_grad_()
        batch = {"visual": {"pixel_values": pixels, "image_grid_thw": torch.ones(1, 3),
                            "feature_indices": torch.tensor([0, 1, 2, 1, 2])}}
        features = encode_images(visual, batch)
        self.assertEqual(visual.calls, 1)
        self.assertFalse(features.requires_grad)
        torch.testing.assert_close(features, pixels.detach()[[0, 1, 2, 1, 2]])

    def test_resume_preserves_updates_optimizer_scheduler_and_rng(self):
        torch.manual_seed(8)
        engine = LocalEngine()
        engine.step()
        with tempfile.TemporaryDirectory() as root:
            path = save_training(engine, root, 1, 3, "signature", "initial")
            draws = random.random(), np.random.random()
            engine.step()
            expected = {key: value.clone() for key, value in engine.module.state_dict().items()}
            restored = LocalEngine()
            self.assertEqual(load_training(restored, root, "signature", "initial"), (1, 3))
            self.assertEqual((random.random(), np.random.random()), draws)
            restored.step()
            for key, value in expected.items():
                torch.testing.assert_close(value, restored.module.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(restored.scheduler.get_last_lr(), engine.scheduler.get_last_lr())
            state = json.loads((path / "complete.json").read_text())
            self.assertFalse(any("reference" in key for key in state))
            with self.assertRaisesRegex(ValueError, "Resume rejected"):
                load_training(restored, path, "changed-signature", "initial")

    def test_resume_signature_allows_runtime_change_but_rejects_training_change(self):
        config = WarmupConfig.load(DEFAULT_CONFIG)
        original = resume_signature(config, "dataset", "initial")
        config.max_steps, config.resume, config.log_steps, config.mode = 100, "somewhere", 20, "smoke"
        self.assertEqual(original, resume_signature(config, "dataset", "initial"))
        config.backbone_lr *= 2
        self.assertNotEqual(original, resume_signature(config, "dataset", "initial"))

    def test_export_reloads_both_modules_and_preserves_tied_backbone(self):
        from safetensors.torch import load_file
        from src.sft1.modules import InteractionCore, WarmupImageAdapter
        config = WarmupConfig(fusion_dim=4, fusion_heads=2, fusion_ffn_dim=8, fusion_score_dim=2)
        model = torch.nn.Module()
        model.decoder = torch.nn.Module()
        model.decoder.embed_tokens = torch.nn.Embedding(6, 4)
        model.lm_head = torch.nn.Linear(4, 6, bias=False)
        model.lm_head.weight = model.decoder.embed_tokens.weight
        model.interaction = InteractionCore(4, 4, 2, 8, 2)
        model.warmup_adapter = WarmupImageAdapter(4, 4, 2, 8, layers=2)
        model.model_config = SimpleNamespace(hidden_size=4, to_dict=lambda: {
            "hidden_size": 4, "stage": "stage2", "latent_size": 9, "reference_fingerprint": "old"})
        class Processor:
            def save_pretrained(self, path):
                (Path(path) / "tokenizer_config.json").write_text("{}")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            export_warmup(SimpleNamespace(module=model), torch.nn.Linear(2, 4), Processor(), output, config, "initial")
            for name, constructor in (("interaction", InteractionCore), ("warmup_adapter", WarmupImageAdapter)):
                structure = json.loads((output / (name + "_config.json")).read_text())
                self.assertEqual(structure["kwargs"]["slots"], 16)
                if name == "interaction":
                    self.assertEqual(structure["kwargs"]["interaction_tokens"], 3)
                module = constructor(**structure["kwargs"])
                module.load_state_dict(load_file(str(output / (name + ".safetensors"))), strict=True)
                for key, tensor in getattr(model, name).state_dict().items():
                    torch.testing.assert_close(tensor, module.state_dict()[key])
            backbone = load_file(str(output / "backbone" / "model.safetensors"))
            self.assertIn("model.embed_tokens.weight", backbone)
            self.assertIn("lm_head.weight", backbone)
            self.assertFalse(any(key.startswith(("interaction.", "warmup_adapter.")) for key in backbone))
            exported_config = json.loads((output / "backbone" / "config.json").read_text())
            self.assertNotIn("latent_size", exported_config)
            self.assertNotIn("reference_fingerprint", exported_config)
            self.assertFalse(json.loads((output / "warmup_recipe.json").read_text())["latent_inference_ready"])


if __name__ == "__main__":
    unittest.main()
