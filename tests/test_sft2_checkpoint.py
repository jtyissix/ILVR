"""CPU checkpoint orchestration; these tests do not emulate distributed ZeRO."""
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from safetensors.torch import load_file

from src.sft2.checkpoint import export_stage2, load_training, resume_signature, save_training
from src.sft2.config import Stage2Config
from tests.test_sft2_training import SmallStudent


class LocalEngine:
    def __init__(self):
        self.module = SmallStudent()
        self.optimizer = torch.optim.AdamW([p for p in self.module.parameters() if p.requires_grad], lr=.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=.9)
        self.global_steps = 0

    def step(self):
        self.optimizer.zero_grad()
        loss = sum((parameter * torch.randn_like(parameter)).square().sum()
                   for parameter in self.module.parameters() if parameter.requires_grad)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.global_steps += 1

    def save_checkpoint(self, root, tag, client_state, save_latest):
        path = Path(root) / tag
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.module.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(), "global_steps": self.global_steps,
                    "client": client_state}, path / "test.pt")

    def load_checkpoint(self, root, tag, **kwargs):
        self.load_options = kwargs
        path = Path(root) / tag
        state = torch.load(path / "test.pt", weights_only=False)
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_steps = state["global_steps"]
        return str(path), state["client"]


class CheckpointTest(unittest.TestCase):
    def test_resume_restores_student_auxiliary_optimizer_and_rng(self):
        torch.manual_seed(19)
        engine = LocalEngine()
        engine.step()
        with tempfile.TemporaryDirectory() as directory:
            saved = save_training(engine, directory, 1, 3, "signature", "warmup", "teacher")
            draws = random.random(), np.random.random()
            engine.step()
            expected = {name: value.clone() for name, value in engine.module.state_dict().items()}
            restored = LocalEngine()
            self.assertEqual(load_training(restored, directory, "signature", "warmup", "teacher"), (1, 3))
            self.assertEqual((random.random(), np.random.random()), draws)
            restored.step()
            for name, value in expected.items():
                torch.testing.assert_close(value, restored.module.state_dict()[name], rtol=0, atol=0)
            self.assertTrue(restored.load_options["load_module_strict"])
            self.assertTrue(restored.load_options["load_optimizer_states"])
            self.assertEqual(restored.scheduler.get_last_lr(), engine.scheduler.get_last_lr())
            for args in (("signature", "other", "teacher"), ("signature", "warmup", "other"),
                         ("other", "warmup", "teacher")):
                with self.assertRaisesRegex(ValueError, "Resume rejected"):
                    load_training(restored, saved, *args)

    def test_signature_includes_loss_weights_teacher_and_interaction_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            ds = Path(directory) / "ds.json"
            ds.write_text('{"zero_optimization": {"stage": 2}}', encoding="utf-8")
            config = Stage2Config(deepspeed=str(ds))
            signature = resume_signature(config, "data", "warmup", "teacher")
            config.log_steps = 777
            self.assertEqual(signature, resume_signature(config, "data", "warmup", "teacher"))
            config.visual_weight = 2
            self.assertNotEqual(signature, resume_signature(config, "data", "warmup", "teacher"))
            self.assertNotEqual(signature, resume_signature(config, "data", "interaction_changed", "teacher"))
            self.assertNotEqual(signature, resume_signature(config, "data", "warmup", "teacher_changed"))

    def test_export_contains_auxiliary_and_declares_helper_plan_requirements(self):
        class Processor:
            def save_pretrained(self, path):
                (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            warmup = root / "warmup"
            warmup.mkdir()
            (warmup / "interaction_config.json").write_text(json.dumps({"kwargs": {"slots": 16}}), encoding="utf-8")
            config = Stage2Config(warmup_path=str(warmup))
            engine = LocalEngine()
            engine.module.aux_decoder.embed_tokens = torch.nn.Identity()
            engine.module.model_config = SimpleNamespace(to_dict=lambda: {
                "model_type": "qwen2_5_vl", "hidden_size": 3, "visual_latent_size": 8,
                "latent_start_id": 6, "latent_end_id": 7})
            export = root / "export"
            export_stage2(engine, torch.nn.Linear(3, 3), Processor(), export, config, "warmup", "teacher")
            recipe = json.loads((export / "stage2_recipe.json").read_text(encoding="utf-8"))
            self.assertEqual(recipe["latent_boundary_ids"], [6, 7])
            self.assertEqual(recipe["boundary_tokens"], ["<|latent_start|>", "<|latent_end|>"])
            self.assertIn("boundaries-v2", recipe["format"])
            self.assertTrue(recipe["requires_helper_images"])
            self.assertTrue(recipe["requires_sequence_plan"])
            self.assertFalse(recipe["helper_free_generation_ready"])
            self.assertFalse(recipe["warmup_adapter_in_student"])
            self.assertFalse((export / "warmup_adapter.safetensors").exists())
            self.assertTrue((export / "interaction.safetensors").is_file())
            auxiliary = load_file(export / "auxiliary_decoder" / "model.safetensors")
            self.assertIn("model.layers.0.weight", auxiliary)
            self.assertIn("lm_head.weight", auxiliary)
            self.assertNotIn("model.embed_tokens.weight", auxiliary)
            backbone_config = json.loads((export / "backbone" / "config.json").read_text(encoding="utf-8"))
            self.assertNotIn("visual_latent_size", backbone_config)


if __name__ == "__main__":
    unittest.main()
