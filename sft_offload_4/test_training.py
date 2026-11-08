"""Actual tiny four-loss loop with CUDA/DS I/O replaced; native migration tested separately."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

class TrainingLoopTest(unittest.TestCase):
    def test_independent_train_loop_reaches_metrics_checkpoint_export_and_completion(self):
        """Run actual tiny four-loss forward/backward; only CUDA/DS are replaced.

        Native offload/collective correctness has a separate multiprocess test.
        This catches dependency/import regressions in the standalone loop itself.
        """
        from src.interaction.data import stable_hash
        from sft_offload_4 import train as training
        from tests import test_sft2_deepspeed as fixture
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            stack.enter_context(patch.object(fixture, "ACCUMULATION", 4))
            config = fixture.training_recipe()
            config.output_dir, config.max_steps, config.num_workers = temporary, 1, 0
            config.warmup_ratio = 0.
            config.epochs = 1
            model = fixture.make_model(config, torch.device("cpu"))
            model.model_config.to_dict = lambda: {"hidden_size": 24}
            visual = torch.nn.Linear(3, 24)
            local = fixture.rank_window(0, 1, 0)
            batches = []
            for batch, features in local:
                batch["sample_count"] = int(batch["sample_mask"].sum())
                batch["fixture_features"] = features
                batches.append(batch)
            class Dataset:
                manifest = {"samples": 4, "fixture": "standalone-offload-loop"}
                records = [{"segments": [], "length": 4, "teacher_record": {"input_ids": [1]}}
                           for _ in range(4)]
                def __len__(self):
                    return 4
            dataset = Dataset()
            receipt = {"dataset_sha256": stable_hash(dataset.manifest), "total_optimizer_steps": 1}
            saved_state = {"warmup_fingerprint": "warmup", "teacher_fingerprint": "teacher", "signature": "fixture"}
            class Engine:
                def __init__(self, model, optimizer, lr_scheduler, **kwargs):
                    self.module, self.optimizer, self.scheduler = model, optimizer, lr_scheduler
                    self.global_steps, self.microsteps = 0, 0
                def train(self):
                    self.module.train()
                def __call__(self, batch, image_features):
                    return self.module(batch, image_features)
                def backward(self, loss):
                    (loss / 4).backward()
                def step(self):
                    self.microsteps += 1
                    if self.microsteps % 4 == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scheduler.step()
                        self.global_steps += 1
            initialized = []
            def initialize(**kwargs):
                engine = Engine(**kwargs)
                initialized.append(engine)
                return engine, None, None, None
            fake_ds = SimpleNamespace(__version__="0.18.0", init_distributed=lambda **kwargs: None,
                                      initialize=initialize)
            fake_cuda = SimpleNamespace(is_available=lambda: True, set_device=lambda index: None,
                is_bf16_supported=lambda: True, manual_seed_all=lambda seed: None,
                get_device_name=lambda: "CPU loop fixture", reset_peak_memory_stats=lambda: None,
                synchronize=lambda: None, max_memory_allocated=lambda: 0)
            class TorchProxy:
                cuda = fake_cuda
                @staticmethod
                def device(*args):
                    return torch.device("cpu")
                def __getattr__(self, name):
                    return getattr(torch, name)
            stack.enter_context(patch.dict(sys.modules, {"deepspeed": fake_ds}))
            patches = [
                patch.object(training, "torch", TorchProxy()),
                patch.object(training, "validate_local_inputs"),
                patch.object(training, "configure_deepspeed", return_value={"zero_optimization": {"stage": 2}}),
                patch.object(training, "resume_context", return_value=(None, saved_state, receipt)),
                patch.object(training, "load_existing", return_value=dataset),
                patch.object(training, "checked_load", return_value=(0, 0)),
                patch.dict(os.environ, {"WORLD_SIZE": "4"}),
                patch("src.sft2.data.Stage2Dataset", return_value=dataset),
                patch("src.sft2.teacher.validate_teacher_cache", return_value={"identity": "teacher"}),
                patch("src.sft2.data.load_processor", return_value=object()),
                patch("src.sft2.model.load_training_components", return_value=(model, visual)),
                patch.object(training, "TrustedCollator", return_value=object()),
                patch.object(training, "create_optimizer", side_effect=lambda student, config:
                             torch.optim.AdamW(training.optimizer_groups(student, config))),
                patch.object(training, "install_ordered_offload", return_value=SimpleNamespace(
                             order_sha256="fixture-order", peak_pending_bytes=0)),
                # This loop fixture uses ordinary AdamW in place of ZeRO.
                patch.object(training.NumericalGuard, "check_state"),
                
                patch.object(training, "DataLoader", return_value=batches),
                patch.object(training, "encode_images", side_effect=lambda visual, batch: batch["fixture_features"]),
            ]
            for context in patches:
                stack.enter_context(context)
            saved = stack.enter_context(patch.object(training, "save_training"))
            exported = stack.enter_context(patch.object(training, "export_stage2"))
            training.train(config)
            self.assertEqual(initialized[0].global_steps, 1)
            saved.assert_called_once()
            exported.assert_called_once()
            result = json.loads((Path(temporary) / "completion.json").read_text(encoding="utf-8"))
            self.assertEqual(result["global_steps"], 1)
            info = json.loads((Path(temporary) / "run_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["optimizer_offload"], "cpu")
            self.assertGreater(info["memory_budget"]["trainable_parameters"], 0)
            metrics = json.loads((Path(temporary) / "metrics.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(set(metrics["loss_counts"]), {"ce", "visual", "action_words", "action_state"})
            self.assertFalse(info["data_hash_checks"])

