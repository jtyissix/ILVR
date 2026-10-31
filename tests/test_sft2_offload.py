"""Independent CPU-offload entry/configuration and existing-cache compatibility."""
import copy
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

from src.sft2.config import Stage2Config
from tests.test_sft2_training import SmallStudent


class OffloadConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "zero.json"
        from src.sft2_offload.config import DEFAULT_CONFIG
        self.config = Stage2Config.load(DEFAULT_CONFIG)
        self.original = json.loads(Path(self.config.deepspeed).read_text(encoding="utf-8"))

    def configured(self, changes=None):
        ds = copy.deepcopy(self.original)
        if changes:
            changes(ds)
        self.path.write_text(json.dumps(ds), encoding="utf-8")
        self.config.deepspeed = str(self.path)
        from src.sft2_offload.config import configure_deepspeed
        return configure_deepspeed(self.config)

    def test_separate_defaults_keep_existing_prepared_and_teacher_paths(self):
        from src.sft2.config import DEFAULT_CONFIG as original
        old = Stage2Config.load(original)
        self.assertEqual(self.config.prepared_dir, old.prepared_dir)
        self.assertEqual(self.config.teacher_cache_dir, old.teacher_cache_dir)
        self.assertEqual(self.config.warmup_path, old.warmup_path)
        self.assertEqual(self.config.sources, old.sources)
        self.assertNotEqual(self.config.output_dir, old.output_dir)
        self.assertNotEqual(self.config.deepspeed, old.deepspeed)
        self.assertIsNone(self.config.resume)

    def test_cpu_optimizer_only_and_batch_dimensions_follow_recipe(self):
        self.config.micro_batch_size = 2
        self.config.gradient_accumulation_steps = 4
        result = self.configured()
        self.assertEqual(result["zero_optimization"]["stage"], 2)
        self.assertEqual(result["zero_optimization"]["offload_optimizer"]["device"], "cpu")
        self.assertTrue(result["zero_optimization"]["offload_optimizer"]["pin_memory"])
        self.assertFalse(result["zero_optimization"].get("offload_param"))
        self.assertEqual(result["train_micro_batch_size_per_gpu"], 2)
        self.assertEqual(result["gradient_accumulation_steps"], 4)
        self.assertEqual(result["train_batch_size"], 8)  # No distributed group in this unit test.
        self.assertNotIn("optimizer", result)

    def test_legacy_recipe_still_rejects_optimizer_offload(self):
        from src.sft2.train import configure_deepspeed
        with self.assertRaisesRegex(ValueError, "offload"):
            configure_deepspeed(self.config)

    def test_rejects_unsupported_state_or_communication_modes(self):
        changes = [
            lambda d: d["zero_optimization"].update(stage=3),
            lambda d: d["zero_optimization"].update(overlap_comm=True),
            lambda d: d["zero_optimization"].update(offload_param={"device": "cpu"}),
            lambda d: d["zero_optimization"].update(offload_optimizer={"device": "nvme"}),
            lambda d: d["zero_optimization"].update(offload_optimizer=None),
            lambda d: d["zero_optimization"].update(round_robin_gradients=True),
            lambda d: d.update(fp16={"enabled": True}),
            lambda d: d.update(bf16={"enabled": False}),
            lambda d: d.update(communication_data_type="fp32"),
            lambda d: d.update(optimizer={"type": "Adam"}),
        ]
        for index, mutate in enumerate(changes):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.configured(mutate)

    def test_cpuadam_uses_unchanged_parameter_groups_and_adamw_semantics(self):
        from src.sft2.train import optimizer_groups
        from src.sft2_offload.train import create_optimizer
        student = SmallStudent()
        captured = {}
        sentinel = object()
        def constructor(groups, **kwargs):
            captured.update(groups=groups, kwargs=kwargs)
            return sentinel
        fake = SimpleNamespace(DeepSpeedCPUAdam=constructor)
        with patch.dict(sys.modules, {"deepspeed.ops.adam": fake}):
            self.assertIs(create_optimizer(student, self.config), sentinel)
        expected = optimizer_groups(student, self.config)
        self.assertEqual(len(captured["groups"]), len(expected))
        for actual, old in zip(captured["groups"], expected):
            self.assertEqual([id(p) for p in actual["params"]], [id(p) for p in old["params"]])
            self.assertEqual(actual["lr"], old["lr"])
            self.assertEqual(actual["weight_decay"], old["weight_decay"])
        self.assertEqual(captured["kwargs"]["betas"], (.9, .999))
        self.assertEqual(captured["kwargs"]["eps"], 1e-8)
        self.assertTrue(captured["kwargs"]["adamw_mode"])


class OffloadLaunchAndCacheTest(unittest.TestCase):
    def test_independent_train_loop_reaches_metrics_checkpoint_export_and_completion(self):
        """Run actual tiny four-loss forward/backward; only CUDA/DS are replaced.

        Native offload/collective correctness has a separate multiprocess test.
        This catches dependency/import regressions in the standalone loop itself.
        """
        from src.interaction.data import stable_hash
        from src.sft2_offload import train as training
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
            receipt = {"dataset_hash": stable_hash(dataset.manifest), "warmup_fingerprint": "warmup"}
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
                patch("src.sft2_offload.prepare.read_preflight", return_value=receipt),
                patch("src.sft2.data.Stage2Dataset", return_value=dataset),
                patch("src.sft2.teacher.validate_teacher_cache", return_value={"identity": "teacher"}),
                patch("src.sft2.data.load_processor", return_value=object()),
                patch("src.sft2.model.load_training_components", return_value=(model, visual)),
                patch("src.sft2.data.Stage2Collator", return_value=object()),
                patch.object(training, "create_optimizer", side_effect=lambda student, config:
                             torch.optim.AdamW(training.optimizer_groups(student, config))),
                patch.object(training, "install_ordered_offload", return_value=SimpleNamespace(
                             order_sha256="fixture-order", peak_pending_bytes=0)),
                # This loop fixture uses ordinary AdamW in place of ZeRO.
                patch.object(training.NumericalGuard, "check_state"),
                patch.object(training, "resume_signature", return_value="fixture-signature"),
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
            self.assertTrue(all("teacher_record" not in record for record in dataset.records))

    def test_launcher_runs_cpu_then_existing_teacher_then_offload_student(self):
        import run_sft2_offload as launcher
        config = Stage2Config(num_gpus=None, gpu_ids=None)
        with patch.object(launcher.Stage2Config, "load", return_value=config), \
             patch.object(launcher, "validate_local_inputs"), \
             patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, \
             patch("torch.cuda.device_count", return_value=8), patch.dict(os.environ, {}, clear=False):
            self.assertEqual(launcher.main([]), 0)
        calls = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(calls), 3)
        self.assertIn("src.sft2_offload.prepare", calls[0])
        self.assertIn("src.sft2.teacher", calls[1])
        self.assertEqual(calls[1][-2:], ["--workers", "8"])
        self.assertIn("src.sft2_offload.train", calls[2])
        self.assertIn("--nproc_per_node=8", calls[2])

    def test_launcher_failure_never_starts_later_stage(self):
        import run_sft2_offload as launcher
        for outcomes in ((7,), (0, 8)):
            with self.subTest(outcomes=outcomes), \
                 patch.object(launcher.Stage2Config, "load", return_value=Stage2Config()), \
                 patch.object(launcher, "validate_local_inputs"), \
                 patch.object(launcher.subprocess, "run", side_effect=[SimpleNamespace(returncode=c) for c in outcomes]) as run, \
                 patch("torch.cuda.device_count", return_value=2), patch.dict(os.environ, {}, clear=False):
                self.assertEqual(launcher.main([]), outcomes[-1])
                self.assertEqual(run.call_count, len(outcomes))

    def test_existing_prepared_records_validate_after_batch_and_gpu_change_without_rebuild(self):
        from tests.test_sft2_data import Stage2DataTest
        from src.sft2.data import Stage2Dataset
        from src.sft2.preprocessing import prepare_dataset
        from src.sft2_offload import prepare as offload_prepare
        fixture = Stage2DataTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        source = fixture.root / "source.jsonl"
        source.write_text(json.dumps(fixture.row) + "\n", encoding="utf-8")
        config = Stage2Config(prepared_dir=str(fixture.root / "prepared"), prepare_workers=1,
                              output_dir=str(fixture.root / "offload-output"),
                              sources=[{**fixture.source, "path": str(source)}])
        inventory = {"config": fixture.model, "metadata": {"fixture": True}}
        with patch("src.sft2.preprocessing.checkpoint_inventory", return_value=inventory), \
             patch("src.sft2.preprocessing.load_processor", return_value=fixture.processor):
            prepare_dataset(config)
        paths = [Path(config.prepared_dir) / name for name in ("records.jsonl", "manifest.json")]
        snapshots = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
        config.micro_batch_size, config.num_gpus, config.gradient_accumulation_steps = 2, 8, 4
        # The actual dataset validates its old processor/source/image signature;
        # only checkpoint fingerprinting and external model loading are mocked.
        with patch.object(offload_prepare, "validate_local_inputs"), \
             patch.object(offload_prepare, "configure_deepspeed"), \
             patch.object(offload_prepare, "check_runtime"), \
             patch.object(offload_prepare, "warmup_fingerprint", return_value="same-warmup"), \
             patch.object(offload_prepare, "warmup_stat_fingerprint", return_value="same-stat"), \
             patch("src.sft1.data.checkpoint_inventory", return_value=inventory), \
             patch("src.sft2.data.load_processor", return_value=fixture.processor), \
             patch.object(offload_prepare, "prepare_dataset", side_effect=AssertionError("must reuse existing cache")):
            receipt = offload_prepare.prepare(config)
        self.assertEqual(len(Stage2Dataset(config.prepared_dir)), 1)
        self.assertEqual(snapshots, [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths])
        self.assertEqual(receipt["warmup_fingerprint"], "same-warmup")

    def test_existing_teacher_identity_and_tensors_reuse_after_offload_and_batch_changes(self):
        from tests.test_sft2_teacher import TeacherCacheTests
        from src.sft2.teacher import prepare_teacher_cache
        fixture = TeacherCacheTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        original = prepare_teacher_cache(fixture.config, fixture.dataset)
        root = Path(fixture.config.teacher_cache_dir)
        tensors = {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
                   for p in root.rglob("*.safetensors")}
        fixture.mock_loader.reset_mock()
        fixture.mock_extract.reset_mock()
        fixture.config.micro_batch_size = 2
        fixture.config.num_gpus = 8
        fixture.config.gradient_accumulation_steps = 4
        fixture.config.output_dir = str(root / "independent-offload-output")
        fixture.config.deepspeed = "configs/sft2_offload/zero2.json"
        reused = prepare_teacher_cache(fixture.config, fixture.dataset, workers=8)
        self.assertEqual(original["identity"], reused["identity"])
        fixture.mock_loader.assert_not_called()
        fixture.mock_extract.assert_not_called()
        self.assertEqual(tensors, {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
                                   for p in root.rglob("*.safetensors")})


if __name__ == "__main__":
    unittest.main()
