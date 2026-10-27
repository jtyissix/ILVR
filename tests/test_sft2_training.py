"""CPU checks of four-loss normalization, optimizer ownership and launch ordering."""
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.sft1.zero2 import install_ordered_zero2, ordered_parameters as original_order
from src.sft2.config import Stage2Config
from src.sft2.train import LOSS_NAMES, combined_loss, optimizer_groups, ordered_parameters, to_device


class SmallDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 3)
        self.layers = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 3)])
        self.norm = nn.LayerNorm(3)


class SmallStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = SmallDecoder()
        self.lm_head = nn.Linear(3, 8)
        self.interaction = nn.Linear(3, 3)
        self.aux_decoder = SmallDecoder()
        self.aux_decoder.embed_tokens.requires_grad_(False)
        self.aux_lm_head = nn.Linear(3, 8)


class TrainingTest(unittest.TestCase):
    def test_each_loss_matches_global_mean_with_unequal_ranks_and_accumulation(self):
        weights = dict(ce=1.0, visual=0.3, action_words=0.8, action_state=0.5)
        # Unequal event counts, including an entirely empty rank/microbatch.
        examples = [dict(ce=[1., 3.], visual=[2.], action_words=[1.], action_state=[4.]),
                    dict(ce=[2.], visual=[], action_words=[], action_state=[]),
                    dict(ce=[5., 6., 7.], visual=[1., 4.], action_words=[2., 3., 4.], action_state=[2., 3.]),
                    {name: [] for name in LOSS_NAMES}]
        counts = {name: sum(len(example[name]) for example in examples) for name in LOSS_NAMES}
        parameter = torch.tensor(0.4, requires_grad=True)
        local_gradients = []
        for example in examples:
            losses = {name: sum(((parameter - y).square() for y in example[name]), parameter * 0)
                      for name in LOSS_NAMES}
            scaled = combined_loss(losses, counts, weights, accumulation_steps=2, world_size=2)
            local_gradients.append(torch.autograd.grad(scaled / 2, parameter)[0])
        actual = sum(local_gradients) / 2  # Two ranks' averaged accumulated gradients.
        expected_loss = sum(weights[name] * sum((parameter - y).square()
                            for example in examples for y in example[name]) / counts[name]
                            for name in LOSS_NAMES)
        expected = torch.autograd.grad(expected_loss, parameter)[0]
        torch.testing.assert_close(actual, expected)

    def test_zero_event_terms_preserve_zero_gradient_dependency(self):
        parameters = {name: torch.tensor(1., requires_grad=True) for name in LOSS_NAMES}
        losses = {name: value * 0 for name, value in parameters.items()}
        loss = combined_loss(losses, {name: 0 for name in LOSS_NAMES},
                             {name: 1 for name in LOSS_NAMES}, 3, 4)
        loss.backward()
        for parameter in parameters.values():
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(parameter.grad.item(), 0)

    def test_optimizer_covers_auxiliary_once_and_preserves_frozen_embedding(self):
        model = SmallStudent()
        config = Stage2Config(backbone_lr=.001, fusion_lr=.002, aux_decoder_lr=.003)
        groups = optimizer_groups(model, config)
        mapping = {id(p): group["lr"] for group in groups for p in group["params"]}
        self.assertEqual(len(mapping), sum(p.requires_grad for p in model.parameters()))
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                self.assertNotIn(id(parameter), mapping)
            else:
                expected = .003 if name.startswith("aux_") else .002 if name.startswith("interaction.") else .001
                self.assertEqual(mapping[id(parameter)], expected)

    def test_auxiliary_order_is_explicit_and_legacy_default_unchanged(self):
        model = SmallStudent()
        groups = [[p for p in model.parameters() if p.requires_grad]]
        stage2 = [name for name, _, _ in ordered_parameters(model, groups)]
        original = [name for name, _, _ in original_order(model, groups)]
        self.assertTrue(stage2[0].startswith("aux_lm_head."))
        self.assertTrue(original[0].startswith("lm_head."))
        self.assertLess(stage2.index("aux_decoder.layers.1.weight"), stage2.index("aux_decoder.layers.0.weight"))
        self.assertLess(stage2.index("aux_decoder.layers.0.weight"), stage2.index("decoder.layers.1.weight"))
        self.assertEqual(set(stage2), set(original))
        self.assertIsNone(inspect.signature(install_ordered_zero2).parameters["parameter_order"].default)
        fake = SimpleNamespace(zero_optimization_stage=lambda: 1)
        self.assertIsNone(install_ordered_zero2(fake))
        self.assertIsNone(install_ordered_zero2(fake, parameter_order=ordered_parameters))

    def test_nested_targets_move_while_position_metadata_stays_cpu(self):
        batch = {"teacher_targets": [torch.ones(2, 3)], "segments": [[{"text": [1, 2]}]]}
        result = to_device(batch, torch.device("cpu"))
        self.assertIs(result["segments"], batch["segments"])
        torch.testing.assert_close(result["teacher_targets"][0], batch["teacher_targets"][0])

    def test_train_validates_teacher_before_releasing_trajectory_then_reaches_cuda_boundary(self):
        from src.interaction.data import stable_hash
        from src.sft2 import train as training
        manifest = {"samples": 1, "records_sha256": "fixture"}
        record = {"record_id": "example", "teacher_record": {"input_ids": [1, 2, 3]}}
        dataset = SimpleNamespace(manifest=manifest, records=[record])
        receipt = {"dataset_hash": stable_hash(manifest), "warmup_fingerprint": "warmup"}
        def verify_teacher(_config, dataset, verify_hashes):
            self.assertIn("teacher_record", dataset.records[0])
            self.assertFalse(verify_hashes)
            return {"identity": "teacher"}
        with patch.object(training, "validate_local_inputs"), patch.object(training, "configure_deepspeed"), \
             patch("src.sft2.prepare.read_preflight", return_value=receipt), \
             patch("src.sft2.data.Stage2Dataset", return_value=dataset), \
             patch("src.sft2.teacher.validate_teacher_cache", side_effect=verify_teacher) as validation, \
             patch("src.sft2.data.load_processor", return_value=object()) as processor, \
             patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.set_device") as device:
            with self.assertRaisesRegex(RuntimeError, "training needs BF16 CUDA"):
                training.train(Stage2Config())
        validation.assert_called_once()
        processor.assert_called_once()
        device.assert_not_called()
        self.assertNotIn("teacher_record", record)
        self.assertEqual(record["record_id"], "example")

    def test_launcher_prepares_then_teacher_then_distributed_and_stops_on_error(self):
        import run_sft2
        config = Stage2Config(mode="train", num_gpus=4, gpu_ids=[1, 3, 5, 7])
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0)
        with patch.object(run_sft2.Stage2Config, "load", return_value=config), \
             patch.object(run_sft2, "validate_local_inputs"), patch.object(run_sft2.subprocess, "run", side_effect=run), \
             patch("torch.cuda.device_count", return_value=4), patch.dict(run_sft2.os.environ, {}, clear=False):
            self.assertEqual(run_sft2.main(["--config", str(Path("fixture.json"))]), 0)
            self.assertEqual(run_sft2.os.environ["CUDA_VISIBLE_DEVICES"], "1,3,5,7")
        self.assertIn("src.sft2.prepare", calls[0])
        self.assertIn("src.sft2.teacher", calls[1])
        self.assertEqual(calls[1][-2:], ["--workers", "4"])
        self.assertIn("torch.distributed.run", calls[2])
        self.assertIn("--nproc_per_node=4", calls[2])
        calls.clear()
        config.mode = "prepare_only"
        with patch.object(run_sft2.Stage2Config, "load", return_value=config), \
             patch.object(run_sft2, "validate_local_inputs"), patch.object(run_sft2.subprocess, "run", side_effect=run), \
             patch.dict(run_sft2.os.environ, {}, clear=False):
            self.assertEqual(run_sft2.main([]), 0)
        self.assertEqual(len(calls), 1)
        config.mode = "train"
        with patch.object(run_sft2.Stage2Config, "load", return_value=config), \
             patch.object(run_sft2, "validate_local_inputs"), \
             patch.object(run_sft2.subprocess, "run", side_effect=[SimpleNamespace(returncode=0),
                                                                  SimpleNamespace(returncode=9)]) as launched, \
             patch("torch.cuda.device_count", return_value=4), patch.dict(run_sft2.os.environ, {}, clear=False):
            self.assertEqual(run_sft2.main([]), 9)
            self.assertEqual(launched.call_count, 2)


if __name__ == "__main__":
    unittest.main()
