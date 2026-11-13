"""No privileged image reads, cache validation, resumes, exports and launch."""
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.interaction.data import file_hash, json_write
from src.sft1.data import _processor_identity
from src.sft2.data import CACHE_VERSION, TOKENIZATION_POLICY
from src.sft2.tokens import BOUNDARY_POLICY
from src.sft3.checkpoint import export_stage3, load_training, resume_signature, save_training
from src.sft3.config import Stage3Config
from src.sft3.data import Stage3Collator, Stage3Dataset, relaxation_record
from tests import test_sft2_data as data_fixtures
from tests.test_sft2_checkpoint import LocalEngine
from tests.test_sft3_model import make_model


class RelaxationDataTests(unittest.TestCase):
    def setUp(self):
        self.fixture = data_fixtures.Stage2DataTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.processor = self.fixture.processor
        self.model = {**self.fixture.model, "latent_start_id": 6, "latent_end_id": 7}

    def records(self):
        row = copy.deepcopy(self.fixture.row)
        row["thought_rewritten"][2:2] = copy.deepcopy(row["thought_rewritten"][:2])
        return [self.fixture.record(row), self.fixture.record()]

    def manifest(self, records):
        return {"version": CACHE_VERSION, "stage": "sft2_latent", "boundary_policy": BOUNDARY_POLICY,
                "tokenization_policy": TOKENIZATION_POLICY, "samples": len(records),
                "processor_identity": _processor_identity(self.processor), "latent_boundary_ids": [6, 7],
                "action_latents": 4, "interaction_tokens": 3, "visual_latents": 16,
                "image_stats": {i["path"]: [Path(i["path"]).stat().st_size, Path(i["path"]).stat().st_mtime_ns]
                                for record in records for i in record["images"]}}

    def test_helper_files_and_teacher_targets_are_never_read(self):
        records = self.records()
        manifest = self.manifest(records)
        for record in records:
            for image in record["images"]:
                if image["kind"] == "helper":
                    image["path"] = str(self.fixture.root / "does-not-exist.png")
        for selection in ("0", "1", "12", "123", "23"):
            config = Stage3Config(collapse_targets=selection)
            collator = Stage3Collator(self.processor, self.model, manifest, config)
            batch = collator([*records, {**records[0], "dummy": True}])
            self.assertNotIn("teacher_targets", batch)
            self.assertEqual(batch["visual"]["image_grid_thw"].shape[0], 1)
            self.assertTrue(all(i["kind"] == "question" for row in batch["images"] for i in row))
            self.assertEqual(batch["sample_mask"].tolist(), [True, True, False])
            self.assertTrue(batch["labels"][2].eq(-100).all())
            for kind in ("text", "interaction", "visual"):
                self.assertEqual(batch["loss_counts"]["collapse_" + kind], int(kind in config.selected_types))
            self.assertEqual(batch["loss_counts"]["ce"], sum(v != -100 for r in records for v in r["labels"][1:]))
            own = relaxation_record(records[0])
            self.assertNotIn("teacher_record", own)
            self.assertEqual(set(own["actions"][0]), {"positions"})
            self.assertEqual(set(own["segments"][0]), {"position", "visual_positions"})
            torch.testing.assert_close(batch["position_ids"][:, 0, :len(own["input_ids"])],
                                       collator([own])["position_ids"][:, 0])

    def test_dataset_accepts_missing_helpers_and_sources_but_checks_questions_and_checksum(self):
        records = self.records()
        manifest = self.manifest(records)
        directory = self.fixture.root / "prepared"
        directory.mkdir()
        cache = directory / "records.jsonl"
        cache.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        manifest["records_sha256"] = file_hash(cache)
        json_write(directory / "manifest.json", manifest)
        for record in records:
            for image in record["images"]:
                if image["kind"] == "helper":
                    Path(image["path"]).unlink(missing_ok=True)
        dataset = Stage3Dataset(directory)
        dataset.validate(Stage3Config(), self.processor)
        self.assertEqual(len(dataset), 2)
        self.assertTrue(dataset[-1]["dummy"])
        with self.assertRaises(IndexError):
            dataset[-2]
        question = Path(dataset[0]["images"][0]["path"])
        question.write_bytes(question.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Question image changed"):
            dataset.validate(Stage3Config(), self.processor)
        cache.write_text(cache.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            Stage3Dataset(directory)


class BackboneEngine(LocalEngine):
    def __init__(self):
        self.module = make_model()
        self.optimizer = torch.optim.AdamW(self.module.parameters(), lr=.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=.9)
        self.global_steps = 0


class RelaxationCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_resume_restores_rng_optimizer_scheduler_and_backbone(self):
        engine = BackboneEngine()
        engine.step()
        with tempfile.TemporaryDirectory() as directory:
            save_training(engine, directory, 1, 3, "signature", "source")
            draws = random.random(), np.random.random()
            engine.step()
            expected = copy.deepcopy(engine.module.state_dict())
            restored = BackboneEngine()
            self.assertEqual(load_training(restored, directory, "signature", "source"), (1, 3))
            self.assertEqual((random.random(), np.random.random()), draws)
            restored.step()
            for name, value in expected.items():
                torch.testing.assert_close(value, restored.module.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(restored.scheduler.get_last_lr(), engine.scheduler.get_last_lr())
            with self.assertRaisesRegex(ValueError, "Resume rejected"):
                load_training(restored, directory, "changed", "source")

    def test_resume_signature_tracks_each_collapse_setting(self):
        config = Stage3Config().resolve_paths()
        original = resume_signature(config, "data", "source")
        for key, value in (("collapse_targets", "123"), ("collapse_weight", .7), ("collapse_margin", .9),
                           ("collapse_alpha", 2), ("collapse_epsilon", .3), ("execution", "recompute")):
            changed = copy.deepcopy(config)
            setattr(changed, key, value)
            self.assertNotEqual(original, resume_signature(changed, "data", "source"))
        config.log_steps = 999
        self.assertEqual(original, resume_signature(config, "data", "source"))

    def test_export_contains_only_backbone_and_declares_latent_runtime(self):
        from safetensors.torch import load_file
        class Processor:
            def save_pretrained(self, path):
                (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        engine = BackboneEngine()
        engine.module.model_config = SimpleNamespace(to_dict=lambda: {
            "model_type": "qwen2_5_vl", "hidden_size": 24, "latent_start_id": 70, "latent_end_id": 72,
            "sft2_helper_conditioned_backbone": True})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            export_stage3(engine, torch.nn.Linear(24, 24), Processor(), output, Stage3Config(), "source")
            self.assertEqual({p.name for p in output.iterdir()}, {"backbone", "stage3_recipe.json"})
            state = load_file(output / "backbone/model.safetensors")
            self.assertTrue(all(name.startswith(("model.", "lm_head.", "visual.")) for name in state))
            torch.testing.assert_close(state["model.embed_tokens.weight"], engine.module.decoder.embed_tokens.weight)
            recipe = json.loads((output / "stage3_recipe.json").read_text(encoding="utf-8"))
            self.assertFalse(recipe["requires_helper_images"])
            self.assertFalse(recipe["requires_teacher"])
            self.assertFalse(recipe["standalone_hf_generation_ready"])
            self.assertEqual(recipe["trainable_components"], ["decoder", "lm_head"])

    def test_launcher_runs_preflight_then_training_without_teacher_process(self):
        import run_sft3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("{}", encoding="utf-8")
            with patch.object(run_sft3, "validate_local_inputs"), patch.object(run_sft3.subprocess, "run") as run, \
                 patch("torch.cuda.device_count", return_value=2), patch.dict(run_sft3.os.environ):
                run.return_value.returncode = 0
                self.assertEqual(run_sft3.main(["--config", str(path)]), 0)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertIn("src.sft3.prepare", commands[0])
            self.assertIn("src.sft3.train", commands[1])
            self.assertIn("--nproc_per_node=2", commands[1])


if __name__ == "__main__":
    unittest.main()
