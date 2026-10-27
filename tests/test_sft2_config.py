import json
from pathlib import Path
import tempfile
import unittest

from src.sft2.config import Stage2Config


class Stage2ConfigTest(unittest.TestCase):
    def test_warmup_bundle_paths_and_teacher_cache_default(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Stage2Config(warmup_path="stage1").validate().resolve_paths(directory)
            self.assertEqual(Path(config.model_path), Path(directory) / "stage1" / "backbone")
            self.assertEqual(Path(config.teacher_cache_dir), Path(config.prepared_dir) / "teacher")
            self.assertNotIn("model_path", config.to_dict())
            self.assertNotIn("adapter_layers", config.to_dict())

    def test_stage1_adapter_parameters_and_model_path_cannot_silently_enter_stage2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for key in ("adapter_layers", "adapter_lr", "adapter_slots", "model_path"):
                path.write_text(json.dumps({key: 1}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Unknown SFT-II"):
                    Stage2Config.load(path)

    def test_slot_contract_and_finite_loss_weights_are_validated(self):
        for kwargs in ({"action_latents": 5}, {"visual_latents": 8}, {"interaction_tokens": 1},
                       {"pool_grid": 3}, {"action_words_weight": float("nan")},
                       {"visual_weight": -1}, {"state_loss_epsilon": 0}, {"execution": "invalid"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Stage2Config(**kwargs).validate()
        self.assertEqual(Stage2Config(mode="smoke", max_steps=10).validate().step_limit, 2)


if __name__ == "__main__":
    unittest.main()
