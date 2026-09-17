import random
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from src.interaction.checkpoint import load_training, save_training


class LocalEngine:
    """Exercise checkpoint orchestration without claiming to emulate ZeRO."""
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
                    "scheduler": self.scheduler.state_dict(), "global_steps": self.global_steps,
                    "client": client_state}, path / "test.pt")

    def load_checkpoint(self, root, tag, **kwargs):
        state = torch.load(Path(root) / tag / "test.pt", weights_only=False)
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_steps = state["global_steps"]
        client = {key: value for key, value in state["client"].items() if key != "global_steps"}
        return str(Path(root) / tag), {**client, "deepspeed_extra_metadata": True}


class CheckpointTest(unittest.TestCase):
    def test_resume_matches_uninterrupted_update_and_random_states(self):
        torch.manual_seed(8)
        engine = LocalEngine()
        engine.step()
        with tempfile.TemporaryDirectory() as root:
            path = save_training(engine, root, 1, 3, "signature", "reference")
            random_draws = random.random(), np.random.random()
            engine.step()
            expected = {k: v.clone() for k, v in engine.module.state_dict().items()}
            restored = LocalEngine()
            self.assertEqual(load_training(restored, root, "signature", "reference"), (1, 3))
            self.assertEqual((random.random(), np.random.random()), random_draws)
            restored.step()
            for key, value in expected.items():
                torch.testing.assert_close(value, restored.module.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(restored.scheduler.get_last_lr(), engine.scheduler.get_last_lr())
            with self.assertRaisesRegex(ValueError, "Resume rejected"):
                load_training(restored, path, "signature", "different-reference")


if __name__ == "__main__":
    unittest.main()
