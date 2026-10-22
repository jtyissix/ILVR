"""Causal, gradient, and label checks for original-image segmented warm-up."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from src.interaction.data import position_ids_for_images
from src.interaction.execution import causal_ce_sum
from src.sft1.config import WarmupConfig
from src.sft1.model import WarmupModel, image_memories
from tests.interaction_fixtures import TinyDecoder


def tiny_config(**changes):
    values = dict(fusion_dim=16, fusion_heads=2, fusion_ffn_dim=32, fusion_score_dim=8,
                  attention_backend="sdpa", ce_chunk_size=3)
    values.update(changes)
    return WarmupConfig(**values)


def model_config(hidden=24):
    return SimpleNamespace(hidden_size=hidden, image_token_id=80,
                           vision_config=SimpleNamespace(spatial_merge_size=2))


def example_batch(specs=None, hidden=24, interaction_tokens=3):
    """Build real-layout image spans, including multi-image and ragged examples."""
    specs = specs or [(2, [[3, 4], "image", [5, 6], "image", [7, 90]]),
                      (0, [[8], "image", "image", [9, 90]]), (0, [[10, 11, 90]])]
    records, features, offset = [], [], 0
    for question_count, steps in specs:
        ids, labels, pictures, grids, segments = [1], [-100], [], [], []
        def picture(kind, image_id):
            nonlocal offset
            grid = [1, 4, 4] if image_id % 2 == 0 else [1, 4, 6]
            count = grid[0] * grid[1] * grid[2] // 4
            boundary_start = len(ids)
            ids.extend([82] + [80] * count + [83])
            labels.extend([-100] * (count + 2))
            grids.append(grid)
            pictures.append(dict(feature_start=offset, feature_end=offset + count,
                                 kind=kind, grid=grid, image_id=image_id, boundary_start=boundary_start))
            features.append(torch.randn(count, hidden))
            offset += count
        for index in range(question_count):
            picture("question", index)
        ids.append(2)
        labels.append(-100)
        text, helper_index = [], 0
        for step in steps:
            if step == "image":
                pos = len(ids)
                ids.extend([0] * interaction_tokens)
                labels.extend([-100] * interaction_tokens)
                segments.append(dict(position=pos, text=text, image_index=len(pictures)))
                text = []
                picture("helper", helper_index)
                helper_index += 1
            else:
                text.extend(range(len(ids), len(ids) + len(step)))
                ids.extend(step)
                labels.extend(step)
        positions = position_ids_for_images(ids, grids, {"image_token_id": 80,
                                                        "vision_config": {"spatial_merge_size": 2}})
        records.append((ids, labels, pictures, segments, positions))
    length = max(len(r[0]) for r in records)
    ids = torch.zeros(len(records), length, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    positions = torch.zeros(3, len(records), length, dtype=torch.long)
    for row, (tokens, targets, _, _, pos) in enumerate(records):
        ids[row, :len(tokens)] = torch.tensor(tokens)
        labels[row, :len(tokens)] = torch.tensor(targets)
        mask[row, :len(tokens)] = True
        positions[:, row, :len(tokens)] = pos
    return dict(input_ids=ids, labels=labels, attention_mask=mask, position_ids=positions,
                images=[r[2] for r in records], segments=[r[3] for r in records]), (
                    torch.cat(features) if features else None)


class WarmupModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(2026)

    def make_model(self, **changes):
        return WarmupModel(TinyDecoder(), nn.Linear(24, 96, bias=False), model_config(), tiny_config(**changes))

    def test_incremental_and_same_history_recompute_match_all_gradients(self):
        original = self.make_model()
        batch, features = example_batch()
        outputs, gradients = [], []
        for execution, checkpointing in (("incremental", False), ("incremental", True), ("recompute", True)):
            model = copy.deepcopy(original)
            model.recipe.execution = execution
            model.recipe.activation_checkpointing = checkpointing
            model.executor.activation_checkpointing = checkpointing
            trajectory = model.trajectory(batch, features)
            loss = causal_ce_sum(trajectory.hidden, batch["labels"], model.lm_head, 3, checkpointing)
            loss.backward()
            outputs.append((trajectory.hidden.detach(), loss.detach(),
                            {k: v.detach() for k, v in trajectory.interactions.items()}))
            gradients.append({n: p.grad.detach().clone() if p.grad is not None else None
                              for n, p in model.named_parameters()})
            for name in ("warmup_adapter.queries", "interaction.query_mlp.0.weight",
                         "interaction.image_projection.weight", "decoder.embed_tokens.weight"):
                self.assertGreater(float(gradients[-1][name].norm()), 0, name)
        for output, gradient in zip(outputs[1:], gradients[1:]):
            torch.testing.assert_close(output[0][batch["attention_mask"]], outputs[0][0][batch["attention_mask"]],
                                       atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(output[1], outputs[0][1], atol=2e-5, rtol=2e-5)
            for key in output[2]:
                torch.testing.assert_close(output[2][key], outputs[0][2][key], atol=2e-5, rtol=2e-5)
            for name in gradient:
                if gradients[0][name] is None:
                    self.assertIsNone(gradient[name], name)
                else:
                    torch.testing.assert_close(gradient[name], gradients[0][name], atol=3e-5, rtol=3e-4, msg=name)

    def test_later_interaction_reads_actual_earlier_b_history(self):
        model = self.make_model(activation_checkpointing=False)
        batch, features = example_batch([(1, [[3, 4], "image", [5, 6], "image", [7, 90]])])
        original = model.trajectory(batch, features)
        count = 0
        offset = torch.randn(1, 24)
        def change_first(_module, _args, output):
            nonlocal count
            count += 1
            return output + offset if count == 1 else output
        hook = model.interaction.register_forward_hook(change_first)
        changed = model.trajectory(batch, features)
        hook.remove()
        torch.testing.assert_close(original.text_states[0, 0], changed.text_states[0, 0])
        self.assertFalse(torch.allclose(original.text_states[0, 1], changed.text_states[0, 1]))
        self.assertFalse(torch.allclose(original.interactions[0, 1], changed.interactions[0, 1]))

    def test_overlapping_cross_sample_interaction_blocks_match_individual_gradients(self):
        batched = self.make_model(activation_checkpointing=True)
        individual = copy.deepcopy(batched)
        batch, features = example_batch([
            (0, [[3], "image", [4], "image", [5, 90]]),
            (0, [[6, 7], "image", [8, 90]]),
            (0, [[9, 10, 11], "image", [12, 90]])])
        positions = [segments[0]["position"] for segments in batch["segments"]]
        self.assertEqual(positions, [3, 4, 5])
        trajectory = batched.trajectory(batch, features)
        batch_loss = batched(batch, features)
        individual_loss = 0
        for row, images in enumerate(batch["images"]):
            length = int(batch["attention_mask"][row].sum())
            start, end = images[0]["feature_start"], images[-1]["feature_end"]
            own_images = copy.deepcopy(images)
            for item in own_images:
                item["feature_start"] -= start
                item["feature_end"] -= start
            own_batch = {key: value[row:row + 1, :length] for key, value in batch.items()
                         if key in ("input_ids", "labels", "attention_mask")}
            own_batch.update(position_ids=batch["position_ids"][:, row:row + 1, :length],
                             images=[own_images], segments=[batch["segments"][row]])
            own_trajectory = individual.trajectory(own_batch, features[start:end])
            torch.testing.assert_close(trajectory.hidden[row, :length], own_trajectory.hidden[0],
                                       atol=3e-5, rtol=3e-5)
            for index in range(len(batch["segments"][row])):
                block = trajectory.interactions[row, index]
                self.assertEqual(block.shape, (3, 24))
                self.assertFalse(torch.allclose(block[0], block[1]))
                torch.testing.assert_close(block, own_trajectory.interactions[0, index], atol=3e-5, rtol=3e-5)
            individual_loss = individual_loss + individual(own_batch, features[start:end])
        torch.testing.assert_close(batch_loss, individual_loss, atol=3e-5, rtol=3e-5)
        batch_loss.backward()
        individual_loss.backward()
        own_parameters = dict(individual.named_parameters())
        for name, parameter in batched.named_parameters():
            torch.testing.assert_close(parameter.grad, own_parameters[name].grad, atol=5e-5, rtol=5e-4, msg=name)

    def test_future_text_and_other_samples_cannot_leak_into_current_interaction(self):
        model = self.make_model(activation_checkpointing=False)
        batch, features = example_batch()
        original = model.trajectory(batch, features)
        changed = copy.deepcopy(batch)
        # Change only final answer and all of a DIFFERENT sample's token values.
        n = int(changed["attention_mask"][0].sum())
        changed["input_ids"][0, n - 2] = 44
        other = changed["attention_mask"][1] & (changed["input_ids"][1] < 70)
        changed["input_ids"][1, other] = 45
        actual = model.trajectory(changed, features)
        for segment in (0, 1):
            torch.testing.assert_close(original.interactions[0, segment], actual.interactions[0, segment])

    def test_ce_includes_first_text_after_each_image_and_final_answer(self):
        model = self.make_model(activation_checkpointing=False)
        batch, features = example_batch([(0, [[3], "image", [4], "image", [5, 90]])])
        trajectory = model.trajectory(batch, features)
        expected = nn.functional.cross_entropy(model.lm_head(trajectory.hidden[:, :-1]).reshape(-1, 96),
                                                batch["labels"][:, 1:].flatten(), reduction="sum")
        actual = model(batch, features)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(batch["labels"][batch["labels"] != -100].tolist(), [3, 4, 5, 90])
        actual.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_zero_helper_batch_and_masked_dummy_have_valid_gradients(self):
        model = self.make_model()
        batch, features = example_batch([(0, [[3, 4, 90]])])
        batch["labels"].fill_(-100)
        loss = model(batch, features)
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_features_and_future_local_text_are_validated(self):
        model = self.make_model()
        batch, features = example_batch()
        with self.assertRaisesRegex(ValueError, "Complete question AND helper"):
            model(batch, features[:-1])
        broken = copy.deepcopy(batch)
        broken["segments"][0][0]["text"].append(broken["segments"][0][0]["position"] + 1)
        with self.assertRaisesRegex(ValueError, "strictly before"):
            model(broken, features)
        broken = copy.deepcopy(batch)
        broken["images"][0][1]["feature_start"] = 0
        with self.assertRaisesRegex(ValueError, "ordered merger"):
            image_memories(features, broken["images"], model_config())

    def test_wide_image_coordinates_stay_exact_with_bfloat16_features(self):
        features = torch.zeros(300, 24, dtype=torch.bfloat16)
        images = [[dict(feature_start=0, feature_end=300, grid=[1, 2, 600],
                        kind="question", image_id=0)]]
        coordinates = image_memories(features, images, model_config())[0][0].coordinates
        self.assertEqual(coordinates[257, 0].item(), 257)
        self.assertTrue(torch.all(coordinates[1:, 0] - coordinates[:-1, 0] == 1))


if __name__ == "__main__":
    unittest.main()
