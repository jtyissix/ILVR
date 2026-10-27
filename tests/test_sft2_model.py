"""Causal order, intentional helper conditioning, four objectives and gradients."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from src.interaction.data import position_ids_for_images
from src.sft2.config import Stage2Config
from src.sft2.model import LatentModel, pool_helper
from tests.interaction_fixtures import TinyDecoder


def recipe(**changes):
    values = dict(fusion_dim=16, fusion_heads=2, fusion_ffn_dim=32, fusion_score_dim=8,
                  attention_backend="sdpa", ce_chunk_size=3, activation_checkpointing=False)
    values.update(changes)
    return Stage2Config(**values)


def model_config(hidden=24):
    return SimpleNamespace(hidden_size=hidden, image_token_id=80, latent_start_id=70, latent_end_id=72,
                           vision_config=SimpleNamespace(spatial_merge_size=2))


def example_batch(specs=None, hidden=24):
    specs = specs or [(2, [[3, 4], "action", "helper", [5], "action", "helper", [6, 90]]),
                      (0, [[7], "action", "helper", "helper", [8, 90]]),
                      (0, [[9, 10, 90]])]
    records, features, feature_offset = [], [], 0
    for question_count, steps in specs:
        ids, labels, images, qgrids = [1], [-100], [], []
        def image(kind, index):
            nonlocal feature_offset
            grid = [1, 4, 6] if index % 2 else [1, 6, 4]
            count = grid[1] * grid[2] // 4
            images.append(dict(kind=kind, image_id=index, grid=grid,
                               feature_start=feature_offset, feature_end=feature_offset + count))
            features.append(torch.randn(count, hidden))
            feature_offset += count
            if kind == "question":
                ids.extend([82] + [80] * count + [83])
                labels.extend([-100] * (count + 2))
                qgrids.append(grid)
        for i in range(question_count):
            image("question", i)
        ids.append(2)
        labels.append(-100)
        memory, actions, segments = [], [], []
        open_block = False
        def boundary(token):
            ids.append(token)
            labels.append(token)
        for step_index, step in enumerate(steps):
            if step == "action":
                boundary(70)
                positions = list(range(len(ids), len(ids) + 4))
                actions.append(dict(positions=positions, target_ids=[15, 16, 90], teacher_position=0))
                ids.extend([0] * 4)
                labels.extend([-100] * 4)
                memory.extend(positions)
                open_block = step_index + 1 < len(steps) and steps[step_index + 1] == "helper"
                if not open_block:
                    boundary(72)
            elif step == "helper":
                if not open_block:
                    boundary(70)
                position = len(ids)
                visual_positions = list(range(position + 3, position + 19))
                segments.append(dict(position=position, visual_positions=visual_positions,
                                     text=list(memory), image_index=len(images)))
                image("helper", len(segments) - 1)
                ids.extend([0] * 19)
                labels.extend([-100] * 19)
                boundary(72)
                open_block = False
                memory = []
            else:
                memory.extend(range(len(ids), len(ids) + len(step)))
                ids.extend(step)
                labels.extend(step)
        pos = position_ids_for_images(ids, qgrids, {"image_token_id": 80, "vision_config": {"spatial_merge_size": 2}})
        records.append((ids, labels, images, actions, segments, pos))
    length = max(len(record[0]) for record in records)
    ids = torch.zeros(len(records), length, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    positions = torch.zeros(3, len(records), length, dtype=torch.long)
    for row, record in enumerate(records):
        n = len(record[0])
        ids[row, :n], labels[row, :n] = torch.tensor(record[0]), torch.tensor(record[1])
        mask[row, :n] = True
        positions[:, row, :n] = record[-1]
    amax = max(len(record[3]) for record in records)
    batch = dict(input_ids=ids, labels=labels, attention_mask=mask, position_ids=positions,
                 latent_boundary_ids=[70, 72],
                 images=[record[2] for record in records], actions=[record[3] for record in records],
                 segments=[record[4] for record in records], sample_mask=torch.ones(len(records), dtype=torch.bool),
                 teacher_targets=torch.randn(len(records), amax, hidden))
    return batch, torch.cat(features) if features else None


class LatentModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(1207)

    def make_model(self, **changes):
        return LatentModel(TinyDecoder(), nn.Linear(24, 96, bias=False), model_config(), recipe(**changes))

    def test_four_losses_train_both_decoders_and_interaction_but_not_targets(self):
        model = self.make_model(activation_checkpointing=True)
        batch, features = example_batch()
        features.requires_grad_(True)
        batch["teacher_targets"].requires_grad_(True)
        losses = model(batch, features)
        self.assertEqual(set(losses), {"ce", "visual", "action_words", "action_state"})
        self.assertTrue(all(float(value) > 0 and bool(torch.isfinite(value)) for value in losses.values()))
        sum(losses.values()).backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(batch["teacher_targets"].grad)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        for name in ("interaction.query_mlp.0.weight", "interaction.visual_projection.weight",
                     "interaction.image_projection.weight", "decoder.embed_tokens.weight",
                     "aux_decoder.layers.0.self_attn.q_proj.weight", "aux_lm_head.weight"):
            self.assertGreater(float(dict(model.named_parameters())[name].grad.norm()), 0, name)
        self.assertFalse(any("warmup_adapter" in name for name, _ in model.named_parameters()))
        self.assertIsInstance(model.aux_decoder.embed_tokens, nn.Identity)

    def test_incremental_recompute_and_checkpoint_gradients_agree(self):
        original = self.make_model()
        batch, features = example_batch([(1, [[3], "action", "helper", [4, 90]]),
                                         (0, [[5, 6], "action", "helper", [7, 90]])])
        observations = []
        for execution, checkpointing in (("incremental", False), ("incremental", True), ("recompute", True)):
            model = copy.deepcopy(original)
            model.recipe.execution = execution
            model.recipe.activation_checkpointing = checkpointing
            model.executor.activation_checkpointing = model.aux_executor.activation_checkpointing = checkpointing
            losses = model(batch, features)
            sum(losses.values()).backward()
            observations.append(({key: value.detach() for key, value in losses.items()},
                                 {name: parameter.grad.clone() for name, parameter in model.named_parameters()}))
        for losses, gradients in observations[1:]:
            for key in losses:
                torch.testing.assert_close(losses[key], observations[0][0][key], atol=5e-5, rtol=5e-5)
            for name in gradients:
                torch.testing.assert_close(gradients[name], observations[0][1][name], atol=1e-4, rtol=2e-3, msg=name)

    def test_overlapping_batch_events_match_individual_losses_and_gradients(self):
        batched = self.make_model(activation_checkpointing=True)
        individual = copy.deepcopy(batched)
        batch, features = example_batch([(0, [[3], "action", "helper", [4], "action", "helper", [5, 90]]),
                                         (0, [[6, 7], "action", "helper", [8, 90]]),
                                         (0, [[9, 10, 11], "action", "helper", [12, 90]])])
        actual = batched(batch, features)
        expected = {key: 0 for key in actual}
        for row, images in enumerate(batch["images"]):
            n = int(batch["attention_mask"][row].sum())
            start, end = images[0]["feature_start"], images[-1]["feature_end"]
            own_images = copy.deepcopy(images)
            for item in own_images:
                item["feature_start"] -= start
                item["feature_end"] -= start
            own = {key: batch[key][row:row + 1, :n] for key in ("input_ids", "labels", "attention_mask")}
            own.update(position_ids=batch["position_ids"][:, row:row + 1, :n], images=[own_images],
                       latent_boundary_ids=batch["latent_boundary_ids"],
                       actions=[batch["actions"][row]], segments=[batch["segments"][row]],
                       sample_mask=batch["sample_mask"][row:row + 1],
                       teacher_targets=batch["teacher_targets"][row:row + 1, :len(batch["actions"][row])])
            losses = individual(own, features[start:end])
            for key in expected:
                expected[key] = expected[key] + losses[key]
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], atol=5e-5, rtol=5e-5, msg=key)
        sum(actual.values()).backward()
        sum(expected.values()).backward()
        expected_params = dict(individual.named_parameters())
        for name, parameter in batched.named_parameters():
            torch.testing.assert_close(parameter.grad, expected_params[name].grad, atol=1e-4, rtol=2e-3, msg=name)

    def test_checkpoint_does_not_retain_full_sequence_storage_per_latent_step(self):
        model = self.make_model(activation_checkpointing=True)
        batch, features = example_batch([(0, [[3], "action", "helper", [4, 90]])])
        full_bytes = batch["input_ids"].numel() * 24 * 4
        storages = set()
        def pack(tensor):
            if tensor.ndim == 3 and tensor.shape[-1] == 24 and tensor.untyped_storage().nbytes() == full_bytes:
                storages.add(tensor.untyped_storage().data_ptr())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            losses = model(batch, features)
        self.assertLessEqual(len(storages), 2, "Full trajectory backing storage was retained once per latent")
        sum(losses.values()).backward()

    def test_helper_prior_intentionally_changes_b_and_visual_but_not_preceding_action(self):
        model = self.make_model()
        batch, features = example_batch([(1, [[3], "action", "helper", [4, 90]])])
        old = model.trajectory(batch, features)
        modified = features.clone()
        helper = batch["images"][0][1]
        modified[helper["feature_start"]:helper["feature_end"]] += torch.randn(24) * 5
        new = model.trajectory(batch, modified)
        for k in range(4):
            torch.testing.assert_close(old.actions[0, 0, k], new.actions[0, 0, k])
        end = batch["actions"][0][0]["positions"][-1]
        torch.testing.assert_close(old.hidden[0, end], new.hidden[0, end])
        self.assertFalse(torch.allclose(old.interactions[0, 0], new.interactions[0, 0]))
        self.assertFalse(torch.allclose(old.visuals[0, 0, 0], new.visuals[0, 0, 0]))
        old_words = model.action_word_loss(old, batch, batch["sample_mask"])
        new_words = model.action_word_loss(new, batch, batch["sample_mask"])
        torch.testing.assert_close(old_words, new_words)

    def test_visual_inputs_are_previous_outputs_not_teacher_features_or_shifted_states(self):
        model = self.make_model()
        batch, features = example_batch([(0, [[3], "action", "helper", [4, 90]])])
        trajectory = model.trajectory(batch, features)
        for index, action in enumerate(batch["actions"][0]):
            for k, position in enumerate(action["positions"]):
                torch.testing.assert_close(trajectory.actions[0, index, k], trajectory.hidden[0, position - 1])
        for index, segment in enumerate(batch["segments"][0]):
            for k, position in enumerate(segment["visual_positions"]):
                torch.testing.assert_close(trajectory.visuals[0, index, k], trajectory.hidden[0, position - 1])
                self.assertFalse(torch.allclose(trajectory.visuals[0, index, k], trajectory.visual_targets[0, index][k]))
        self.assertEqual(len(trajectory.visuals), 16)
        self.assertEqual(len(trajectory.actions), 4)
        action = batch["actions"][0][0]
        target = batch["teacher_targets"][0, 0]
        actual_state = model(batch, features)["action_state"]
        expected_state = (trajectory.hidden[0, action["positions"][-1]] - target).square().mean() / target.square().mean()
        torch.testing.assert_close(actual_state, expected_state)

    def test_visual_loss_reaches_interaction_and_action_generator(self):
        model = self.make_model()
        batch, features = example_batch([(1, [[3], "action", "helper", [4, 90]])])
        model(batch, features)["visual"].backward()
        self.assertGreater(float(model.interaction.query_mlp[0].weight.grad.norm()), 0)
        self.assertGreater(float(model.decoder.embed_tokens.weight.grad.norm()), 0)
        self.assertIsNone(model.aux_lm_head.weight.grad)

    def test_current_action_losses_do_not_update_future_interaction(self):
        model = self.make_model()
        batch, features = example_batch([(1, [[3], "action", "helper", [4, 90]])])
        losses = model(batch, features)
        (losses["action_words"] + losses["action_state"]).backward()
        # Zero dependencies may supply zero-valued gradients; neither objective
        # may use the helper-conditioned future state to recover this action.
        for parameter in model.interaction.parameters():
            self.assertTrue(parameter.grad is None or float(parameter.grad.norm()) == 0)

    def test_mixed_text_memory_includes_all_four_action_outputs(self):
        model = self.make_model()
        batch, features = example_batch([(0, [[3], "action", "helper", [4, 90]])])
        result = model.trajectory(batch, features)
        segment = batch["segments"][0][0]
        self.assertEqual(len(result.text_states[0, 0]), 5)
        torch.testing.assert_close(result.text_states[0, 0], result.hidden[0, segment["text"]])
        for position in batch["actions"][0][0]["positions"]:
            self.assertEqual(int(batch["labels"][0, position]), -100)
            self.assertIn(position, segment["text"])

    def test_earlier_interaction_affects_later_actual_context(self):
        model = self.make_model()
        batch, features = example_batch([(0, [[3], "action", "helper", [4], "action", "helper", [5, 90]])])
        old = model.trajectory(batch, features)
        calls = [0]
        offset = torch.randn(1, 1, 24)
        def perturb(_module, _args, output):
            calls[0] += 1
            return output + offset if calls[0] == 1 else output
        hook = model.interaction.register_forward_hook(perturb)
        new = model.trajectory(batch, features)
        hook.remove()
        torch.testing.assert_close(old.text_states[0, 0], new.text_states[0, 0])
        self.assertFalse(torch.allclose(old.text_states[0, 1], new.text_states[0, 1]))
        self.assertFalse(torch.allclose(old.actions[0, 1, 0], new.actions[0, 1, 0]))

    def test_no_actions_no_helpers_and_dummy_losses_are_zero_with_complete_gradients(self):
        for specs in ([(0, [[3, 90]])], [(0, [[3], "action", "helper", [4, 90]])]):
            model = self.make_model()
            batch, features = example_batch(specs)
            batch["sample_mask"].fill_(False)
            losses = model(batch, features)
            self.assertTrue(all(float(value) == 0 for value in losses.values()))
            sum(losses.values()).backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertEqual(float(parameter.grad.norm()), 0, name)

    def test_action_without_helper_is_supported(self):
        model = self.make_model()
        batch, _ = example_batch([(0, [[3], "action", [4, 90]])])
        losses = model(batch)
        self.assertEqual(float(losses["visual"]), 0)
        self.assertGreater(float(losses["action_words"]), 0)
        self.assertGreater(float(losses["action_state"]), 0)

    def test_boundary_ce_and_start_hidden_drive_the_continuous_block(self):
        model = self.make_model()
        batch, features = example_batch([(0, [[3], "action", "helper", [4, 90]])])
        trajectory = model.trajectory(batch, features)
        start = batch["actions"][0][0]["positions"][0] - 1
        end = batch["segments"][0][0]["visual_positions"][-1] + 1
        self.assertEqual(end - start + 1, 25)
        torch.testing.assert_close(trajectory.actions[0, 0, 0], trajectory.hidden[0, start])
        # The last visual output predicts END; its loss differentiates through
        # interaction, the four action inputs, and START's embedding.
        loss = F.cross_entropy(model.lm_head(trajectory.hidden[:, end - 1]), torch.tensor([72]))
        loss.backward()
        self.assertGreater(float(model.decoder.embed_tokens.weight.grad[70].norm()), 0)
        self.assertGreater(float(model.interaction.query_mlp[0].weight.grad.norm()), 0)
        self.assertGreater(float(model.lm_head.weight.grad[72].norm()), 0)

    def test_missing_unbalanced_and_unsupervised_boundaries_are_rejected(self):
        model = self.make_model()
        batch, features = example_batch([(0, [[3], "action", "helper", [4, 90]])])
        start = batch["actions"][0][0]["positions"][0] - 1
        end = batch["segments"][0][0]["visual_positions"][-1] + 1
        for position in (start, end):
            broken = copy.deepcopy(batch)
            broken["labels"][0, position] = -100
            with self.assertRaisesRegex(ValueError, "CE labels"):
                model(broken, features)
            broken = copy.deepcopy(batch)
            broken["input_ids"][0, position] = 3
            with self.assertRaisesRegex(ValueError, "Unbalanced"):
                model(broken, features)
        del batch["latent_boundary_ids"]
        with self.assertRaisesRegex(ValueError, "start/end metadata"):
            model(batch, features)

    def test_future_answer_and_other_row_do_not_change_current_prior_path(self):
        model = self.make_model()
        batch, features = example_batch()
        old = model.trajectory(batch, features)
        altered = copy.deepcopy(batch)
        length = int(altered["attention_mask"][0].sum())
        altered["input_ids"][0, length - 2] = 44
        allowed = altered["attention_mask"][1] & (altered["input_ids"][1] > 0) & (altered["input_ids"][1] < 70)
        altered["input_ids"][1, allowed] = 45
        new = model.trajectory(altered, features)
        for index in range(2):
            torch.testing.assert_close(old.interactions[0, index], new.interactions[0, index])
            for k in range(16):
                torch.testing.assert_close(old.visuals[0, index, k], new.visuals[0, index, k])

    def test_invalid_positions_targets_and_images_fail_before_silent_training(self):
        model = self.make_model()
        batch, features = example_batch()
        broken = copy.deepcopy(batch)
        broken["actions"][0][0]["positions"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 4"):
            model(broken, features)
        broken = copy.deepcopy(batch)
        broken["teacher_targets"] = None
        with self.assertRaisesRegex(ValueError, "teacher targets"):
            model(broken, features)
        broken = copy.deepcopy(batch)
        broken["segments"][0][0]["text"].append(broken["segments"][0][0]["position"])
        with self.assertRaisesRegex(ValueError, "strictly before"):
            model(broken, features)
        with self.assertRaisesRegex(ValueError, "Missing or incomplete"):
            model(batch, features[:-1])


class PoolingTests(unittest.TestCase):
    def test_spatial_order_rectangular_grid_and_no_tail_discard(self):
        features = torch.arange(5 * 9, dtype=torch.float32)[:, None].requires_grad_(True)
        result = pool_helper(features, [1, 10, 18], 2)
        expected = F.adaptive_avg_pool2d(features.detach().T.reshape(1, 1, 5, 9), (4, 4)).flatten()[:, None]
        torch.testing.assert_close(result, expected)
        self.assertEqual(result.shape, (16, 1))
        self.assertFalse(result.requires_grad)
        altered = features.detach().clone()
        altered[-1] += 100
        changed = pool_helper(altered, [1, 10, 18], 2)
        self.assertGreater(float(changed[-1]), float(result[-1]))

    def test_tiny_grid_and_bfloat16(self):
        features = torch.tensor([[1., 3.], [5., 7.]], dtype=torch.bfloat16)
        result = pool_helper(features, [1, 2, 4], 2)
        self.assertEqual(result.dtype, features.dtype)
        self.assertEqual(result.shape, (16, 2))
        self.assertTrue(bool(torch.isfinite(result).all()))
        with self.assertRaisesRegex(ValueError, "exact post-merger"):
            pool_helper(features, [1, 4, 4], 2)
        with self.assertRaisesRegex(ValueError, "still-image"):
            pool_helper(features, [2, 2, 2], 2)


if __name__ == "__main__":
    unittest.main()
