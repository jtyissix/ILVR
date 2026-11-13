"""Differentiable relaxation, exact CE masking and per-type diversity."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from src.sft3.config import Stage3Config, parse_collapse_targets
from src.sft3.loss import LOSS_NAMES, distance_weighted_diversity
from src.sft3.model import RelaxationModel
from src.sft3.train import combined_loss, loss_weights, optimizer_groups
from tests.interaction_fixtures import TinyDecoder
from tests.test_sft2_model import example_batch, model_config


def recipe(**changes):
    return Stage3Config(**{**dict(attention_backend="sdpa", activation_checkpointing=False,
                                 ce_chunk_size=3), **changes})


def example(specs=None, hidden=24):
    batch, features = example_batch(specs, hidden)
    kept, rows, offset = [], [], 0
    for images in batch["images"]:
        row = []
        for item in images:
            if item["kind"] != "question":
                continue
            value = features[item["feature_start"]:item["feature_end"]]
            kept.append(value)
            row.append({**item, "feature_start": offset, "feature_end": offset + len(value)})
            offset += len(value)
        rows.append(row)
    batch["images"] = rows
    batch.pop("teacher_targets", None)
    return batch, torch.cat(kept) if kept else None


def make_model(**changes):
    return RelaxationModel(TinyDecoder(), nn.Linear(24, 96, bias=False), model_config(), recipe(**changes))


def own_row(batch, features, row):
    n = int(batch["attention_mask"][row].sum())
    own = {key: batch[key][row:row + 1, :n] for key in ("input_ids", "labels", "attention_mask")}
    images = copy.deepcopy(batch["images"][row])
    start, end = (images[0]["feature_start"], images[-1]["feature_end"]) if images else (0, 0)
    for image in images:
        image["feature_start"] -= start
        image["feature_end"] -= start
    own.update(position_ids=batch["position_ids"][:, row:row + 1, :n], images=[images],
               latent_boundary_ids=batch["latent_boundary_ids"], actions=[batch["actions"][row]],
               segments=[batch["segments"][row]], sample_mask=batch["sample_mask"][row:row + 1])
    return own, features[start:end] if images else None


class RelaxationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(1903)

    def test_all_selector_combinations_and_invalid_values(self):
        for selection in (0, 1, 2, 3, 12, 13, 23, 123, 321):
            self.assertEqual(parse_collapse_targets(selection), parse_collapse_targets(str(selection)))
            recipe(collapse_targets=selection).validate()
        for selection in (True, None, [], 1.0, "", "01", "11", "14", "1,2", -1):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                recipe(collapse_targets=selection).validate()
        for changes in ({"collapse_weight": -1}, {"collapse_weight": float("nan")},
                        {"collapse_alpha": .5}, {"collapse_margin": 1.1}, {"collapse_epsilon": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                recipe(**changes).validate()

    def test_diversity_matches_hand_computed_distance_weights(self):
        # All centroids equal: three pairs at distances 1, 2, 1.
        steps = torch.ones(3, 4, 5, requires_grad=True)
        actual = distance_weighted_diversity(steps, margin=.8, alpha=2, epsilon=.1)
        expected = .2 * (2 * (.1 + .9 * .25) + 1)
        self.assertAlmostEqual(actual.item(), expected, places=6)
        self.assertEqual(distance_weighted_diversity(torch.eye(3)[:, None], margin=.8).item(), 0)
        for count in (0, 1):
            value = torch.zeros(count, 4, 5, requires_grad=True)
            zero = distance_weighted_diversity(value)
            self.assertEqual(zero.item(), 0)
            zero.backward()
            self.assertIsNotNone(value.grad)

    def test_diversity_gradient_and_bfloat16_zero_vectors_are_finite(self):
        steps = (torch.ones(3, 4, 5) + torch.randn(3, 4, 5) * .1).requires_grad_()
        loss = distance_weighted_diversity(steps)
        loss.backward()
        self.assertGreater(steps.grad.norm().item(), 0)
        value = torch.zeros(3, 4, 5, dtype=torch.bfloat16, requires_grad=True)
        distance_weighted_diversity(value).backward()
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_all_latents_are_previous_hidden_states_and_ce_is_text_only(self):
        model = make_model()
        batch, features = example([(1, [[3], "action", "helper", [4, 90]])])
        features.requires_grad_()
        trajectory = model.trajectory(batch, features)
        groups = {"text": batch["actions"][0][0]["positions"],
                  "interaction": list(range(batch["segments"][0][0]["position"], batch["segments"][0][0]["position"] + 3)),
                  "visual": batch["segments"][0][0]["visual_positions"]}
        for kind, positions in groups.items():
            torch.testing.assert_close(trajectory.latents[kind][0, 0], trajectory.hidden[0, [p - 1 for p in positions]])
        losses = model(batch, features)
        expected = F.cross_entropy(model.lm_head(trajectory.hidden[:, :-1]).reshape(-1, 96),
                                   batch["labels"][:, 1:].reshape(-1), reduction="sum", ignore_index=-100)
        torch.testing.assert_close(losses["ce"], expected)
        losses["ce"].backward()
        self.assertIsNone(features.grad)
        self.assertEqual(set(losses), set(LOSS_NAMES))
        for name, parameter in model.named_parameters():
            self.assertTrue(name.startswith(("decoder.", "lm_head.")), name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        # Placeholder embedding is never consumed by an attended latent slot.
        self.assertEqual(model.decoder.embed_tokens.weight.grad[0].abs().sum().item(), 0)
        self.assertFalse(hasattr(model, "interaction"))
        self.assertFalse(hasattr(model, "aux_decoder"))
        self.assertEqual(sum(len(g["params"]) for g in optimizer_groups(model, model.recipe)), len(list(model.parameters())))

    def test_ce_reaches_pre_latent_states_through_recurrence(self):
        model = make_model()
        batch, features = example([(0, [[3], "action", "helper", [4, 90]])])
        # Only final ordinary text carries CE. Retain each latent-step output.
        last = batch["segments"][0][0]["visual_positions"][-1]
        captured = []
        original = model.executor.chunk
        def chunk(*args, **kwargs):
            output = original(*args, **kwargs)
            output[0].retain_grad()
            captured.append(output[0])
            return output
        with patch.object(model.executor, "chunk", side_effect=chunk):
            trajectory = model.trajectory(batch, features)
        loss = F.cross_entropy(model.lm_head(trajectory.hidden[0, last + 1:last + 3]),
                               batch["labels"][0, last + 2:last + 4], reduction="sum")
        loss.backward()
        self.assertGreater(len(captured), 23)
        for hidden in captured[:-1]:
            self.assertGreater(hidden.grad[:, -1].norm().item(), 0)

    def test_disabled_loss_never_calls_diversity_and_selections_are_exact(self):
        batch, features = example([(0, [[3], "action", "helper", [4], "action", "helper", [5, 90]])])
        for selection in ("0", "1", "2", "3", "12", "13", "23", "123"):
            model = make_model(collapse_targets=selection, collapse_margin=0)
            with patch("src.sft3.model.distance_weighted_diversity", wraps=distance_weighted_diversity) as call:
                losses = model(batch, features)
            self.assertEqual(call.call_count, len(model.recipe.selected_types))
            for kind in ("text", "interaction", "visual"):
                if kind not in model.recipe.selected_types:
                    self.assertEqual(losses["collapse_" + kind].item(), 0)
        with patch("src.sft3.model.distance_weighted_diversity", side_effect=AssertionError("disabled")):
            make_model(collapse_targets="123", collapse_weight=0)(batch, features)

    def test_each_selected_loss_has_a_backbone_gradient(self):
        batch, features = example([(0, [[3], "action", "helper", [4], "action", "helper", [5, 90]])])
        for selection, kind in (("1", "text"), ("2", "interaction"), ("3", "visual")):
            model = make_model(collapse_targets=selection, collapse_margin=0)
            model(batch, features)["collapse_" + kind].backward()
            self.assertGreater(model.decoder.layers[0].self_attn.q_proj.weight.grad.norm().item(), 0)

    def test_incremental_checkpoint_and_recompute_values_and_gradients_agree(self):
        original = make_model(collapse_targets="123", collapse_margin=0)
        batch, features = example()
        results = []
        for execution, checkpointing in (("incremental", False), ("incremental", True), ("recompute", True)):
            model = copy.deepcopy(original)
            model.recipe.execution = execution
            model.executor.activation_checkpointing = model.recipe.activation_checkpointing = checkpointing
            losses = model(batch, features)
            sum(losses.values()).backward()
            results.append((losses, {n: p.grad for n, p in model.named_parameters()}))
        for losses, gradients in results[1:]:
            for key in LOSS_NAMES:
                torch.testing.assert_close(losses[key], results[0][0][key], rtol=5e-5, atol=5e-5)
            for name, value in gradients.items():
                torch.testing.assert_close(value, results[0][1][name], rtol=2e-3, atol=1e-4, msg=name)

    def test_ragged_batch_matches_individual_examples_without_cross_sample_pairs(self):
        batched = make_model(collapse_targets="123", collapse_margin=0)
        individual = copy.deepcopy(batched)
        batch, features = example()
        losses = batched(batch, features)
        expected = dict.fromkeys(LOSS_NAMES, 0)
        for row in range(len(batch["input_ids"])):
            values = individual(*own_row(batch, features, row))
            expected = {k: expected[k] + values[k] for k in LOSS_NAMES}
        for key in LOSS_NAMES:
            torch.testing.assert_close(losses[key], expected[key], atol=5e-5, rtol=5e-5)
        sum(losses.values()).backward()
        sum(expected.values()).backward()
        reference = dict(individual.named_parameters())
        for name, parameter in batched.named_parameters():
            torch.testing.assert_close(parameter.grad, reference[name].grad, rtol=2e-3, atol=1e-4, msg=name)

    def test_dummy_only_has_zero_loss_and_all_parameter_dependencies(self):
        model = make_model(collapse_targets="123")
        batch, features = example()
        batch["sample_mask"].zero_()
        batch["labels"].fill_(-100)
        losses = model(batch, features)
        self.assertTrue(all(v.item() == 0 for v in losses.values()))
        sum(losses.values()).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertEqual(parameter.grad.abs().sum().item(), 0, name)

    def test_bad_labels_overlaps_boundaries_and_helper_inputs_fail(self):
        original, features = example([(0, [[3], "action", "helper", [4, 90]])])
        model = make_model()
        for mutation in (lambda b: b["labels"][0].__setitem__(b["actions"][0][0]["positions"][0], 3),
                         lambda b: b["segments"][0][0].__setitem__("position", b["actions"][0][0]["positions"][0]),
                         lambda b: b.__setitem__("latent_boundary_ids", [70, 71]),
                         lambda b: b["images"][0].append({"kind": "helper"})):
            batch = copy.deepcopy(original)
            mutation(batch)
            with self.assertRaises(ValueError):
                model(batch, features)

    def test_global_counts_compensate_rank_and_accumulation_averages(self):
        config = recipe(collapse_targets="23", collapse_weight=.6)
        counts = dict(ce=17, collapse_text=0, collapse_interaction=3, collapse_visual=7)
        values = {name: torch.tensor(float(i + 1), requires_grad=True) for i, name in enumerate(LOSS_NAMES)}
        combined_loss(values, counts, loss_weights(config), 3, 2).backward()
        expected = dict(ce=6 / 17, collapse_text=0, collapse_interaction=.3 * 6 / 3,
                        collapse_visual=.3 * 6 / 7)
        for name in LOSS_NAMES:
            self.assertAlmostEqual(values[name].grad.item(), expected[name], places=6)


if __name__ == "__main__":
    unittest.main()
