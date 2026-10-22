"""Numerical and gradient contracts for the warm-up interaction modules."""
import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from src.sft1.modules import ImageMemory, InteractionCore, WarmupImageAdapter


class SFT1ModulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(41)
        self.hidden, self.dim = 24, 16

    def core(self, **kwargs):
        return InteractionCore(self.hidden, self.dim, heads=2, ffn_dim=32, score_dim=7, **kwargs)

    def adapter(self):
        return WarmupImageAdapter(self.hidden, self.dim, heads=2, ffn_dim=32)

    def image(self, length, index, requires_grad=False):
        return ImageMemory(
            torch.randn(length, self.hidden, requires_grad=requires_grad),
            torch.stack((torch.arange(length) % 3, torch.arange(length) // 3), -1), index,
        )

    def inputs(self):
        text = torch.randn(2, 5, self.hidden, requires_grad=True)
        latents = torch.randn(2, 16, self.hidden, requires_grad=True)
        mask = torch.tensor([[True] * 5, [True, True, False, False, False]])
        images = [[self.image(4, 0, True), self.image(7, 1, True)], [self.image(3, 0, True)]]
        return text, latents, mask, images

    def test_ce_reaches_adapter_core_and_text_but_not_frozen_images(self):
        core, adapter = self.core(), self.adapter()
        text, _, mask, images = self.inputs()
        helper = torch.randn(2, 6, self.hidden, requires_grad=True)
        coordinates = torch.randn(2, 6, 2)
        vectors = adapter(helper, coordinates, torch.ones(2, 6, dtype=torch.bool))
        self.assertEqual(vectors.shape, (2, 16, self.hidden))
        vectors.retain_grad()
        logits = nn.Linear(self.hidden, 13)(core(text, vectors, mask, images))
        self.assertEqual(logits.shape, (2, 3, 13))
        F.cross_entropy(logits.reshape(-1, 13), torch.tensor([3, 4, 5, 8, 9, 10])).backward()
        for module in (core, adapter):
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for name in ("visual_projection.weight", "query_mlp.0.weight", "query_mlp.2.weight",
                     "image_projection.weight", "image_attention.v.weight", "image_gate.weight",
                     "score_query.weight", "score_read.weight", "score_index.weight", "score_output.weight",
                     "image_index_encoding.projection.weight", "spatial_encoding.projection.weight"):
            self.assertGreater(float(dict(core.named_parameters())[name].grad.norm()), 0, name)
        self.assertGreater(float(adapter.queries.grad.norm()), 0)
        self.assertGreater(float(adapter.input_projection.weight.grad.norm()), 0)
        self.assertGreater(float(vectors.grad.norm()), 0)
        self.assertGreater(float(text.grad[mask].norm()), 0)
        self.assertEqual(float(text.grad[~mask].abs().sum()), 0)
        self.assertIsNone(helper.grad)
        for sample in images:
            for memory in sample:
                self.assertIsNone(memory.features.grad)

    def test_query_is_ordered_sixteen_slot_concat_mlp(self):
        core = self.core()
        self.assertEqual(core.query_mlp[0].in_features, 16 * self.dim)
        latents = torch.randn(2, 16, self.hidden)
        query, projected = core.build_query(latents)
        expected = core.query_mlp[2](F.silu(core.query_mlp[0](projected.flatten(1)))).reshape(2, 3, self.dim)
        torch.testing.assert_close(query, expected, rtol=0, atol=0)
        permuted, _ = core.build_query(latents.flip(1))
        self.assertFalse(torch.allclose(query, permuted))
        for size in (0, 8, 15, 17):
            with self.assertRaisesRegex(ValueError, "latents must have shape"):
                core.build_query(torch.randn(1, size, self.hidden))

    def test_disabled_image_branch_is_exact_shared_core_baseline(self):
        core = self.core()
        baseline = copy.deepcopy(core)
        baseline.image_branch_enabled = False
        text, latents, mask, images = self.inputs()
        without_images = baseline(text, latents, mask, images)
        torch.testing.assert_close(core(text, latents, mask, None), without_images, rtol=0, atol=0)
        with torch.no_grad():
            core.alpha_image.zero_()
        torch.testing.assert_close(core(text, latents, mask, images), without_images, rtol=0, atol=0)
        # A disabled branch must not even inspect its unused memory payload.
        torch.testing.assert_close(baseline(text, latents, mask, ["unused"]), without_images, rtol=0, atol=0)

    def test_image_branch_mode_does_not_change_parameter_initialization(self):
        torch.manual_seed(99)
        with_images = self.core(image_branch_enabled=True)
        torch.manual_seed(99)
        without_images = self.core(image_branch_enabled=False)
        for key, value in with_images.state_dict().items():
            torch.testing.assert_close(value, without_images.state_dict()[key], rtol=0, atol=0)

    def test_images_are_isolated_between_samples_and_padding_is_ignored(self):
        core = self.core()
        text, latents, mask, images = self.inputs()
        expected = core(text, latents, mask, images)
        altered = [[images[0][0], images[0][1]], [self.image(11, 0), self.image(2, 1)]]
        changed = core(text, latents, mask, altered)
        torch.testing.assert_close(changed[0], expected[0], rtol=0, atol=0)
        self.assertFalse(torch.allclose(changed[1], expected[1], rtol=1e-6, atol=1e-7))
        padded = []
        for sample in images:
            local = []
            for memory in sample:
                length = memory.features.shape[0]
                local.append(ImageMemory(
                    torch.cat((memory.features, torch.full((5, self.hidden), float("nan")))),
                    torch.cat((memory.coordinates.float(), torch.full((5, 2), float("nan")))),
                    memory.image_index,
                    torch.tensor([True] * length + [False] * 5),
                ))
            padded.append(local)
        padded_text = text.detach().clone()
        padded_text[~mask] = float("nan")
        torch.testing.assert_close(core(padded_text, latents, mask, padded), expected, rtol=1e-5, atol=1e-6)

    def test_question_image_position_and_occurrence_are_used(self):
        core = self.core()
        text, latents, mask, images = self.inputs()
        original = core(text, latents, mask, images)
        image0 = images[0][0]
        shifted = [[ImageMemory(image0.features, image0.coordinates + 3, image0.image_index), images[0][1]], images[1]]
        reindexed = [[ImageMemory(image0.features, image0.coordinates, 5), images[0][1]], images[1]]
        self.assertFalse(torch.allclose(original[0], core(text, latents, mask, shifted)[0], rtol=1e-6, atol=1e-7))
        self.assertFalse(torch.allclose(original[0], core(text, latents, mask, reindexed)[0], rtol=1e-6, atol=1e-7))

    def test_empty_text_images_and_all_masked_rows_are_finite(self):
        core = self.core()
        latents = torch.randn(2, 16, self.hidden)
        empty_text = torch.empty(2, 0, self.hidden)
        empty_mask = torch.empty(2, 0, dtype=torch.bool)
        expected = core(empty_text, latents, empty_mask, [[], []])
        all_mask = torch.zeros(2, 4, dtype=torch.bool)
        invalid_image = ImageMemory(torch.full((3, self.hidden), float("nan")),
                                    torch.full((3, 2), float("nan")), 0, torch.zeros(3, dtype=torch.bool))
        result = core(torch.full((2, 4, self.hidden), float("nan")), latents, all_mask,
                      [[invalid_image, self.image(0, 1)], []])
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        result.square().sum().backward()
        self.assertTrue(torch.isfinite(core.visual_projection.weight.grad).all())
        self.assertEqual(core(torch.empty(0, 0, self.hidden), torch.empty(0, 16, self.hidden),
                              torch.empty(0, 0, dtype=torch.bool), []).shape, (0, 3, self.hidden))

    def test_adapter_padding_empty_images_and_coordinates(self):
        adapter = self.adapter()
        features, coordinates = torch.randn(2, 4, self.hidden), torch.randn(2, 4, 2)
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        expected = adapter(features, coordinates, mask)
        features[~mask] = float("nan")
        coordinates[~mask] = float("nan")
        torch.testing.assert_close(adapter(features, coordinates, mask), expected, rtol=0, atol=0)
        padded_features = F.pad(features, (0, 0, 0, 3), value=float("nan"))
        padded_coordinates = F.pad(coordinates, (0, 0, 0, 3), value=float("nan"))
        padded_mask = F.pad(mask, (0, 3), value=False)
        torch.testing.assert_close(adapter(padded_features, padded_coordinates, padded_mask), expected, rtol=1e-5, atol=1e-6)
        shifted = adapter(features, coordinates + 2, mask)
        self.assertFalse(torch.allclose(expected, shifted))
        no_pixels = adapter(torch.empty(2, 0, self.hidden), torch.empty(2, 0, 2))
        masked_pixels = adapter(features, coordinates, torch.zeros_like(mask))
        torch.testing.assert_close(no_pixels, masked_pixels, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(no_pixels).all())
        self.assertEqual(adapter(torch.empty(0, 0, self.hidden), torch.empty(0, 0, 2)).shape,
                         (0, 16, self.hidden))

    def test_gate_initialization_and_core_restore(self):
        core = self.core()
        self.assertEqual(float(core.image_gate.weight.abs().sum()), 0)
        self.assertAlmostEqual(float(core.image_gate.bias.sigmoid()), 0.1, places=6)
        for name in ("alpha_text", "beta_text", "alpha_latent", "beta_final", "alpha_image"):
            self.assertAlmostEqual(float(getattr(core, name)), 0.1, places=6)
        # Loading only the shared core requires no warm-up adapter parameters.
        restored = self.core()
        restored.load_state_dict(core.state_dict(), strict=True)
        inputs = self.inputs()
        torch.testing.assert_close(restored(*inputs), core(*inputs), rtol=0, atol=0)

    def test_invalid_configuration_and_memory_shapes_fail_clearly(self):
        for kwargs in ({"hidden_size": 0}, {"dim": 15, "heads": 2}, {"score_dim": 0}, {"image_branch_enabled": 1}, {"slots": 0}, {"slots": 16.0}):
            args = dict(hidden_size=24, dim=16, heads=2, ffn_dim=32, score_dim=7)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                InteractionCore(**args)
        for kwargs in ({"slots": 0}, {"slots": 16.0}, {"layers": 0}, {"heads": 3}):
            args = dict(hidden_size=24, dim=16, heads=2, ffn_dim=32)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                WarmupImageAdapter(**args)
        core = self.core()
        text, latents, mask, images = self.inputs()
        with self.assertRaisesRegex(ValueError, "one image list"):
            core(text, latents, mask, images[:1])
        with self.assertRaisesRegex(ValueError, "boolean dtype"):
            core(text, latents, mask.float(), images)
        images[0][0].coordinates = torch.zeros(3, 2)
        with self.assertRaisesRegex(ValueError, "coordinates"):
            core(text, latents, mask, images)

    def test_bfloat16_autocast_has_finite_outputs_and_gradients(self):
        core, adapter = self.core(), self.adapter()
        text, _, mask, images = self.inputs()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            latents = adapter(torch.randn(2, 5, self.hidden), torch.randn(2, 5, 2))
            output = core(text, latents, mask, images)
            loss = F.cross_entropy(nn.Linear(self.hidden, 13)(output).reshape(-1, 13),
                                   torch.tensor([2, 3, 4, 5, 6, 7]))
        self.assertTrue(torch.isfinite(output).all())
        loss.backward()
        for module in (core, adapter):
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
