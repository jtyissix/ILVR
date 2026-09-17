import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import torch
from torch import nn
from src.interaction.config import TrainConfig
from src.interaction.data import expand_record, find_segments, normalize_sample, position_ids_for_images, tensorize
from src.interaction.distributed import GlobalBatchSampler, normalized_loss
from src.interaction.execution import DecoderExecutor, causal_ce_sum, run_trajectory
from src.interaction.fusion import RoundTripFusion
from src.interaction.generation import generate_continuous
from src.interaction.model import InteractionStudent, reference_contexts
from tests.interaction_fixtures import TinyDecoder, latent_batch


class InteractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)

    def test_incremental_matches_recompute_hidden_ce_and_every_gradient(self):
        original = TinyDecoder()
        ids = torch.randint(1, 60, (3, 13))
        lengths = torch.tensor([13, 10, 5])
        mask = torch.arange(13)[None] < lengths[:, None]
        positions = torch.arange(13)[None, None].expand(3, 3, -1)
        events = {3: [(0, 0, 0)], 4: [(0, 0, 1)], 6: [(1, 0, 0)],
                  7: [(1, 0, 1)], 9: [(0, 1, 0)], 10: [(0, 1, 1)]}
        labels = ids.clone()
        labels[~mask] = -100
        labels[:, :2] = -100
        for pos, entries in events.items():
            for row, _, _ in entries:
                labels[row, pos] = -100
        head = nn.Linear(24, 96, bias=False)
        outputs, gradients = [], []
        for execution, recompute, align in (("incremental", False, False), ("incremental", True, False),
                                            ("recompute", True, False), ("incremental", True, True),
                                            ("recompute", True, True)):
            decoder, local_head = copy.deepcopy(original), copy.deepcopy(head)
            executor = DecoderExecutor(decoder, "sdpa", recompute)
            global_events = sorted(set(events) | {2, 8, 11, 12}) if align else None
            trajectory = run_trajectory(executor, decoder.embed_tokens(ids), mask, positions, events,
                                        execution=execution, events=global_events)
            first_latent = trajectory.latents[0, 0, 0]
            first_latent.retain_grad()
            loss = causal_ce_sum(trajectory.hidden, labels, local_head, 3, recompute)
            loss.backward()
            self.assertGreater(float(first_latent.grad.norm()), 0)
            outputs.append((trajectory.hidden.detach(), loss.detach()))
            gradients.append({name: p.grad.detach().clone() for name, p in decoder.named_parameters()})
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients[-1].values()))
        for result, gradient in zip(outputs[1:], gradients[1:]):
            torch.testing.assert_close(result[0][mask], outputs[0][0][mask], rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(result[1], outputs[0][1], rtol=2e-5, atol=2e-5)
            for name in gradient:
                torch.testing.assert_close(gradient[name], gradients[0][name], rtol=2e-4, atol=2e-5, msg=name)

    def test_frozen_reference_and_full_fusion_gradient(self):
        reference = TinyDecoder().requires_grad_(False).eval()
        before = {n: p.clone() for n, p in reference.named_parameters()}
        context = reference_contexts(reference, latent_batch(), None, 80, "sdpa")
        self.assertEqual(context[0, 0][0].shape, (1, 24))
        self.assertEqual(context[0, 0][1].shape, (8, 24))
        self.assertFalse(context[0, 0][1].requires_grad)
        config = TrainConfig(fusion_dim=16, fusion_heads=2, fusion_ffn_dim=32, attention_backend="sdpa", ce_chunk_size=3)
        student = InteractionStudent(copy.deepcopy(reference).requires_grad_(True), nn.Linear(24, 96),
                                     SimpleNamespace(hidden_size=24, image_token_id=80), config)
        loss = student(latent_batch(1), contexts=context)
        loss.backward()
        for name, param in student.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        self.assertGreater(sum(float(p.grad.norm()) for p in student.fusion.parameters()), 0)
        for name, param in reference.named_parameters():
            self.assertIsNone(param.grad)
            torch.testing.assert_close(param, before[name], rtol=0, atol=0)

    def test_empty_text_and_no_latent_microbatch(self):
        fusion = RoundTripFusion(24, 16, 2, 32)
        visual = torch.randn(2, 8, 24)
        value = fusion(torch.zeros(2, 0, 24), visual, torch.zeros(2, 0, dtype=torch.bool))
        self.assertTrue(torch.isfinite(value).all())
        value.square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters()))
        batch = latent_batch(1)
        batch["segments"], batch["latent_positions"] = [[]], {}
        config = TrainConfig(fusion_dim=16, fusion_heads=2, fusion_ffn_dim=32, attention_backend="sdpa")
        student = InteractionStudent(TinyDecoder(), nn.Linear(24, 96), SimpleNamespace(hidden_size=24, image_token_id=80), config)
        student(batch, contexts={}).backward()
        self.assertTrue(all(p.grad is not None for p in student.fusion.parameters()))

    def test_chunked_ce_matches_full_head_and_zero_labels(self):
        h1 = torch.randn(2, 7, 24, requires_grad=True)
        h2 = h1.detach().clone().requires_grad_()
        head1, head2 = nn.Linear(24, 96), nn.Linear(24, 96)
        head2.load_state_dict(head1.state_dict())
        labels = torch.randint(0, 96, (2, 7))
        labels[0, :4] = -100
        expected = nn.functional.cross_entropy(head1(h1[:, :-1]).reshape(-1, 96), labels[:, 1:].reshape(-1), reduction="sum")
        actual = causal_ce_sum(h2, labels, head2, 2, True, padded_count=20)
        expected.backward()
        actual.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(h2.grad, h1.grad)
        torch.testing.assert_close(head2.weight.grad, head1.weight.grad)
        zero = causal_ce_sum(h2, torch.full_like(labels, -100), head2)
        self.assertEqual(float(zero), 0)

    def test_global_batches_tail_and_resume(self):
        records = [{"length": i+10, "segments": [None]*(i % 3)} for i in range(19)]
        all_real = []
        for rank in range(3):
            sampler = GlobalBatchSampler(records, 2, 2, 3, rank, seed=7)
            batches = list(sampler)
            self.assertTrue(all(len(batch) == 2 for batch in batches))
            all_real.extend(i for batch in batches for i in batch if i >= 0)
            resumed = list(GlobalBatchSampler(records, 2, 2, 3, rank, seed=7, start_window=1))
            self.assertEqual(resumed, batches[2:])
        self.assertEqual(sorted(all_real), list(range(19)))

    def test_cross_rank_accumulation_normalization(self):
        p = torch.tensor(2.0, requires_grad=True)
        # Emulate DS's GAS division then DDP's mean, with unequal token counts and a dummy microbatch.
        micros = [[torch.tensor([1., 3.]), torch.tensor([2.])], [torch.tensor([4., 5., 6.]), torch.tensor([])]]
        total_count = sum(len(x) for rank in micros for x in rank)
        loss = sum(normalized_loss(((p-x)**2).sum(), total_count, 2, 2)/2/2 for rank in micros for x in rank)
        loss.backward()
        q = p.detach().clone().requires_grad_()
        expected = ((q-torch.cat([x for rank in micros for x in rank]))**2).mean()
        expected.backward()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(p.grad, q.grad)

    def test_segment_expansion_and_local_text_boundaries(self):
        ids = {"start": 70, "pad": 71, "end": 72}
        values = [1, 2, 3, 70] + [71]*8 + [72, 4, 70] + [71]*8 + [72, 5]
        segments = find_segments(values, 2, {0, 70, 71, 72}, ids)
        self.assertEqual([s["text"] for s in segments], [[2], [13]])
        record = {"input_ids": values, "labels": values, "assistant_start": 2, "segments": segments}
        expanded = expand_record(record, 1, ids, {0, 70, 71, 72})
        self.assertTrue(all(len(s["pads"]) == 9 for s in expanded["segments"]))
        self.assertEqual(expanded["segments"][1]["text"], [14])
        self.assertEqual(expanded["labels"][4], -100)
        with self.assertRaises(ValueError):
            find_segments([1, 70, 71, 72], 0, set(), ids)

    def test_inference_exact_steps_and_first_embedding(self):
        class ScriptedHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, x):
                self.calls += 1
                output = x.new_full((1, 96), -100)
                output[:, 70 if self.calls in (1, 2) else 90] = 100
                return output
        decoder = TinyDecoder().eval()
        embeds = decoder.embed_tokens(torch.tensor([[1, 2, 3]]))
        positions = torch.arange(3)[None, None].expand(3, 1, -1)
        for steps in (8, 9):
            head = ScriptedHead()
            calls = []
            original_chunk = DecoderExecutor.chunk
            def capture(executor, values, *args, **kwargs):
                output = original_chunk(executor, values, *args, **kwargs)
                calls.append((values.detach().clone(), output[0].detach().clone()))
                return output
            with mock.patch.object(DecoderExecutor, "chunk", capture):
                result = generate_continuous(decoder, head, embeds, positions, {"start": 70, "pad": 71, "end": 72},
                                             {90}, steps, 64, "sdpa")
            self.assertEqual(result["segment_steps"], [steps, steps])
            self.assertEqual(result["token_ids"], ([70] + [71]*steps + [72])*2 + [90])
            self.assertFalse(result["ended_mid_latent"])
            self.assertEqual(head.calls, 3)  # no vocabulary projection for latent slots or forced end
            torch.testing.assert_close(calls[2][0], calls[1][1][:, -1:])  # b_inf == h(latent_start)

    def test_flash_layout_excludes_finished_rows_and_handles_all_padding(self):
        from src.interaction.execution import AttentionLayout, attend
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        layout = AttentionLayout.make(mask, 2, 2, "flash_attention_2")
        self.assertEqual(layout.cu_q.tolist(), [0, 2])
        self.assertEqual(layout.cu_k.tolist(), [0, 4])
        self.assertEqual(layout.k_indices.tolist(), [0, 1, 2, 3])
        empty = AttentionLayout.make(torch.tensor([[1, 1, 0]], dtype=torch.bool), 2, 1, "flash_attention_2")
        q = torch.randn(1, 3, 1, 8, requires_grad=True)
        k = torch.randn(1, 1, 3, 8, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        value = attend(q, k, v, empty, "flash_attention_2")
        value.sum().backward()
        self.assertTrue(value.eq(0).all())
        self.assertTrue(all(x.grad is not None for x in (q, k, v)))

    def test_image_path_root_and_missing_helper(self):
        with tempfile.TemporaryDirectory() as root:
            image = Path(root) / "images_comt" / "creation" / "x.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"path validation only")
            sample = {"text_input": "question", "image_input": ["images_comt/creation/x.png"],
                      "sequence_plan": [{"type": "text", "content": "reason"},
                                        {"type": "latent", "helper_image": "images_comt/creation/x.png"}]}
            paths, _ = normalize_sample(sample, root)
            self.assertEqual(paths, [str(image.resolve())])
            sample["sequence_plan"][1]["helper_image"] = "absent.png"
            with self.assertRaises(FileNotFoundError):
                normalize_sample(sample, root)
            self.assertEqual(normalize_sample(sample, root, check_helpers=False)[0], paths)

    def test_configs_have_matched_controls(self):
        configs = [TrainConfig.load(f"configs/interaction/{mode}.json") for mode in ("baseline_ct", "interaction_ce", "no_refinement")]
        for key in ("epochs", "backbone_lr", "warmup_ratio", "seed", "micro_batch_size", "gradient_accumulation_steps"):
            self.assertEqual(len({getattr(c, key) for c in configs}), 1)
        self.assertEqual([c.latent_steps for c in configs], [8, 9, 9])
        mb2 = TrainConfig.load("configs/interaction/interaction_ce_mb2.json")
        self.assertEqual(mb2.micro_batch_size * mb2.gradient_accumulation_steps,
                         configs[1].micro_batch_size * configs[1].gradient_accumulation_steps)


if __name__ == "__main__":
    unittest.main()
