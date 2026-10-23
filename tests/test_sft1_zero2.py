"""CPU autograd checks for ordering; native multi-rank checks are separate."""
import copy
import random
from types import SimpleNamespace
import unittest

import torch

from src.sft1.model import WarmupModel
from src.sft1.train import optimizer_groups
from src.sft1.zero2 import OrderedGradientReduction, check_zero2_version, finish_gradient_partition, ordered_parameters
from tests.interaction_fixtures import TinyDecoder
from tests.test_sft1_model import example_batch, model_config, tiny_config


class OrderedZero2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ready_permutations_preserve_values_and_group_order_without_copies(self):
        params = [torch.nn.Parameter(torch.randn(i + 1)) for i in range(8)]
        entries = [(str(i), p, i % 3) for i, p in enumerate(params)]
        observed = []
        def reduce(param, group):
            observed.append((id(param), group, param.grad.clone()))
            param.grad = None  # Like native ZeRO, release a consumed leaf gradient.
        reducer = OrderedGradientReduction(entries, reduce)
        rng = random.Random(4)
        for iteration in range(20):
            reducer.begin()
            permutation = list(range(len(params)))
            rng.shuffle(permutation)
            observed.clear()
            for index in permutation:
                params[index].grad = torch.full_like(params[index], index + iteration)
                reducer.ready(params[index], index % 3)
            reducer.finish()
            self.assertEqual([(p, g) for p, g, _ in observed], [(id(p), g) for _, p, g in entries])
            for index, (_, _, grad) in enumerate(observed):
                torch.testing.assert_close(grad, torch.full_like(params[index], index + iteration))
            self.assertEqual(reducer.pending_bytes, 0)
            self.assertTrue(all(p.grad is None for p in params))
        self.assertEqual(reducer.completed_backwards, 20)

    def test_real_warmup_graphs_match_plain_gradients_and_stream_before_end(self):
        torch.manual_seed(17)
        config = tiny_config(activation_checkpointing=True, adapter_lr=7e-5)
        model = WarmupModel(TinyDecoder(), torch.nn.Linear(24, 96, bias=False), model_config(), config)
        groups = [group["params"] for group in optimizer_groups(model, config)]
        entries = ordered_parameters(model, groups)
        captured, actual_hook_orders, orders = {}, [], []
        names = {id(p): name for name, p in model.named_parameters()}
        def reduce(param, group):
            captured[names[id(param)]] = param.grad.clone()
            orders.append(names[id(param)])
            param.grad = None
        reducer = OrderedGradientReduction(entries, reduce)
        handles = []
        for _, param, group in entries:
            def hook(p, group=group):
                actual_hook_orders[-1].append(names[id(p)])
                reducer.ready(p, group)
            handles.append(param.register_post_accumulate_grad_hook(hook))
        cases = [([(2, [[3, 4], "image", [5], "image", [6, 90]])], False),
                 ([(0, [[7], "image", "image", [8, 90]])], False),
                 ([(0, [[9, 10, 90]])], False), ([(0, [[11, 90]])], True)]
        try:
            for specs, masked in cases:
                batch, features = example_batch(specs)
                if masked:
                    batch["labels"].fill_(-100)
                # Build the oracle without copying the reduction hooks.
                reference = WarmupModel(TinyDecoder(), torch.nn.Linear(24, 96, bias=False), model_config(), config)
                reference.load_state_dict(model.state_dict())
                reference(batch, features).backward()
                captured.clear()
                orders.clear()
                actual_hook_orders.append([])
                reducer.begin()
                model(batch, features).backward()
                reducer.finish()
                self.assertEqual(orders, [n for n, _, _ in entries])
                for name, param in reference.named_parameters():
                    self.assertIsNotNone(param.grad, name)
                    torch.testing.assert_close(captured[name], param.grad, rtol=0, atol=0, msg=name)
                total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                self.assertLess(reducer.peak_pending_bytes, total_bytes)
            self.assertGreater(len({tuple(order) for order in actual_hook_orders}), 1)
        finally:
            for handle in handles:
                handle.remove()

    def test_tied_embedding_is_unique_and_last_and_groups_stay_intact(self):
        config = tiny_config()
        model = WarmupModel(TinyDecoder(), torch.nn.Linear(24, 96, bias=False), model_config(), config)
        model.lm_head.weight = model.decoder.embed_tokens.weight
        groups = [group["params"] for group in optimizer_groups(model, config)]
        entries = ordered_parameters(model, groups)
        self.assertIs(entries[-1][1], model.decoder.embed_tokens.weight)
        self.assertEqual(len({id(p) for _, p, _ in entries}), sum(len(g) for g in groups))
        for _, param, group in entries:
            self.assertTrue(any(param is candidate for candidate in groups[group]))
        with self.assertRaisesRegex(RuntimeError, "match all trainable"):
            ordered_parameters(model, [groups[0]])

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    def test_native_qwen_bfloat16_cuda_gradients_unchanged_by_streaming(self):
        from src.interaction.distributed import to_device
        from tests.test_interaction_qwen import tiny_qwen
        full = tiny_qwen()
        config = tiny_config(activation_checkpointing=True)
        model = WarmupModel(full.model, full.lm_head, full.config, config).to("cuda", torch.bfloat16)
        reference = copy.deepcopy(model)
        batch, features = example_batch([(1, [[3], "image", [4], "image", [5, 90]])], hidden=64)
        batch, features = to_device(batch, torch.device("cuda")), features.to("cuda", torch.bfloat16)
        reference(batch, features).backward()
        groups = [group["params"] for group in optimizer_groups(model, config)]
        entries = ordered_parameters(model, groups)
        captured = {}
        names = {id(p): name for name, p in model.named_parameters()}
        def reduce(param, group):
            captured[names[id(param)]] = param.grad.detach().clone()
            param.grad = None
        reducer = OrderedGradientReduction(entries, reduce)
        handles = [param.register_post_accumulate_grad_hook(lambda p, group=group: reducer.ready(p, group))
                   for _, param, group in entries]
        try:
            reducer.begin()
            model(batch, features).backward()
            reducer.finish()
            for name, param in reference.named_parameters():
                torch.testing.assert_close(captured[name], param.grad, rtol=0, atol=0, msg=name)
        finally:
            for handle in handles:
                handle.remove()

    def test_missing_duplicate_and_outside_backward_gradients_fail(self):
        a, b = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))
        reducer = OrderedGradientReduction([("a", a, 0), ("b", b, 0)], lambda *_: None)
        with self.assertRaisesRegex(RuntimeError, "engine.backward"):
            reducer.ready(a, 0)
        reducer.begin()
        a.grad = torch.ones_like(a)
        reducer.ready(a, 0)
        with self.assertRaisesRegex(RuntimeError, "twice"):
            reducer.ready(a, 0)
        with self.assertRaisesRegex(RuntimeError, "missing gradients.*b"):
            reducer.finish()

    def test_unsupported_deepspeed_version_fails_closed(self):
        check_zero2_version("0.18.0")
        check_zero2_version("0.18.0+local")
        for version in ("0.17.0", "0.18.1", "0.19.0", "0.18.0.dev0"):
            with self.subTest(version=version), self.assertRaisesRegex(RuntimeError, "deepspeed==0.18.0"):
                check_zero2_version(version)

    def test_partition_accumulation_retains_every_microbatch_and_resets_each_window(self):
        # Reproduces the 0.18.0 failure: averaged_gradients is absent/None until
        # the boundary, while all_grad_tensors already holds earlier microbatches.
        for accumulation_steps in (1, 2, 8):
            params = [torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(2))]
            optimizer = SimpleNamespace(
                params_already_reduced=[True, True], params_in_partition=[[p] for p in params],
                gradient_accumulation_dtype=torch.float32, all_grad_tensors={}, averaged_gradients={},
                first_offset=[1, 0], partition_size=[2, 2], single_partition_of_fp32_groups=params,
                reduce_ipg_grads=lambda: None, _release_ipg_buffers=lambda: None)
            optimizer.get_all_grad_tensors = lambda group, dtype: [p.grad for p in group]
            optimizer.get_flat_partition = lambda group, start, size, dtype, device, param_group_idx, return_tensor_list: [
                optimizer.all_grad_tensors[param_group_idx][0].flatten()[start:start + size]]
            def zero_grad(set_to_none):
                for param in params:
                    param.grad = None
            optimizer.zero_grad = zero_grad
            for window in range(2):
                expected = 0
                for micro in range(accumulation_steps):
                    value = window * 10 + micro + 1
                    expected += value
                    for param in params:
                        param.grad = torch.full_like(param, value)
                    optimizer.is_gradient_accumulation_boundary = micro == accumulation_steps - 1
                    finish_gradient_partition(optimizer)
                    self.assertEqual(optimizer.params_already_reduced, [False, False])
                    self.assertTrue(all(p.grad is None for p in params))
                for group in range(2):
                    self.assertIsNone(optimizer.all_grad_tensors[group])
                    torch.testing.assert_close(optimizer.averaged_gradients[group][0], torch.full((2,), float(expected)))
                    optimizer.averaged_gradients[group] = None  # Native optimizer.step releases it.


if __name__ == "__main__":
    unittest.main()
