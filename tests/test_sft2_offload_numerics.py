"""Numerical protections: large finite norms, bad gradients and DP fail-together."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist

from src.sft2_offload.numerics import NumericalGuard
from src.sft2_offload.zero2 import offload_clip, offload_group_norm


def optimizer_fixture(gradient):
    parameter = torch.nn.Parameter(torch.zeros_like(gradient))
    return SimpleNamespace(
        bit16_groups=[[parameter]], params_in_partition=[[parameter]],
        norm_for_param_grads={0: gradient.double().norm()}, get_param_id=lambda p: 0,
        ignore_unused_parameters=False, dp_process_group=None, device=torch.device("cpu"),
        loss_scale=1., clip_grad=1.), parameter


def distributed_failure_worker(rank, rendezvous, output):
    from datetime import timedelta
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        guard = NumericalGuard(output, "cpu")
        gradient = torch.full((4,), 1e20)
        optimizer, parameter = optimizer_fixture(gradient)
        norm = offload_group_norm(optimizer, [parameter])
        offload_clip(optimizer, [gradient], norm)
        norm_after = gradient.double().square().sum()
        dist.all_reduce(norm_after)
        torch.testing.assert_close(norm_after.sqrt(), torch.tensor(1., dtype=torch.float64))
        # Only one rank is bad. BOTH must fail before an optimizer update.
        calls = 0
        try:
            guard.observe_gradient("raw_gradient", "test.weight",
                                   torch.tensor([float("nan") if rank else 1.]))
            guard.check("backward", gradients=True)
            calls += 1
        except FloatingPointError:
            pass
        else:
            raise AssertionError("All ranks must reject a remote non-finite gradient")
        assert calls == 0
        optimizer.norm_for_param_grads[0] = torch.tensor(float("nan") if rank else 1., dtype=torch.float64)
        try:
            offload_group_norm(optimizer, [parameter])
        except FloatingPointError:
            pass
        else:
            raise AssertionError("Invalid group norm must not become a -1 sentinel")
    finally:
        dist.destroy_process_group()


class NumericalProtectionTest(unittest.TestCase):
    def test_huge_finite_gradient_clips_to_one(self):
        for magnitude in (1., 1e20, 1e38):
            gradient = torch.full((4,), magnitude)
            optimizer, parameter = optimizer_fixture(gradient)
            norm = offload_group_norm(optimizer, [parameter])
            self.assertEqual(norm.dtype, torch.float64)
            torch.testing.assert_close(norm, gradient.double().norm())
            offload_clip(optimizer, [gradient], norm)
            self.assertLessEqual(float(gradient.double().norm()), 1.000001)

    def test_normal_clipping_matches_pytorch(self):
        for scale in (1., 128.):
            gradient = torch.tensor([3., -4., .25, -.5]) * scale
            optimizer, parameter = optimizer_fixture(gradient)
            optimizer.loss_scale = scale
            expected = torch.nn.Parameter(torch.zeros_like(gradient))
            expected.grad = gradient.clone() / scale
            torch.nn.utils.clip_grad_norm_([expected], 1.)
            offload_clip(optimizer, [gradient], offload_group_norm(optimizer, [parameter]))
            torch.testing.assert_close(gradient, expected.grad)

    def test_nonfinite_norm_fails_without_mutating_gradient(self):
        for value in (float("nan"), float("inf")):
            gradient = torch.tensor([value])
            optimizer, parameter = optimizer_fixture(gradient)
            with self.assertRaises(FloatingPointError):
                offload_group_norm(optimizer, [parameter])
            with self.assertRaises(FloatingPointError):
                offload_clip(optimizer, [gradient], torch.tensor(value))

    def test_gradient_failure_has_parameter_and_sample_context(self):
        with tempfile.TemporaryDirectory() as output:
            guard = NumericalGuard(output, "cpu")
            guard.context = {"record_ids": ["record-97"], "update": 97, "microbatch": 7}
            guard.observe_gradient("accumulated_gradient", "interaction.weight", torch.tensor([float("inf")]))
            with self.assertRaises(FloatingPointError):
                guard.check("backward", gradients=True)
            failure = json.loads(next((Path(output) / "numerics").glob("failure*.json")).read_text())
            self.assertEqual(failure["record_ids"], ["record-97"])
            self.assertIn("accumulated_gradient:interaction.weight", failure["local_nonfinite"])

    def test_state_and_bf16_copy_are_checked(self):
        with tempfile.TemporaryDirectory() as output:
            guard = NumericalGuard(output, "cpu")
            master = torch.nn.Parameter(torch.ones(2))
            master.grad = torch.ones(2)
            optimizer = SimpleNamespace(single_partition_of_fp32_groups=[master],
                optimizer=SimpleNamespace(state={master: {"exp_avg": torch.zeros(2), "exp_avg_sq": torch.ones(2)}}),
                bit16_groups_flat=[torch.ones(2, dtype=torch.bfloat16)], _global_grad_norm=2., clip_grad=1.)
            guard.check_state(optimizer, "startup")
            guard.check_state(optimizer, "before_update")
            optimizer.optimizer.state[master]["exp_avg_sq"][0] = float("nan")
            with self.assertRaises(FloatingPointError):
                guard.check_state(optimizer, "after_update")
            optimizer.optimizer.state[master]["exp_avg_sq"][0] = 1.
            optimizer.bit16_groups_flat[0][0] = float("inf")
            with self.assertRaises(FloatingPointError):
                guard.check_state(optimizer, "after_update")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_bf16_observation_and_cpu_gradient_clipping(self):
        with tempfile.TemporaryDirectory() as output:
            gradient = torch.full((4,), 1e20, device="cuda", dtype=torch.bfloat16)
            optimizer, parameter = optimizer_fixture(gradient)
            guard = NumericalGuard(output, "cuda")
            guard.observe_gradient("accumulated_gradient", "test.weight", gradient)
            guard.check("backward", gradients=True)
            norm = offload_group_norm(optimizer, [parameter])
            self.assertEqual(norm.device.type, "cpu")
            cpu_gradient = gradient.float().cpu()
            offload_clip(optimizer, [cpu_gradient], norm)
            self.assertAlmostEqual(float(cpu_gradient.double().norm()), 1., places=6)
            gradient[0] = float("nan")
            guard.observe_gradient("raw_gradient", "test.weight", gradient)
            with self.assertRaises(FloatingPointError):
                guard.check("backward", gradients=True)

    def test_two_ranks_clip_and_fail_together(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as output:
            rendezvous = (Path(output) / "rendezvous").as_uri()
            mp.spawn(distributed_failure_worker, args=(rendezvous, output), nprocs=2, join=True)
            failures = list((Path(output) / "numerics").glob("failure*.json"))
            self.assertEqual(len(failures), 2)
            records = [json.loads(path.read_text()) for path in failures]
            self.assertTrue(all(record["failed_any_rank"] for record in records))
            self.assertEqual(sum(bool(record["local_nonfinite"]) for record in records), 1)


if __name__ == "__main__":
    unittest.main()
