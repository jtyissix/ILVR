"""Native multi-rank ZeRO-2 optimizer-offload acceptance with a tiny Stage2Model.

Linux production optimizer/transport: python -m tests.test_sft2_offload_deepspeed
CPU/Gloo numerical fixture only: python -m tests.test_sft2_offload_deepspeed --cpu --cpu-test-optimizer
Add --bf16 for a BF16 mixed-precision resume/finite-gradient smoke check.

Default uses real DeepSpeedCPUAdam and CUDA/NCCL. The explicit CPU fixture uses
ordinary torch AdamW on CPU, retaining native DS CPU-offload partitions and
collectives. It does NOT validate CPUAdam kernels, pinned DMA, NCCL, BF16 or 7B
memory/speed. Four microsteps detect the upstream intermediate-copy bug which
two microsteps cannot expose. Three windows include unequal events, empty
branches, dummy samples, oversized buckets, optimizer state and resume checks.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import socket
import tempfile


ACCUMULATION = 4
BUCKET_SIZE = 257


def recipe():
    from tests import test_sft2_deepspeed as base
    base.ACCUMULATION = ACCUMULATION
    config = base.training_recipe()
    config.backbone_lr, config.fusion_lr, config.aux_decoder_lr = .002, .004, .003
    config.weight_decay = .01
    return config


def initialize(config, device, cpu_test_optimizer, bf16=False):
    import deepspeed
    import torch
    from unittest.mock import patch
    from src.interaction.distributed import world
    from src.sft2.train import optimizer_groups
    from src.sft2_offload.train import create_optimizer
    from src.sft2_offload.zero2 import install_ordered_offload
    from tests.test_sft2_deepspeed import make_model
    model = make_model(config, device)
    if bf16:
        model = model.to(dtype=torch.bfloat16)
    if cpu_test_optimizer:
        optimizer = torch.optim.AdamW(optimizer_groups(model, config), betas=(.9, .999), eps=1e-8)
    else:
        optimizer = create_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1. / (1. + .1 * step))
    ds = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": ACCUMULATION,
          "train_batch_size": ACCUMULATION * world(), "fp16": {"enabled": False}, "bf16": {"enabled": bf16},
          "communication_data_type": "bf16" if bf16 else "fp32", "gradient_clipping": 0., "prescale_gradients": False,
          "gradient_predivide_factor": 1., "zero_allow_untested_optimizer": True,
          "zero_force_ds_cpu_optimizer": not cpu_test_optimizer, "steps_per_print": 100000,
          "zero_optimization": {"stage": 2, "overlap_comm": False, "reduce_scatter": True,
                                "contiguous_gradients": True, "reduce_bucket_size": BUCKET_SIZE,
                                "allgather_bucket_size": 258,
                                "offload_optimizer": {"device": "cpu", "pin_memory": device.type == "cuda"}}}
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, lr_scheduler=scheduler,
                                          config=ds, dist_init_required=False)
    if cpu_test_optimizer:
        # Production rejects non-CPUAdam optimizers. This explicit test-only
        # substitution permits Windows/Gloo numerical validation, never silently.
        with patch("deepspeed.ops.adam.DeepSpeedCPUAdam", torch.optim.AdamW):
            reducer = install_ordered_offload(engine)
    else:
        reducer = install_ordered_offload(engine)
    if reducer is None or not engine.optimizer.cpu_offload:
        raise AssertionError("The actual ordered native CPU-offload path was not installed")
    if not any(p.numel() > BUCKET_SIZE for p in engine.module.parameters()):
        raise AssertionError("Fixture must include parameters larger than the reduction bucket")
    for master in engine.optimizer.single_partition_of_fp32_groups:
        if master.device.type != "cpu" or master.grad.device.type != "cpu":
            raise AssertionError("FP32 master and gradient partitions must reside on CPU")
    from src.sft2_offload.numerics import NumericalGuard
    guard = NumericalGuard(tempfile.mkdtemp(prefix="sft2-offload-numerics-"), device)
    engine.optimizer._sft2_numerics = guard
    guard.install_trajectory(engine.module)
    guard.check_state(engine.optimizer, "startup")
    engine.train()
    return engine


def distributed_update(engine, config, world_size, rank_id, window, device):
    import torch
    import torch.distributed as dist
    from deepspeed.utils import safe_get_full_grad
    from src.interaction.distributed import sum_scalar
    from src.sft2.train import LOSS_NAMES, combined_loss, loss_weights, to_device
    from tests.test_sft2_deepspeed import parameters, rank_window
    local = rank_window(rank_id, world_size, window)
    counts = {name: sum_scalar(sum(batch["loss_counts"][name] for batch, _ in local), device) for name in LOSS_NAMES}
    totals, ready, orders = {name: 0. for name in LOSS_NAMES}, [], []
    names = {name for name, p in engine.module.named_parameters() if p.requires_grad}
    handles = [p.register_hook(lambda grad, name=name: ready.append(name))
               for name, p in engine.module.named_parameters() if p.requires_grad]
    initial, gradients = engine.global_steps, {}
    guard = engine.optimizer._sft2_numerics
    for microstep, (batch, features) in enumerate(local):
        ready.clear()
        dtype = next(engine.module.parameters()).dtype
        guard.begin(initial + 1, 0, window, microstep, batch, [])
        batch = to_device(batch, device)
        features = None if features is None else features.to(device=device, dtype=dtype)
        guard.inputs(batch, features)
        losses = engine(batch, features)
        guard.check("forward", losses)
        for name in LOSS_NAMES:
            totals[name] += float(losses[name].detach())
        engine.backward(combined_loss(losses, counts, loss_weights(config), ACCUMULATION, world_size))
        guard.check("backward", gradients=True)
        if len(ready) != len(names) or set(ready) != names:
            raise AssertionError("Every trainable leaf must be ready exactly once, including empty branches")
        gathered = [None] * world_size
        dist.all_gather_object(gathered, tuple(ready))
        orders.append(len(set(gathered)))
        if microstep == ACCUMULATION - 1:
            guard.check_state(engine.optimizer, "before_update")
            for name, parameter in engine.module.named_parameters():
                if parameter.requires_grad:
                    gradient = safe_get_full_grad(parameter)
                    if gradient is None or not bool(torch.isfinite(gradient).all()):
                        raise AssertionError(f"Missing or nonfinite offloaded gradient: {name}")
                    gradients[name] = gradient.detach().clone()
        engine.step()
        if microstep == ACCUMULATION - 1:
            guard.check_state(engine.optimizer, "after_update")
        if engine.global_steps != initial + int(microstep == ACCUMULATION - 1):
            raise AssertionError("Wrong optimizer accumulation boundary")
    for handle in handles:
        handle.remove()
    if orders[0] < 2:
        raise AssertionError("Fixture did not exercise unequal gradient readiness order")
    means = {name: sum_scalar(totals[name], device) / max(counts[name], 1) for name in LOSS_NAMES}
    for state in engine.optimizer.optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key not in state or state[key].device.type != "cpu" or state[key].dtype != torch.float32:
                raise AssertionError("Adam moments must be initialized and remain FP32 on CPU")
    return counts, means, parameters(engine.module), gradients, orders


def compare_gradients(actual, expected):
    import torch
    error = 0.
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=8e-4, atol=1e-5, msg=name + " full gradient")
        error = max(error, float((actual[name] - expected[name]).abs().max()))
    return error


def destroy_engine(engine):
    engine.destroy()
    engine.optimizer = None


def bf16_resume_acceptance(config, device, rank_id, world_size, directory, cpu, cpu_test_optimizer):
    """Production precision smoke; FP32 mode supplies the independent gradient oracle."""
    import copy
    import torch
    from src.sft2.checkpoint import load_training, save_training
    from tests.test_sft1_deepspeed import assert_rank_agreement
    from tests.test_sft2_deepspeed import parameters
    engine = initialize(config, device, cpu_test_optimizer, bf16=True)
    initial = parameters(engine.module)
    _, _, first, _, first_orders = distributed_update(engine, config, world_size, rank_id, 0, device)
    if all(torch.equal(first[name], initial[name]) for name in initial):
        raise AssertionError("BF16 smoke did not update any model parameters")
    assert_rank_agreement(engine.module)
    signature, source, teacher = "sft2-offload-bf16-smoke-v1", "tiny", "synthetic"
    checkpoint = save_training(engine, directory, 0, 1, signature, source, teacher)
    saved_scheduler = copy.deepcopy(engine.lr_scheduler.state_dict())
    _, second_means, continued, continued_gradients, second_orders = distributed_update(
        engine, config, world_size, rank_id, 1, device)
    assert_rank_agreement(engine.module)
    destroy_engine(engine)
    del engine
    gc.collect()
    restored = initialize(config, device, cpu_test_optimizer, bf16=True)
    if load_training(restored, checkpoint, signature, source, teacher) != (0, 1):
        raise AssertionError("BF16 checkpoint cursor mismatch")
    if restored.lr_scheduler.state_dict() != saved_scheduler:
        raise AssertionError("BF16 checkpoint scheduler state mismatch")
    _, replay_means, replay, replay_gradients, _ = distributed_update(
        restored, config, world_size, rank_id, 1, device)
    for name in continued:
        torch.testing.assert_close(replay[name], continued[name], rtol=0, atol=0, msg=name)
        torch.testing.assert_close(replay_gradients[name], continued_gradients[name], rtol=0, atol=0, msg=name)
    if replay_means != second_means:
        raise AssertionError("BF16 resumed losses differ")
    assert_rank_agreement(restored.module)
    _, _, _, _, third_orders = distributed_update(restored, config, world_size, rank_id, 2, device)
    assert_rank_agreement(restored.module)
    if rank_id == 0:
        result = {"status": "passed", "backend": "cpu-gloo" if cpu else "cuda-nccl",
                  "optimizer": "torch-AdamW-test-substitute" if cpu_test_optimizer else "DeepSpeedCPUAdam",
                  "production_cpuadam_tested": not cpu_test_optimizer, "precision": "bf16",
                  "world_size": world_size, "gradient_accumulation_steps": ACCUMULATION,
                  "optimizer_offload": "cpu", "cpu_master_gradient_and_moments": "fp32",
                  "checkpoint_continuation_matches": True, "scheduler_resume_matches": True,
                  "all_rank_parameters_identical": True,
                  "readiness_order_counts": [first_orders, second_orders, third_orders],
                  "note": "BF16 finite-gradient and exact-resume smoke, not the independent FP32 numerical oracle."}
        (Path(directory) / "acceptance_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    destroy_engine(restored)
    del restored
    gc.collect()


def worker(rank_id, world_size, port, directory, cpu, cpu_test_optimizer, bf16):
    import copy
    from datetime import timedelta
    import importlib
    import torch
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank_id),
                      WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank_id), LOCAL_SIZE=str(world_size))
    torch.set_num_threads(1)
    if cpu:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(rank_id)
        device = torch.device("cuda", rank_id)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    import deepspeed
    if cpu:
        comm = importlib.import_module("deepspeed.comm.torch")
        if hasattr(comm, "build_shm_op"):
            comm.build_shm_op = lambda: None
    deepspeed.init_distributed(dist_backend="gloo" if cpu else "nccl", auto_mpi_discovery=False,
                              timeout=timedelta(minutes=3))
    from src.sft2.checkpoint import load_training, save_training
    from src.sft2.train import optimizer_groups
    from tests.test_sft1_deepspeed import assert_rank_agreement
    from tests.test_sft2_deepspeed import compare, make_model, reference_update
    config = recipe()
    if bf16:
        bf16_resume_acceptance(config, device, rank_id, world_size, directory, cpu, cpu_test_optimizer)
        dist.barrier()
        dist.destroy_process_group()
        return
    engine = initialize(config, device, cpu_test_optimizer)
    reference = make_model(config, device)
    reference.load_state_dict(copy.deepcopy(engine.module.state_dict()))
    optimizer = torch.optim.AdamW(optimizer_groups(reference, config), betas=(.9, .999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1. / (1. + .1 * step))
    counts, expected_means, gradients, expected = reference_update(reference, optimizer, config, world_size, 0, device)
    scheduler.step()
    actual_counts, actual_means, actual, actual_gradients, first_orders = distributed_update(
        engine, config, world_size, rank_id, 0, device)
    if actual_counts != counts:
        raise AssertionError("Four global loss denominators differ from the independent union")
    gradient_error = compare_gradients(actual_gradients, gradients)
    error = compare(actual, expected, actual_means, expected_means)
    assert_rank_agreement(engine.module)
    signature, source, teacher = "sft2-cpu-offload-acceptance-v1", "random-tiny-model", "synthetic-action-states"
    checkpoint = save_training(engine, directory, 0, 1, signature, source, teacher)
    saved_scheduler = copy.deepcopy(engine.lr_scheduler.state_dict())
    _, expected_means, gradients, expected = reference_update(reference, optimizer, config, world_size, 1, device)
    scheduler.step()
    _, second_means, continued, actual_gradients, second_orders = distributed_update(
        engine, config, world_size, rank_id, 1, device)
    gradient_error = max(gradient_error, compare_gradients(actual_gradients, gradients))
    error = max(error, compare(continued, expected, second_means, expected_means))
    assert_rank_agreement(engine.module)
    destroy_engine(engine)
    del engine
    gc.collect()
    if not cpu:
        torch.cuda.empty_cache()
    restored = initialize(config, device, cpu_test_optimizer)
    if load_training(restored, checkpoint, signature, source, teacher) != (0, 1):
        raise AssertionError("Checkpoint cursor mismatch")
    if restored.lr_scheduler.state_dict() != saved_scheduler:
        raise AssertionError("Checkpoint scheduler state mismatch")
    _, replay_means, replay, _, _ = distributed_update(restored, config, world_size, rank_id, 1, device)
    compare(replay, continued, replay_means, second_means)
    assert_rank_agreement(restored.module)
    _, expected_means, gradients, expected = reference_update(reference, optimizer, config, world_size, 2, device)
    scheduler.step()
    _, final_means, final, actual_gradients, third_orders = distributed_update(
        restored, config, world_size, rank_id, 2, device)
    gradient_error = max(gradient_error, compare_gradients(actual_gradients, gradients))
    error = max(error, compare(final, expected, final_means, expected_means))
    assert_rank_agreement(restored.module)
    if rank_id == 0:
        result = {"status": "passed", "backend": "cpu-gloo" if cpu else "cuda-nccl",
                  "optimizer": "torch-AdamW-test-substitute" if cpu_test_optimizer else "DeepSpeedCPUAdam",
                  "production_cpuadam_tested": not cpu_test_optimizer,
                  "deepspeed": deepspeed.__version__, "precision": "fp32", "world_size": world_size,
                  "zero_stage": 2, "optimizer_offload": "cpu", "gradient_accumulation_steps": ACCUMULATION,
                  "max_weight_error": error, "max_full_gradient_error": gradient_error,
                  "four_loss_counts": counts, "readiness_order_counts": [first_orders, second_orders, third_orders],
                  "checkpoint_continuation_matches": True, "scheduler_resume_matches": True,
                  "all_rank_parameters_identical": True, "checkpoint": str(checkpoint)}
        (Path(directory) / "acceptance_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    destroy_engine(restored)
    del restored, reference, optimizer, scheduler
    gc.collect()
    dist.barrier()
    dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cpu-test-optimizer", action="store_true",
                        help="Explicit numerical fixture: ordinary CPU AdamW, not a CPUAdam kernel acceptance")
    parser.add_argument("--bf16", action="store_true", help="Mixed-precision finite-gradient and resume smoke")
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--checkpoint-dir")
    args = parser.parse_args(argv)
    if args.cpu_test_optimizer and not args.cpu:
        parser.error("--cpu-test-optimizer requires --cpu; CUDA acceptance must use real CPUAdam")
    if args.cpu:
        os.environ["DS_ACCELERATOR"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("USE_LIBUV", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from tests.test_sft2_deepspeed import prerequisites
    reason = prerequisites(args.cpu, args.processes)
    if reason:
        print(json.dumps({"status": "skipped", "reason": reason, "note": "No distributed acceptance ran."}, indent=2))
        return 0
    import torch.multiprocessing as mp
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    directory = args.checkpoint_dir or tempfile.mkdtemp(prefix="sft2-offload-acceptance-")
    Path(directory).mkdir(parents=True, exist_ok=True)
    mp.spawn(worker, args=(args.processes, port, str(Path(directory).resolve()), args.cpu, args.cpu_test_optimizer, args.bf16),
             nprocs=args.processes, join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
