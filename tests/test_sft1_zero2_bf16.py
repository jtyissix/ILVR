"""Optional real multi-GPU BF16 ZeRO-2/AdamW acceptance; no model downloads.

Run ``python -m tests.test_sft1_zero2_bf16``. Edit settings below; torchrun is
started automatically. Ordinary BF16 backward + all-reduce verifies gradient
partitions. An independent FP32-master AdamW verifies optimizer/clipping math
using those checked gradients, avoiding near-zero BF16 gradient sign sensitivity.
Checkpoint replay must match bitwise. An unsupported machine reports skipped.
"""
from __future__ import annotations

NUM_GPUS = 4
ACCUMULATION_STEPS = 8
GRADIENT_CLIPPING = 1.0
REDUCE_BUCKET_SIZE = 257
CHECKPOINT_DIR = None
GRADIENT_RTOL = 0.02
GRADIENT_ATOL = 1e-4
GRADIENT_RELATIVE_L2_TOL = 0.01

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run_check(device, directory):
    """Run after initializing DeepSpeed/Torch distributed; also usable with CPU/Gloo."""
    import copy
    import deepspeed
    import torch
    import torch.distributed as dist
    from tests import test_sft1_deepspeed as fixture
    from src.sft1.train import optimizer_groups
    from src.sft1.zero2 import install_ordered_zero2
    from src.sft1.checkpoint import save_training, load_training
    from src.interaction.distributed import normalized_loss, sum_scalar, to_device
    rank, size = dist.get_rank(), dist.get_world_size()
    fixture.ACCUMULATION_STEPS = ACCUMULATION_STEPS
    config = fixture.training_recipe()
    config.weight_decay = 0.01
    config.backbone_lr, config.fusion_lr, config.adapter_lr = 0.003, 0.005, 0.007

    def initialize():
        model = fixture.make_model(config, device).to(torch.bfloat16)
        optimizer = torch.optim.AdamW(optimizer_groups(model, config), betas=(0.9, 0.95),
                                     eps=1e-8, fused=device.type == "cuda")
        ds = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": ACCUMULATION_STEPS,
              "train_batch_size": size * ACCUMULATION_STEPS, "bf16": {"enabled": True},
              "communication_data_type": "bf16", "gradient_clipping": GRADIENT_CLIPPING,
              "zero_optimization": {"stage": 2, "overlap_comm": False, "reduce_scatter": True,
                                    "contiguous_gradients": True, "reduce_bucket_size": REDUCE_BUCKET_SIZE,
                                    "allgather_bucket_size": 258},
              "zero_allow_untested_optimizer": True, "steps_per_print": 1000000}
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds, dist_init_required=False)
        install_ordered_zero2(engine)
        return engine

    engine = initialize()
    reference = fixture.make_model(config, device).to(torch.bfloat16)
    reference.load_state_dict(copy.deepcopy(engine.module.state_dict()))
    ref_parameters = dict(reference.named_parameters())
    master = {name: torch.nn.Parameter(param.detach().float().clone()) for name, param in ref_parameters.items()}
    by_id = {id(param): name for name, param in ref_parameters.items()}
    master_groups = [{**group, "params": [master[by_id[id(p)]] for p in group["params"]]}
                     for group in optimizer_groups(reference, config)]
    master_optimizer = torch.optim.AdamW(master_groups, betas=(0.9, 0.95), eps=1e-8, fused=False)
    offsets, cursor = {}, 0
    for name, param in ref_parameters.items():
        offsets[name] = (cursor, cursor + param.numel())
        cursor += param.numel()

    def update(current, window, oracle=True):
        local = fixture.rank_window(rank, size, window)
        count = int(sum_scalar(fixture.token_count(local), device))
        accumulated = torch.zeros(cursor, dtype=torch.bfloat16, device=device)
        max_gradient_error, max_weight_error, max_master_error, norm = 0.0, 0.0, 0.0, 0.0
        gradient_error_squared, gradient_squared, local_loss = 0.0, 0.0, 0.0
        initial_step = current.global_steps
        names = {id(param): name for name, param in current.module.named_parameters()}
        for step, (batch, features) in enumerate(local):
            batch = to_device(batch, device)
            features = None if features is None else features.to(device=device, dtype=torch.bfloat16)
            if oracle:
                reference.zero_grad(set_to_none=True)
                ref_loss = reference(batch, features)
                (ref_loss * size / count).backward()
                reduced = torch.cat([param.grad.flatten() for param in ref_parameters.values()])
                reduced.div_(size)
                dist.all_reduce(reduced)
                accumulated.add_(reduced)
            loss = current(batch, features)
            local_loss += float(loss.detach())
            current.backward(normalized_loss(loss, count, ACCUMULATION_STEPS, size))
            boundary = step == ACCUMULATION_STEPS - 1
            if boundary and oracle:
                verified_gradients = {}
                for group_index, group in enumerate(current.optimizer.bit16_groups):
                    flat = torch.cat([accumulated[slice(*offsets[names[id(p)]])] for p in group])
                    partition_size = int(current.optimizer.partition_size[group_index])
                    flat = torch.nn.functional.pad(flat, (0, size * partition_size - flat.numel()))
                    expected = flat[rank * partition_size:(rank + 1) * partition_size]
                    actual = torch.cat([part.flatten() for part in current.optimizer.averaged_gradients[group_index]])
                    torch.testing.assert_close(actual.float(), expected.float(), rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL)
                    max_gradient_error = max(max_gradient_error, float((actual.float()-expected.float()).abs().max()))
                    gradient_error_squared += float((actual.float()-expected.float()).square().sum())
                    gradient_squared += float(expected.float().square().sum())
                    # BF16 reduction tree rounding can flip near-zero gradients.
                    # Check optimizer math separately with these verified inputs.
                    shards = [torch.empty_like(actual) for _ in range(size)]
                    dist.all_gather(shards, actual)
                    complete, position = torch.cat(shards), 0
                    for param in group:
                        verified_gradients[names[id(param)]] = complete[position:position+param.numel()].float().reshape_as(param).clone()
                        position += param.numel()
                for name, param in master.items():
                    param.grad = verified_gradients[name]
                norm = float(torch.nn.utils.clip_grad_norm_(list(master.values()), GRADIENT_CLIPPING))
                master_optimizer.step()
                master_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for name, param in ref_parameters.items():
                        param.copy_(master[name])
            current.step()
            if current.global_steps != initial_step + int(boundary):
                raise AssertionError("Expected exactly one optimizer update per complete accumulation window")
        if oracle:
            for group_index, group in enumerate(current.optimizer.bit16_groups):
                expected = torch.cat([master[names[id(param)]].detach().flatten() for param in group])
                partition_size = int(current.optimizer.partition_size[group_index])
                expected = torch.nn.functional.pad(expected, (0, size * partition_size - expected.numel()))
                expected = expected[rank * partition_size:(rank + 1) * partition_size]
                actual = current.optimizer.single_partition_of_fp32_groups[group_index].detach()
                torch.testing.assert_close(actual, expected, rtol=5e-5, atol=1e-7)
                max_master_error = max(max_master_error, float((actual-expected).abs().max()))
            for name, param in current.module.named_parameters():
                max_weight_error = max(max_weight_error, float((param.float()-ref_parameters[name].float()).abs().max()))
                # The FP32-master assertion above is the strict optimizer check;
                # casting nearly identical FP32 values can differ by one BF16 ULP.
                torch.testing.assert_close(param.float(), ref_parameters[name].float(), rtol=0.008, atol=1e-5)
        fixture.assert_rank_agreement(current.module)
        relative_error = (sum_scalar(gradient_error_squared, device) /
                          max(sum_scalar(gradient_squared, device), 1e-20)) ** 0.5
        if relative_error >= GRADIENT_RELATIVE_L2_TOL:
            raise AssertionError(f"BF16 gradient relative L2 error too large: {relative_error}")
        return dict(gradient_max_error=max_gradient_error, gradient_relative_l2_error=relative_error,
                    master_weight_max_error=max_master_error, bf16_weight_max_error=max_weight_error,
                    unclipped_gradient_norm=norm, mean_ce=sum_scalar(local_loss, device)/count)

    first = update(engine, 0)
    checkpoint = save_training(engine, directory, 0, 1, "bf16-adamw-gas8", "tiny-local")
    second = update(engine, 1)
    if max(first["unclipped_gradient_norm"], second["unclipped_gradient_norm"]) <= GRADIENT_CLIPPING:
        raise AssertionError("The acceptance fixture did not exercise active gradient clipping")
    continued = fixture.parameters(engine.module)
    restored = initialize()
    if load_training(restored, checkpoint, "bf16-adamw-gas8", "tiny-local") != (0, 1):
        raise AssertionError("Checkpoint cursor did not restore")
    replay = update(restored, 1, oracle=False)
    for name, param in restored.module.named_parameters():
        torch.testing.assert_close(param, continued[name], rtol=0, atol=0)
    if replay["mean_ce"] != second["mean_ce"]:
        raise AssertionError("Checkpoint replay loss differs")
    return dict(status="passed", precision="bf16", backend=dist.get_backend(), zero_stage=2,
                world_size=size, gradient_accumulation_steps=ACCUMULATION_STEPS, optimizer="AdamW",
                gradient_clipping=GRADIENT_CLIPPING, first=first, second=second,
                checkpoint_replay_bitwise_equal=True, checkpoint=str(checkpoint))


def main():
    from tests import test_sft1_deepspeed as fixture
    fixture.NUM_GPUS = NUM_GPUS
    worker = "RANK" in os.environ
    error = fixture.prerequisites(worker=worker)
    if error:
        print(json.dumps(dict(status="skipped", reason=error,
                              note="No distributed BF16 acceptance test was run."), indent=2))
        return 0
    if worker:
        from datetime import timedelta
        import deepspeed
        import torch
        import torch.distributed as dist
        torch.set_num_threads(1)
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        deepspeed.init_distributed(dist_backend="nccl", timeout=timedelta(minutes=3))
        directory = [str(Path(CHECKPOINT_DIR).resolve()) if CHECKPOINT_DIR else
                     tempfile.mkdtemp(prefix="sft1-zero2-bf16-") if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(directory, src=0)
        result = run_check(device, directory[0])
        if dist.get_rank() == 0:
            print(json.dumps(result, indent=2), flush=True)
        dist.barrier()
        return 0
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node", str(NUM_GPUS), "--module", "tests.test_sft1_zero2_bf16"]
    return subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
