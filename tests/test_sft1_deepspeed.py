"""Optional Linux multi-GPU acceptance test; no checkpoint/model download.

From the ILVR directory, run just:
    python -m tests.test_sft1_deepspeed

The parent starts torchrun automatically. Settings are below, not CLI flags.
This intentionally uses FP32 + SDPA to test token normalization and ordered
ZeRO-2 reduction against one ordinary model, followed by the ZeRO-1 fallback.
It is not a BF16 numerical-quality or production-size memory test.
An unsupported machine reports ``skipped``; that must not be counted as a pass.
"""
from __future__ import annotations

# --------------------------- Editable settings ---------------------------
NUM_GPUS = 4
ZERO_STAGES = (2, 1)
REDUCE_BUCKET_SIZE = 257  # Force multiple buckets and oversized-parameter reductions.
BACKEND = "sdpa"
ACCUMULATION_STEPS = 2
BACKBONE_LR = 0.03
INTERACTION_LR = 0.05
ADAPTER_LR = 0.07
SGD_MOMENTUM = 0.9
CHECKPOINT_DIR = None  # None: create and retain a temporary directory for inspection.
WEIGHT_RTOL = 3e-5
WEIGHT_ATOL = 3e-6
GRAD_RTOL = 3e-4
GRAD_ATOL = 5e-6
# -------------------------------------------------------------------------

import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def prerequisites(worker=False):
    """Check without initializing CUDA/NCCL or importing DeepSpeed in the parent."""
    if not sys.platform.startswith("linux"):
        return "This acceptance test requires Linux, CUDA and at least two GPUs."
    if importlib.util.find_spec("torch") is None:
        return "PyTorch is not installed."
    if importlib.util.find_spec("deepspeed") is None:
        return "DeepSpeed is not installed; use the project's training environment."
    import torch
    if not torch.cuda.is_available():
        return "CUDA is not available."
    requested = int(os.environ.get("WORLD_SIZE", "1")) if worker else NUM_GPUS
    if requested < 2:
        return "At least two ranks are required to verify distributed reduction."
    if torch.cuda.device_count() < requested:
        return f"Need {requested} visible GPUs; found {torch.cuda.device_count()}."
    if BACKEND != "sdpa":
        return "This FP32 reduction test requires BACKEND='sdpa'; BF16/FlashAttention is a separate acceptance test."
    return None


def training_recipe():
    from tests.test_sft1_model import tiny_config
    return tiny_config(attention_backend=BACKEND, activation_checkpointing=True,
                       micro_batch_size=1, gradient_accumulation_steps=ACCUMULATION_STEPS,
                       backbone_lr=BACKBONE_LR, fusion_lr=INTERACTION_LR,
                       adapter_lr=ADAPTER_LR, weight_decay=0.0)


def make_model(config, device):
    import torch
    from src.sft1.model import WarmupModel
    from tests.interaction_fixtures import TinyDecoder
    from tests.test_sft1_model import model_config
    torch.manual_seed(20261022)
    return WarmupModel(TinyDecoder(), torch.nn.Linear(24, 96, bias=False),
                       model_config(), config).to(device=device, dtype=torch.float32)


def initialize(config, device, zero_stage):
    import deepspeed
    import torch
    from src.interaction.distributed import world
    from src.sft1.train import optimizer_groups
    from src.sft1.zero2 import install_ordered_zero2
    model = make_model(config, device)
    optimizer = torch.optim.SGD(optimizer_groups(model, config), momentum=SGD_MOMENTUM)
    ds = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": ACCUMULATION_STEPS,
        "train_batch_size": ACCUMULATION_STEPS * world(),
        "fp16": {"enabled": False}, "bf16": {"enabled": False},
        "communication_data_type": "fp32", "gradient_clipping": 0.0,
        "prescale_gradients": False, "gradient_predivide_factor": 1.0,
        "zero_optimization": {"stage": zero_stage, "overlap_comm": False,
                              "reduce_scatter": zero_stage == 2, "contiguous_gradients": True,
                              "reduce_bucket_size": REDUCE_BUCKET_SIZE,
                              "allgather_bucket_size": 258},
        "zero_allow_untested_optimizer": True, "steps_per_print": 1000000,
    }
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer,
                                          config=ds, dist_init_required=False)
    ordered = install_ordered_zero2(engine)
    if (ordered is not None) != (zero_stage == 2):
        raise AssertionError("Ordered reduction must be installed exactly for ZeRO-2")
    engine.train()
    return engine


def rank_window(rank_id, world_size, window_index):
    """Two distinct graphs per rank, including no-helper and all-masked examples.

    Rank 0: multiple question images and two helpers, then a text-only sample.
    Rank 1: no question image and one helper, then a zero-helper dummy.
    On the continuation window rank 1 has consecutive helpers instead, checking
    an empty local-text segment and a different gradient-hook order as well.
    On the third window rank 1 has no helper while rank 0 still has two.
    """
    import torch
    from tests.test_sft1_model import example_batch
    batches = []
    with torch.random.fork_rng(devices=[]):
        for microstep in range(ACCUMULATION_STEPS):
            torch.manual_seed(1000 + 100 * window_index + 10 * rank_id + microstep)
            if microstep == 0 and rank_id % 2 == 0:
                specs = [(2 + rank_id % 3, [[3, 4], "image", [5], "image", [6, 90]])]
            elif microstep == 0 and window_index % 3 != 2:
                middle = ["image", "image"] if window_index % 2 else ["image"]
                specs = [(0, [[10], *middle, [11, 12, 90]])]
            else:
                specs = [(0, [[13, 14, 15, 90]])]
            batch, features = example_batch(specs)
            # Unequal CE counts on different ranks are essential to this test.
            if rank_id % 2 == 1 and microstep == 0:
                first_target = batch["labels"][0].ne(-100).nonzero()[0, 0]
                batch["labels"][0, first_target] = -100
            if rank_id == world_size - 1 and microstep == ACCUMULATION_STEPS - 1:
                batch["labels"].fill_(-100)
            batches.append((batch, features))
    return batches


def token_count(batches):
    return sum(int(batch["labels"][:, 1:].ne(-100).sum()) for batch, _ in batches)


def parameters(model):
    return {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}


def reference_update(model, optimizer, config, world_size, window_index, device):
    """One ordinary model sees the union of every rank's two microbatches."""
    import torch
    from src.interaction.distributed import to_device
    union = [item for rank_id in range(world_size)
             for item in rank_window(rank_id, world_size, window_index)]
    total = token_count(union)
    if total <= 0:
        raise AssertionError("The test window must contain supervised tokens")
    optimizer.zero_grad(set_to_none=True)
    summed_loss = 0.0
    for batch, features in union:
        loss = model(to_device(batch, device), None if features is None else features.to(device))
        summed_loss += float(loss.detach())
        (loss / total).backward()
    gradients = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None or not torch.isfinite(param.grad).all():
            raise AssertionError(f"Reference gradient missing/nonfinite: {name}")
        gradients[name] = param.grad.detach().clone()
    optimizer.step()
    return total, summed_loss / total, gradients, parameters(model)


def distributed_update(engine, world_size, rank_id, window_index, device):
    import torch.distributed as dist
    from src.interaction.distributed import normalized_loss, sum_scalar, to_device
    local = rank_window(rank_id, world_size, window_index)
    total = int(sum_scalar(token_count(local), device))
    initial_step = engine.global_steps
    local_loss = 0.0
    ready_order = []
    expected_names = {name for name, param in engine.module.named_parameters() if param.requires_grad}
    # Observe readiness without replacing or modifying a gradient. This proves
    # the acceptance fixture exercises the ordering hazard it is meant to catch.
    handles = [param.register_hook(lambda _grad, name=name: ready_order.append(name))
               for name, param in engine.module.named_parameters() if param.requires_grad]
    distinct_ready_orders = []
    for microstep, (batch, features) in enumerate(local):
        ready_order.clear()
        loss = engine(to_device(batch, device), None if features is None else features.to(device))
        local_loss += float(loss.detach())
        engine.backward(normalized_loss(loss, total, ACCUMULATION_STEPS, world_size))
        if len(ready_order) != len(expected_names) or set(ready_order) != expected_names:
            raise AssertionError("Every trainable parameter must become ready exactly once per microbatch")
        all_orders = [None] * world_size
        dist.all_gather_object(all_orders, tuple(ready_order))
        distinct_ready_orders.append(len(set(all_orders)))
        engine.step()
        expected_steps = initial_step + int(microstep == ACCUMULATION_STEPS - 1)
        if engine.global_steps != expected_steps:
            raise AssertionError("DeepSpeed did not perform exactly one update per complete accumulation window")
    for handle in handles:
        handle.remove()
    if distinct_ready_orders[0] < 2:
        raise AssertionError("The fixture failed to exercise different gradient readiness orders across ranks")
    return total, sum_scalar(local_loss, device) / total, parameters(engine.module), distinct_ready_orders


def assert_rank_agreement(model):
    import torch
    import torch.distributed as dist
    vector = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
    primary = vector.clone()
    dist.broadcast(primary, src=0)
    # Stages 1 and 2 replicate the same all-gathered updated parameter values.
    torch.testing.assert_close(vector, primary, rtol=0, atol=0,
                               msg="ZeRO parameters differ across ranks")


def compare_update(actual, expected, actual_loss, expected_loss):
    import torch
    torch.testing.assert_close(torch.tensor(actual_loss), torch.tensor(expected_loss),
                               rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL, msg="global token-normalized CE")
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=WEIGHT_RTOL,
                                   atol=WEIGHT_ATOL, msg=name)
    return max(float((actual[name] - expected[name]).abs().max()) for name in expected)


def check_first_sgd_gradient(before, after, reference_gradients, model, config):
    """First-step SGD has no prior momentum: (before-after)/lr is the reduced g.

    This checks the effective optimizer gradient without relying on private
    DeepSpeed partition buffers. Small tolerances allow FP32 subtraction error.
    Subsequent updates also exercise the saved/restored momentum state.
    """
    import torch
    from src.sft1.train import optimizer_groups
    rates = {id(param): group["lr"] for group in optimizer_groups(model, config) for param in group["params"]}
    largest = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        effective = (before[name] - after[name]) / rates[id(param)]
        expected = reference_gradients[name]
        torch.testing.assert_close(effective, expected, rtol=GRAD_RTOL, atol=GRAD_ATOL,
                                   msg=f"effective globally averaged gradient: {name}")
        largest = max(largest, float((effective - expected).abs().max()))
    return largest


def worker_main():
    error = prerequisites(worker=True)
    if error:
        raise RuntimeError(error)
    import copy
    from datetime import timedelta
    import deepspeed
    import torch
    import torch.distributed as dist
    from src.sft1.checkpoint import load_training, save_training
    from src.sft1.train import optimizer_groups
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    deepspeed.init_distributed(dist_backend="nccl", timeout=timedelta(minutes=3))
    rank_id, world_size = dist.get_rank(), dist.get_world_size()
    zero_stage = int(os.environ["SFT1_ACCEPTANCE_ZERO_STAGE"])
    config = training_recipe()
    engine = initialize(config, device, zero_stage)
    # The baseline is independent of DeepSpeed's flattened/partitioned optimizer
    # storage, but begins from exactly the weights distributed initialization used.
    reference = make_model(config, device)
    reference.load_state_dict(copy.deepcopy(engine.module.state_dict()))
    baseline_optimizer = torch.optim.SGD(optimizer_groups(reference, config), momentum=SGD_MOMENTUM)
    before = parameters(engine.module)
    total, expected_loss, gradients, expected = reference_update(reference, baseline_optimizer, config, world_size, 0, device)
    actual_count, actual_loss, actual, first_orders = distributed_update(engine, world_size, rank_id, 0, device)
    if total != actual_count:
        raise AssertionError("Global CE token count differs from the single-model union")
    weight_error = compare_update(actual, expected, actual_loss, expected_loss)
    gradient_error = check_first_sgd_gradient(before, actual, gradients, engine.module, config)
    assert_rank_agreement(engine.module)
    directory = [str(Path(CHECKPOINT_DIR).resolve() / f"zero{zero_stage}") if CHECKPOINT_DIR else
                 tempfile.mkdtemp(prefix="sft1-deepspeed-acceptance-") if rank_id == 0 else None]
    dist.broadcast_object_list(directory, src=0)
    signature, source = f"sft1-fp32-zero{zero_stage}-acceptance-v2", "random-tiny-model-no-download"
    checkpoint = save_training(engine, directory[0], 0, 1, signature, source)
    _, second_loss, _, second_expected = reference_update(reference, baseline_optimizer, config, world_size, 1, device)
    _, continued_loss, continued, second_orders = distributed_update(engine, world_size, rank_id, 1, device)
    weight_error = max(weight_error, compare_update(continued, second_expected, continued_loss, second_loss))
    assert_rank_agreement(engine.module)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    restored = initialize(config, device, zero_stage)
    if load_training(restored, checkpoint, signature, source) != (0, 1):
        raise AssertionError("Checkpoint continuation cursor was not restored")
    _, restored_loss, restored_state, _ = distributed_update(restored, world_size, rank_id, 1, device)
    compare_update(restored_state, continued, restored_loss, continued_loss)
    assert_rank_agreement(restored.module)
    _, third_loss, _, third_expected = reference_update(reference, baseline_optimizer, config, world_size, 2, device)
    _, final_loss, final_state, third_orders = distributed_update(restored, world_size, rank_id, 2, device)
    weight_error = max(weight_error, compare_update(final_state, third_expected, final_loss, third_loss))
    assert_rank_agreement(restored.module)
    if rank_id == 0:
        print(json.dumps({"status": "passed", "precision": "fp32", "attention_backend": BACKEND,
                          "zero_stage": zero_stage, "world_size": world_size,
                          "reduce_bucket_size": REDUCE_BUCKET_SIZE,
                          "microsteps_per_update": ACCUMULATION_STEPS,
                          "distinct_rank_readiness_orders_per_microstep": [first_orders, second_orders, third_orders],
                          "global_ce_tokens_first_update": total, "max_weight_error": weight_error,
                          "max_effective_gradient_error": gradient_error,
                          "single_model_union_matches": True, "all_rank_parameters_identical": True,
                          "checkpoint_continuation_matches": True, "checkpoint": str(checkpoint)}, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    if ACCUMULATION_STEPS != 2:
        raise SystemExit("This acceptance fixture requires ACCUMULATION_STEPS=2.")
    if not ZERO_STAGES or any(stage not in (1, 2) for stage in ZERO_STAGES):
        raise SystemExit("ZERO_STAGES must contain only supported stages 1 and 2.")
    if "RANK" in os.environ:
        worker_main()
        return 0
    if isinstance(NUM_GPUS, bool) or not isinstance(NUM_GPUS, int) or NUM_GPUS < 2:
        raise SystemExit("Set NUM_GPUS to an integer of at least 2 at the top of this file.")
    error = prerequisites()
    if error:
        print(json.dumps({"status": "skipped", "reason": error,
                          "note": "No distributed acceptance test was run."}, indent=2))
        return 0
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node", str(NUM_GPUS), "--module", "tests.test_sft1_deepspeed"]
    for zero_stage in ZERO_STAGES:
        environment["SFT1_ACCEPTANCE_ZERO_STAGE"] = str(zero_stage)
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=environment, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
