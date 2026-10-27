"""Native multi-process ZeRO-2 acceptance for the actual Stage2Model trajectory.

python -m tests.test_sft2_deepspeed             # two CUDA/NCCL ranks
python -m tests.test_sft2_deepspeed --cpu       # two CPU/Gloo ranks

The decoder is tiny and deterministic; all Stage2Model modules, four losses,
continuous recurrence, auxiliary decoder, native DeepSpeed reducer/optimizer,
accumulation and checkpoint paths are real. This is FP32/SDPA numerical
acceptance, not a production 7B/BF16 memory benchmark. Missing prerequisites
produce an explicit skipped report, never a claimed distributed pass.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile


ACCUMULATION = 2
BUCKET_SIZE = 257
MOMENTUM = 0.9
WEIGHT_RTOL, WEIGHT_ATOL = 5e-5, 5e-6
GRAD_RTOL, GRAD_ATOL = 8e-4, 1e-5


def prerequisites(cpu, processes):
    if processes < 2:
        return "At least two processes are required."
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("deepspeed") is None:
        return "PyTorch and pinned DeepSpeed 0.18.0 must be installed."
    import torch
    if not cpu and (not torch.cuda.is_available() or torch.cuda.device_count() < processes):
        return f"Need {processes} CUDA GPUs; use --cpu for separate Gloo numerical acceptance."
    if not cpu and os.name == "nt":
        return "NCCL acceptance requires Linux; use --cpu on Windows."
    if cpu and not torch.distributed.is_gloo_available():
        return "This PyTorch build has no Gloo backend."
    return None


def training_recipe():
    from tests.test_sft2_model import recipe
    return recipe(activation_checkpointing=True, gradient_accumulation_steps=ACCUMULATION,
                  backbone_lr=.02, fusion_lr=.04, aux_decoder_lr=.03, weight_decay=0.,
                  ce_weight=1., visual_weight=.3, action_words_weight=.8, action_state_weight=.4)


def make_model(config, device):
    import torch
    from src.sft2.model import Stage2Model
    from tests.interaction_fixtures import TinyDecoder
    from tests.test_sft2_model import model_config
    torch.manual_seed(20261026)
    return Stage2Model(TinyDecoder(), torch.nn.Linear(24, 96, bias=False), model_config(), config).to(device)


def initialize(config, device):
    import deepspeed
    import torch
    from src.interaction.distributed import world
    from src.sft1.zero2 import install_ordered_zero2
    from src.sft2.train import optimizer_groups, ordered_parameters
    model = make_model(config, device)
    optimizer = torch.optim.SGD(optimizer_groups(model, config), momentum=MOMENTUM)
    ds = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": ACCUMULATION,
          "train_batch_size": ACCUMULATION * world(), "fp16": {"enabled": False}, "bf16": {"enabled": False},
          "communication_data_type": "fp32", "gradient_clipping": 0., "prescale_gradients": False,
          "gradient_predivide_factor": 1., "zero_allow_untested_optimizer": True, "steps_per_print": 100000,
          "zero_optimization": {"stage": 2, "overlap_comm": False, "reduce_scatter": True,
                                "contiguous_gradients": True, "reduce_bucket_size": BUCKET_SIZE,
                                "allgather_bucket_size": 258}}
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds, dist_init_required=False)
    if install_ordered_zero2(engine, parameter_order=ordered_parameters) is None:
        raise AssertionError("Native ordered ZeRO-2 was not installed")
    engine.train()
    return engine


def rank_window(rank_id, world_size, window):
    """Unequal action/helper/word counts, no-image actions, consecutive helpers and dummies."""
    import torch
    from src.sft2.train import LOSS_NAMES
    from tests.test_sft2_model import example_batch
    result = []
    with torch.random.fork_rng(devices=[]):
        for microstep in range(ACCUMULATION):
            torch.manual_seed(7000 + window * 100 + rank_id * 10 + microstep)
            if microstep == 0 and rank_id % 2 == 0:
                specs = [(2, [[3, 4], "action", "helper", [5], "action", "helper", [6, 90]])]
            elif microstep == 0 and window % 3 == 0:
                specs = [(0, [[7], "action", "helper", [8, 90]])]
            elif microstep == 0 and window % 3 == 1:
                specs = [(0, [[7], "helper", "helper", [8, 90]])]
            elif microstep == 0:
                specs = [(0, [[7], "action", [8, 90]])]
            else:
                specs = [(0, [[9, 10, 11, 90]])]
            batch, features = example_batch(specs)
            for index, action in enumerate(batch["actions"][0]):
                action["target_ids"] = [15] * (1 + index + rank_id % 2) + [90]
            dummy = rank_id == world_size - 1 and microstep == ACCUMULATION - 1
            if dummy:
                batch["sample_mask"].fill_(False)
                batch["labels"].fill_(-100)
            counts = {"ce": int(batch["labels"][:, 1:].ne(-100).sum()),
                      "visual": 16 * len(batch["segments"][0]),
                      "action_words": sum(len(a["target_ids"]) for a in batch["actions"][0]),
                      "action_state": len(batch["actions"][0])}
            batch["loss_counts"] = {name: 0 if dummy else counts[name] for name in LOSS_NAMES}
            result.append((batch, features))
    return result


def parameters(model):
    return {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}


def reference_update(model, optimizer, config, world_size, window, device):
    """Independent ordinary optimizer over the exact union of all rank samples."""
    import torch
    from src.sft2.train import LOSS_NAMES, combined_loss, loss_weights, to_device
    union = [item for owner in range(world_size) for item in rank_window(owner, world_size, window)]
    counts = {name: sum(batch["loss_counts"][name] for batch, _ in union) for name in LOSS_NAMES}
    values = {name: 0. for name in LOSS_NAMES}
    optimizer.zero_grad(set_to_none=True)
    for batch, features in union:
        losses = model(to_device(batch, device), None if features is None else features.to(device))
        for name in LOSS_NAMES:
            values[name] += float(losses[name].detach())
        combined_loss(losses, counts, loss_weights(config), 1, 1).backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise AssertionError(f"Missing/nonfinite reference gradient: {name}")
            gradients[name] = parameter.grad.detach().clone()
    optimizer.step()
    return counts, {name: values[name] / max(counts[name], 1) for name in LOSS_NAMES}, gradients, parameters(model)


def distributed_update(engine, config, world_size, rank_id, window, device):
    import torch.distributed as dist
    from src.interaction.distributed import sum_scalar
    from src.sft2.train import LOSS_NAMES, combined_loss, loss_weights, to_device
    local = rank_window(rank_id, world_size, window)
    counts = {name: sum_scalar(sum(batch["loss_counts"][name] for batch, _ in local), device)
              for name in LOSS_NAMES}
    totals, ready, orders = {name: 0. for name in LOSS_NAMES}, [], []
    names = {name for name, p in engine.module.named_parameters() if p.requires_grad}
    handles = [p.register_hook(lambda grad, name=name: ready.append(name))
               for name, p in engine.module.named_parameters() if p.requires_grad]
    initial = engine.global_steps
    for microstep, (batch, features) in enumerate(local):
        ready.clear()
        losses = engine(to_device(batch, device), None if features is None else features.to(device))
        for name in LOSS_NAMES:
            totals[name] += float(losses[name].detach())
        engine.backward(combined_loss(losses, counts, loss_weights(config), ACCUMULATION, world_size))
        if len(ready) != len(names) or set(ready) != names:
            raise AssertionError("Every trainable leaf must become ready exactly once, including empty auxiliary branches")
        gathered = [None] * world_size
        dist.all_gather_object(gathered, tuple(ready))
        orders.append(len(set(gathered)))
        engine.step()
        if engine.global_steps != initial + int(microstep == ACCUMULATION - 1):
            raise AssertionError("Incorrect DeepSpeed accumulation boundary")
    for handle in handles:
        handle.remove()
    if orders[0] < 2:
        raise AssertionError("The fixture did not exercise unequal gradient readiness orders")
    means = {name: sum_scalar(totals[name], device) / max(counts[name], 1) for name in LOSS_NAMES}
    return counts, means, parameters(engine.module), orders


def compare(actual, expected, actual_means, expected_means):
    import torch
    for name in actual_means:
        torch.testing.assert_close(torch.tensor(actual_means[name]), torch.tensor(expected_means[name]),
                                   rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL, msg=name + " global mean")
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL, msg=name)
    return max(float((actual[name] - expected[name]).abs().max()) for name in expected)


def worker(rank_id, world_size, port, directory, cpu):
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
        # Disable only the optional SHM extension, retaining native Torch/Gloo
        # collectives and the real DeepSpeed ZeRO reducer and optimizer.
        comm = importlib.import_module("deepspeed.comm.torch")
        if hasattr(comm, "build_shm_op"):
            comm.build_shm_op = lambda: None
    deepspeed.init_distributed(dist_backend="gloo" if cpu else "nccl", auto_mpi_discovery=False,
                              timeout=timedelta(minutes=3))
    from src.sft2.checkpoint import load_training, save_training
    from src.sft2.train import optimizer_groups
    from tests.test_sft1_deepspeed import assert_rank_agreement
    config = training_recipe()
    engine = initialize(config, device)
    reference = make_model(config, device)
    reference.load_state_dict(copy.deepcopy(engine.module.state_dict()))
    optimizer = torch.optim.SGD(optimizer_groups(reference, config), momentum=MOMENTUM)
    before = parameters(engine.module)
    counts, expected_means, gradients, expected = reference_update(reference, optimizer, config, world_size, 0, device)
    actual_counts, actual_means, actual, first_orders = distributed_update(engine, config, world_size, rank_id, 0, device)
    if actual_counts != counts:
        raise AssertionError("Four global loss denominators differ from the independent union")
    error = compare(actual, expected, actual_means, expected_means)
    rates = {id(p): group["lr"] for group in optimizer_groups(engine.module, config) for p in group["params"]}
    gradient_error = 0.
    for name, p in engine.module.named_parameters():
        if p.requires_grad:
            effective = (before[name] - actual[name]) / rates[id(p)]
            torch.testing.assert_close(effective, gradients[name], rtol=GRAD_RTOL, atol=GRAD_ATOL, msg=name)
            gradient_error = max(gradient_error, float((effective - gradients[name]).abs().max()))
    assert_rank_agreement(engine.module)
    signature, source, teacher = "sft2-native-zero2-acceptance-v1", "random-tiny-model", "synthetic-boundary-targets"
    checkpoint = save_training(engine, directory, 0, 1, signature, source, teacher)
    _, expected_second_means, _, expected_second = reference_update(reference, optimizer, config, world_size, 1, device)
    _, second_means, continued, second_orders = distributed_update(engine, config, world_size, rank_id, 1, device)
    error = max(error, compare(continued, expected_second, second_means, expected_second_means))
    assert_rank_agreement(engine.module)
    engine.destroy()
    engine.optimizer = None  # __del__ must not destroy the same native optimizer after backend teardown.
    del engine
    gc.collect()
    if not cpu:
        torch.cuda.empty_cache()
    restored = initialize(config, device)
    if load_training(restored, checkpoint, signature, source, teacher) != (0, 1):
        raise AssertionError("Checkpoint cursor mismatch")
    _, replay_means, replay, _ = distributed_update(restored, config, world_size, rank_id, 1, device)
    compare(replay, continued, replay_means, second_means)
    assert_rank_agreement(restored.module)
    _, expected_third_means, _, expected_third = reference_update(reference, optimizer, config, world_size, 2, device)
    _, third_means, final, third_orders = distributed_update(restored, config, world_size, rank_id, 2, device)
    error = max(error, compare(final, expected_third, third_means, expected_third_means))
    assert_rank_agreement(restored.module)
    if rank_id == 0:
        result = {"status": "passed", "backend": "cpu-gloo" if cpu else "cuda-nccl",
                  "deepspeed": deepspeed.__version__, "precision": "fp32", "world_size": world_size,
                  "zero_stage": 2, "four_loss_counts": counts, "max_weight_error": error,
                  "max_effective_gradient_error": gradient_error,
                  "readiness_order_counts": [first_orders, second_orders, third_orders],
                  "checkpoint_continuation_matches": True, "all_rank_parameters_identical": True,
                  "checkpoint": str(checkpoint)}
        (Path(directory) / "acceptance_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    # Native engine destructors remove hooks while the communication backend is
    # still alive. Do not defer them until after destroy_process_group.
    restored.destroy()
    restored.optimizer = None
    del restored, reference, optimizer
    gc.collect()
    dist.barrier()
    dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--checkpoint-dir")
    args = parser.parse_args(argv)
    if args.cpu:
        os.environ["DS_ACCELERATOR"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("USE_LIBUV", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    reason = prerequisites(args.cpu, args.processes)
    if reason:
        print(json.dumps({"status": "skipped", "reason": reason, "note": "No distributed acceptance ran."}, indent=2))
        return 0
    import torch.multiprocessing as mp
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    directory = args.checkpoint_dir or tempfile.mkdtemp(prefix="sft2-zero2-acceptance-")
    Path(directory).mkdir(parents=True, exist_ok=True)
    mp.spawn(worker, args=(args.processes, port, str(Path(directory).resolve()), args.cpu),
             nprocs=args.processes, join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
