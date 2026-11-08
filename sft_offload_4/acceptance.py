"""Native 8 -> 4 ZeRO-2 save/load/update test on a small model.

Default: CPU/Gloo with torch AdamW as an explicit CPUAdam substitute.
Use --cuda for real CPUAdam/NCCL (requires eight visible GPUs).
"""
import argparse
import copy
from datetime import timedelta
import importlib
import json
import os
from pathlib import Path
import socket
import tempfile

import torch
import torch.distributed as dist

from .state import (MODEL_FILE, equal, load, optimizer_file, optimizer_path, reshard_rank,
                    validate_source, verify_rank)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(5, 7)
        self.interaction = torch.nn.Linear(7, 7)
        self.aux_decoder = torch.nn.Linear(7, 3)

    def forward(self, value):
        return self.aux_decoder(torch.tanh(self.interaction(torch.tanh(self.decoder(value)))))


def make_engine(world, device, bf16):
    import deepspeed
    from unittest.mock import patch
    from src.sft2_offload.zero2 import install_ordered_offload
    torch.manual_seed(123)
    model = Model().to(device=device, dtype=torch.bfloat16 if bf16 else torch.float32)
    groups = [{"params": [p for p in model.parameters() if p.ndim == ndim],
               "lr": .002 if ndim == 2 else .003, "weight_decay": .01} for ndim in (2, 1)]
    if device.type == "cpu":
        optimizer = torch.optim.AdamW(groups)
    else:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam(groups, adamw_mode=True, fp32_optimizer_states=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1/(1+.1*step))
    config = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 16//world,
              "train_batch_size": 16, "bf16": {"enabled": bf16}, "gradient_clipping": 1.,
              "communication_data_type": "bf16" if bf16 else "fp32", "steps_per_print": 100000,
              "zero_allow_untested_optimizer": True, "zero_force_ds_cpu_optimizer": device.type != "cpu",
              "zero_optimization": {"stage": 2, "overlap_comm": False, "contiguous_gradients": True,
                                    "reduce_scatter": True, "reduce_bucket_size": 17, "allgather_bucket_size": 18,
                                    "offload_optimizer": {"device": "cpu", "pin_memory": device.type != "cpu"}}}
    engine, *_ = deepspeed.initialize(model=model, optimizer=optimizer, lr_scheduler=scheduler,
                                      config=config, dist_init_required=False)
    if device.type == "cpu":
        with patch("deepspeed.ops.adam.DeepSpeedCPUAdam", torch.optim.AdamW):
            install_ordered_offload(engine)
    else:
        install_ordered_offload(engine)
    return engine


def update(engine, world, rank, window, device):
    dtype = next(engine.module.parameters()).dtype
    total = torch.zeros((), device=device, dtype=torch.float64)
    for micro in range(16//world):
        sample = window*16 + micro*world + rank
        value = torch.tensor([[(sample + i) % 11 / 11 for i in range(5)]], device=device, dtype=dtype)
        target = torch.tensor([[(sample + i) % 7 / 7 for i in range(3)]], device=device, dtype=dtype)
        loss = (engine(value).float()-target.float()).square().mean()
        total += loss.detach().double()
        engine.backward(loss)
        engine.step()
    dist.all_reduce(total)
    return float(total/16)


def worker(rank, world, port, root, phase, cuda, bf16):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world),
                      LOCAL_RANK=str(rank), LOCAL_SIZE=str(world))
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(rank)
    import deepspeed
    if not cuda:
        comm = importlib.import_module("deepspeed.comm.torch")
        if hasattr(comm, "build_shm_op"):
            comm.build_shm_op = lambda: None
    deepspeed.init_distributed(dist_backend="nccl" if cuda else "gloo", auto_mpi_discovery=False,
                              timeout=timedelta(minutes=3))
    engine = make_engine(world, device, bf16)
    root = Path(root)
    if phase == "save":
        update(engine, world, rank, 0, device)
        from src.sft2.checkpoint import save_training
        save_training(engine, root / "source", 0, 1, "acceptance", "toy", "toy")
        expected_loss = update(engine, world, rank, 1, device)
        if rank == 0:
            torch.save({"module": engine.module.state_dict(), "loss": expected_loss,
                        "scheduler": engine.lr_scheduler.state_dict()}, root / "reference.pt")
    else:
        from types import SimpleNamespace
        from .checkpoint import checked_load
        config = SimpleNamespace(resume=str(root / "target/global_step00000001"))
        assert checked_load(engine, config, {}) == (0, 1)
        assert engine.global_steps == 1
        saved = load(optimizer_path(root / "target/global_step00000001", rank))["optimizer_state_dict"]
        for i, master in enumerate(engine.optimizer.single_partition_of_fp32_groups):
            source = saved["single_partition_of_fp32_groups"][i]
            equal(master.detach()[:source.numel()], source, "native restored FP32 master")
            pid = saved["base_optimizer_state"]["param_groups"][i]["params"][0]
            equal(engine.optimizer.optimizer.state[master], saved["base_optimizer_state"]["state"][pid],
                  "native restored Adam")
        equal(engine.optimizer._param_slice_mappings, saved["param_slice_mappings"], "native new slice mappings")
        actual_loss = update(engine, world, rank, 1, device)
        reference = load(root / "reference.pt")
        error = 0.
        for name, value in engine.module.state_dict().items():
            expected = reference["module"][name].to(device)
            torch.testing.assert_close(value, expected, rtol=1e-2 if bf16 else 1e-5,
                                       atol=2e-3 if bf16 else 1e-7)
            error = max(error, float((value.float()-expected.float()).abs().max()))
        equal(engine.lr_scheduler.state_dict(), reference["scheduler"], "native scheduler")
        assert abs(actual_loss-reference["loss"]) < (2e-3 if bf16 else 1e-6)
        assert engine.global_steps == 2 and engine.global_samples == 32
        # Check every new rank, then save and reload a native four-card checkpoint.
        from src.sft2.checkpoint import save_training
        path = save_training(engine, root / "continued", 0, 2, "acceptance", "toy", "toy")
        config.resume = str(path)
        assert checked_load(engine, config, {}) == (0, 2)
        assert engine.global_steps == 2
        if rank == 0:
            result = {"source_world": 8, "target_world": 4, "bf16": bf16, "passed": True,
                      "max_next_update_weight_error": error, "loss_error": abs(actual_loss-reference["loss"]),
                      "restored_master_and_adam_exact": True, "four_rank_save_reload": True,
                      "optimizer": "CPUAdam" if cuda else "torch.AdamW test substitute",
                      "backend": "nccl" if cuda else "gloo"}
            (root / "acceptance_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
    engine.destroy()
    engine.optimizer = None
    del engine
    import gc
    gc.collect()
    dist.barrier()
    dist.destroy_process_group()


def launch(world, root, phase, cuda, bf16):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.multiprocessing.spawn(worker, args=(world, port, root, phase, cuda, bf16), nprocs=world, join=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="sft_offload_4/test-results")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args(argv)
    if not args.cuda:
        os.environ.update(DS_ACCELERATOR="cpu", CUDA_VISIBLE_DEVICES="", USE_LIBUV="0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    launch(8, str(root), "save", args.cuda, args.bf16)
    model = load(root / "source/global_step00000001" / MODEL_FILE)
    shards = [load(optimizer_path(root / "source/global_step00000001", r)) for r in range(8)]
    sizes = validate_source(model, shards, 8, 1)
    destination = root / "target/global_step00000001"
    destination.mkdir(parents=True, exist_ok=True)
    ds = {**model["ds_config"], "gradient_accumulation_steps": 4}
    for rank in range(4):
        converted = reshard_rank(shards, model["param_shapes"], sizes, rank, 4, ds)
        path = destination / optimizer_file(rank, args.bf16)
        torch.save(converted, path)
        verify_rank(path, shards, sizes, rank, 4, converted)
    torch.save({**model, "dp_world_size": 4, "world_size": 4, "ds_config": ds}, destination / MODEL_FILE)
    import shutil
    state = json.loads((root / "source/global_step00000001/complete.json").read_text())
    state["world_size"] = 4
    (destination / "complete.json").write_text(json.dumps(state))
    for rank in range(4):
        shutil.copyfile(root / "source/global_step00000001" / f"rng_rank{rank}.pt", destination / f"rng_rank{rank}.pt")
    launch(4, str(root), "load", args.cuda, args.bf16)


if __name__ == "__main__":
    main()
