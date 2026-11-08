"""Checkpoint metadata and exact startup checks, without dataset content hashes."""
import json
from pathlib import Path

import torch
import torch.distributed as dist

from src.interaction.data import stable_hash
from src.sft2.checkpoint import FORMAT, load_training
from .state import equal, load, optimizer_file, optimizer_path, require


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not (path / "complete.json").is_file():
        tag = (path / "latest").read_text(encoding="utf-8").strip()
        require(tag and Path(tag).name == tag and tag not in {".", ".."}, "Invalid latest pointer")
        path = path / tag
    require((path / "complete.json").is_file(), "Checkpoint has no complete.json")
    return path


def signature(config, dataset_hash, warmup_hash, teacher_hash, world_size):
    """Same small recipe signature as the 8-card trainer; no data files hashed."""
    values = config.to_dict()
    for key in ("output_dir", "resume", "max_steps", "smoke_steps", "mode", "save_steps", "log_steps",
                "num_workers", "prefetch_factor", "prepared_dir", "gpu_ids", "num_gpus",
                "prepare_workers", "prepare_log_seconds", "warmup_path", "teacher_cache_dir"):
        values.pop(key, None)
    values["deepspeed"] = read_json(config.deepspeed)
    base = stable_hash({"format": FORMAT, "config": values, "dataset": dataset_hash,
                        "warmup": warmup_hash, "teacher": teacher_hash, "world_size": world_size})
    return stable_hash({"recipe": "sft2-optimizer-cpu-offload-v1", "base": base})


def resume_context(config):
    path = resolve_checkpoint(config.resume)
    state = read_json(path / "complete.json")
    migration = path / "migration.json"
    info = read_json(migration if migration.is_file() else path.parent.parent / "run_info.json")
    require(state["world_size"] == 4 and info["world_size"] == 4, "Convert the 8-card checkpoint first")
    require(info["signature"] == state["signature"], "Checkpoint/run metadata mismatch")
    expected = signature(config, info["dataset_sha256"], state["warmup_fingerprint"],
                         state["teacher_fingerprint"], 4)
    require(expected == state["signature"], "Training settings changed since conversion/checkpoint")
    require(info["effective_batch"] == 4 * config.micro_batch_size * config.gradient_accumulation_steps,
            "Global batch changed")
    return path, state, info


def checked_load(engine, config, info):
    """Use native loading, then verify loaded CPU masters, Adam and LR exactly."""
    path = resolve_checkpoint(config.resume)
    state = read_json(path / "complete.json")
    cursor = load_training(engine, path, state["signature"], state["warmup_fingerprint"],
                           state["teacher_fingerprint"])
    error = None
    try:
        saved = load(optimizer_path(path, dist.get_rank()))
        zero = saved["optimizer_state_dict"]
        optimizer = engine.optimizer
        require(len(optimizer.single_partition_of_fp32_groups) ==
                len(zero["single_partition_of_fp32_groups"]), "Restored master group count differs")
        for actual, expected in zip(optimizer.optimizer.param_groups,
                                    zero["base_optimizer_state"]["param_groups"]):
            equal({k: v for k, v in actual.items() if k != "params"},
                  {k: v for k, v in expected.items() if k != "params"}, "loaded.optimizer_hyperparameters")
        for i, master in enumerate(optimizer.single_partition_of_fp32_groups):
            reference = zero["single_partition_of_fp32_groups"][i]
            equal(master.detach()[:reference.numel()], reference, f"loaded.master.{i}")
            require(bool((master.detach()[reference.numel():] == 0).all()), "Loaded padding differs")
            pid = zero["base_optimizer_state"]["param_groups"][i]["params"][0]
            equal(optimizer.optimizer.state[master], zero["base_optimizer_state"]["state"][pid],
                  f"loaded.adam.{i}")
        equal(optimizer._param_slice_mappings, zero["param_slice_mappings"], "loaded.parameter_order")
        # The shared model file is already loaded by DS. Read its small scheduler
        # metadata through mmap; do not allocate another full model copy.
        from .state import MODEL_FILE
        model = load(path / MODEL_FILE)
        equal(engine.lr_scheduler.state_dict(), model["lr_scheduler"], "loaded.scheduler")
        require(int(engine.lr_scheduler.last_epoch) == state["optimizer_steps"], "Loaded LR step differs")
        require(engine.global_steps == state["optimizer_steps"] and
                engine.global_samples == model["global_samples"], "Restored step/sample counters differ")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f"4-card checkpoint numerical verification failed: {errors}")
    return cursor
