"""CPU-only conversion: python -m sft_offload_4.convert --help."""
import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import shutil

from .checkpoint import read_json, resolve_checkpoint, signature
from .state import MODEL_FILE, equal, load, optimizer_file, optimizer_path, require, reshard_rank, validate_source, verify_rank


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def convert(checkpoint, config_path, output, training_output=None):
    import torch
    from src.sft2.config import Stage2Config
    from src.sft2_offload.config import configure_deepspeed
    source = resolve_checkpoint(checkpoint)
    output = Path(output).expanduser().resolve()
    require(not output.exists(), "Output already exists; choose a NEW conversion directory")
    require(source != output and source not in output.parents and output not in source.parents,
            "Conversion output must be separate from the source checkpoint")
    config = Stage2Config.load(config_path)
    configure_deepspeed(config)
    old = read_json(source / "complete.json")
    info = read_json(source.parent.parent / "run_info.json")
    require(old["world_size"] == 8 and info["world_size"] == 8, "This production entry converts exactly 8 to 4")
    require(info.get("optimizer_offload") == "cpu", "Expected CPU-offload source checkpoint")
    require(old["signature"] == info["signature"] == signature(
        config, info["dataset_sha256"], old["warmup_fingerprint"], old["teacher_fingerprint"], 8),
        "Source config/run_info/checkpoint signatures do not match; use the actual 8-card training config")
    old_ds = read_json(config.deepspeed)
    # Read only the tiny manifest for sample count. No records/images/teacher
    # tensors are processed or hashed, and none are written during conversion.
    samples = read_json(Path(config.prepared_dir) / "manifest.json")["samples"]
    batch = 8 * config.micro_batch_size * config.gradient_accumulation_steps
    windows = math.ceil(samples / batch)
    require(info["effective_batch"] == batch and info["total_optimizer_steps"] == windows * config.epochs,
            "Source dataset size/schedule mismatch")
    require(0 <= old["epoch"] <= config.epochs and 0 <= old["next_window"] < windows and
            old["optimizer_steps"] == old["epoch"] * windows + old["next_window"],
            "Checkpoint cursor does not match optimizer step")
    model = load(source / MODEL_FILE)
    for key, value in old.items():
        equal(model[key], value, "source.client." + key)
    require(model["lr_scheduler"]["last_epoch"] == old["optimizer_steps"], "Scheduler step mismatch")
    require(model["ds_config"]["zero_optimization"].get("load_from_fp32_weights", True),
            "Continuation must load the FP32 masters, not reconstruct them from BF16 weights")
    require(model["global_samples"] == old["optimizer_steps"] * batch, "Sample counter mismatch")
    for key, value in old_ds.items():
        if key not in {"train_batch_size", "train_micro_batch_size_per_gpu", "gradient_accumulation_steps"}:
            equal(model["ds_config"].get(key), value, "source.DeepSpeed." + key)
    require(model["ds_config"]["gradient_accumulation_steps"] == config.gradient_accumulation_steps and
            model["ds_config"]["train_micro_batch_size_per_gpu"] == config.micro_batch_size and
            model["ds_config"]["train_batch_size"] == batch, "Saved batch dimensions differ")
    for key in ("round_robin_gradients", "elastic_checkpoint", "fp16_master_weights_and_gradients"):
        require(not model["ds_config"]["zero_optimization"].get(key), f"Unsupported {key}")
    require(len(list(source.glob("*optim_states.pt"))) == 8 and
            len(list(source.glob("*model_states.pt"))) == 1, "Unexpected model/optimizer shard inventory")
    shards = [load(optimizer_path(source, rank)) for rank in range(8)]
    print("Checking all source FP32 masters, Adam states, parameter mappings and BF16 model weights...", flush=True)
    sizes = validate_source(model, shards, 8, old["optimizer_steps"])
    equal(model["lr_scheduler"].get("_last_lr"),
          [group["lr"] for group in shards[0]["optimizer_state_dict"]["base_optimizer_state"]["param_groups"]],
          "source.scheduler_optimizer_lr")
    for rank in range(8):
        require((source / f"rng_rank{rank}.pt").is_file(), f"Missing source RNG rank {rank}")
    target_step = output / source.name
    config.num_gpus, config.gpu_ids = 4, None
    config.gradient_accumulation_steps *= 2
    config.output_dir = str(Path(training_output).expanduser().resolve() if training_output else output / "training")
    require(Path(config.output_dir) != source.parent.parent, "Use a separate 4-card training output directory")
    config.resume = str(target_step)
    config.deepspeed = str(output / "zero2.json")
    output.mkdir(parents=True)
    target_step.mkdir()
    write_json(config.deepspeed, old_ds)
    write_json(output / "latent.json", config.to_dict())
    new = {**old, "world_size": 4}
    new["signature"] = signature(config, info["dataset_sha256"], old["warmup_fingerprint"],
                                 old["teacher_fingerprint"], 4)
    target_ds = copy.deepcopy(model["ds_config"])
    target_ds["gradient_accumulation_steps"] = config.gradient_accumulation_steps
    target_ds["train_batch_size"] = batch
    # No all-model FP32 concatenation: only one destination rank is materialized.
    for rank in range(4):
        print(f"Converting and verifying optimizer partition {rank + 1}/4...", flush=True)
        converted = reshard_rank(shards, model["param_shapes"], sizes, rank, 4, target_ds)
        path = target_step / optimizer_file(rank, model["ds_config"].get("bf16", {}).get("enabled", False))
        torch.save(converted, path)
        verify_rank(path, shards, sizes, rank, 4, converted)
        del converted
        gc.collect()
    saved_model = {**model, **new, "dp_world_size": 4, "ds_config": target_ds}
    torch.save(saved_model, target_step / MODEL_FILE)
    equal(load(target_step / MODEL_FILE), saved_model, "saved.model_and_scheduler")
    for rank in range(4):
        shutil.copyfile(source / f"rng_rank{rank}.pt", target_step / f"rng_rank{rank}.pt")
        require((source / f"rng_rank{rank}.pt").read_bytes() ==
                (target_step / f"rng_rank{rank}.pt").read_bytes(), "RNG copy failed")
    report = {**info, "world_size": 4, "signature": new["signature"], "dataset_samples": samples,
              "effective_batch": batch, "source_checkpoint": str(source), "source_world_size": 8,
              "target_world_size": 4, "optimizer_step": old["optimizer_steps"],
              "source_accumulation": config.gradient_accumulation_steps // 2,
              "target_accumulation": config.gradient_accumulation_steps,
              "verified_all_elements": True, "max_conversion_error": 0.0,
              "rng_policy": "new rank r inherits old rank r; changed DP reduction/RNG usage is not bitwise replay",
              "data_prepared_or_hashed": False}
    write_json(target_step / "migration.json", report)
    # Completion markers are LAST: incomplete/failed conversions cannot resume.
    write_json(target_step / "complete.json", new)
    (output / "latest").write_text(source.name, encoding="utf-8")
    print(json.dumps({"converted": str(target_step), "config": str(output / "latent.json"),
                      "max_conversion_error": 0.0, "global_batch": batch}, ensure_ascii=False), flush=True)
    return output / "latent.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="8-card complete checkpoint directory (or checkpoints root)")
    parser.add_argument("--config", required=True, help="The exact old 8-card latent.json / train_config.json")
    parser.add_argument("--output", default="sft_offload_4/converted", help="NEW conversion output directory")
    parser.add_argument("--training-output", help="Separate directory for resumed 4-card training")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    require(args.threads > 0, "--threads must be positive")
    # Conversion never initializes CUDA, distributed groups or CPUAdam kernels.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["DS_ACCELERATOR"] = "cpu"
    os.environ.setdefault("DS_BUILD_OPS", "0")
    import torch
    torch.set_num_threads(args.threads)
    convert(args.checkpoint, args.config, args.output, args.training_output)


if __name__ == "__main__":
    main()
