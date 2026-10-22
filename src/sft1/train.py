"""CE-only SFT-I with differentiable segments and one frozen image encoding per batch."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ..interaction.data import json_write, stable_hash
from ..interaction.distributed import GlobalBatchSampler, cosine_factor, normalized_loss, rank, sum_scalar, to_device, world
from .checkpoint import export_warmup, load_training, resume_signature, save_training
from .config import DEFAULT_CONFIG, WarmupConfig, validate_local_inputs


def primary_call(function):
    """Broadcast startup failures so other ranks do not wait forever at a barrier."""
    result = [None]
    if rank() == 0:
        try:
            result[0] = {"value": function()}
        except Exception as exc:
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise RuntimeError(result[0]["error"])
    return result[0]["value"]


def optimizer_groups(student, config):
    groups = {}
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("warmup_adapter."):
            lr = config.adapter_lr if config.adapter_lr is not None else config.fusion_lr
        elif name.startswith("interaction."):
            lr = config.fusion_lr
        elif name.startswith(("decoder.", "lm_head.")):
            lr = config.backbone_lr
        else:
            raise ValueError(f"Unclassified trainable parameter: {name}")
        decay = config.weight_decay if param.ndim > 1 else 0.0
        groups.setdefault((lr, decay), []).append(param)
    return [{"params": params, "lr": lr, "weight_decay": decay} for (lr, decay), params in groups.items()]


def configure_deepspeed(config):
    ds = json.loads(Path(config.deepspeed).read_text(encoding="utf-8"))
    zero = ds.get("zero_optimization", {})
    stage = zero.get("stage")
    if stage != 1:
        raise ValueError("SFT-I requires ZeRO-1 with overlap_comm=false: variable helper graphs can produce "
                         "different gradient hook order across ranks, so ordinary ZeRO-2 buckets are unsafe. "
                         "ZeRO-3 also needs aligned module call schedules that are not implemented.")
    if zero.get("overlap_comm") is not False:
        raise ValueError("Set overlap_comm=false explicitly: SFT-I reduces in fixed parameter order after backward")
    if zero.get("reduce_scatter") is not False or zero.get("contiguous_gradients") is not True:
        raise ValueError("SFT-I requires reduce_scatter=false and contiguous_gradients=true")
    if any(zero.get(key) for key in ("offload_optimizer", "offload_param", "zenflow")) or ds.get("zenflow") or ds.get("deepcompile"):
        raise ValueError("Optimizer/parameter offload, ZenFlow and DeepCompile are not supported by this SFT-I recipe")
    if not ds.get("bf16", {}).get("enabled"):
        raise ValueError("SFT-I requires BF16 in the DeepSpeed configuration")
    if "optimizer" in ds or "scheduler" in ds:
        raise ValueError("Optimizer and scheduler are controlled by warmup.json; remove them from DeepSpeed JSON")
    ds.update({"train_micro_batch_size_per_gpu": config.micro_batch_size,
               "gradient_accumulation_steps": config.gradient_accumulation_steps,
               "train_batch_size": config.micro_batch_size * config.gradient_accumulation_steps * world()})
    return ds


@torch.no_grad()
def encode_images(visual, batch):
    """Encode originals and helpers together once; reuse features in both paths."""
    inputs = batch.get("visual")
    if not inputs:
        return None
    dtype = next(visual.parameters()).dtype
    features = visual(inputs["pixel_values"].to(dtype=dtype), grid_thw=inputs["image_grid_thw"])
    indices = inputs.get("feature_indices")
    return features.index_select(0, indices) if indices is not None else features


def train(config):
    from .data import WarmupCollator, WarmupDataset, load_processor
    from .model import load_training_components
    from .prepare import read_preflight
    validate_local_inputs(config)
    configure_deepspeed(config)  # Reject unsupported configurations before distributed work.
    receipt = read_preflight(config)
    # Load the verified cache before NCCL initialization as well. The launcher
    # already hashed records and scanned all source/images in a separate process.
    dataset = WarmupDataset(config.prepared_dir, verify_checksum=False)
    dataset_hash = stable_hash(dataset.manifest)
    if dataset_hash != receipt["dataset_hash"]:
        raise ValueError("Prepared manifest changed after preflight; restart with python run_sft1.py")
    source = receipt["initial_backbone_fingerprint"]
    processor = load_processor(config)
    if not torch.cuda.is_available():
        raise RuntimeError("SFT-I training requires a BF16-capable CUDA GPU; prepare_only and CPU tests are separate")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"GPU {local_rank} does not support the required BF16 training")
    device = torch.device("cuda", local_rank)
    import deepspeed
    deepspeed.init_distributed(dist_backend="nccl")
    random.seed(config.seed + rank())
    np.random.seed(config.seed + rank())
    torch.manual_seed(config.seed)  # Identical newly initialized adapter/interaction on each rank.
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    output = Path(config.output_dir)

    if not len(dataset):
        raise ValueError("Prepared SFT-I dataset is empty")
    student, visual = load_training_components(config, device)
    visual.requires_grad_(False).eval()
    collator = WarmupCollator(processor, student.model_config.to_dict(), dataset.manifest)
    total_windows = math.ceil(len(dataset) / (world() * config.micro_batch_size * config.gradient_accumulation_steps))
    total_steps = total_windows * config.epochs
    optimizer = torch.optim.AdamW(optimizer_groups(student, config), betas=(0.9, 0.999), eps=1e-8, fused=True)
    # max_steps only stops the run: it does not alter the resumable epoch-based LR schedule.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_factor(step, total_steps, config.warmup_ratio))
    ds_config = configure_deepspeed(config)
    engine, _, _, _ = deepspeed.initialize(model=student, optimizer=optimizer, lr_scheduler=scheduler,
                                          config=ds_config, dist_init_required=False)
    signature = resume_signature(config, dataset_hash, source)
    epoch_start, start_window = 0, 0
    if config.resume:
        epoch_start, start_window = load_training(engine, config.resume, signature, source)
    if not (0 <= epoch_start <= config.epochs and 0 <= start_window < total_windows):
        raise ValueError("Checkpoint epoch/window is outside this dataset's training schedule")
    if rank() == 0:
        json_write(output / "run_info.json", {
            "world_size": world(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "deepspeed": deepspeed.__version__, "zero_stage": 1,
            "effective_batch": world() * config.micro_batch_size * config.gradient_accumulation_steps,
            "total_optimizer_steps": total_steps, "step_limit": config.step_limit,
            "initial_backbone_fingerprint": source, "dataset_sha256": dataset_hash, "signature": signature,
            "loss": "supervised_text_ce_only", "execution": config.execution})
    engine.train()
    last_saved = -1
    stop = config.step_limit is not None and engine.global_steps >= config.step_limit
    final_epoch, final_window = epoch_start, start_window
    for epoch in range(epoch_start, config.epochs):
        if stop:
            break
        sampler = GlobalBatchSampler(dataset.records, config.micro_batch_size, config.gradient_accumulation_steps,
                                     world(), rank(), config.seed, epoch, start_window if epoch == epoch_start else 0)
        loader_args = {"dataset": dataset, "batch_sampler": sampler, "collate_fn": collator,
                       "num_workers": config.num_workers, "pin_memory": True,
                       "generator": torch.Generator().manual_seed(config.seed + epoch * world() + rank())}
        if config.num_workers:
            loader_args.update(prefetch_factor=config.prefetch_factor, multiprocessing_context="spawn",
                               persistent_workers=True)
        iterator = iter(DataLoader(**loader_args))
        window_index = sampler.start_window
        while window_index < total_windows:
            tick = time.perf_counter()
            window = [next(iterator) for _ in range(config.gradient_accumulation_steps)]
            data_seconds = time.perf_counter() - tick
            count = sum_scalar(sum(batch["ce_count"] for batch in window), device)
            samples = sum_scalar(sum(batch["sample_count"] for batch in window), device)
            if count <= 0:
                raise ValueError("An optimizer window contains no supervised tokens")
            local_loss = torch.zeros((), device=device, dtype=torch.float32)
            torch.cuda.reset_peak_memory_stats()
            before_step = engine.global_steps
            for cpu_batch in window:
                batch = to_device(cpu_batch, device)
                features = encode_images(visual, batch)
                # All segment states are connected in this one logical forward. Backward only
                # after all helpers and the final answer have contributed to the CE sum.
                loss_sum = engine(batch, image_features=features)
                local_loss += loss_sum.detach().float()
                engine.backward(normalized_loss(loss_sum, count, config.gradient_accumulation_steps, world()))
                engine.step()
                del loss_sum, features, batch
            if engine.global_steps != before_step + 1:
                raise RuntimeError("DeepSpeed accumulation did not produce exactly one optimizer step per window")
            window_index += 1
            final_epoch, final_window = (epoch + 1, 0) if window_index == total_windows else (epoch, window_index)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - tick
            loss_value = sum_scalar(local_loss.item(), device) / count
            perf = torch.tensor([seconds, torch.cuda.max_memory_allocated() / 1024**3],
                                device=device, dtype=torch.float64)
            if dist.is_initialized():
                dist.all_reduce(perf, op=dist.ReduceOp.MAX)
            seconds, peak_gib = perf.tolist()
            if rank() == 0 and (engine.global_steps % config.log_steps == 0 or engine.global_steps == 1):
                record = {"step": engine.global_steps, "epoch": epoch, "next_window": window_index,
                          "ce_loss": loss_value, "effective_tokens": count, "samples": samples,
                          "tokens_per_second": count / seconds, "samples_per_second": samples / seconds,
                          "step_seconds": seconds, "data_seconds": data_seconds,
                          "peak_allocated_gib": peak_gib, "lr": scheduler.get_last_lr()}
                print(json.dumps(record), flush=True)
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            stop = config.step_limit is not None and engine.global_steps >= config.step_limit
            if engine.global_steps % config.save_steps == 0 or stop:
                save_training(engine, output / "checkpoints", final_epoch, final_window, signature, source)
                last_saved = engine.global_steps
            if stop:
                break
    if last_saved != engine.global_steps:
        save_training(engine, output / "checkpoints", final_epoch, final_window, signature, source)
    export_warmup(engine, visual, processor, output / "export", config, source)
    primary_call(lambda: json_write(output / "completion.json", {
        "finished_all_epochs": final_epoch >= config.epochs, "global_steps": engine.global_steps,
        "epoch": final_epoch, "next_window": final_window, "export": str(output / "export")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    config = WarmupConfig.load(args.config)
    if config.mode == "prepare_only":
        from .prepare import prepare
        prepare(config)
    else:
        train(config)


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
