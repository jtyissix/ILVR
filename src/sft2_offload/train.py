"""Independent SFT-II optimizer-offload loop; shared model/data/loss semantics."""
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
from ..interaction.distributed import GlobalBatchSampler, cosine_factor, rank, sum_scalar, world
from ..sft1.train import encode_images, primary_call
from ..sft1.zero2 import check_zero2_version
from ..sft2.checkpoint import export_stage2, load_training, resume_signature, save_training
from ..sft2.train import LOSS_NAMES, combined_loss, loss_weights, optimizer_groups, to_device
from .config import DEFAULT_CONFIG, Stage2Config, configure_deepspeed, validate_local_inputs
from .zero2 import install_ordered_offload


def create_optimizer(student, config):
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    return DeepSpeedCPUAdam(optimizer_groups(student, config), betas=(0.9, 0.999), eps=1e-8,
                           adamw_mode=True, fp32_optimizer_states=True)


def memory_budget(student, visual):
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    frozen_bytes = sum(p.numel() * p.element_size() for p in visual.parameters())
    gib = 1024 ** 3
    accumulation_bytes = 2 * trainable if student.recipe.gradient_accumulation_steps > 1 else 0
    return {"trainable_parameters": trainable,
            "gpu_weights_gib_per_rank": (sum(p.numel() * p.element_size() for p in student.parameters()) + frozen_bytes) / gib,
            "cpu_fp32_master_and_adam_gib_node": 12 * trainable / gib,
            "cpu_fp32_gradient_gib_node": 4 * trainable / gib,
            "cpu_bf16_update_buffer_gib_node": 2 * trainable / gib,
            "cpu_bf16_accumulation_gib_node_approx": accumulation_bytes / gib,
            "cpu_model_states_and_buffers_gib_node_approx": (18 * trainable + accumulation_bytes) / gib,
            "note": "State estimates only: exclude activations, boundary duplication, per-rank scratch, data workers, initialization and save peaks."}


def train(config):
    from ..sft2.data import Stage2Collator, Stage2Dataset, load_processor
    from ..sft2.model import load_training_components
    from .prepare import read_preflight
    from ..sft2.teacher import validate_teacher_cache

    validate_local_inputs(config)
    configure_deepspeed(config)
    receipt = read_preflight(config)
    dataset = Stage2Dataset(config.prepared_dir, verify_checksum=False)
    dataset_hash = stable_hash(dataset.manifest)
    if dataset_hash != receipt["dataset_hash"]:
        raise ValueError("Prepared manifest changed after preflight; restart with python run_sft2_offload.py")
    teacher = validate_teacher_cache(config, dataset=dataset, verify_hashes=False)
    teacher_hash = stable_hash(teacher)
    del teacher
    # Identity checks above still see the complete on-disk teacher trajectories.
    # The student/loader needs only cached boundary states; avoid replicating
    # full original-text/image token arrays into every spawned worker.
    for record in dataset.records:
        record.pop("teacher_record", None)
    source = receipt["warmup_fingerprint"]
    processor = load_processor(config)
    if not torch.cuda.is_available():
        raise RuntimeError("SFT-II training needs BF16 CUDA; prepare_only and CPU unit tests are separate")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"GPU {local_rank} does not support BF16")
    device = torch.device("cuda", local_rank)
    import deepspeed
    if configure_deepspeed(config)["zero_optimization"]["stage"] == 2:
        check_zero2_version(deepspeed.__version__)
    deepspeed.init_distributed(dist_backend="nccl")
    random.seed(config.seed + rank())
    np.random.seed(config.seed + rank())
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    if not len(dataset):
        raise ValueError("Prepared SFT-II dataset is empty")
    output = Path(config.output_dir)
    student, visual = load_training_components(config, device)
    visual.requires_grad_(False).eval()
    collator = Stage2Collator(processor, student.model_config.to_dict(), dataset.manifest,
                              teacher_cache_dir=config.teacher_cache_dir)
    total_windows = math.ceil(len(dataset) / (world() * config.micro_batch_size * config.gradient_accumulation_steps))
    total_steps = total_windows * config.epochs
    optimizer = create_optimizer(student, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_factor(step, total_steps, config.warmup_ratio))
    ds_config = configure_deepspeed(config)
    engine, _, _, _ = deepspeed.initialize(model=student, optimizer=optimizer, lr_scheduler=scheduler,
                                          config=ds_config, dist_init_required=False)
    reducer = install_ordered_offload(engine)
    signature = stable_hash({"recipe": "sft2-optimizer-cpu-offload-v1",
                             "base": resume_signature(config, dataset_hash, source, teacher_hash)})
    epoch_start, start_window = 0, 0
    if config.resume:
        epoch_start, start_window = load_training(engine, config.resume, signature, source, teacher_hash)
    if not (0 <= epoch_start <= config.epochs and 0 <= start_window < total_windows):
        raise ValueError("Checkpoint epoch/window is outside this dataset's training schedule")
    weights = loss_weights(config)
    if rank() == 0:
        json_write(output / "run_info.json", {
            "world_size": world(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "deepspeed": deepspeed.__version__, "zero_stage": ds_config["zero_optimization"]["stage"],
            "gradient_order_sha256": reducer.order_sha256 if reducer else None,
            "gradient_reduction": "ordered_auxiliary_then_student_cpu_offload",
            "optimizer_offload": "cpu", "optimizer": "DeepSpeedCPUAdam",
            "memory_budget": memory_budget(student, visual),
            "effective_batch": world() * config.micro_batch_size * config.gradient_accumulation_steps,
            "total_optimizer_steps": total_steps, "step_limit": config.step_limit,
            "warmup_fingerprint": source, "teacher_fingerprint": teacher_hash,
            "dataset_sha256": dataset_hash, "signature": signature, "loss_weights": weights,
            "loss_normalization": "separate global accumulation-window denominators",
            "helper_prior": "ground_truth_pooled_helper_features", "execution": config.execution})
    if rank() == 0:
        print(json.dumps({"stage": "sft2_offload_ready", **memory_budget(student, visual)}), flush=True)
    engine.train()
    last_saved = -1
    stop = config.step_limit is not None and engine.global_steps >= config.step_limit
    final_epoch, final_window = epoch_start, start_window
    for epoch in range(epoch_start, config.epochs):
        if stop:
            break
        sampler = GlobalBatchSampler(dataset.records, config.micro_batch_size, config.gradient_accumulation_steps,
                                     world(), rank(), config.seed, epoch, start_window if epoch == epoch_start else 0)
        loader_args = dict(dataset=dataset, batch_sampler=sampler, collate_fn=collator,
                           num_workers=config.num_workers, pin_memory=True,
                           generator=torch.Generator().manual_seed(config.seed + epoch * world() + rank()))
        if config.num_workers:
            loader_args.update(prefetch_factor=config.prefetch_factor, multiprocessing_context="spawn", persistent_workers=True)
        iterator = iter(DataLoader(**loader_args))
        window_index = sampler.start_window
        while window_index < total_windows:
            tick = time.perf_counter()
            window = [next(iterator) for _ in range(config.gradient_accumulation_steps)]
            data_seconds = time.perf_counter() - tick
            counts = {name: sum_scalar(sum(batch["loss_counts"][name] for batch in window), device)
                      for name in LOSS_NAMES}
            samples = sum_scalar(sum(batch["sample_count"] for batch in window), device)
            if counts["ce"] <= 0:
                raise ValueError("An optimizer window contains no supervised text tokens")
            local_losses = {name: torch.zeros((), device=device, dtype=torch.float32) for name in LOSS_NAMES}
            torch.cuda.reset_peak_memory_stats()
            before_step, pending_peak_bytes = engine.global_steps, 0
            for cpu_batch in window:
                batch = to_device(cpu_batch, device)
                features = encode_images(visual, batch)
                losses = engine(batch, image_features=features)
                for name in LOSS_NAMES:
                    local_losses[name] += losses[name].detach().float()
                loss = combined_loss(losses, counts, weights, config.gradient_accumulation_steps, world())
                engine.backward(loss)
                if reducer:
                    pending_peak_bytes = max(pending_peak_bytes, reducer.peak_pending_bytes)
                engine.step()
                del losses, loss, features, batch
            if engine.global_steps != before_step + 1:
                raise RuntimeError("DeepSpeed accumulation did not produce one optimizer step per window")
            window_index += 1
            final_epoch, final_window = (epoch + 1, 0) if window_index == total_windows else (epoch, window_index)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - tick
            means = {name: sum_scalar(local_losses[name].item(), device) / max(counts[name], 1)
                     for name in LOSS_NAMES}
            perf = torch.tensor([seconds, torch.cuda.max_memory_allocated() / 1024**3,
                                 pending_peak_bytes / 1024**3], device=device, dtype=torch.float64)
            if dist.is_initialized():
                dist.all_reduce(perf, op=dist.ReduceOp.MAX)
            seconds, peak_gib, pending_peak_gib = perf.tolist()
            if rank() == 0 and (engine.global_steps % config.log_steps == 0 or engine.global_steps == 1):
                record = {"step": engine.global_steps, "epoch": epoch, "next_window": window_index,
                          **{name + "_loss": means[name] for name in LOSS_NAMES},
                          "weighted_loss": sum(weights[name] * means[name] for name in LOSS_NAMES),
                          "loss_counts": counts, "samples": samples, "samples_per_second": samples / seconds,
                          "step_seconds": seconds, "data_seconds": data_seconds, "peak_allocated_gib": peak_gib,
                          "zero2_pending_grad_peak_gib": pending_peak_gib, "lr": scheduler.get_last_lr()}
                print(json.dumps(record), flush=True)
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            stop = config.step_limit is not None and engine.global_steps >= config.step_limit
            if engine.global_steps % config.save_steps == 0 or stop:
                save_training(engine, output / "checkpoints", final_epoch, final_window, signature, source, teacher_hash)
                last_saved = engine.global_steps
            if stop:
                break
    if last_saved != engine.global_steps:
        save_training(engine, output / "checkpoints", final_epoch, final_window, signature, source, teacher_hash)
    export_stage2(engine, visual, processor, output / "export", config, source, teacher_hash)
    primary_call(lambda: json_write(output / "completion.json", {
        "finished_all_epochs": final_epoch >= config.epochs, "global_steps": engine.global_steps,
        "epoch": final_epoch, "next_window": final_window, "export": str(output / "export")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    config = Stage2Config.load(args.config)
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
