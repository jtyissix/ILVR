"""Launch with torchrun; DeepSpeed owns optimizer steps and gradient accumulation."""
import argparse
from contextlib import nullcontext
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
from .checkpoint import export_inference, load_training, resume_signature, save_training, source_fingerprint
from .config import TrainConfig
from .data import Collator, PreparedDataset, json_write, prepare_dataset, token_ids
from .distributed import GlobalBatchSampler, align_zero3, cosine_factor, normalized_loss, rank, sum_scalar, to_device, world
from .model import assert_local_transformers, load_training_components, reference_contexts


def primary_call(function):
    """Propagate rank-zero startup failures instead of leaving peers at a barrier."""
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
        fusion = name.startswith("fusion.")
        decay = param.ndim > 1
        groups.setdefault((fusion, decay), []).append(param)
    return [{"params": params, "lr": config.fusion_lr if fusion else config.backbone_lr,
             "weight_decay": config.weight_decay if decay else 0.0}
            for (fusion, decay), params in groups.items()]


def configure_deepspeed(config, device):
    ds = json.loads(Path(config.deepspeed).read_text(encoding="utf-8"))
    stage = ds.get("zero_optimization", {}).get("stage")
    if stage not in {2, 3}:
        raise ValueError("This recipe supports DeepSpeed ZeRO-2 or ZeRO-3")
    if not ds.get("bf16", {}).get("enabled"):
        raise ValueError("This controlled recipe requires BF16")
    ds.update({"train_micro_batch_size_per_gpu": config.micro_batch_size,
               "gradient_accumulation_steps": config.gradient_accumulation_steps,
               "train_batch_size": config.micro_batch_size * config.gradient_accumulation_steps * world()})
    if "optimizer" in ds or "scheduler" in ds:
        raise ValueError("Optimizer and scheduler are defined by the experiment; remove them from DeepSpeed JSON")
    return ds, stage


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    for key in ("model_path", "reference_path", "data_path", "image_root", "prepared_dir", "output_dir", "deepspeed", "resume", "mode", "execution"):
        parser.add_argument("--" + key)
    for key in ("micro_batch_size", "gradient_accumulation_steps", "num_workers", "max_steps", "profile_steps",
                "log_steps", "save_steps", "epochs", "max_seq_length"):
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None)
    values = vars(parser.parse_args())
    filename = values.pop("config")
    values.pop("local_rank")
    return TrainConfig.load(filename, {k: v for k, v in values.items() if v is not None})


class StepTimer:
    def __init__(self):
        self.events = {}

    def start(self, name):
        pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        self.events.setdefault(name, []).append(pair)
        pair[0].record()

    def end(self, name):
        self.events[name][-1][1].record()

    def milliseconds(self):
        return {name + "_ms": sum(start.elapsed_time(end) for start, end in pairs)
                for name, pairs in self.events.items()}


def train(config):
    assert_local_transformers()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a BF16-capable CUDA GPU; use the CPU tests for local verification")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    import deepspeed
    deepspeed.init_distributed(dist_backend="nccl")
    random.seed(config.seed + rank())
    np.random.seed(config.seed + rank())
    torch.manual_seed(config.seed)  # identical newly initialized fusion weights across ranks
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    output = Path(config.output_dir)

    def startup():
        output.mkdir(parents=True, exist_ok=True)
        if (output / "checkpoints" / "latest").exists() and config.resume is None:
            raise ValueError("Output already contains training checkpoints; specify --resume or a new output_dir")
        if not (Path(config.prepared_dir) / "manifest.json").is_file():
            prepare_dataset(config.data_path, config.image_root, config.model_path, config.prepared_dir, config.max_seq_length)
        source = source_fingerprint(config.model_path, output / "initial_source.json")
        if config.mode == "interaction_ce" and config.reference_path:
            reference = source_fingerprint(config.reference_path, output / "reference_source.json")
            if reference != source:
                raise ValueError("Reference must be the SAME initial CoMT checkpoint; shared frozen vision requires identical weights")
        json_write(output / "train_config.json", config.to_dict())
        return source

    source = primary_call(startup)
    dataset = PreparedDataset(config.prepared_dir)
    primary_call(lambda: dataset.validate_sources(config))
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config.model_path, local_files_only=True)
    ds_config, stage = configure_deepspeed(config, device)
    student, visual, reference = load_training_components(config, device)
    token_ids(processor, student.model_config.to_dict())
    collator = Collator(processor, student.model_config.to_dict(), config.interaction_steps, dataset.manifest)
    total_windows = math.ceil(len(dataset) / (world() * config.micro_batch_size * config.gradient_accumulation_steps))
    total_steps = total_windows * config.epochs
    groups = optimizer_groups(student, config)
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_factor(step, total_steps, config.warmup_ratio))
    engine, _, _, _ = deepspeed.initialize(model=student, optimizer=optimizer, lr_scheduler=scheduler,
                                          config=ds_config, dist_init_required=False)
    signature = resume_signature(config, dataset.manifest["source_sha256"], source)
    epoch_start, start_window = 0, 0
    if config.resume:
        epoch_start, start_window = load_training(engine, config.resume, signature, source)
    if rank() == 0:
        json_write(output / "run_info.json", {"world_size": world(), "gpu": torch.cuda.get_device_name(),
                   "torch": torch.__version__, "deepspeed": deepspeed.__version__, "zero_stage": stage,
                   "effective_batch": world()*config.micro_batch_size*config.gradient_accumulation_steps,
                   "total_optimizer_steps": total_steps, "reference_fingerprint": source, "signature": signature,
                   "dataset_sha256": dataset.manifest["source_sha256"]})
    engine.train()
    last_saved = -1
    stop = config.max_steps is not None and engine.global_steps >= config.max_steps
    final_epoch, final_window = epoch_start, start_window
    for epoch in range(epoch_start, config.epochs):
        if stop:
            break
        sampler = GlobalBatchSampler(dataset.records, config.micro_batch_size, config.gradient_accumulation_steps,
                                     world(), rank(), config.seed, epoch, start_window if epoch == epoch_start else 0)
        loader_args = {"dataset": dataset, "batch_sampler": sampler, "collate_fn": collator,
                       "num_workers": config.num_workers, "pin_memory": True,
                       "generator": torch.Generator().manual_seed(config.seed + epoch*world() + rank())}
        if config.num_workers:
            loader_args.update(prefetch_factor=config.prefetch_factor, multiprocessing_context="spawn", persistent_workers=True)
        iterator = iter(DataLoader(**loader_args))
        window_index = sampler.start_window
        while window_index < total_windows:
            tick = time.perf_counter()
            window = [next(iterator) for _ in range(config.gradient_accumulation_steps)]
            data_seconds = time.perf_counter()-tick
            count = sum_scalar(sum(b["ce_count"] for b in window), device)
            samples = sum_scalar(sum(b["sample_count"] for b in window), device)
            if count <= 0:
                raise ValueError("An optimizer window contains no supervised tokens")
            timer = StepTimer()
            local_loss = torch.zeros((), device=device, dtype=torch.float32)
            torch.cuda.reset_peak_memory_stats()
            do_profile = config.profile_steps > 0 and engine.global_steps < config.profile_steps
            profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                              record_shapes=True, profile_memory=True) if do_profile else nullcontext()
            with profile as profiler:
                for cpu_batch in window:
                    batch = to_device(cpu_batch, device)
                    timer.start("vision")
                    with torch.no_grad():
                        features = visual(batch["visual"]["pixel_values"].to(dtype=torch.bfloat16),
                                          grid_thw=batch["visual"]["image_grid_thw"]) if batch["visual"] else None
                        if features is not None:
                            features = features.index_select(0, batch["visual"]["feature_indices"])
                    timer.end("vision")
                    timer.start("reference")
                    contexts = reference_contexts(reference, batch["reference"], features, student.model_config.image_token_id,
                                                  config.attention_backend, config.execution) if reference is not None else None
                    timer.end("reference")
                    events, head_count = align_zero3(batch["student"], processor.tokenizer.pad_token_id, device) if stage == 3 else (None, None)
                    timer.start("student")
                    loss_sum = engine(batch["student"], features, contexts, events, head_count)
                    local_loss += loss_sum.detach().float()
                    timer.end("student")
                    timer.start("backward")
                    engine.backward(normalized_loss(loss_sum, count, config.gradient_accumulation_steps, world()))
                    timer.end("backward")
                    timer.start("optimizer")
                    engine.step()
                    timer.end("optimizer")
                    del loss_sum, contexts, features, batch
            window_index += 1
            final_epoch, final_window = (epoch + 1, 0) if window_index == total_windows else (epoch, window_index)
            # A single window boundary synchronization also makes timing and peak-memory reports honest.
            torch.cuda.synchronize()
            seconds = time.perf_counter()-tick
            loss_value = sum_scalar(local_loss.item(), device)/count
            timings = timer.milliseconds()
            peak = torch.cuda.max_memory_allocated()/1024**3
            perf = torch.tensor([seconds, peak, *timings.values()], device=device, dtype=torch.float64)
            if dist.is_initialized():
                dist.all_reduce(perf, op=dist.ReduceOp.MAX)
            perf = perf.tolist()
            if rank() == 0 and (engine.global_steps % config.log_steps == 0 or engine.global_steps == 1):
                record = {"step": engine.global_steps, "epoch": epoch, "next_window": window_index,
                          "ce_loss": loss_value, "effective_tokens": count, "samples": samples,
                          "tokens_per_second": count/perf[0], "samples_per_second": samples/perf[0],
                          "step_seconds": perf[0], "data_seconds": data_seconds, "peak_allocated_gib": perf[1],
                          **dict(zip(timings, perf[2:])), "lr": scheduler.get_last_lr()}
                print(json.dumps(record), flush=True)
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            if do_profile:
                trace = output / "profiles" / f"rank{rank()}_step{engine.global_steps}.json"
                trace.parent.mkdir(parents=True, exist_ok=True)
                profiler.export_chrome_trace(str(trace))
            stop = config.max_steps is not None and engine.global_steps >= config.max_steps
            if engine.global_steps % config.save_steps == 0 or stop:
                save_training(engine, output / "checkpoints", final_epoch, final_window, signature, source)
                last_saved = engine.global_steps
            if stop:
                break
    if last_saved != engine.global_steps:
        save_training(engine, output / "checkpoints", final_epoch, final_window, signature, source)
    export_inference(engine, visual, processor, output / "inference", config, source)
    if rank() == 0:
        json_write(output / "completion.json", {"finished_all_epochs": final_epoch >= config.epochs,
                   "global_steps": engine.global_steps, "epoch": final_epoch, "next_window": final_window})


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
