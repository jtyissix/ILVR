"""Deterministic global batches, loss normalization and ZeRO-3 call alignment."""
import math
import random
import torch
import torch.distributed as dist
from torch.nn import functional as F


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def sum_scalar(value, device):
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor.item()


def normalized_loss(local_sum, global_token_count, accumulation_steps, world_size):
    # DeepSpeed divides by GAS; its data parallel reduction averages by world size.
    return local_sum * (world_size * accumulation_steps / max(global_token_count, 1))


class GlobalBatchSampler:
    """Every source sample occurs once per epoch; -1 fills only the final window.

    Each window is one optimizer update. A complete last window avoids leaving
    DeepSpeed with unflushed partial accumulation; dummy labels contribute zero.
    """
    def __init__(self, records, micro_batch, accumulation, world_size=1, rank_id=0, seed=42, epoch=0, start_window=0):
        self.micro = micro_batch
        self.accumulation = accumulation
        self.world_size = world_size
        self.rank_id = rank_id
        self.global_size = micro_batch * accumulation * world_size
        rng = random.Random(seed + epoch)
        indices = list(range(len(records)))
        rng.shuffle(indices)
        ordered = []
        bucket_size = self.global_size * 32
        for start in range(0, len(indices), bucket_size):
            bucket = indices[start:start+bucket_size]
            bucket.sort(key=lambda i: (len(records[i]["segments"]), records[i]["length"]))
            ordered.extend(bucket)
        self.windows = [ordered[i:i+self.global_size] for i in range(0, len(ordered), self.global_size)]
        rng.shuffle(self.windows)
        self.start_window = start_window

    def __len__(self):
        return max(0, len(self.windows)-self.start_window) * self.accumulation

    def __iter__(self):
        for window in self.windows[self.start_window:]:
            window = window + [-1] * (self.global_size-len(window))
            for microstep in range(self.accumulation):
                start = (microstep*self.world_size + self.rank_id)*self.micro
                yield window[start:start+self.micro]


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        # CPU segment/event metadata must stay on CPU.
        return {key: value if key in {"segments", "latent_positions", "source_lines"} else to_device(value, device)
                for key, value in batch.items()}
    return batch


def align_zero3(batch, pad_id, device):
    """Same decoder event boundaries and head-call count on all ZeRO-3 ranks."""
    size = torch.tensor([batch["input_ids"].shape[1], int(batch["labels"][:, 1:].ne(-100).sum())], device=device)
    if dist.is_initialized():
        dist.all_reduce(size, op=dist.ReduceOp.MAX)
    target_length, head_count = size.tolist()
    extra = target_length-batch["input_ids"].shape[1]
    if extra:
        for key, fill in (("input_ids", pad_id), ("attention_mask", False), ("labels", -100), ("position_ids", 1)):
            batch[key] = F.pad(batch[key], (0, extra), value=fill)
    # Tensor collectives avoid pickle/object collectives and NCCL object staging.
    flags = torch.zeros(target_length, dtype=torch.int32, device=device)
    if batch["latent_positions"]:
        flags[list(batch["latent_positions"])] = 1
    if dist.is_initialized():
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    events = flags.nonzero().flatten().tolist()
    return events, head_count


def cosine_factor(step, total, warmup_ratio):
    warmup = math.ceil(total * warmup_ratio)
    if warmup and step < warmup:
        return step / warmup
    progress = min(1.0, (step-warmup) / max(1, total-warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))
