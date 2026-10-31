"""Synchronized finite checks and detached diagnostics, local to CPU offload.

Collectives run at fixed microbatch boundaries, never in variable-length
trajectories or gradient hooks. No NaN replacement or skipped updates.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist


def extrema(value):
    """Two scalars without a full-size abs/isfinite temporary or FP32 copy."""
    value = value.detach()
    if not value.numel():
        return torch.zeros(2, device=value.device, dtype=torch.float64)
    low, high = torch.aminmax(value)
    return torch.stack((low, high)).to(dtype=torch.float64)


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


class NumericalGuard:
    def __init__(self, output, device):
        self.device = torch.device(device)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.directory = Path(output) / "numerics"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"rank-{self.rank:05d}.jsonl"
        self.run_id = time.time_ns()
        self.context = {}
        self.observations = {}
        self.gradients = []

    def write(self, record):
        record = clean_json({"run_id": self.run_id, "rank": self.rank,
                             "time_ns": time.time_ns(), **self.context, **record})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return record

    def begin(self, step, epoch, window, microbatch, batch, window_records):
        self.context = {"update": step, "epoch": epoch, "window": window,
                        "microbatch": microbatch, "record_ids": batch.get("record_ids", []),
                        "source_lines": batch.get("source_lines", []),
                        "window_record_ids": window_records,
                        "sequence_lengths": batch["attention_mask"].sum(-1).tolist(),
                        "padded_length": batch["input_ids"].shape[1],
                        "actions_per_sample": [len(x) for x in batch["actions"]],
                        "helpers_per_sample": [len(x) for x in batch["segments"]],
                        "loss_counts": batch["loss_counts"]}
        self.observations = {}
        self.gradients = []
        self.write({"stage": "begin"})

    def observe(self, name, value):
        if value is not None:
            self.observations[name] = extrema(value)

    def observe_vectors(self, name, values):
        values = list(values)
        if values:
            vectors = torch.cat([v.detach().reshape(-1, v.shape[-1]) for v in values])
            self.observe(name + ".l2", torch.linalg.vector_norm(vectors.double(), dim=-1))

    def install_trajectory(self, student):
        native = student.trajectory

        def trajectory(batch, image_features=None):
            result = native(batch, image_features)
            with torch.no_grad():
                self.observe("student_hidden", result.hidden)
                self.observe_vectors("action_input", result.actions.values())
                self.observe_vectors("visual_input", result.visuals.values())
                self.observe_vectors("interaction", result.interactions.values())
                self.observe_vectors("helper_target", result.visual_targets.values())
                self.observe_vectors("action_output", (
                    result.hidden[row, action["positions"][-1]]
                    for row, actions in enumerate(batch["actions"]) for action in actions))
            return result

        student.trajectory = trajectory

    def inputs(self, batch, features):
        self.observe("image_features", features)
        teacher = batch.get("teacher_targets")
        if teacher is not None:
            mask = batch.get("teacher_action_mask")
            selected = teacher[mask] if mask is not None else teacher
            if selected.numel():
                self.observe("teacher_target", selected)
                # Valid actions only; the state-loss denominator BEFORE clamping.
                self.observe("teacher_mean_square", selected.detach().double().square().mean(-1))
        self.check("inputs")

    def observe_gradient(self, stage, name, gradient):
        self.gradients.append((stage, name, extrema(gradient)))

    def gradient_summary(self):
        if not self.gradients:
            return {}, []
        values = torch.stack([item[2] for item in self.gradients]).cpu().tolist()
        groups, bad = {}, []
        for (stage, name, _), (low, high) in zip(self.gradients, values):
            magnitude = max(abs(low), abs(high))
            if not math.isfinite(low) or not math.isfinite(high):
                bad.append(f"{stage}:{name}")
            groups.setdefault(stage, []).append((magnitude, name))
        result = {stage: [{"parameter": name, "abs_max": magnitude}
                          for magnitude, name in sorted(items, reverse=True)[:5]]
                  for stage, items in groups.items()}
        self.gradients = []
        return result, bad

    def check(self, stage, tensors=None, extra=None, gradients=False):
        for name, value in (tensors or {}).items():
            self.observe(name, value)
        stats, bad, devices = {}, [], {}
        for name, value in self.observations.items():
            devices.setdefault(value.device, []).append((name, value))
        for entries in devices.values():
            values = torch.stack([v for _, v in entries]).cpu().tolist()
            for (name, _), (low, high) in zip(entries, values):
                stats[name] = {"min": low, "max": high}
                if not math.isfinite(low) or not math.isfinite(high):
                    bad.append(name)
        grad_stats = {}
        if gradients:
            grad_stats, bad_gradients = self.gradient_summary()
            bad.extend(bad_gradients)
        flag = torch.tensor(int(bool(bad)), device=self.device, dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        failed = bool(flag.item())
        record = self.write({"stage": stage, "failed_any_rank": failed,
                             "local_nonfinite": bad[:32], "local_nonfinite_count": len(bad),
                             "tensor_ranges": stats, "gradient_top5": grad_stats, **(extra or {})})
        if failed:
            failure = self.directory / f"failure-{self.run_id}-rank-{self.rank:05d}.json"
            failure.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            raise FloatingPointError(
                f"SFT-II offload non-finite at {stage}; all ranks stop without saving a checkpoint. "
                f"Local tensors: {bad[:5]}; diagnostics: {failure}")

    def check_state(self, optimizer, stage):
        """Scan state without full-size temporaries; a bad update cannot roll back."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.observations = {}
        for i, master in enumerate(optimizer.single_partition_of_fp32_groups):
            self.observe(f"master.{i}", master)
            if stage == "before_update":
                self.observe(f"gradient_partition.{i}", master.grad)
            else:
                for name, value in optimizer.optimizer.state.get(master, {}).items():
                    if isinstance(value, torch.Tensor):
                        self.observe(f"adam.{i}.{name}", value)
        if stage != "before_update":
            for i, value in enumerate(optimizer.bit16_groups_flat):
                self.observe(f"model_partition_group.{i}", value)
        extra = {}
        if stage == "after_update":
            norm = getattr(optimizer, "_global_grad_norm", None)
            if norm is not None:
                self.observe("global_grad_norm", torch.as_tensor(norm))
                extra["unclipped_global_grad_norm"] = float(norm)
                extra["clip_threshold"] = optimizer.clip_grad
                extra["gradient_multiplier"] = (min(1., optimizer.clip_grad / (float(norm) + 1e-6))
                                                  if optimizer.clip_grad > 0 else 1.)
        self.check(stage, extra=extra)
        self.observations = {}
