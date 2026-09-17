"""Validated, serializable experiment settings."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass
class TrainConfig:
    mode: str = "interaction_ce"
    model_path: str = "checkpoints/ilvr_comt"
    reference_path: str | None = None
    data_path: str = "data/comt/TRAIN.jsonl"
    image_root: str = "data/comt"
    prepared_dir: str = "data/comt/prepared"
    output_dir: str = "outputs/interaction_ce"
    deepspeed: str = "configs/interaction/zero2.json"
    epochs: int = 3
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    backbone_lr: float = 2e-6
    fusion_lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_seq_length: int = 32768
    visual_latent_size: int = 8
    fusion_dim: int = 1024
    fusion_heads: int = 8
    fusion_ffn_dim: int = 4096
    attention_backend: str = "flash_attention_2"
    activation_checkpointing: bool = True
    ce_chunk_size: int = 256
    num_workers: int = 4
    prefetch_factor: int = 2
    seed: int = 42
    save_steps: int = 100
    log_steps: int = 10
    max_steps: int | None = None
    resume: str | None = None
    profile_steps: int = 0
    execution: str = "incremental"

    @property
    def interaction_steps(self):
        return int(self.mode != "baseline_ct")

    @property
    def latent_steps(self):
        return self.visual_latent_size + self.interaction_steps

    def validate(self):
        if self.mode not in {"baseline_ct", "interaction_ce", "no_refinement"}:
            raise ValueError(f"Unknown experiment mode: {self.mode}")
        if self.attention_backend not in {"flash_attention_2", "sdpa"}:
            raise ValueError("attention_backend must be flash_attention_2 or sdpa")
        if self.execution not in {"incremental", "recompute"}:
            raise ValueError("execution must be incremental or recompute")
        for name in ("epochs", "micro_batch_size", "gradient_accumulation_steps",
                     "max_seq_length", "visual_latent_size", "fusion_dim", "fusion_heads",
                     "fusion_ffn_dim", "ce_chunk_size", "log_steps", "save_steps", "prefetch_factor"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.visual_latent_size != 8:
            raise ValueError("These controlled experiments require 8 visual latent steps")
        if self.fusion_dim % self.fusion_heads:
            raise ValueError("fusion_dim must be divisible by fusion_heads")
        if not 0 <= self.warmup_ratio < 1 or self.num_workers < 0:
            raise ValueError("Invalid warmup_ratio or num_workers")
        if self.backbone_lr <= 0 or self.fusion_lr <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path, overrides=None):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values.update(overrides or {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values).validate()
