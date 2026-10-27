"""Second-stage configuration; the first-stage export is the sole initialization."""
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path

from ..sft1.config import PROJECT_ROOT, WarmupConfig, default_sources

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sft2" / "latent.json"


@dataclass
class Stage2Config:
    mode: str = "train"
    warmup_path: str = "outputs/sft1_warmup/export"
    sources: list[dict] = field(default_factory=default_sources)
    prepared_dir: str = "data/sft2/prepared"
    teacher_cache_dir: str | None = None
    output_dir: str = "outputs/sft2_latent"
    deepspeed: str = "configs/sft2/zero2.json"
    gpu_ids: list[int] | None = None
    num_gpus: int | None = None
    epochs: int = 3
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    backbone_lr: float = 2e-6
    fusion_lr: float = 3e-5
    aux_decoder_lr: float = 2e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_seq_length: int = 32768
    min_image_pixels: int = 3136
    max_image_pixels: int = 12845056
    fusion_dim: int = 1024
    fusion_heads: int = 8
    fusion_ffn_dim: int = 4096
    fusion_score_dim: int = 256
    image_branch_enabled: bool = True
    action_latents: int = 4
    visual_latents: int = 16
    pool_grid: int = 4
    interaction_tokens: int = 3
    ce_weight: float = 1.0
    visual_weight: float = 1.0
    action_words_weight: float = 1.0
    action_state_weight: float = 1.0
    state_loss_epsilon: float = 1e-6
    shrink_overlong_images: bool = True
    skip_overlong_samples: bool = True
    prepare_workers: int = 4
    prepare_log_seconds: int = 5
    attention_backend: str = "flash_attention_2"
    activation_checkpointing: bool = True
    ce_chunk_size: int = 256
    execution: str = "incremental"
    num_workers: int = 4
    prefetch_factor: int = 2
    seed: int = 42
    save_steps: int = 100
    log_steps: int = 10
    max_steps: int | None = None
    smoke_steps: int = 2
    resume: str | None = None

    @property
    def model_path(self):
        return str(Path(self.warmup_path) / "backbone")

    @property
    def step_limit(self):
        return min(self.max_steps or self.smoke_steps, self.smoke_steps) if self.mode == "smoke" else self.max_steps

    def validate(self):
        # Reuse the established source, image, optimizer and launcher validation.
        common = {f.name for f in fields(WarmupConfig)}
        WarmupConfig(**{k: v for k, v in self.to_dict().items() if k in common},
                     model_path=self.model_path, adapter_slots=self.visual_latents).validate()
        for name in ("action_latents", "visual_latents", "pool_grid"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (self.action_latents, self.visual_latents, self.interaction_tokens) != (4, 16, 3):
            raise ValueError("This stage implements exactly 4 action, 16 visual and 3 interaction tokens")
        if self.pool_grid ** 2 != self.visual_latents:
            raise ValueError("pool_grid squared must equal visual_latents")
        for name in ("ce_weight", "visual_weight", "action_words_weight", "action_state_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("aux_decoder_lr", "state_loss_epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not any(getattr(self, n) for n in ("ce_weight", "visual_weight", "action_words_weight", "action_state_weight")):
            raise ValueError("At least one loss weight must be positive")
        for name in ("warmup_path", "prepared_dir", "output_dir", "deepspeed"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty path")
        if not isinstance(self.image_branch_enabled, bool) or not isinstance(self.activation_checkpointing, bool):
            raise ValueError("image_branch_enabled and activation_checkpointing must be boolean")
        return self

    def resolve_paths(self, root=PROJECT_ROOT):
        def resolved(value):
            path = Path(value).expanduser()
            return str((Path(root) / path).resolve() if not path.is_absolute() else path.resolve())
        for name in ("warmup_path", "prepared_dir", "output_dir", "deepspeed", "resume"):
            if getattr(self, name) is not None:
                setattr(self, name, resolved(getattr(self, name)))
        self.teacher_cache_dir = (resolved(self.teacher_cache_dir) if self.teacher_cache_dir is not None
                                  else str(Path(self.prepared_dir) / "teacher"))
        self.sources = [{**source, "path": resolved(source["path"]),
                         "image_root": resolved(source["image_root"])} for source in self.sources]
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=DEFAULT_CONFIG):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown SFT-II config keys: {sorted(unknown)}")
        return cls(**values).validate().resolve_paths()


def interaction_kwargs(config, hidden_size):
    return {"hidden_size": hidden_size, "dim": config.fusion_dim, "heads": config.fusion_heads,
            "ffn_dim": config.fusion_ffn_dim, "score_dim": config.fusion_score_dim,
            "image_branch_enabled": config.image_branch_enabled, "slots": config.visual_latents,
            "interaction_tokens": config.interaction_tokens}


def validate_local_inputs(config):
    missing = []
    root = Path(config.warmup_path)
    for name in ("backbone/config.json", "interaction.safetensors", "interaction_config.json", "warmup_recipe.json",
                 "warmup_adapter.safetensors", "warmup_adapter_config.json"):
        if not (root / name).is_file():
            missing.append(str(root / name))
    for source in config.sources:
        if not Path(source["path"]).is_file():
            missing.append(f"sources.{source['name']}.path: {source['path']}")
        if not Path(source["image_root"]).is_dir():
            missing.append(f"sources.{source['name']}.image_root: {source['image_root']}")
    if not Path(config.deepspeed).is_file():
        missing.append(f"deepspeed: {config.deepspeed}")
    if missing:
        raise FileNotFoundError("Set actual local SFT-I export and source paths in configs/sft2/latent.json:\n  "
                                + "\n  ".join(missing))
    metadata = json.loads((root / "interaction_config.json").read_text(encoding="utf-8"))
    backbone = json.loads((root / "backbone/config.json").read_text(encoding="utf-8"))
    if metadata.get("format") != "sft1-interaction-v1" or metadata.get("kwargs") != interaction_kwargs(config, backbone["hidden_size"]):
        raise ValueError("Stage-2 interaction architecture must exactly match interaction_config.json in the warmup export")


load_config = Stage2Config.load
TrainConfig = Stage2Config
