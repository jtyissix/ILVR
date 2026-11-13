"""SFT-III resumes the stage-II backbone and its prepared student sequences."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

from ..sft1.config import PROJECT_ROOT, WarmupConfig

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sft3" / "relaxation.json"
LATENT_TYPES = {"1": "text", "2": "interaction", "3": "visual"}


def parse_collapse_targets(value):
    """Accept 0 or any non-repeating combination of 1/2/3, as JSON int/string."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("collapse_targets must be 0 or a combination of 1, 2, 3")
    value = str(value)
    if value == "0":
        return ()
    if not value or any(c not in LATENT_TYPES for c in value) or len(set(value)) != len(value):
        raise ValueError("collapse_targets must be 0, 1, 2, 3, 12, 13, 23, or 123 (any order)")
    return tuple(LATENT_TYPES[c] for c in sorted(value))


@dataclass
class Stage3Config:
    mode: str = "train"
    stage2_path: str = "outputs/sft2_latent/export"
    prepared_dir: str = "data/sft2/prepared"
    output_dir: str = "outputs/sft3_relaxation"
    deepspeed: str = "configs/sft3/zero2.json"
    gpu_ids: list[int] | None = None
    num_gpus: int | None = None
    epochs: int = 3
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    backbone_lr: float = 2e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_seq_length: int = 32768
    min_image_pixels: int = 3136
    max_image_pixels: int = 12845056
    action_latents: int = 4  # text/action latents from SFT-II
    interaction_tokens: int = 3
    visual_latents: int = 16
    collapse_targets: str | int = "0"
    collapse_weight: float = 0.2
    collapse_margin: float = 0.8
    collapse_alpha: float = 1.0
    collapse_epsilon: float = 0.1
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
        return str(Path(self.stage2_path) / "backbone")

    @property
    def step_limit(self):
        return min(self.max_steps or self.smoke_steps, self.smoke_steps) if self.mode == "smoke" else self.max_steps

    @property
    def selected_types(self):
        return parse_collapse_targets(self.collapse_targets) if self.collapse_weight else ()

    def validate(self):
        common = {f.name for f in fields(WarmupConfig)}
        WarmupConfig(**{k: v for k, v in self.to_dict().items() if k in common},
                     model_path=self.model_path).validate()
        parse_collapse_targets(self.collapse_targets)
        for name in ("backbone_lr", "weight_decay", "warmup_ratio", "collapse_weight", "collapse_margin",
                     "collapse_alpha", "collapse_epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.collapse_weight < 0:
            raise ValueError("collapse_weight must be nonnegative")
        if not 0 <= self.collapse_margin <= 1 or not 0 <= self.collapse_epsilon <= 1 or self.collapse_alpha < 1:
            raise ValueError("Require collapse_margin/epsilon in [0,1] and collapse_alpha >= 1")
        for name, expected in (("action_latents", 4), ("interaction_tokens", 3), ("visual_latents", 16)):
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected}, matching the SFT-II sequence")
        if not isinstance(self.activation_checkpointing, bool):
            raise ValueError("activation_checkpointing must be boolean")
        if type(self.num_workers) is not int or self.num_workers < 0 or type(self.seed) is not int:
            raise ValueError("num_workers must be a nonnegative integer and seed must be an integer")
        for name in ("stage2_path", "prepared_dir", "output_dir", "deepspeed", "resume"):
            value = getattr(self, name)
            if name == "resume" and value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty path")
        return self

    def resolve_paths(self, root=PROJECT_ROOT):
        for name in ("stage2_path", "prepared_dir", "output_dir", "deepspeed", "resume"):
            value = getattr(self, name)
            if value is not None:
                path = Path(value).expanduser()
                setattr(self, name, str((Path(root) / path).resolve() if not path.is_absolute() else path.resolve()))
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
            raise ValueError(f"Unknown SFT-III config keys: {sorted(unknown)}")
        return cls(**values).validate().resolve_paths()


def validate_local_inputs(config):
    required = [Path(config.model_path) / "config.json", Path(config.stage2_path) / "stage2_recipe.json",
                Path(config.prepared_dir) / "manifest.json", Path(config.prepared_dir) / "records.jsonl",
                Path(config.deepspeed)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Set stage2_path and prepared_dir in configs/sft3/relaxation.json:\n  "
                                + "\n  ".join(missing))
    from ..sft2.checkpoint import FORMAT
    recipe = json.loads(required[1].read_text(encoding="utf-8"))
    if recipe.get("format") != FORMAT:
        raise ValueError("SFT-III requires the current SFT-II export with latent boundaries")
    for key in ("action_latents", "interaction_tokens", "visual_latents"):
        if recipe.get("config", {}).get(key) != getattr(config, key):
            raise ValueError(f"SFT-II export {key} differs from the relaxation recipe")


TrainConfig = Stage3Config
