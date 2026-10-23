"""One JSON file controls the SFT-I job; relative paths use the ILVR root."""
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sft1" / "warmup.json"


def default_sources():
    return [
        {"name": "comt", "path": "../comt_rewrite_output/hard/TRAIN.jsonl",
         "image_root": "data/comt", "plan_field": "thought_rewritten",
         "answer_field": "original_final_answer", "require_structured_actions": True, "allow_append_final_answer": True,
         "skip_missing_plan": True, "skip_empty_answer": True},
        {"name": "refocus", "path": "../refocus_bbox_output/hard_cropped.jsonl",
         "image_root": "../refocus_bbox_output", "plan_field": "sequence_plan",
         "question_field": "question", "images_field": "image", "answer_field": "answer",
         "require_structured_actions": True, "allow_append_final_answer": True},
        {"name": "visual_cot", "path": "../visual_cot_grounded_causal_v2_output/train.comt.jsonl",
         "image_root": "../visual_cot_grounded_causal_v2_output", "plan_field": "sequence_plan",
         "answer_field": "original_final_answer", "require_structured_actions": True},
        {"name": "zebra_cot_subset", "path": "../zebra_rewrite_v2_output/hard_rewritten.jsonl",
         "image_root": "../zebra_rewrite_v2_output", "plan_field": "sequence_plan",
         "answer_field": "original_final_answer", "require_structured_actions": True},
    ]


@dataclass
class WarmupConfig:
    mode: str = "train"  # train | prepare_only | smoke (real training, short run)
    model_path: str = "checkpoints/ilvr_comt"
    sources: list[dict] = field(default_factory=default_sources)
    prepared_dir: str = "data/sft1/prepared"
    output_dir: str = "outputs/sft1_warmup"
    deepspeed: str = "configs/sft1/zero2.json"
    gpu_ids: list[int] | None = None  # null = keep CUDA_VISIBLE_DEVICES unchanged
    num_gpus: int | None = None  # null = use all visible GPUs
    epochs: int = 3
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    backbone_lr: float = 2e-6
    fusion_lr: float = 5e-5
    adapter_lr: float | None = None  # null = fusion_lr
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_seq_length: int = 32768
    min_image_pixels: int = 3136  # 4 * 28 * 28
    max_image_pixels: int = 12845056  # 16384 * 28 * 28 (Qwen2.5-VL default)
    fusion_dim: int = 1024
    fusion_heads: int = 8
    fusion_ffn_dim: int = 4096
    fusion_score_dim: int = 256
    image_branch_enabled: bool = True
    adapter_layers: int = 2
    adapter_slots: int = 16
    interaction_tokens: int = 3
    shrink_overlong_images: bool = True  # Try a smaller per-image budget before excluding a row.
    skip_overlong_samples: bool = True  # Skip only if the enabled resize policy still cannot fit.
    prepare_workers: int = 4  # CPU processes; 1 runs serially for debugging.
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
    def step_limit(self):
        if self.mode == "smoke":
            return min(self.max_steps or self.smoke_steps, self.smoke_steps)
        return self.max_steps

    def validate(self):
        if self.mode not in {"train", "prepare_only", "smoke"}:
            raise ValueError("mode must be train, prepare_only, or smoke")
        if self.attention_backend not in {"flash_attention_2", "sdpa"}:
            raise ValueError("attention_backend must be flash_attention_2 or sdpa")
        if self.execution not in {"incremental", "recompute"}:
            raise ValueError("execution must be incremental or recompute")
        for name in ("epochs", "micro_batch_size", "gradient_accumulation_steps", "max_seq_length",
                     "min_image_pixels", "max_image_pixels", "fusion_dim", "fusion_heads", "fusion_ffn_dim",
                     "fusion_score_dim", "adapter_layers", "adapter_slots", "interaction_tokens", "ce_chunk_size", "prefetch_factor",
                     "save_steps", "log_steps", "smoke_steps", "prepare_workers", "prepare_log_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_steps", "num_gpus"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be null or a positive integer")
        if self.fusion_dim % self.fusion_heads:
            raise ValueError("fusion_dim must be divisible by fusion_heads")
        if not isinstance(self.shrink_overlong_images, bool):
            raise ValueError("shrink_overlong_images must be boolean")
        if not isinstance(self.skip_overlong_samples, bool):
            raise ValueError("skip_overlong_samples must be boolean")
        if self.min_image_pixels > self.max_image_pixels:
            raise ValueError("min_image_pixels cannot exceed max_image_pixels")
        if not 0 <= self.warmup_ratio < 1 or self.num_workers < 0:
            raise ValueError("Invalid warmup_ratio or num_workers")
        if self.backbone_lr <= 0 or self.fusion_lr <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.adapter_lr is not None and self.adapter_lr <= 0:
            raise ValueError("adapter_lr must be null or positive")
        if self.gpu_ids is not None:
            if (not isinstance(self.gpu_ids, list) or not self.gpu_ids
                    or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.gpu_ids)
                    or len(set(self.gpu_ids)) != len(self.gpu_ids)):
                raise ValueError("gpu_ids must be null or a nonempty list of unique nonnegative integers")
            if self.num_gpus is not None and self.num_gpus > len(self.gpu_ids):
                raise ValueError("num_gpus exceeds the configured gpu_ids")
        if not isinstance(self.sources, list) or not self.sources:
            raise ValueError("sources must contain at least one explicitly configured rewritten dataset")
        allowed = {"name", "path", "image_root", "plan_field", "answer_field", "question_field", "images_field",
                   "allow_append_final_answer", "json_records_key", "require_structured_actions",
                   "skip_missing_plan", "skip_empty_answer"}
        names = set()
        for source in self.sources:
            if not isinstance(source, dict) or set(source) - allowed:
                raise ValueError(f"Invalid source configuration: {source}")
            for key in ("name", "path", "image_root", "plan_field"):
                if not isinstance(source.get(key), str) or not source[key].strip():
                    raise ValueError(f"Each source requires a nonempty {key}")
            if source["name"] in names:
                raise ValueError(f"Duplicate source name: {source['name']}")
            names.add(source["name"])
            for key in ("allow_append_final_answer", "require_structured_actions", "skip_missing_plan", "skip_empty_answer"):
                if key in source and not isinstance(source[key], bool):
                    raise ValueError(f"{key} must be boolean")
        return self

    def resolve_paths(self, root=PROJECT_ROOT):
        root = Path(root)
        def resolved(value):
            path = Path(value).expanduser()
            return str((root / path).resolve() if not path.is_absolute() else path.resolve())
        for key in ("model_path", "prepared_dir", "output_dir", "deepspeed", "resume"):
            if getattr(self, key) is not None:
                setattr(self, key, resolved(getattr(self, key)))
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
            raise ValueError(f"Unknown SFT-I config keys: {sorted(unknown)}")
        return cls(**values).validate().resolve_paths()


def validate_local_inputs(config):
    """Report all missing local paths together before creating a distributed job."""
    missing = []
    if not (Path(config.model_path) / "config.json").is_file():
        missing.append(f"model_path: {config.model_path} (local Qwen2.5-VL/ILVR checkpoint)")
    for source in config.sources:
        if not Path(source["path"]).is_file():
            missing.append(f"sources.{source['name']}.path: {source['path']}")
        if not Path(source["image_root"]).is_dir():
            missing.append(f"sources.{source['name']}.image_root: {source['image_root']}")
    if not Path(config.deepspeed).is_file():
        missing.append(f"deepspeed: {config.deepspeed}")
    if missing:
        raise FileNotFoundError("Set the server's real paths in configs/sft1/warmup.json; nothing is downloaded:\n  "
                                + "\n  ".join(missing))


# Convenient compatibility for callers that use the existing config class naming.
TrainConfig = WarmupConfig
