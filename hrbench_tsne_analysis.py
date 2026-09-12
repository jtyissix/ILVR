"""Capture ILVR original float16 features and optionally fit joint 3D t-SNE.

Edit the global configuration and run this script from the model repository.
Upload the sibling hrbench_tsne_common.py with this file. Existing PCA scripts
are independent. tsne_only mode reads saved high-dimensional features without
loading model weights. See the shared t-SNE README for full usage.
"""

from __future__ import annotations

import base64
import gc
import inspect
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from PIL import Image
# Resolve the portable helper beside this script, including python -m usage.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrbench_tsne_common import (
    sample_indices, save_features, run_tsne, validate_settings,
    prepare_capture_directory, json_compatible as _json_compatible,
    SampleDeadline, SampleCaptureTimeout, validate_timeout_settings,
    append_sample_progress, skipped_sample_record, write_empty_capture, check_timeout_budget,
)


# =============================================================================
# Global configuration -- edit values here; no CLI arguments are used
# =============================================================================

MODEL_PATH = "/home/fit/renjujty/jty/lmllms/ilvr/"
HRBENCH_PATH = "/home/fit/renjujty/jty/lmllms/hrbench/hr_bench_4k.parquet"
OUTPUT_DIR = "outputs/hrbench_tsne"
CACHE_DIR: str | None = None


# "sequential": START_INDEX ... START_INDEX + NUM_SAMPLES
# "random": deterministic sampling without replacement using RANDOM_SEED
SELECTION_MODE = "random"
START_INDEX = 0
NUM_SAMPLES = 800
RANDOM_SEED = 0

DEVICE_MAP = "auto"
TORCH_DTYPE = "bfloat16"
TRUST_REMOTE_CODE = False
ATTN_IMPLEMENTATION = "flash_attention_2"
FALLBACK_TO_EAGER_ATTENTION = True

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 8192 * 28 * 28
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.0
TOP_P = 1.0
USE_CACHE = True

VOCAB_EMBEDDING_BATCH_SIZE = 8192

# Keep temporary float16 captures only when an error occurs.
KEEP_TEMP_CAPTURE_ON_ERROR = True
# Cooperative deadline: checked between GPU decoding steps. None disables it.
SAMPLE_TIMEOUT_SECONDS: float | None = 300.0
MAX_SAMPLE_TIMEOUTS = 20


# t-SNE and persistent original-feature configuration.
# Modes: "capture_only", "capture_and_tsne", "tsne_only".
RUN_MODE = "capture_and_tsne"
VOCAB_SAMPLE_SIZE = 30_000
IMAGE_SAMPLE_SIZE = 30_000
# None means OUTPUT_DIR/features. Used only by tsne_only mode.
FEATURE_CACHE_DIR: str | None = None
# Each analysis writes OUTPUT_DIR/projections/TSNE_RUN_NAME; never overwritten.
TSNE_RUN_NAME = "tsne_001"
PCA_COMPONENTS = 50
TSNE_PERPLEXITY = 30.0
TSNE_MAX_ITER = 1500
TSNE_RANDOM_SEED = 0
TSNE_N_JOBS = -1

REQUIRED_COLUMNS = {
    "index",
    "question",
    "answer",
    "category",
    "A",
    "B",
    "C",
    "D",
    "cycle_category",
    "image",
}
MAX_PATH_CANDIDATE_LENGTH = 4096
LATENT_TOKEN_NAMES = {
    "pad": "<|latent_pad|>",
    "start": "<|latent_start|>",
    "end": "<|latent_end|>",
}


def select_sample_indices(
    total: int,
    mode: str = SELECTION_MODE,
    start_index: int = START_INDEX,
    count: int = NUM_SAMPLES,
    seed: int = RANDOM_SEED,
) -> list[int]:
    if total <= 0:
        raise ValueError("HRBench is empty.")
    if count <= 0:
        raise ValueError("NUM_SAMPLES must be positive.")
    if count > total:
        raise ValueError(f"NUM_SAMPLES={count} exceeds dataset size {total}.")
    if mode == "sequential":
        if start_index < 0 or start_index + count > total:
            raise ValueError(
                f"Sequential range [{start_index}, {start_index + count}) "
                f"is outside dataset size {total}."
            )
        return list(range(start_index, start_index + count))
    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(total, size=count, replace=False).tolist()
    raise ValueError("SELECTION_MODE must be 'sequential' or 'random'.")


def validate_configuration() -> tuple[Path, Path, Path]:
    if not MODEL_PATH:
        raise ValueError("MODEL_PATH is empty. Edit the global configuration first.")
    if not HRBENCH_PATH:
        raise ValueError("HRBENCH_PATH is empty. Edit the global configuration first.")
    if VOCAB_EMBEDDING_BATCH_SIZE <= 0:
        raise ValueError("VOCAB_EMBEDDING_BATCH_SIZE must be positive.")
    if MAX_NEW_TOKENS <= 0:
        raise ValueError("MAX_NEW_TOKENS must be positive.")
    if TEMPERATURE < 0:
        raise ValueError("TEMPERATURE cannot be negative.")
    if not 0 < TOP_P <= 1:
        raise ValueError("TOP_P must be in (0, 1].")

    model_path = Path(MODEL_PATH).expanduser()
    dataset_path = Path(HRBENCH_PATH).expanduser()
    output_path = Path(OUTPUT_DIR).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"MODEL_PATH is not a model directory: {model_path}. "
            "Edit the global variable."
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"HRBENCH_PATH is not a parquet file: {dataset_path}. "
            "Edit the global variable."
        )
    output_path.mkdir(parents=True, exist_ok=True)
    return model_path.resolve(), dataset_path.resolve(), output_path.resolve()


def load_hrbench_rows(
    dataset_path: Path,
) -> tuple[list[dict[str, Any]], list[int]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet", data_files=str(dataset_path), split="train"
    )
    missing = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise ValueError(
            "Unexpected HRBench schema; missing columns: "
            + ", ".join(sorted(missing))
        )
    selected = select_sample_indices(len(dataset))
    return [dict(dataset[index]) for index in selected], selected


def _open_image_path(path_value: str, dataset_dir: Path) -> Image.Image | None:
    if not path_value or "\x00" in path_value:
        return None
    try:
        possible_path = Path(path_value).expanduser()
        if not possible_path.is_absolute():
            possible_path = dataset_dir / possible_path
        if not possible_path.is_file():
            return None
    except (OSError, RuntimeError):
        return None
    try:
        return Image.open(possible_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to open image file: {possible_path}") from exc


def _open_image_bytes(image_bytes: bytes, description: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to decode {description} as an image.") from exc


def _decode_base64_image(encoded: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(encoded, validate=False)
    except Exception as exc:
        raise ValueError("The HRBench image contains invalid base64 data.") from exc
    return _open_image_bytes(image_bytes, "HRBench base64 data")


def decode_hrbench_image(value: Any, dataset_dir: Path) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return _open_image_bytes(value["bytes"], "HRBench byte data")
        if value.get("path"):
            image = _open_image_path(str(value["path"]), dataset_dir)
            if image is not None:
                return image
            raise ValueError(f"Image path does not exist: {value['path']}")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _open_image_bytes(bytes(value), "HRBench byte data")
    if not isinstance(value, str):
        raise TypeError(f"Unsupported HRBench image value: {type(value)!r}")

    value = value.strip()
    if value.startswith("data:image/"):
        if "," not in value:
            raise ValueError("Malformed image data URI: missing comma separator.")
        return _decode_base64_image(value.split(",", 1)[1])
    if len(value) <= MAX_PATH_CANDIDATE_LENGTH:
        image = _open_image_path(value, dataset_dir)
        if image is not None:
            return image
    return _decode_base64_image(value)


def build_question(row: dict[str, Any]) -> str:
    choices = "\n".join(f"({letter}) {row[letter]}" for letter in "ABCD")
    return (
        f"Question: {row['question']} The choices are listed below:\n"
        f"{choices}\nPut your final answer in \\boxed{{}}."
    )


def _torch_dtype_from_name(name: str) -> Any:
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(
            "TORCH_DTYPE must be 'float16', 'bfloat16', or 'float32'."
        ) from exc


def _single_token_id(tokenizer: Any, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise RuntimeError(
            f"Checkpoint token {token!r} must encode to exactly one token; "
            f"got {token_ids}. Do not add tokens during analysis."
        )
    return int(token_ids[0])


def discover_latent_token_ids(tokenizer: Any) -> dict[str, int]:
    return {
        name: _single_token_id(tokenizer, token)
        for name, token in LATENT_TOKEN_NAMES.items()
    }


def _validate_custom_ilvr_runtime(
    model: Any, tokenizer: Any, token_ids: dict[str, int]
) -> dict[str, Any]:
    forward_parameters = inspect.signature(model.forward).parameters
    required_parameters = {"generate_mode", "latent_hidden_states"}
    missing = required_parameters.difference(forward_parameters)
    if missing:
        raise RuntimeError(
            "The loaded Qwen2.5-VL class is not ILVR's customized Transformers "
            f"implementation (missing forward parameters: {sorted(missing)}). "
            "Install this repository's fork with `pip install -e ./transformers` "
            "and restart Python."
        )

    config_ids = {
        "pad": getattr(model.config, "latent_token_id", None),
        "start": getattr(model.config, "latent_start_id", None),
        "end": getattr(model.config, "latent_end_id", None),
    }
    mismatches = {
        name: {"tokenizer": token_ids[name], "config": int(config_value)}
        for name, config_value in config_ids.items()
        if config_value is not None and int(config_value) != token_ids[name]
    }
    if mismatches:
        raise RuntimeError(
            "Latent token IDs disagree between tokenizer and model config: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    latent_size = int(getattr(model.config, "latent_size", 0))
    if latent_size <= 0:
        raise RuntimeError(
            "The checkpoint config has no positive latent_size; this does not "
            "look like a complete ILVR checkpoint."
        )
    return {
        "latent_size": latent_size,
        "latent_token_ids": token_ids,
        "tokenizer_class": type(tokenizer).__name__,
    }


def load_model_and_processor(
    model_path: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    try:
        import transformers
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "ILVR's customized Transformers package is not importable. Install "
            "this repository's fork with `pip install -e ./transformers` and "
            "restart Python."
        ) from exc

    dtype = _torch_dtype_from_name(TORCH_DTYPE)
    cache_dir = CACHE_DIR or None
    processor_kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
    }
    processor = AutoProcessor.from_pretrained(str(model_path), **processor_kwargs)

    model_kwargs = {
        "device_map": DEVICE_MAP,
        "torch_dtype": dtype,
        "cache_dir": cache_dir,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "attn_implementation": ATTN_IMPLEMENTATION,
    }
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), **model_kwargs
        )
    except Exception as exc:
        if not FALLBACK_TO_EAGER_ATTENTION or ATTN_IMPLEMENTATION == "eager":
            raise
        print(
            "[ILVR analysis] attention initialization failed; retrying with "
            f"eager attention: {exc}"
        )
        model_kwargs["attn_implementation"] = "eager"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), **model_kwargs
        )

    model.eval()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    token_ids = discover_latent_token_ids(processor.tokenizer)
    ilvr_details = _validate_custom_ilvr_runtime(
        model, processor.tokenizer, token_ids
    )
    embedding = model.get_input_embeddings()
    vocab_size, hidden_size = map(int, embedding.weight.shape)
    details = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "embedding_dtype": str(embedding.weight.dtype),
        "embedding_device": str(embedding.weight.device),
        "transformers_version": transformers.__version__,
        "transformers_path": str(Path(transformers.__file__).resolve()),
        "attention_implementation": model_kwargs["attn_implementation"],
        **ilvr_details,
    }
    print("[ILVR analysis] model capture configuration:")
    print(json.dumps(details, ensure_ascii=False, indent=2))
    return model, processor, details


def export_vocabulary_embeddings(
    model: Any, temporary_dir: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    weight = model.get_input_embeddings().weight
    vocab_size, hidden_size = map(int, weight.shape)
    path = temporary_dir / "vocabulary_embeddings.npy"
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float16,
        shape=(vocab_size, hidden_size),
    )
    with torch.inference_mode():
        for start in range(0, vocab_size, VOCAB_EMBEDDING_BATCH_SIZE):
            end = min(start + VOCAB_EMBEDDING_BATCH_SIZE, vocab_size)
            output[start:end] = (
                weight[start:end]
                .detach()
                .to(device="cpu", dtype=torch.float16)
                .numpy()
            )
    output.flush()
    del output
    return np.load(path, mmap_mode="r"), {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "dtype": "float16",
        "batch_size": VOCAB_EMBEDDING_BATCH_SIZE,
    }


def _iter_tensors(value: Any) -> Iterable[Any]:
    import torch

    if isinstance(value, torch.Tensor):
        yield value
        return
    preferred_names = ("pooler_output", "last_hidden_state")
    for name in preferred_names:
        tensor = getattr(value, name, None)
        if isinstance(tensor, torch.Tensor):
            yield tensor
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_tensors(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            yield from _iter_tensors(nested)


def _select_feature_tensor(output: Any, hidden_size: int, label: str) -> Any:
    candidates = []
    seen = set()
    for tensor in _iter_tensors(output):
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        if tensor.ndim in (2, 3) and int(tensor.shape[-1]) == hidden_size:
            candidates.append(tensor)
    if not candidates:
        shapes = [tuple(tensor.shape) for tensor in _iter_tensors(output)]
        raise RuntimeError(
            f"{label} returned no tensor with hidden size {hidden_size}. "
            f"Observed tensor shapes: {shapes}"
        )
    for tensor in candidates:
        if tensor.ndim == 2:
            return tensor
    return candidates[0]


def locate_visual_module(model: Any) -> tuple[Any, str]:
    import torch.nn as nn

    candidates = ("visual", "model.visual")
    for candidate in candidates:
        value = model
        for component in candidate.split("."):
            value = getattr(value, component, None)
            if value is None:
                break
        if isinstance(value, nn.Module):
            return value, candidate
    raise RuntimeError(
        "Could not locate Qwen2.5-VL's visual module. Checked: "
        + ", ".join(candidates)
    )


class ImageFeatureCapture:
    """Capture the visual tower output used during one batch-size-one prefill."""

    def __init__(self, visual: Any, hidden_size: int):
        self.visual = visual
        self.hidden_size = hidden_size
        self.outputs: list[np.ndarray] = []
        self._handle: Any | None = None

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        import torch

        features = _select_feature_tensor(output, self.hidden_size, "Visual hook")
        if features.ndim == 3:
            if int(features.shape[0]) != 1:
                raise RuntimeError(
                    "Analysis expects batch size 1, but the visual hook returned "
                    f"shape {tuple(features.shape)}."
                )
            features = features[0]
        self.outputs.append(
            features.detach().to(device="cpu", dtype=torch.float16).numpy()
        )

    def __enter__(self) -> "ImageFeatureCapture":
        self._handle = self.visual.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def injected_features(self, expected_count: int) -> np.ndarray:
        if len(self.outputs) != 1:
            raise RuntimeError(
                "Expected exactly one visual-tower call for one HRBench sample, "
                f"but captured {len(self.outputs)}."
            )
        features = self.outputs[0]
        if len(features) < expected_count:
            raise RuntimeError(
                "Visual feature count is smaller than the prompt image-token "
                f"count: {len(features)} < {expected_count}."
            )
        # ILVR's customized Qwen forward truncates excess merged features to
        # the image-token count for batch size one before masked_scatter.
        return features[:expected_count]


def _move_processor_inputs(inputs: Any, device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def _generation_step_vector(step_output: Any, hidden_size: int) -> np.ndarray:
    import torch

    candidates = [
        tensor
        for tensor in _iter_tensors(step_output)
        if tensor.ndim in (2, 3) and int(tensor.shape[-1]) == hidden_size
    ]
    if not candidates:
        observed = [tuple(tensor.shape) for tensor in _iter_tensors(step_output)]
        raise RuntimeError(
            "A generation hidden-state step has no compatible final hidden "
            f"state. Observed shapes: {observed}"
        )
    # Standard Transformers returns all layers; ILVR returns the final tensor
    # directly. Taking the last compatible tensor handles both layouts.
    hidden = candidates[-1]
    if hidden.ndim == 3:
        if int(hidden.shape[0]) != 1:
            raise RuntimeError(
                "Analysis expects generation batch size 1, got hidden state "
                f"shape {tuple(hidden.shape)}."
            )
        hidden = hidden[0, -1]
    else:
        hidden = hidden[-1]
    return hidden.detach().to(device="cpu", dtype=torch.float16).numpy()


def _latent_indices_within_blocks(
    output_ids: np.ndarray,
    latent_steps: np.ndarray,
    token_ids: dict[str, int],
) -> np.ndarray:
    index_by_step: dict[int, int] = {}
    in_block = False
    latent_index = 0
    for step, token_id_value in enumerate(output_ids.tolist()):
        token_id = int(token_id_value)
        if token_id == token_ids["start"]:
            in_block = True
            latent_index = 0
        elif token_id == token_ids["end"]:
            in_block = False
            latent_index = 0
        elif token_id == token_ids["pad"]:
            index_by_step[step] = latent_index
            latent_index += 1
            # Retain an index even for a malformed out-of-block pad, while the
            # result metadata still makes the generated token stream inspectable.
            if not in_block:
                latent_index = 0
    return np.asarray(
        [index_by_step[int(step)] for step in latent_steps], dtype=np.int32
    )


def run_inference(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    selected_indices: list[int],
    dataset_dir: Path,
    capture_dir: Path,
    model_details: dict[str, Any],
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    import torch
    from transformers import LogitsProcessorList, StoppingCriteriaList

    project_src = Path(__file__).resolve().parent / "src"
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from utils_deepseed import LatentTemplateLogitsProcessor

    hidden_size = int(model_details["hidden_size"])
    latent_size = int(model_details["latent_size"])
    token_ids = {key: int(value) for key, value in model_details["latent_token_ids"].items()}
    image_token_id = int(model.config.image_token_id)
    input_device = model.get_input_embeddings().weight.device
    visual, visual_name = locate_visual_module(model)
    model_details["visual_module"] = visual_name
    logits_processor = LogitsProcessorList(
        [
            LatentTemplateLogitsProcessor(
                token_ids["start"],
                token_ids["end"],
                token_ids["pad"],
                latent_size,
            )
        ]
    )

    capture_paths: list[Path] = []
    sample_records: list[dict[str, Any]] = []
    per_sample_image_counts: list[int] = []
    per_sample_latent_counts: list[int] = []

    for sample_ordinal, (row, dataset_ordinal) in enumerate(
        zip(rows, selected_indices)
    ):
        deadline = SampleDeadline(SAMPLE_TIMEOUT_SECONDS)
        image = None
        inputs = prompt_ids = generated = full_ids = hidden_steps = None
        image_capture = image_vectors = latent_vectors = output_ids = None
        request_id = f"hf-{sample_ordinal:06d}"
        capture_path = capture_dir / f"capture_{sample_ordinal:06d}.npz"
        try:
            image = decode_hrbench_image(row["image"], dataset_dir)
            deadline.check()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": build_question(row)},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = _move_processor_inputs(inputs, input_device)
            prompt_ids = inputs["input_ids"][0]
            prompt_length = int(prompt_ids.shape[0])
            image_positions = (
                (prompt_ids == image_token_id)
                .nonzero(as_tuple=True)[0]
                .to(device="cpu", dtype=torch.int64)
                .numpy()
                .astype(np.int32, copy=False)
            )
            if not len(image_positions):
                raise RuntimeError(
                    f"Sample {sample_ordinal} contains no image tokens after processing."
                )

            deadline.check()
            with (
                torch.inference_mode(),
                ImageFeatureCapture(visual, hidden_size) as image_capture,
            ):
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    stopping_criteria=StoppingCriteriaList([deadline]),
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=TEMPERATURE > 0,
                    use_cache=USE_CACHE,
                    logits_processor=logits_processor,
                    tokenizer=processor.tokenizer,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )

            deadline.check()
            full_ids = generated.sequences[0]
            output_ids = (
                full_ids[prompt_length:]
                .detach()
                .to(device="cpu", dtype=torch.int64)
                .numpy()
            )
            hidden_steps = generated.hidden_states
            if hidden_steps is None:
                raise RuntimeError(
                    "generate() returned no hidden_states. The customized ILVR "
                    "generation implementation is required."
                )
            consumed_count = min(len(hidden_steps), len(output_ids))
            latent_steps = np.flatnonzero(
                output_ids[:consumed_count] == token_ids["pad"]
            ).astype(np.int32, copy=False)
            if len(latent_steps):
                latent_vectors = np.stack(
                    [
                        _generation_step_vector(hidden_steps[int(step)], hidden_size)
                        for step in latent_steps
                    ],
                    axis=0,
                ).astype(np.float16, copy=False)
            else:
                latent_vectors = np.empty((0, hidden_size), dtype=np.float16)
            latent_indices = _latent_indices_within_blocks(
                output_ids[:consumed_count], latent_steps, token_ids
            )
            latent_sequence_positions = (
                latent_steps + prompt_length
            ).astype(np.int32, copy=False)
            image_vectors = image_capture.injected_features(len(image_positions))

            deadline.check()
            np.savez(
                capture_path,
                image_vectors=image_vectors.astype(np.float16, copy=False),
                image_sequence_positions=image_positions,
                image_generation_steps=np.full(
                    len(image_vectors), -1, dtype=np.int32
                ),
                latent_vectors=latent_vectors,
                latent_sequence_positions=latent_sequence_positions,
                latent_generation_steps=latent_steps,
                latent_indices=latent_indices,
            )
            decoded = processor.batch_decode(
                [output_ids.tolist()],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
            sample_records.append(
                {
                    "status": "ok",
                    "elapsed_seconds": deadline.elapsed,
                    "sample_ordinal": sample_ordinal,
                    "dataset_ordinal": int(dataset_ordinal),
                    "dataset_index": _json_compatible(row["index"]),
                    "question": _json_compatible(row["question"]),
                    "answer": _json_compatible(row["answer"]),
                    "category": _json_compatible(row["category"]),
                    "cycle_category": _json_compatible(row["cycle_category"]),
                    "choices": {
                        letter: _json_compatible(row[letter]) for letter in "ABCD"
                    },
                    "request_id": request_id,
                    "raw_output_text": decoded,
                    "output_token_ids": [int(value) for value in output_ids],
                    "prompt_token_count": prompt_length,
                    "consumed_output_token_count": consumed_count,
                    "unconsumed_output_token_ids": [
                        int(value) for value in output_ids[consumed_count:]
                    ],
                    "capture_counts": {
                        "image_feature": int(len(image_vectors)),
                        "latent": int(len(latent_vectors)),
                    },
                    "latent_generation_steps": latent_steps.tolist(),
                    "latent_indices": latent_indices.tolist(),
                }
            )
            capture_paths.append(capture_path)
            per_sample_image_counts.append(int(len(image_vectors)))
            per_sample_latent_counts.append(int(len(latent_vectors)))
            append_sample_progress(capture_dir, sample_records[-1])
            print(
                f"[{sample_ordinal + 1}/{len(rows)}] "
                f"dataset={dataset_ordinal}, image={len(image_vectors)}, "
                f"latent={len(latent_vectors)}, output={len(output_ids)}"
            )
        except SampleCaptureTimeout as exc:
            write_empty_capture(capture_path, hidden_size, request_id)
            capture_paths.append(capture_path)
            per_sample_image_counts.append(0)
            per_sample_latent_counts.append(0)
            record = skipped_sample_record(row, sample_ordinal, dataset_ordinal, request_id, str(exc), deadline.elapsed)
            sample_records.append(record)
            append_sample_progress(capture_dir, record)
            print(f"[Timeout: skipped] sample={sample_ordinal}, dataset={dataset_ordinal}: {exc}", flush=True)
            check_timeout_budget(sample_records, MAX_SAMPLE_TIMEOUTS)
        finally:
            if image is not None:
                image.close()
            inputs = prompt_ids = generated = full_ids = hidden_steps = None
            image_capture = image_vectors = latent_vectors = output_ids = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not any(record.get("status") == "ok" for record in sample_records):
        raise RuntimeError("No samples completed successfully; refusing a vocabulary-only analysis.")

    return (
        capture_paths,
        sample_records,
        {
            "per_sample_image_counts": per_sample_image_counts,
            "per_sample_latent_counts": per_sample_latent_counts,
            "timed_out_samples": sum(r.get("status") == "skipped_timeout" for r in sample_records),
        },
    )


def sample_image_positions(available_count: int, target_count: int) -> tuple[np.ndarray, bool]:
    """Sample without replacement; if fewer image rows exist, keep all."""
    return sample_indices(available_count, target_count, RANDOM_SEED), False


def extract_sampled_images_and_latents(
    capture_paths: list[Path],
    target_image_count: int,
    hidden_size: int,
    temporary_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    image_counts = np.zeros(len(capture_paths), dtype=np.int64)
    latent_counts = np.zeros(len(capture_paths), dtype=np.int64)
    for sample_ordinal, path in enumerate(capture_paths):
        with np.load(path, allow_pickle=False) as data:
            image_counts[sample_ordinal] = len(data["image_vectors"])
            latent_counts[sample_ordinal] = len(data["latent_vectors"])

    total_images = int(image_counts.sum())
    total_latents = int(latent_counts.sum())
    chosen_images, used_replacement = sample_image_positions(
        total_images, target_image_count
    )
    target_image_count = len(chosen_images)
    image_path = temporary_dir / "sampled_image_embeddings.npy"
    image_vectors = np.lib.format.open_memmap(
        image_path,
        mode="w+",
        dtype=np.float16,
        shape=(target_image_count, hidden_size),
    )
    latent_vectors = np.empty((total_latents, hidden_size), dtype=np.float16)
    image_sample_ordinals = np.empty(target_image_count, dtype=np.int32)
    image_sequence_positions = np.empty(target_image_count, dtype=np.int32)
    image_generation_steps = np.empty(target_image_count, dtype=np.int32)
    latent_sample_ordinals = np.empty(total_latents, dtype=np.int32)
    latent_sequence_positions = np.empty(total_latents, dtype=np.int32)
    latent_generation_steps = np.empty(total_latents, dtype=np.int32)
    latent_indices = np.empty(total_latents, dtype=np.int32)
    latent_trajectory_steps = np.empty(total_latents, dtype=np.int32)

    image_global_start = 0
    latent_output_start = 0
    for sample_ordinal, path in enumerate(capture_paths):
        with np.load(path, allow_pickle=False) as data:
            captured_images = data["image_vectors"]
            captured_latents = data["latent_vectors"]
            if captured_images.ndim != 2 or captured_images.shape[1] != hidden_size:
                raise RuntimeError(
                    f"Image hidden-size mismatch in {path}: {captured_images.shape}."
                )
            if captured_latents.ndim != 2 or captured_latents.shape[1] != hidden_size:
                raise RuntimeError(
                    f"Latent hidden-size mismatch in {path}: {captured_latents.shape}."
                )

            image_global_end = image_global_start + len(captured_images)
            chosen_start = np.searchsorted(
                chosen_images, image_global_start, side="left"
            )
            chosen_end = np.searchsorted(
                chosen_images, image_global_end, side="left"
            )
            if chosen_end > chosen_start:
                local_positions = (
                    chosen_images[chosen_start:chosen_end] - image_global_start
                )
                output_slice = slice(chosen_start, chosen_end)
                image_vectors[output_slice] = captured_images[local_positions]
                image_sample_ordinals[output_slice] = sample_ordinal
                image_sequence_positions[output_slice] = data[
                    "image_sequence_positions"
                ][local_positions]
                image_generation_steps[output_slice] = data[
                    "image_generation_steps"
                ][local_positions]
            image_global_start = image_global_end

            latent_output_end = latent_output_start + len(captured_latents)
            if latent_output_end > latent_output_start:
                output_slice = slice(latent_output_start, latent_output_end)
                latent_vectors[output_slice] = captured_latents
                latent_sample_ordinals[output_slice] = sample_ordinal
                latent_sequence_positions[output_slice] = data[
                    "latent_sequence_positions"
                ]
                latent_generation_steps[output_slice] = data[
                    "latent_generation_steps"
                ]
                latent_indices[output_slice] = data["latent_indices"]
                latent_trajectory_steps[output_slice] = np.arange(
                    len(captured_latents), dtype=np.int32
                )
            latent_output_start = latent_output_end

    image_vectors.flush()
    latent_offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(latent_counts, dtype=np.int64),
        )
    )
    metadata = {
        "image_global_positions": chosen_images,
        "image_sample_ordinal": image_sample_ordinals,
        "image_sequence_positions": image_sequence_positions,
        "image_generation_steps": image_generation_steps,
        "latent_sample_ordinal": latent_sample_ordinals,
        "latent_sequence_positions": latent_sequence_positions,
        "latent_generation_steps": latent_generation_steps,
        "latent_indices": latent_indices,
        "latent_trajectory_steps": latent_trajectory_steps,
        "latent_sample_offsets": latent_offsets,
    }
    statistics = {
        "available_image_features": total_images,
        "sampled_image_features": target_image_count,
        "image_sampling_with_replacement": used_replacement,
        "latent_vectors": total_latents,
        "per_sample_image_counts": image_counts.tolist(),
        "per_sample_latent_counts": latent_counts.tolist(),
    }
    return image_vectors, latent_vectors, metadata, statistics


def global_config_snapshot() -> dict[str, Any]:
    return {name: value for name, value in globals().items()
            if name.isupper() and isinstance(value, (str, int, float, bool, type(None), list, dict, tuple))}


def tsne_options() -> dict[str, Any]:
    return {"pca_components": PCA_COMPONENTS, "perplexity": TSNE_PERPLEXITY,
            "max_iter": TSNE_MAX_ITER, "random_state": TSNE_RANDOM_SEED,
            "n_jobs": TSNE_N_JOBS}


def main() -> None:
    options = tsne_options()
    validate_settings(RUN_MODE, VOCAB_SAMPLE_SIZE, IMAGE_SAMPLE_SIZE, options, TSNE_RUN_NAME)
    output_path = Path(OUTPUT_DIR).expanduser().resolve()
    projection_path = output_path / "projections" / TSNE_RUN_NAME
    if RUN_MODE == "tsne_only":
        feature_dir = Path(FEATURE_CACHE_DIR).expanduser() if FEATURE_CACHE_DIR else output_path / "features"
        run_tsne(feature_dir, projection_path, options, expected_model='ILVR')
        return
    validate_timeout_settings(SAMPLE_TIMEOUT_SECONDS, MAX_SAMPLE_TIMEOUTS)
    model_path, dataset_path, output_path = validate_configuration()
    prepare_capture_directory(output_path)
    rows, selected_indices = load_hrbench_rows(dataset_path)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=".ilvr_tsne_capture_", dir=output_path)
    )
    succeeded = False
    try:
        model, processor, model_details = load_model_and_processor(model_path)
        vocabulary_vectors, vocabulary_export = export_vocabulary_embeddings(
            model, temporary_dir
        )
        capture_paths, sample_records, inference_statistics = run_inference(
            model,
            processor,
            rows,
            selected_indices,
            dataset_path.parent,
            temporary_dir,
            model_details,
        )
        image_vectors, latent_vectors, metadata, capture_statistics = (
            extract_sampled_images_and_latents(
                capture_paths,
                target_image_count=IMAGE_SAMPLE_SIZE,
                hidden_size=int(vocabulary_vectors.shape[1]),
                temporary_dir=temporary_dir,
            )
        )
        token_ids = sample_indices(len(vocabulary_vectors), VOCAB_SAMPLE_SIZE, RANDOM_SEED)
        selected_vocabulary = vocabulary_vectors[token_ids]
        metadata["vocabulary_token_ids"] = token_ids
        capture_statistics["available_vocabulary_embeddings"] = len(vocabulary_vectors)
        capture_statistics["sampled_vocabulary_embeddings"] = len(token_ids)
        capture_config = global_config_snapshot()
        capture_config["capture_details"] = {"model_capture": model_details, "inference_statistics": inference_statistics}
        save_features(
            output_path, [selected_vocabulary, image_vectors, latent_vectors],
            metadata, sample_records, capture_config, capture_statistics,
            model_name='ILVR', semantics='Vocabulary input embeddings; visual features; final decoder states producing generated <|latent_pad|> tokens and fed back as their input embeddings. Multiple latent blocks remain one ordered trajectory.',
        )
        del vocabulary_vectors, selected_vocabulary, image_vectors, latent_vectors
        del model, processor
        gc.collect()
        succeeded = True
    finally:
        # Only expendable captures are cleaned; features/ is never removed.
        if succeeded or not KEEP_TEMP_CAPTURE_ON_ERROR:
            shutil.rmtree(temporary_dir, ignore_errors=True)
        elif temporary_dir.exists():
            print(f"Temporary captures kept for debugging: {temporary_dir}")
        gc.collect()

    if RUN_MODE == "capture_and_tsne":
        run_tsne(output_path / "features", projection_path, options, expected_model='ILVR')
    else:
        print(f"Capture complete. Original features: {output_path / 'features'}")


if __name__ == "__main__":
    main()
