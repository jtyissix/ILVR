"""Run ILVR on HRBench and export latent/answer attention distributions.

Edit the global variables below and run this file directly.  There is no
command-line interface.  The analysis uses this repository's customized
Transformers generation loop and never requests full-sequence attention
outputs from ``generate``.
"""

from __future__ import annotations

import base64
import csv
import gc
import inspect
import io
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,6,7"

# =============================================================================
# Global configuration -- edit values here; no CLI arguments are used
# =============================================================================

MODEL_PATH = "/home/fit/renjujty/jty/lmllms/ilvr/"
HRBENCH_PATH = "/home/fit/renjujty/jty/lmllms/hrbench/hr_bench_4k.parquet"
OUTPUT_DIR = "outputs/hrbench_attention"
CACHE_DIR: str | None = None

RESULTS_FILE = "results.jsonl"
RUN_CONFIG_FILE = "run_config.json"
CATEGORY_ATTENTION_CSV_FILE = "category_attention.csv"
LATENT_TOPK_CSV_FILE = "latent_topk.csv"
ATTENTION_SUBDIR = "attention"
PLOT_SUBDIR = "plots"

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

LATENT_TOP_K = 20
ATTENTION_STORAGE_DTYPE = "float16"
PLOT_LAYER = -1
PLOT_DPI = 180
PLOT_MAX_TOKEN_LABELS = 80
PLOT_FIGURE_WIDTH = 18.0
PLOT_ROW_HEIGHT = 0.38
WRITE_PLOTS = True
KEEP_TEMP_CAPTURE_ON_ERROR = False


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

SOURCE_KIND_NAMES = np.asarray(
    [
        "input_text",
        "input_visual",
        "latent",
        "cot_text",
        "answer_history",
        "special",
    ],
    dtype=np.str_,
)
SOURCE_INPUT_TEXT = 0
SOURCE_INPUT_VISUAL = 1
SOURCE_LATENT = 2
SOURCE_COT_TEXT = 3
SOURCE_ANSWER_HISTORY = 4
SOURCE_SPECIAL = 5

TARGET_KIND_NAMES = np.asarray(
    [
        "input_text",
        "input_visual",
        "latent",
        "cot_text",
    ],
    dtype=np.str_,
)
TARGET_SOURCE_CODES = (
    SOURCE_INPUT_TEXT,
    SOURCE_INPUT_VISUAL,
    SOURCE_LATENT,
    SOURCE_COT_TEXT,
)

QUERY_KIND_NAMES = np.asarray(["latent", "answer"], dtype=np.str_)
QUERY_LATENT = 0
QUERY_ANSWER = 1


def select_sample_indices(
    total: int,
    mode: str = SELECTION_MODE,
    start_index: int = START_INDEX,
    count: int = NUM_SAMPLES,
    seed: int = RANDOM_SEED,
) -> list[int]:
    if total <= 0:
        raise ValueError("HRBench is empty.")
    if count <= 0 or count > total:
        raise ValueError(f"NUM_SAMPLES must be in [1, {total}], received {count}.")
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
    if MAX_NEW_TOKENS <= 0 or LATENT_TOP_K <= 0:
        raise ValueError("MAX_NEW_TOKENS and LATENT_TOP_K must be positive.")
    if TEMPERATURE < 0 or not 0 < TOP_P <= 1:
        raise ValueError("TEMPERATURE must be non-negative and TOP_P in (0, 1].")
    if ATTENTION_STORAGE_DTYPE not in {"float16", "float32"}:
        raise ValueError("ATTENTION_STORAGE_DTYPE must be float16 or float32.")

    model_path = Path(MODEL_PATH).expanduser()
    dataset_path = Path(HRBENCH_PATH).expanduser()
    output_path = Path(OUTPUT_DIR).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"MODEL_PATH is not a directory: {model_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"HRBENCH_PATH is not a file: {dataset_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / ATTENTION_SUBDIR).mkdir(exist_ok=True)
    (output_path / PLOT_SUBDIR).mkdir(exist_ok=True)
    return model_path.resolve(), dataset_path.resolve(), output_path.resolve()


def load_hrbench_rows(dataset_path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=str(dataset_path), split="train")
    missing = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise ValueError(
            "Unexpected HRBench schema; missing columns: " + ", ".join(sorted(missing))
        )
    selected = select_sample_indices(len(dataset))
    return [dict(dataset[index]) for index in selected], selected


def _open_image_path(path_value: str, dataset_dir: Path) -> Image.Image | None:
    if not path_value or "\x00" in path_value:
        return None
    try:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = dataset_dir / path
        if not path.is_file():
            return None
    except (OSError, RuntimeError):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to open image path: {path}") from exc


def _open_image_bytes(value: bytes, description: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(value)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to decode {description} as an image.") from exc


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
            raise ValueError("Malformed image data URI.")
        value = value.split(",", 1)[1]
    elif len(value) <= MAX_PATH_CANDIDATE_LENGTH:
        image = _open_image_path(value, dataset_dir)
        if image is not None:
            return image
    try:
        decoded = base64.b64decode(value, validate=False)
    except Exception as exc:
        raise ValueError("The HRBench image contains invalid base64 data.") from exc
    return _open_image_bytes(decoded, "HRBench base64 data")


def build_question(row: dict[str, Any]) -> str:
    choices = "\n".join(f"({letter}) {row[letter]}" for letter in "ABCD")
    return (
        f"Question: {row['question']} The choices are listed below:\n"
        f"{choices}\nPut your final answer in \\boxed{{}}."
    )


def clean_latent_output(text: str) -> str:
    pattern = re.compile(
        re.escape(LATENT_TOKEN_NAMES["start"])
        + r".*?"
        + re.escape(LATENT_TOKEN_NAMES["end"]),
        flags=re.DOTALL,
    )
    replacement = LATENT_TOKEN_NAMES["start"] + "<latent>" + LATENT_TOKEN_NAMES["end"]
    return pattern.sub(replacement, text)


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
        raise ValueError("Unsupported TORCH_DTYPE: " + name) from exc


def _single_token_id(tokenizer: Any, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"{token!r} must encode to one token, received {ids}.")
    return int(ids[0])


def discover_latent_token_ids(tokenizer: Any) -> dict[str, int]:
    return {
        name: _single_token_id(tokenizer, token)
        for name, token in LATENT_TOKEN_NAMES.items()
    }


def load_model_and_processor(model_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        cache_dir=CACHE_DIR or None,
        trust_remote_code=TRUST_REMOTE_CODE,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    kwargs = {
        "device_map": DEVICE_MAP,
        "torch_dtype": _torch_dtype_from_name(TORCH_DTYPE),
        "cache_dir": CACHE_DIR or None,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "attn_implementation": ATTN_IMPLEMENTATION,
    }
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), **kwargs
        )
    except Exception as exc:
        if not FALLBACK_TO_EAGER_ATTENTION or ATTN_IMPLEMENTATION == "eager":
            raise
        print(f"[ILVR attention] FlashAttention unavailable; using eager: {exc}")
        kwargs["attn_implementation"] = "eager"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), **kwargs
        )
    model.eval()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    parameters = inspect.signature(model.forward).parameters
    missing = {"generate_mode", "latent_hidden_states"}.difference(parameters)
    if missing:
        raise RuntimeError(
            "The imported Transformers is not ILVR's customized fork; missing "
            f"forward parameters: {sorted(missing)}. Install ./transformers."
        )
    token_ids = discover_latent_token_ids(processor.tokenizer)
    generation_token_ids = {"pad": 151665, "start": 151666, "end": 151667}
    if token_ids != generation_token_ids:
        raise RuntimeError(
            "Checkpoint latent token IDs do not match ILVR's customized "
            f"generation loop: {token_ids} != {generation_token_ids}."
        )
    latent_size = int(getattr(model.config, "latent_size", 0))
    if latent_size <= 0:
        raise RuntimeError("Checkpoint config has no positive latent_size.")
    config_ids = {
        "pad": getattr(model.config, "latent_token_id", None),
        "start": getattr(model.config, "latent_start_id", None),
        "end": getattr(model.config, "latent_end_id", None),
    }
    for name, config_id in config_ids.items():
        if config_id is not None and int(config_id) != token_ids[name]:
            raise RuntimeError(
                f"Tokenizer/config latent {name} IDs differ: "
                f"{token_ids[name]} != {config_id}."
            )
    details = {
        "transformers_version": transformers.__version__,
        "transformers_path": str(Path(transformers.__file__).resolve()),
        "attention_implementation": kwargs["attn_implementation"],
        "latent_size": latent_size,
        "latent_token_ids": token_ids,
        "num_hidden_layers": int(model.config.num_hidden_layers),
    }
    print(json.dumps(details, ensure_ascii=False, indent=2))
    return model, processor, details


class ILVRAttentionRecorder:
    """Duck-typed recorder consumed by the customized generation/attention code."""

    def __init__(
        self,
        capture_dir: Path,
        prompt_token_ids: list[int],
        image_positions: list[int],
        special_token_ids: set[int],
        latent_token_ids: dict[str, int],
        layer_count: int,
        top_k: int = LATENT_TOP_K,
        storage_dtype: str = ATTENTION_STORAGE_DTYPE,
    ) -> None:
        self.capture_dir = capture_dir
        self.prompt_token_ids = list(map(int, prompt_token_ids))
        self.prompt_length = len(self.prompt_token_ids)
        self.image_positions = list(map(int, image_positions))
        self.special_token_ids = set(map(int, special_token_ids))
        self.latent_token_ids = {
            key: int(value) for key, value in latent_token_ids.items()
        }
        self.special_token_ids.update(self.latent_token_ids.values())
        self.layer_count = int(layer_count)
        self.top_k = int(top_k)
        self.storage_dtype = np.dtype(storage_dtype)
        self.latent_path = capture_dir / "latent.bin"
        self.answer_path = capture_dir / "answer.bin"
        self.latent_path.write_bytes(b"")
        self.answer_path.write_bytes(b"")
        self.generated_token_ids: list[int] = []
        self.latent_records: list[dict[str, Any]] = []
        self.answer_records: list[dict[str, Any]] = []
        self.latent_topk: list[dict[str, Any]] = []
        self.saw_latent = False
        self.in_latent = False
        self.latent_block_index = -1
        self.next_latent_index = 0
        self._current: dict[str, Any] | None = None

    def begin_generation(
        self, model: Any, prompt_input_ids: Any, latent_size: int
    ) -> None:
        del model, latent_size
        if int(prompt_input_ids.shape[0]) != 1:
            raise RuntimeError("Attention analysis requires generation batch size 1.")
        observed = prompt_input_ids[0].detach().to(device="cpu").tolist()
        if list(map(int, observed)) != self.prompt_token_ids:
            raise RuntimeError("Recorder prompt IDs differ from generation input IDs.")

    def begin_step(
        self,
        model: Any,
        query_sequence_position: int,
        output_index: int,
        is_latent_query: bool,
        latent_vector: Any,
    ) -> None:
        if self._current is not None:
            raise RuntimeError("Previous ILVR attention step was not committed.")
        capture = bool(is_latent_query or not self.in_latent)
        self._current = {
            "query_sequence_position": int(query_sequence_position),
            "output_index": int(output_index),
            "is_latent_query": bool(is_latent_query),
            "capture": capture,
            "layers": {},
            "latent_block_index": self.latent_block_index,
            "latent_index": self.next_latent_index if is_latent_query else -1,
        }
        if is_latent_query:
            if latent_vector is None:
                raise RuntimeError("A latent query has no injected latent vector.")
            self._capture_latent_topk(model, latent_vector)

    def wants_current_step(self) -> bool:
        return bool(self._current is not None and self._current["capture"])

    def record_layer_attention(self, layer_index: int, attention: Any) -> None:
        import torch

        if not self.wants_current_step():
            return
        assert self._current is not None
        if layer_index in self._current["layers"]:
            raise RuntimeError(f"Layer {layer_index} recorded twice in one step.")
        if not isinstance(attention, torch.Tensor) or attention.ndim != 2:
            raise RuntimeError("Recorder expects [batch, source] attention.")
        if int(attention.shape[0]) != 1:
            raise RuntimeError("Recorder supports attention batch size 1 only.")
        expected = int(self._current["query_sequence_position"]) + 1
        if int(attention.shape[1]) != expected:
            raise RuntimeError(
                f"Attention source length {attention.shape[1]} != {expected}."
            )
        self._current["layers"][int(layer_index)] = (
            attention[0]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
            .astype(self.storage_dtype, copy=False)
        )

    def _capture_latent_topk(self, model: Any, latent_vector: Any) -> None:
        import torch

        assert self._current is not None
        vector = latent_vector[:, -1, :]
        output_head = model.get_output_embeddings()
        if output_head is None:
            raise RuntimeError("Model has no output embedding head.")
        try:
            output_device = next(output_head.parameters()).device
        except StopIteration:
            output_device = vector.device
        with torch.inference_mode():
            logits = output_head(vector.to(output_device)).float()
            k = min(self.top_k, int(logits.shape[-1]))
            values, token_ids = torch.topk(logits[0], k=k)
        self._current["topk"] = {
            "query_sequence_position": int(self._current["query_sequence_position"]),
            "latent_block_index": int(self._current["latent_block_index"]),
            "latent_index": int(self._current["latent_index"]),
            "token_ids": token_ids.detach().to(device="cpu").tolist(),
            "logits": values.detach().to(device="cpu").tolist(),
        }

    def _append_record(
        self, stream: str, current: dict[str, Any], **extra: Any
    ) -> None:
        layers = current["layers"]
        if sorted(layers) != list(range(self.layer_count)):
            raise RuntimeError(
                "Not every decoder layer recorded attention: "
                f"{sorted(layers)} vs 0..{self.layer_count - 1}."
            )
        matrix = np.stack([layers[index] for index in range(self.layer_count)])
        path = self.latent_path if stream == "latent" else self.answer_path
        offset = path.stat().st_size // self.storage_dtype.itemsize
        with path.open("ab") as handle:
            matrix.tofile(handle)
        record = {
            "query_sequence_position": int(current["query_sequence_position"]),
            "source_count": int(matrix.shape[1]),
            "layer_count": int(matrix.shape[0]),
            "offset": int(offset),
            **extra,
        }
        if stream == "latent":
            self.latent_records.append(record)
        else:
            self.answer_records.append(record)

    def end_step(self, predicted_token_ids: Any) -> None:
        if self._current is None:
            raise RuntimeError("No active recorder step to commit.")
        ids = predicted_token_ids.detach().to(device="cpu").reshape(-1).tolist()
        if len(ids) != 1:
            raise RuntimeError("Attention analysis requires batch size 1.")
        token_id = int(ids[0])
        current = self._current

        if current["is_latent_query"]:
            self._append_record(
                "latent",
                current,
                latent_block_index=int(current["latent_block_index"]),
                latent_index=int(current["latent_index"]),
            )
            self.latent_topk.append(current["topk"])
            self.next_latent_index += 1

        output_index = len(self.generated_token_ids)
        self.generated_token_ids.append(token_id)
        if token_id == self.latent_token_ids["start"]:
            self.saw_latent = True
            self.in_latent = True
            self.latent_block_index += 1
            self.next_latent_index = 0
            self.answer_records.clear()
            self.answer_path.write_bytes(b"")
        elif token_id == self.latent_token_ids["end"]:
            self.in_latent = False
        elif (
            not self.in_latent
            and token_id not in self.special_token_ids
            and not current["is_latent_query"]
        ):
            self._append_record(
                "answer",
                current,
                output_index=output_index,
                predicted_token_id=token_id,
            )
        self._current = None

    def end_generation(self, input_ids: Any) -> None:
        if self._current is not None:
            raise RuntimeError("Generation ended with an uncommitted recorder step.")
        observed = input_ids[0, self.prompt_length :].detach().to(device="cpu").tolist()
        if list(map(int, observed)) != self.generated_token_ids:
            raise RuntimeError("Recorder output IDs differ from generated IDs.")
        if len(self.latent_topk) != len(self.latent_records):
            raise RuntimeError("Every latent query must have one top-k record.")

    def manifest(self) -> dict[str, Any]:
        return {
            "storage_dtype": self.storage_dtype.name,
            "layer_count": self.layer_count,
            "prompt_length": self.prompt_length,
            "prompt_token_ids": self.prompt_token_ids,
            "generated_token_ids": self.generated_token_ids,
            "image_positions": self.image_positions,
            "special_token_ids": sorted(self.special_token_ids),
            "latent_records": self.latent_records,
            "answer_records": self.answer_records,
            "latent_topk": self.latent_topk,
            "latent_spool": self.latent_path.name,
            "answer_spool": self.answer_path.name,
            "no_latent_fallback": not self.saw_latent,
        }


@contextmanager
def install_attention_recorder(model: Any, recorder: ILVRAttentionRecorder):
    if hasattr(model, "_ilvr_attention_recorder"):
        raise RuntimeError("Model already has an ILVR attention recorder.")
    attention_modules = [
        module
        for module in model.modules()
        if callable(getattr(module, "_record_ilvr_analysis_attention", None))
    ]
    if len(attention_modules) != recorder.layer_count:
        raise RuntimeError(
            f"Found {len(attention_modules)} analysis-capable attention modules; "
            f"expected {recorder.layer_count}."
        )
    setattr(model, "_ilvr_attention_recorder", recorder)
    for module in attention_modules:
        setattr(module, "_ilvr_attention_recorder", recorder)
    try:
        yield
    finally:
        if getattr(model, "_ilvr_attention_recorder", None) is recorder:
            delattr(model, "_ilvr_attention_recorder")
        for module in attention_modules:
            if getattr(module, "_ilvr_attention_recorder", None) is recorder:
                delattr(module, "_ilvr_attention_recorder")


def generated_source_kinds(
    generated_ids: list[int],
    latent_ids: dict[str, int],
    special_ids: set[int],
    latent_positions: set[int],
    prompt_length: int,
) -> np.ndarray:
    """Classify generated positions after the complete sequence is known."""
    final_end = max(
        (
            index
            for index, token in enumerate(generated_ids)
            if token == latent_ids["end"]
        ),
        default=-1,
    )
    saw_latent = any(token == latent_ids["start"] for token in generated_ids)
    kinds = np.full(len(generated_ids), SOURCE_ANSWER_HISTORY, dtype=np.uint8)
    in_latent = False
    for index, token_id in enumerate(generated_ids):
        sequence_position = prompt_length + index
        if sequence_position in latent_positions:
            kinds[index] = SOURCE_LATENT
        elif token_id == latent_ids["start"]:
            in_latent = True
            kinds[index] = SOURCE_SPECIAL
        elif token_id == latent_ids["end"]:
            in_latent = False
            kinds[index] = SOURCE_SPECIAL
        elif in_latent or token_id in special_ids:
            kinds[index] = SOURCE_SPECIAL
        elif saw_latent and (final_end < 0 or index < final_end):
            kinds[index] = SOURCE_COT_TEXT
        else:
            kinds[index] = SOURCE_ANSWER_HISTORY
    return kinds


def validate_complete_latent_blocks(
    manifest: dict[str, Any], latent_ids: dict[str, int], latent_size: int
) -> None:
    counts: dict[int, int] = {}
    for record in manifest["latent_records"]:
        block = int(record["latent_block_index"])
        counts[block] = counts.get(block, 0) + 1
    block = -1
    in_latent = False
    completed: list[int] = []
    for token_id in map(int, manifest["generated_token_ids"]):
        if token_id == latent_ids["start"]:
            block += 1
            in_latent = True
        elif token_id == latent_ids["end"] and in_latent:
            completed.append(block)
            in_latent = False
    for block_index in completed:
        observed = counts.get(block_index, 0)
        if observed != latent_size:
            raise RuntimeError(
                f"Complete latent block {block_index} has {observed} recorded "
                f"latent inputs; expected {latent_size}."
            )


def classify_source_positions(
    source_positions: np.ndarray,
    manifest: dict[str, Any],
    latent_ids: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    prompt_ids = list(map(int, manifest["prompt_token_ids"]))
    generated_ids = list(map(int, manifest["generated_token_ids"]))
    prompt_length = int(manifest["prompt_length"])
    image_positions = set(map(int, manifest["image_positions"]))
    special_ids = set(map(int, manifest["special_token_ids"]))
    latent_positions = {
        int(record["query_sequence_position"]) for record in manifest["latent_records"]
    }
    generated_kinds = generated_source_kinds(
        generated_ids, latent_ids, special_ids, latent_positions, prompt_length
    )

    kinds = np.empty(len(source_positions), dtype=np.uint8)
    token_ids = np.full(len(source_positions), -1, dtype=np.int32)
    for ordinal, position_value in enumerate(source_positions):
        position = int(position_value)
        if position < prompt_length:
            token_id = prompt_ids[position]
            if position in image_positions:
                kind = SOURCE_INPUT_VISUAL
            elif token_id in special_ids:
                kind = SOURCE_SPECIAL
            else:
                kind = SOURCE_INPUT_TEXT
        else:
            output_index = position - prompt_length
            token_id = generated_ids[output_index]
            kind = int(generated_kinds[output_index])
        kinds[ordinal] = kind
        token_ids[ordinal] = token_id
    return kinds, token_ids


def normalize_attention_groups(
    raw: np.ndarray,
    source_kinds: np.ndarray,
    target_codes: tuple[int, ...],
) -> np.ndarray:
    normalized = np.zeros_like(raw, dtype=np.float32)
    mask = np.isin(source_kinds, target_codes)
    if not mask.any():
        return normalized
    denominator = raw[:, mask].sum(axis=1, keepdims=True, dtype=np.float32)
    valid = denominator[:, 0] > 0
    normalized[np.ix_(valid, mask)] = (
        raw[np.ix_(valid, mask)].astype(np.float32) / denominator[valid]
    )
    return normalized


def _read_record_matrix(
    spool: np.memmap, record: dict[str, Any], dtype: np.dtype
) -> np.ndarray:
    count = int(record["layer_count"]) * int(record["source_count"])
    offset = int(record["offset"])
    return np.asarray(spool[offset : offset + count]).reshape(
        int(record["layer_count"]), int(record["source_count"])
    )


def assemble_sample_archive(
    manifest: dict[str, Any],
    capture_dir: Path,
    latent_ids: dict[str, int],
) -> dict[str, np.ndarray]:
    dtype = np.dtype(manifest["storage_dtype"])
    layer_count = int(manifest["layer_count"])
    query_offsets = [0]
    query_kinds: list[int] = []
    query_positions: list[int] = []
    query_output_indices: list[int] = []
    query_predicted_ids: list[int] = []
    query_block_indices: list[int] = []
    query_latent_indices: list[int] = []
    source_positions_all: list[np.ndarray] = []
    source_kinds_all: list[np.ndarray] = []
    source_token_ids_all: list[np.ndarray] = []
    raw_blocks: list[np.ndarray] = []
    normalized_blocks: list[np.ndarray] = []
    target_masses: list[np.ndarray] = []
    raw_masses: list[np.ndarray] = []

    streams = [
        ("latent", QUERY_LATENT, (SOURCE_INPUT_TEXT, SOURCE_INPUT_VISUAL)),
        ("answer", QUERY_ANSWER, TARGET_SOURCE_CODES),
    ]
    for stream, query_kind, target_codes in streams:
        records = manifest[f"{stream}_records"]
        spool_path = capture_dir / manifest[f"{stream}_spool"]
        expected = sum(
            int(item["layer_count"]) * int(item["source_count"]) for item in records
        )
        actual = spool_path.stat().st_size // dtype.itemsize
        if actual != expected:
            raise RuntimeError(f"{stream} spool size mismatch: {actual} != {expected}")
        spool = np.memmap(spool_path, mode="r", dtype=dtype) if expected else None
        try:
            for record in records:
                if int(record["layer_count"]) != layer_count:
                    raise RuntimeError("Layer count changed within one sample.")
                source_count = int(record["source_count"])
                positions = np.arange(source_count, dtype=np.int32)
                kinds, token_ids = classify_source_positions(
                    positions, manifest, latent_ids
                )
                assert spool is not None
                raw = _read_record_matrix(spool, record, dtype)
                tolerance = 5e-3 if dtype == np.dtype("float16") else 1e-4
                if not np.allclose(
                    raw.astype(np.float32).sum(axis=1),
                    1.0,
                    atol=tolerance,
                    rtol=tolerance,
                ):
                    raise RuntimeError(
                        "Raw attention does not sum to one at query position "
                        f"{record['query_sequence_position']}."
                    )
                normalized = normalize_attention_groups(raw, kinds, target_codes)
                visible_mass = np.stack(
                    [
                        normalized[:, kinds == code].sum(axis=1, dtype=np.float32)
                        for code in TARGET_SOURCE_CODES
                    ],
                    axis=-1,
                )
                if not np.allclose(
                    visible_mass.sum(axis=-1), 1.0, atol=1e-5, rtol=1e-5
                ):
                    raise RuntimeError(
                        "Target attention categories cannot be normalized at "
                        f"query position {record['query_sequence_position']}."
                    )
                complete_mass = np.stack(
                    [
                        raw[:, kinds == code].sum(axis=1, dtype=np.float32)
                        for code in range(len(SOURCE_KIND_NAMES))
                    ],
                    axis=-1,
                )
                raw_blocks.append(raw)
                normalized_blocks.append(normalized.astype(dtype, copy=False))
                target_masses.append(visible_mass)
                raw_masses.append(complete_mass)
                source_positions_all.append(positions)
                source_kinds_all.append(kinds)
                source_token_ids_all.append(token_ids)
                query_offsets.append(query_offsets[-1] + source_count)
                query_kinds.append(query_kind)
                query_positions.append(int(record["query_sequence_position"]))
                query_output_indices.append(int(record.get("output_index", -1)))
                query_predicted_ids.append(int(record.get("predicted_token_id", -1)))
                query_block_indices.append(int(record.get("latent_block_index", -1)))
                query_latent_indices.append(int(record.get("latent_index", -1)))
        finally:
            if spool is not None:
                del spool

    total_sources = query_offsets[-1]
    raw_attention = (
        np.concatenate(raw_blocks, axis=1)
        if raw_blocks
        else np.empty((layer_count, 0), dtype=dtype)
    )
    normalized_attention = (
        np.concatenate(normalized_blocks, axis=1)
        if normalized_blocks
        else np.empty((layer_count, 0), dtype=dtype)
    )
    if raw_attention.shape != (layer_count, total_sources):
        raise RuntimeError("Ragged attention assembly produced an invalid shape.")

    topk = manifest["latent_topk"]
    if len(topk) != len(manifest["latent_records"]):
        raise RuntimeError("Every latent query must have exactly one top-k row.")
    topk_ids = (
        np.asarray([item["token_ids"] for item in topk], dtype=np.int32)
        if topk
        else np.empty((0, LATENT_TOP_K), dtype=np.int32)
    )
    topk_logits = (
        np.asarray([item["logits"] for item in topk], dtype=np.float32)
        if topk
        else np.empty((0, LATENT_TOP_K), dtype=np.float32)
    )

    def concatenate(values: list[np.ndarray], dtype_: Any) -> np.ndarray:
        return (
            np.concatenate(values).astype(dtype_, copy=False)
            if values
            else np.empty(0, dtype=dtype_)
        )

    return {
        "raw_attention": raw_attention,
        "group_normalized_attention": normalized_attention,
        "query_source_offsets": np.asarray(query_offsets, dtype=np.int64),
        "query_kind_codes": np.asarray(query_kinds, dtype=np.uint8),
        "query_sequence_positions": np.asarray(query_positions, dtype=np.int32),
        "query_output_indices": np.asarray(query_output_indices, dtype=np.int32),
        "query_predicted_token_ids": np.asarray(query_predicted_ids, dtype=np.int32),
        "query_latent_block_indices": np.asarray(query_block_indices, dtype=np.int32),
        "query_latent_indices": np.asarray(query_latent_indices, dtype=np.int32),
        "source_sequence_positions": concatenate(source_positions_all, np.int32),
        "source_kind_codes": concatenate(source_kinds_all, np.uint8),
        "source_token_ids": concatenate(source_token_ids_all, np.int32),
        "category_attention_distribution": (
            np.stack(target_masses).astype(np.float32)
            if target_masses
            else np.empty((0, layer_count, 4), dtype=np.float32)
        ),
        "raw_category_attention_mass": (
            np.stack(raw_masses).astype(np.float32)
            if raw_masses
            else np.empty((0, layer_count, len(SOURCE_KIND_NAMES)), dtype=np.float32)
        ),
        "source_kind_names": SOURCE_KIND_NAMES,
        "target_kind_names": TARGET_KIND_NAMES,
        "query_kind_names": QUERY_KIND_NAMES,
        "layer_names": np.asarray(
            [f"model.layers.{index}" for index in range(layer_count)], dtype=np.str_
        ),
        "latent_topk_token_ids": topk_ids,
        "latent_topk_logits": topk_logits,
        "latent_topk_sequence_positions": np.asarray(
            [item["query_sequence_position"] for item in topk], dtype=np.int32
        ),
        "latent_topk_block_indices": np.asarray(
            [item["latent_block_index"] for item in topk], dtype=np.int32
        ),
        "latent_topk_indices": np.asarray(
            [item["latent_index"] for item in topk], dtype=np.int32
        ),
        "no_latent_fallback": np.asarray(bool(manifest["no_latent_fallback"])),
    }


def token_piece(tokenizer: Any, token_id: int) -> str:
    if token_id < 0:
        return "<unknown>"
    return str(tokenizer.convert_ids_to_tokens(token_id)).replace("\n", "\\n")


def query_label(data: dict[str, np.ndarray], tokenizer: Any, index: int) -> str:
    kind = int(data["query_kind_codes"][index])
    position = int(data["query_sequence_positions"][index])
    if kind == QUERY_LATENT:
        block = int(data["query_latent_block_indices"][index])
        latent = int(data["query_latent_indices"][index])
        return f"latent[{block}:{latent}]@{position}"
    token_id = int(data["query_predicted_token_ids"][index])
    return f"answer:{token_piece(tokenizer, token_id)}@{position}"


def plot_attention_heatmap(
    data: dict[str, np.ndarray], tokenizer: Any, query_kind: int, path: Path
) -> None:
    if not WRITE_PLOTS:
        return
    import matplotlib.pyplot as plt

    indices = np.flatnonzero(data["query_kind_codes"] == query_kind)
    if not len(indices):
        return
    layer_index = PLOT_LAYER % len(data["layer_names"])
    max_sources = max(
        int(
            data["query_source_offsets"][index + 1]
            - data["query_source_offsets"][index]
        )
        for index in indices
    )
    raw = np.full((len(indices), max_sources), np.nan, dtype=np.float32)
    normalized = np.full_like(raw, np.nan)
    for row, query_index in enumerate(indices):
        start = int(data["query_source_offsets"][query_index])
        end = int(data["query_source_offsets"][query_index + 1])
        positions = data["source_sequence_positions"][start:end]
        kinds = data["source_kind_codes"][start:end]
        allowed = (
            np.isin(kinds, (SOURCE_INPUT_TEXT, SOURCE_INPUT_VISUAL))
            if query_kind == QUERY_LATENT
            else np.isin(kinds, TARGET_SOURCE_CODES)
        )
        selected_positions = positions[allowed]
        raw[row, selected_positions] = data["raw_attention"][layer_index, start:end][
            allowed
        ]
        normalized[row, selected_positions] = data["group_normalized_attention"][
            layer_index, start:end
        ][allowed]

    height = max(3.5, PLOT_ROW_HEIGHT * len(indices))
    figure, axes = plt.subplots(1, 2, figsize=(PLOT_FIGURE_WIDTH, height))
    for axis, matrix, title in zip(
        axes, (raw, normalized), ("raw target attention", "target-normalized attention")
    ):
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
        axis.set_title(title)
        axis.set_xlabel("absolute source sequence position")
        tick_step = max(1, len(indices) // PLOT_MAX_TOKEN_LABELS)
        ticks = np.arange(0, len(indices), tick_step)
        axis.set_yticks(ticks)
        axis.set_yticklabels(
            [query_label(data, tokenizer, int(indices[tick])) for tick in ticks],
            fontsize=7,
        )
        figure.colorbar(image, ax=axis, fraction=0.025)
    figure.suptitle(f"{QUERY_KIND_NAMES[query_kind]} attention, layer {layer_index}")
    figure.tight_layout()
    figure.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def plot_category_attention(
    data: dict[str, np.ndarray], tokenizer: Any, path: Path
) -> None:
    if not WRITE_PLOTS or not len(data["query_kind_codes"]):
        return
    import matplotlib.pyplot as plt

    layer_index = PLOT_LAYER % len(data["layer_names"])
    matrix = data["category_attention_distribution"][:, layer_index, :]
    height = max(3.5, PLOT_ROW_HEIGHT * len(matrix))
    figure, axis = plt.subplots(figsize=(8.5, height))
    image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(TARGET_KIND_NAMES)))
    axis.set_xticklabels(TARGET_KIND_NAMES.tolist(), rotation=25, ha="right")
    tick_step = max(1, len(matrix) // PLOT_MAX_TOKEN_LABELS)
    ticks = np.arange(0, len(matrix), tick_step)
    axis.set_yticks(ticks)
    axis.set_yticklabels(
        [query_label(data, tokenizer, int(tick)) for tick in ticks], fontsize=7
    )
    axis.set_title(f"Target category attention, layer {layer_index}")
    figure.colorbar(image, ax=axis, fraction=0.03)
    figure.tight_layout()
    figure.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def decode_topk(data: dict[str, np.ndarray], tokenizer: Any) -> list[dict[str, Any]]:
    records = []
    for block, latent, position, ids, logits in zip(
        data["latent_topk_block_indices"],
        data["latent_topk_indices"],
        data["latent_topk_sequence_positions"],
        data["latent_topk_token_ids"],
        data["latent_topk_logits"],
    ):
        records.append(
            {
                "latent_block_index": int(block),
                "latent_index": int(latent),
                "sequence_position": int(position),
                "candidates": [
                    {
                        "rank": rank,
                        "token_id": int(token_id),
                        "decoded_text": tokenizer.decode(
                            [int(token_id)], skip_special_tokens=False
                        ),
                        "raw_logit": float(logit),
                    }
                    for rank, (token_id, logit) in enumerate(zip(ids, logits), start=1)
                ],
            }
        )
    return records


def category_attention_csv_fieldnames() -> list[str]:
    return [
        "sample_ordinal",
        "dataset_ordinal",
        "dataset_index",
        "request_id",
        "query_ordinal",
        "query_kind",
        "query_sequence_position",
        "query_output_index",
        "query_predicted_token_id",
        "query_predicted_text",
        "query_latent_block_index",
        "query_latent_index",
        "layer_index",
        "layer_name",
        *TARGET_KIND_NAMES.tolist(),
    ]


def latent_topk_csv_fieldnames(top_k: int = LATENT_TOP_K) -> list[str]:
    fields = [
        "sample_ordinal",
        "dataset_ordinal",
        "dataset_index",
        "request_id",
        "latent_ordinal",
        "latent_block_index",
        "latent_index",
        "sequence_position",
    ]
    for rank in range(1, top_k + 1):
        fields.extend(
            [
                f"top{rank}_text",
                f"top{rank}_token_id",
                f"top{rank}_raw_logit",
            ]
        )
    return fields


def build_category_attention_csv_rows(
    data: dict[str, np.ndarray], tokenizer: Any, **metadata: Any
) -> Iterable[dict[str, Any]]:
    masses = data["category_attention_distribution"]
    for query_ordinal, query_kind_value in enumerate(data["query_kind_codes"]):
        predicted_id = int(data["query_predicted_token_ids"][query_ordinal])
        common = {
            **metadata,
            "query_ordinal": query_ordinal,
            "query_kind": str(QUERY_KIND_NAMES[int(query_kind_value)]),
            "query_sequence_position": int(
                data["query_sequence_positions"][query_ordinal]
            ),
            "query_output_index": int(data["query_output_indices"][query_ordinal]),
            "query_predicted_token_id": predicted_id,
            "query_predicted_text": (
                tokenizer.decode([predicted_id], skip_special_tokens=False)
                if predicted_id >= 0
                else ""
            ),
            "query_latent_block_index": int(
                data["query_latent_block_indices"][query_ordinal]
            ),
            "query_latent_index": int(data["query_latent_indices"][query_ordinal]),
        }
        for layer_index, layer_name in enumerate(data["layer_names"]):
            row = {**common, "layer_index": layer_index, "layer_name": str(layer_name)}
            for kind_index, name in enumerate(TARGET_KIND_NAMES):
                row[str(name)] = float(masses[query_ordinal, layer_index, kind_index])
            yield row


def build_latent_topk_csv_rows(
    decoded: list[dict[str, Any]], top_k: int = LATENT_TOP_K, **metadata: Any
) -> Iterable[dict[str, Any]]:
    for ordinal, record in enumerate(decoded):
        candidates = record["candidates"]
        if len(candidates) != top_k:
            raise RuntimeError(
                f"Expected {top_k} top-k entries, got {len(candidates)}."
            )
        row = {
            **metadata,
            "latent_ordinal": ordinal,
            "latent_block_index": record["latent_block_index"],
            "latent_index": record["latent_index"],
            "sequence_position": record["sequence_position"],
        }
        for rank, candidate in enumerate(candidates, start=1):
            row[f"top{rank}_text"] = candidate["decoded_text"]
            row[f"top{rank}_token_id"] = candidate["token_id"]
            row[f"top{rank}_raw_logit"] = candidate["raw_logit"]
        yield row


def _move_processor_inputs(inputs: Any, device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def global_config_snapshot() -> dict[str, Any]:
    names = [
        "MODEL_PATH",
        "HRBENCH_PATH",
        "OUTPUT_DIR",
        "CACHE_DIR",
        "RESULTS_FILE",
        "RUN_CONFIG_FILE",
        "CATEGORY_ATTENTION_CSV_FILE",
        "LATENT_TOPK_CSV_FILE",
        "ATTENTION_SUBDIR",
        "PLOT_SUBDIR",
        "SELECTION_MODE",
        "START_INDEX",
        "NUM_SAMPLES",
        "RANDOM_SEED",
        "DEVICE_MAP",
        "TORCH_DTYPE",
        "TRUST_REMOTE_CODE",
        "ATTN_IMPLEMENTATION",
        "FALLBACK_TO_EAGER_ATTENTION",
        "MIN_PIXELS",
        "MAX_PIXELS",
        "MAX_NEW_TOKENS",
        "TEMPERATURE",
        "TOP_P",
        "USE_CACHE",
        "LATENT_TOP_K",
        "ATTENTION_STORAGE_DTYPE",
        "PLOT_LAYER",
        "PLOT_DPI",
        "PLOT_MAX_TOKEN_LABELS",
        "WRITE_PLOTS",
        "KEEP_TEMP_CAPTURE_ON_ERROR",
    ]
    return {name: _json_compatible(globals()[name]) for name in names}


def main() -> None:
    import torch
    from transformers import LogitsProcessorList

    project_src = Path(__file__).resolve().parent / "src"
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from utils_deepseed import LatentTemplateLogitsProcessor

    model_path, dataset_path, output_path = validate_configuration()
    rows, selected_indices = load_hrbench_rows(dataset_path)
    temporary_root = Path(tempfile.mkdtemp(prefix=".ilvr_attention_", dir=output_path))
    succeeded = False
    result_tmp = output_path / f".{RESULTS_FILE}.tmp"
    category_tmp = output_path / f".{CATEGORY_ATTENTION_CSV_FILE}.tmp"
    topk_tmp = output_path / f".{LATENT_TOPK_CSV_FILE}.tmp"
    capture_statistics: list[dict[str, Any]] = []
    try:
        model, processor, model_details = load_model_and_processor(model_path)
        tokenizer = processor.tokenizer
        latent_ids = model_details["latent_token_ids"]
        special_ids = set(map(int, tokenizer.all_special_ids))
        special_ids.update(latent_ids.values())
        logits_processor = LogitsProcessorList(
            [
                LatentTemplateLogitsProcessor(
                    latent_ids["start"],
                    latent_ids["end"],
                    latent_ids["pad"],
                    int(model_details["latent_size"]),
                )
            ]
        )
        input_device = model.get_input_embeddings().weight.device

        with (
            result_tmp.open("w", encoding="utf-8") as result_handle,
            category_tmp.open("w", encoding="utf-8-sig", newline="") as category_handle,
            topk_tmp.open("w", encoding="utf-8-sig", newline="") as topk_handle,
        ):
            category_writer = csv.DictWriter(
                category_handle, fieldnames=category_attention_csv_fieldnames()
            )
            topk_writer = csv.DictWriter(
                topk_handle, fieldnames=latent_topk_csv_fieldnames()
            )
            category_writer.writeheader()
            topk_writer.writeheader()

            for sample_ordinal, (row, dataset_ordinal) in enumerate(
                zip(rows, selected_indices)
            ):
                sample_capture = temporary_root / f"sample_{sample_ordinal:06d}"
                sample_capture.mkdir()
                sample_succeeded = False
                image = decode_hrbench_image(row["image"], dataset_path.parent)
                try:
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
                    prompt_id_list = list(
                        map(int, prompt_ids.detach().to(device="cpu").tolist())
                    )
                    image_token_id = int(model.config.image_token_id)
                    image_positions = (
                        (prompt_ids == image_token_id)
                        .nonzero(as_tuple=True)[0]
                        .detach()
                        .to(device="cpu")
                        .tolist()
                    )
                    if not image_positions:
                        raise RuntimeError(
                            "Processed HRBench sample has no image tokens."
                        )
                    recorder = ILVRAttentionRecorder(
                        sample_capture,
                        prompt_id_list,
                        image_positions,
                        special_ids,
                        latent_ids,
                        int(model_details["num_hidden_layers"]),
                    )
                    with (
                        torch.inference_mode(),
                        install_attention_recorder(model, recorder),
                    ):
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            do_sample=TEMPERATURE > 0,
                            use_cache=USE_CACHE,
                            logits_processor=logits_processor,
                            tokenizer=tokenizer,
                            return_dict_in_generate=True,
                            output_attentions=False,
                            output_hidden_states=False,
                        )
                    output_ids = generated.sequences[0, len(prompt_id_list) :]
                    output_id_list = list(
                        map(int, output_ids.detach().to(device="cpu").tolist())
                    )
                    manifest = recorder.manifest()
                    if output_id_list != manifest["generated_token_ids"]:
                        raise RuntimeError(
                            "Generated IDs do not match recorder manifest."
                        )
                    validate_complete_latent_blocks(
                        manifest, latent_ids, int(model_details["latent_size"])
                    )
                    data = assemble_sample_archive(manifest, sample_capture, latent_ids)
                    stem = f"sample_{sample_ordinal:06d}"
                    archive_rel = Path(ATTENTION_SUBDIR) / f"{stem}.npz"
                    np.savez_compressed(output_path / archive_rel, **data)

                    latent_plot_rel = Path(PLOT_SUBDIR) / f"{stem}_latent_attention.png"
                    answer_plot_rel = Path(PLOT_SUBDIR) / f"{stem}_answer_attention.png"
                    category_plot_rel = (
                        Path(PLOT_SUBDIR) / f"{stem}_category_attention.png"
                    )
                    plot_attention_heatmap(
                        data, tokenizer, QUERY_LATENT, output_path / latent_plot_rel
                    )
                    plot_attention_heatmap(
                        data, tokenizer, QUERY_ANSWER, output_path / answer_plot_rel
                    )
                    plot_category_attention(
                        data, tokenizer, output_path / category_plot_rel
                    )

                    request_id = f"hf-{sample_ordinal:06d}"
                    metadata = {
                        "sample_ordinal": sample_ordinal,
                        "dataset_ordinal": int(dataset_ordinal),
                        "dataset_index": _json_compatible(row["index"]),
                        "request_id": request_id,
                    }
                    category_writer.writerows(
                        build_category_attention_csv_rows(data, tokenizer, **metadata)
                    )
                    decoded_topk = decode_topk(data, tokenizer)
                    topk_writer.writerows(
                        build_latent_topk_csv_rows(decoded_topk, **metadata)
                    )
                    answer_mask = data["query_kind_codes"] == QUERY_ANSWER
                    answer_ids = data["query_predicted_token_ids"][answer_mask].tolist()
                    decoded = tokenizer.decode(
                        output_id_list,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    result = {
                        **metadata,
                        "question": _json_compatible(row["question"]),
                        "answer": _json_compatible(row["answer"]),
                        "category": _json_compatible(row["category"]),
                        "cycle_category": _json_compatible(row["cycle_category"]),
                        "choices": {
                            letter: _json_compatible(row[letter]) for letter in "ABCD"
                        },
                        "raw_output_text": decoded,
                        "cleaned_output_text": clean_latent_output(decoded),
                        "output_token_ids": output_id_list,
                        "prompt_token_count": len(prompt_id_list),
                        "finish_reason": (
                            "length"
                            if len(output_id_list) >= MAX_NEW_TOKENS
                            else "stop"
                        ),
                        "no_latent_fallback": bool(data["no_latent_fallback"]),
                        "answer_output_indices": data["query_output_indices"][
                            answer_mask
                        ].tolist(),
                        "answer_token_ids": answer_ids,
                        "answer_text": tokenizer.decode(
                            answer_ids, skip_special_tokens=False
                        ),
                        "latent_output_head_topk": decoded_topk,
                        "capture_counts": {
                            "latent": int(
                                np.count_nonzero(
                                    data["query_kind_codes"] == QUERY_LATENT
                                )
                            ),
                            "answer": int(np.count_nonzero(answer_mask)),
                        },
                        "attention_archive": str(archive_rel),
                        "plots": {
                            "latent_attention": (
                                str(latent_plot_rel)
                                if (output_path / latent_plot_rel).exists()
                                else None
                            ),
                            "answer_attention": (
                                str(answer_plot_rel)
                                if (output_path / answer_plot_rel).exists()
                                else None
                            ),
                            "category_attention": (
                                str(category_plot_rel)
                                if (output_path / category_plot_rel).exists()
                                else None
                            ),
                        },
                    }
                    result_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    result_handle.flush()
                    category_handle.flush()
                    topk_handle.flush()
                    capture_statistics.append(
                        {
                            "sample_ordinal": sample_ordinal,
                            "latent_queries": int(
                                np.count_nonzero(
                                    data["query_kind_codes"] == QUERY_LATENT
                                )
                            ),
                            "answer_queries": int(np.count_nonzero(answer_mask)),
                            "ragged_source_entries": int(
                                data["query_source_offsets"][-1]
                            ),
                        }
                    )
                    print(
                        f"[{sample_ordinal + 1}/{len(rows)}] dataset={dataset_ordinal}, "
                        f"latent={capture_statistics[-1]['latent_queries']}, "
                        f"answer={capture_statistics[-1]['answer_queries']}"
                    )
                    sample_succeeded = True
                finally:
                    image.close()
                    if sample_capture.exists() and (
                        sample_succeeded or not KEEP_TEMP_CAPTURE_ON_ERROR
                    ):
                        shutil.rmtree(sample_capture, ignore_errors=True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        os.replace(result_tmp, output_path / RESULTS_FILE)
        os.replace(category_tmp, output_path / CATEGORY_ATTENTION_CSV_FILE)
        os.replace(topk_tmp, output_path / LATENT_TOPK_CSV_FILE)
        run_config = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": global_config_snapshot(),
            "model": model_details,
            "selected_dataset_ordinals": selected_indices,
            "capture_statistics": capture_statistics,
            "schema": {
                "source_kind_names": SOURCE_KIND_NAMES.tolist(),
                "target_kind_names": TARGET_KIND_NAMES.tolist(),
                "query_kind_names": QUERY_KIND_NAMES.tolist(),
                "raw_attention": "[decoder_layer, concatenated_ragged_source]",
                "group_normalized_attention": (
                    "latent: input_text+input_visual; answer: "
                    "input_text+input_visual+latent+cot_text"
                ),
                "answer_alignment": "query that predicted each answer token",
            },
            "outputs": {
                "results": RESULTS_FILE,
                "category_attention_csv": CATEGORY_ATTENTION_CSV_FILE,
                "latent_topk_csv": LATENT_TOPK_CSV_FILE,
                "attention_directory": ATTENTION_SUBDIR,
                "plot_directory": PLOT_SUBDIR,
            },
        }
        (output_path / RUN_CONFIG_FILE).write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        succeeded = True
        print(f"Attention analysis complete: {output_path}")
    finally:
        for temporary in (result_tmp, category_tmp, topk_tmp):
            if temporary.exists():
                temporary.unlink()
        if succeeded or not KEEP_TEMP_CAPTURE_ON_ERROR:
            shutil.rmtree(temporary_root, ignore_errors=True)
        elif temporary_root.exists():
            print(f"Temporary captures kept for debugging: {temporary_root}")
        gc.collect()


if __name__ == "__main__":
    main()
