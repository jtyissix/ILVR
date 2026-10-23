"""Strict rewritten-plan preparation for CE-only, image-interleaved SFT 1.

This cache is intentionally separate from the latent/reference-training cache.
No tokenizer tokens are added: masked ordinary pad tokens reserve each ordered
interaction block, immediately before that helper's complete vision block.
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

from src.interaction.data import (
    TOKENS, checkpoint_inventory, file_hash, json_write,
    position_ids_for_images, resolve_image, stable_hash,
)

from .config import WarmupConfig

CACHE_VERSION = 4
IMAGE_POLICY = "min_side_upscale_bicubic_and_per_record_pixel_cap_v1"
_MISSING = object()
_RESERVED = (*TOKENS.values(), "<|im_start|>", "<|im_end|>",
             "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>")


class OverlengthTrajectoryError(ValueError):
    def __init__(self, length, limit):
        self.length, self.limit = length, limit
        super().__init__(f"Overlength trajectory: {length} > {limit}; no truncation performed")


def processor_settings(config):
    """Image size limits are part of the cache identity, not CLI-only options."""
    return {name: getattr(config, name, None)
            for name in ("min_image_pixels", "max_image_pixels")}


def load_processor(config):
    from transformers import AutoProcessor
    options = {name.replace("_image", ""): value
               for name, value in processor_settings(config).items() if value is not None}
    processor = AutoProcessor.from_pretrained(config.model_path, local_files_only=True, **options)
    # Set explicitly as well: older processor wrappers may consume kwargs themselves.
    for name, value in options.items():
        setattr(processor.image_processor, name, value)
        if hasattr(processor.image_processor, "size"):
            processor.image_processor.size["shortest_edge" if name == "min_pixels" else "longest_edge"] = value
    return processor


def _field(sample, name, default=None):
    """An explicit dotted field is useful for preserved source metadata."""
    value = sample
    for part in name.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _audit_demo(value):
    if isinstance(value, dict):
        return bool(value.get("audit_demo_only")) or any(_audit_demo(v) for v in value.values())
    if isinstance(value, list):
        return any(_audit_demo(v) for v in value)
    return False


def _text_content(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or not content:
        raise ValueError("Text content must be a string or a non-empty analysis/action list")
    parts = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Each structured text part must be an object")
        kind = part.get("sub_type")
        if kind not in {"analysis", "action"}:
            raise ValueError(f"Unknown text sub_type: {kind!r}")
        key = "original_text" if kind == "action" else "content"
        text = part.get(key)
        if not isinstance(text, str) or (kind == "action" and not text.strip()):
            raise ValueError(f"{kind}.{key} must contain original text")
        parts.append(text)
    return "\n".join(parts)


def _answer_in_text(answer, text):
    # Whitespace formatting is immaterial, but token boundaries are not: answer
    # '2' must not pass merely because the rationale mentions '12'.
    answer, text = " ".join(answer.split()), " ".join(text.split())
    return re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", text) is not None


def normalize_sample(sample, source):
    """Read only the selected rewritten field; never fall back to an older plan.

    ReFocus retains question/image/answer, so its source config explicitly maps
    question_field, images_field and answer_field. Other converters emit COMT
    field names. A final answer must already be in the last text step unless
    allow_append_final_answer is explicitly enabled for this particular source.
    """
    if not isinstance(sample, dict):
        raise ValueError("Each training sample must be an object")
    if _audit_demo(sample):
        raise ValueError("audit_demo_only records cannot enter training")
    if "answer_field" in source and _field(sample, source["answer_field"], _MISSING) is _MISSING:
        raise ValueError(f"Configured answer_field={source['answer_field']!r} is missing; "
                         f"available top-level fields: {sorted(sample)}")
    plan_field = source.get("plan_field")
    if not isinstance(plan_field, str) or not plan_field:
        raise ValueError("Every source must explicitly specify plan_field")
    question = _field(sample, source.get("question_field", "text_input"))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("The configured question field must be a non-empty string")
    paths = _field(sample, source.get("images_field", "image_input"), [])
    paths = [paths] if isinstance(paths, str) else paths
    if not isinstance(paths, list):
        raise ValueError("The configured images field must be a path or list of paths")
    question_images = [str(resolve_image(source["image_root"], path)) for path in paths]
    sequence = _field(sample, plan_field)
    if isinstance(sequence, str):
        try:
            sequence = json.loads(sequence)
        except ValueError as exc:
            raise ValueError(f"{plan_field} is not a JSON-encoded step list") from exc
    if not isinstance(sequence, list) or not sequence:
        raise ValueError(f"{plan_field} must contain a non-empty step list")
    normalized, structured_actions = [], 0
    for index, step in enumerate(sequence):
        if not isinstance(step, dict):
            raise ValueError(f"Step {index} must be an object")
        kind = step.get("type")
        if kind == "text":
            if step.get("helper_image") is not None:
                raise ValueError(f"Text step {index} unexpectedly contains a helper image")
            normalized.append({"type": "text", "content": _text_content(step.get("content")),
                               "is_final_answer": bool(step.get("is_final_answer"))})
            if isinstance(step.get("content"), list):
                structured_actions += sum(part.get("sub_type") == "action" for part in step["content"])
        elif kind == "latent":
            normalized.append({"type": "image", "path": str(resolve_image(
                source["image_root"], step.get("helper_image")))})
        else:
            raise ValueError(f"Unknown sequence step type at {index}: {kind!r}")
    if (source.get("require_structured_actions", False) and not structured_actions
            and any(step["type"] == "image" for step in normalized)):
        raise ValueError("require_structured_actions: helper-image plans need at least one structured "
                         "action.original_text; select the rewritten structured plan, not a legacy/flattened plan")
    answer = _field(sample, source.get("answer_field", "original_final_answer"))
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        answer = str(answer)
    if answer is not None and (not isinstance(answer, str) or not answer.strip()):
        raise ValueError("The configured authoritative answer must be non-empty text")
    answer = answer.strip() if answer is not None else None
    final = normalized[-1]
    marked = [i for i, step in enumerate(normalized) if step.get("is_final_answer")]
    if marked and marked != [len(normalized) - 1]:
        raise ValueError("The single is_final_answer step must end the sequence plan")
    has_answer = final["type"] == "text" and bool(final["content"].strip()) and (
        _answer_in_text(answer, final["content"]) if answer is not None else bool(marked))
    if not has_answer:
        if source.get("allow_append_final_answer", False) and answer is not None and not marked:
            normalized.append({"type": "text", "content": "The final answer is: " + answer,
                               "is_final_answer": True})
        else:
            raise ValueError("Final answer is missing from the last plan text step; metadata alone is not a CE target")
    # Answerless legacy plans need either an explicit final marker or an explicit
    # authoritative answer_field. We do not guess from the last reasoning sentence.
    if answer is None:
        answer = normalized[-1]["content"]
    for text in [question, *(s["content"] for s in normalized if s["type"] == "text")]:
        if any(token in text for token in _RESERVED):
            raise ValueError("Raw text contains reserved chat/image/latent delimiters")
    return {"question": question, "question_images": question_images,
            "sequence": normalized, "answer": answer,
            "final_answer_text": normalized[-1]["content"]}


def _expanded_prompt(prompt, grids, processor):
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    pieces = prompt.split(image_token)
    if len(pieces) != len(grids) + 1:
        raise ValueError("Prompt image placeholder count does not match question images")
    merge = processor.image_processor.merge_size ** 2
    value = pieces[0]
    for grid, tail in zip(grids, pieces[1:]):
        value += image_token * (grid[0] * grid[1] * grid[2] // merge) + tail
    return value


def load_training_image(path, image_processor, return_original_size=False):
    """Use the same aspect-preserving tiny-image repair in prepare and training."""
    from PIL import Image
    with Image.open(path) as original:
        original_size = list(original.size)
        picture = original.convert("RGB")
    factor = getattr(image_processor, "patch_size", 14) * image_processor.merge_size
    width, height = picture.size
    if min(width, height) < factor:
        scale = factor / min(width, height)
        picture = picture.resize((math.ceil(width * scale), math.ceil(height * scale)), Image.Resampling.BICUBIC)
    return (picture, original_size) if return_original_size else picture


def image_processor_with_cap(image_processor, cap):
    if cap is None:
        return image_processor
    result = copy.deepcopy(image_processor)
    result.max_pixels = cap
    if hasattr(result, "size"):
        result.size["longest_edge"] = cap
    return result


def prepare_sample(sample, source, processor, model_config, max_seq_length,
                   interaction_tokens=WarmupConfig.interaction_tokens,
                   shrink_overlong_images=WarmupConfig.shrink_overlong_images):
    if isinstance(interaction_tokens, bool) or not isinstance(interaction_tokens, int) or interaction_tokens <= 0:
        raise ValueError("interaction_tokens must be a positive integer")
    normalized = normalize_sample(sample, source)
    question_paths = normalized["question_images"]
    all_paths = question_paths + [s["path"] for s in normalized["sequence"] if s["type"] == "image"]
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        raise ValueError("The checkpoint tokenizer needs an existing pad_token_id")
    # Qwen may reserve extra embedding/head rows. Token IDs must fit; the
    # tokenizer length does not need to equal the model's padded capacity.
    if len(tokenizer) > model_config["vocab_size"]:
        raise ValueError(f"Tokenizer vocabulary ({len(tokenizer)}) exceeds model capacity "
                         f"({model_config['vocab_size']}); use the matching checkpoint tokenizer")
    for text in [normalized["question"], *(s["content"] for s in normalized["sequence"] if s["type"] == "text")]:
        if any(token and token in text for token in getattr(tokenizer, "all_special_tokens", [])):
            raise ValueError("Raw training text contains a tokenizer special token")
    user = {"role": "user", "content": [
        *({"type": "image", "image": path} for path in question_paths),
        {"type": "text", "text": normalized["question"]},
    ]}
    assistant = {"role": "assistant", "content": [
        {"type": "text", "text": step["content"] + "\n"} if step["type"] == "text" else
        {"type": "image", "image": step["path"]}
        for step in normalized["sequence"]
    ]}
    prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template([user, assistant], tokenize=False, add_generation_prompt=False)
    if not full_text.startswith(prompt):
        raise ValueError("Chat template does not have a consistent assistant prefix")
    loaded = [load_training_image(path, processor.image_processor, return_original_size=True) for path in all_paths]
    pictures = [picture for picture, _ in loaded]
    original_sizes = [size for _, size in loaded]
    pixel_cap, original_length = None, None
    original_grids = None
    minimum = getattr(processor.image_processor, "min_pixels", 3136)
    patch = getattr(processor.image_processor, "patch_size", 14)
    merge = processor.image_processor.merge_size
    if pictures:
        minimum = max(minimum, math.ceil(max(max(p.size) / min(p.size) for p in pictures) * (patch * merge) ** 2))
    while True:
        active = processor
        if pixel_cap is not None:
            active = copy.copy(processor)
            active.image_processor = image_processor_with_cap(processor.image_processor, pixel_cap)
        encoded = active(text=[full_text], images=pictures or None, return_tensors="pt")
        original_ids = encoded["input_ids"][0].tolist()
        grids = encoded["image_grid_thw"].tolist() if "image_grid_thw" in encoded else []
        if original_grids is None:
            original_grids = grids
        length = len(original_ids) + interaction_tokens * (len(all_paths) - len(question_paths))
        original_length = original_length or length
        if length <= max_seq_length:
            break
        visual_tokens = sum(t * h * w // (merge * merge) for t, h, w in grids)
        available = max_seq_length - (length - visual_tokens)
        if (not shrink_overlong_images or not grids or available <= 0
                or pixel_cap is not None and pixel_cap <= minimum):
            raise OverlengthTrajectoryError(length, max_seq_length)
        current = min(pixel_cap or math.inf, max(h * w * patch * patch for _, h, w in grids))
        next_cap = max(minimum, int(current * min(0.8, 0.9 * available / visual_tokens)))
        if pixel_cap is not None and next_cap >= pixel_cap:
            raise OverlengthTrajectoryError(length, max_seq_length)
        pixel_cap = next_cap
    if any(token < 0 or token >= model_config["vocab_size"] for token in
           (*original_ids, tokenizer.pad_token_id)):
        raise ValueError("Encoded token ID exceeds model vocabulary capacity")
    if len(grids) != len(all_paths):
        raise ValueError("Processor did not preserve every question/helper image")
    prefix = tokenizer.encode(_expanded_prompt(prompt, grids[:len(question_paths)], processor),
                              add_special_tokens=False)
    if original_ids[:len(prefix)] != prefix:
        raise ValueError("Assistant label boundary differs from the expanded prompt")
    image_token = model_config["image_token_id"]
    vision_start = model_config["vision_start_token_id"]
    vision_end = model_config["vision_end_token_id"]
    starts = [i for i, token in enumerate(original_ids) if token == vision_start]
    if len(starts) != len(grids):
        raise ValueError("Vision boundaries differ from image count")
    ids, labels, images, segments = [], [], [], []
    cursor, local_start, feature_offset = 0, len(prefix), 0
    merge = model_config["vision_config"]["spatial_merge_size"]
    if merge != processor.image_processor.merge_size:
        raise ValueError("Processor and model spatial merge size differ")
    special = set(tokenizer.all_special_ids)

    def append_text(end):
        nonlocal cursor
        for original_position in range(cursor, end):
            token = original_ids[original_position]
            ids.append(token)
            labels.append(token if original_position >= len(prefix) else -100)
        cursor = end

    for index, (start, grid, path) in enumerate(zip(starts, grids, all_paths)):
        if len(grid) != 3 or any(v <= 0 for v in grid) or grid[1] % merge or grid[2] % merge:
            raise ValueError(f"Invalid image grid: {grid}")
        count = grid[0] * (grid[1] // merge) * (grid[2] // merge)
        end = start + count + 2
        expected = [vision_start] + [image_token] * count + [vision_end]
        if original_ids[start:end] != expected or start < cursor:
            raise ValueError("Image tokens/grid/boundary mismatch")
        append_text(start)
        kind = "question" if index < len(question_paths) else "helper"
        if kind == "helper":
            position = len(ids)
            segments.append({"position": position,
                             "text": [i for i in range(local_start, position)
                                      if labels[i] != -100 and ids[i] not in special],
                             "image_index": index})
            ids.extend([tokenizer.pad_token_id] * interaction_tokens)
            labels.extend([-100] * interaction_tokens)
        boundary_start = len(ids)
        ids.extend(expected)
        labels.extend([-100] * len(expected))
        initial_t, initial_h, initial_w = original_grids[index]
        images.append({"path": path, "grid": grid, "kind": kind, "max_pixels": pixel_cap,
                       "original_size": original_sizes[index],
                       "initial_processed_size": [initial_w * patch, initial_h * patch],
                       "processed_size": [grid[2] * patch, grid[1] * patch],
                       "original_visual_tokens": initial_t * initial_h * initial_w // (merge * merge),
                       "visual_tokens": count,
                       "image_id": index if kind == "question" else index - len(question_paths),
                       "feature_start": feature_offset, "feature_end": feature_offset + count,
                       "token_start": boundary_start + 1, "token_end": boundary_start + 1 + count,
                       "boundary_start": boundary_start, "boundary_end": boundary_start + len(expected)})
        feature_offset += count
        cursor = end
        if kind == "helper":
            local_start = len(ids)
    append_text(len(original_ids))
    if len(ids) > max_seq_length:
        raise OverlengthTrajectoryError(len(ids), max_seq_length)
    end_tokens = {tokenizer.eos_token_id} - {None}
    # Qwen's chat template ends an assistant turn with im_end even when a
    # tokenizer's generic eos_token is endoftext (including stock constructors).
    chat_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(chat_end) == 1 and chat_end[0] in special:
        end_tokens.add(chat_end[0])
    if not any(label in end_tokens for label in labels[local_start:]):
        raise ValueError("The assistant answer must include a supervised end-of-answer token")
    if not any(label != -100 for label in labels[1:]):
        raise ValueError("No supervised CE targets")
    visual_tokens = sum(image["visual_tokens"] for image in images)
    return {"input_ids": ids, "labels": labels, "assistant_start": len(prefix),
            "segments": segments, "images": images, "image_grid_thw": grids,
            "length": len(ids), "original_length": original_length, "image_budget_reduced": pixel_cap is not None,
            "original_visual_tokens": sum(image["original_visual_tokens"] for image in images),
            "visual_tokens": visual_tokens, "non_visual_tokens": len(ids) - visual_tokens,
            "answer": normalized["answer"],
            "final_answer_text": normalized["final_answer_text"]}


def _source_inventory(sources):
    result, names = [], set()
    for source in sources:
        if not isinstance(source, dict) or any(not source.get(k) for k in ("name", "path", "image_root", "plan_field")):
            raise ValueError("Each source requires name, path, image_root and explicit plan_field")
        if source["name"] in names:
            raise ValueError(f"Duplicate source name: {source['name']}")
        names.add(source["name"])
        path = Path(source["path"]).resolve()
        result.append({**source, "path": str(path), "image_root": str(Path(source["image_root"]).resolve()),
                       "sha256": file_hash(path)})
    if not result:
        raise ValueError("At least one rewritten data source is required")
    return result


def _iter_rows(source):
    path = Path(source["path"])
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as handle:
            for line, raw in enumerate(handle, 1):
                if raw.strip():
                    yield line, json.loads(raw)
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        if source.get("json_records_key"):
            rows = _field(rows, source["json_records_key"])
        if not isinstance(rows, list):
            raise ValueError("JSON input must be a row list; configure json_records_key for wrapped lists")
        yield from enumerate(rows, 1)
    else:
        raise ValueError("Prepared training sources must be rewritten .jsonl or .json files")


def _processor_identity(processor):
    return stable_hash({"image_processor": processor.image_processor.to_dict(),
                        "chat_template": processor.chat_template,
                        "pad_id": processor.tokenizer.pad_token_id,
                        "eos_id": processor.tokenizer.eos_token_id,
                        "vocab_size": len(processor.tokenizer)})


def validate_tokenizer_capacity(processor, model_config):
    """Check the shared tokenizer once before scanning thousands of samples."""
    tokenizer = processor.tokenizer
    capacity = model_config["vocab_size"]
    vocab = tokenizer.get_vocab()
    if not vocab or len(tokenizer) > capacity or any(i < 0 or i >= capacity for i in vocab.values()):
        largest = max(vocab.values(), default=-1)
        raise ValueError(f"Tokenizer IDs do not fit the model: tokenizer_size={len(tokenizer)}, "
                         f"max_token_id={largest}, model_vocab_size={capacity}. "
                         "Use the matching checkpoint tokenizer; no vocabulary resize is performed.")
    if tokenizer.pad_token_id is None or not 0 <= tokenizer.pad_token_id < capacity:
        raise ValueError("The checkpoint tokenizer needs an existing pad_token_id within model capacity")
    for token, field in (("<|image_pad|>", "image_token_id"),
                         ("<|vision_start|>", "vision_start_token_id"),
                         ("<|vision_end|>", "vision_end_token_id")):
        if vocab.get(token) != model_config[field]:
            raise ValueError(f"Tokenizer/model special token mismatch: {token} maps to "
                             f"{vocab.get(token)}, but {field}={model_config[field]}")


def prepare_dataset(config):
    # Lazy import keeps the record-level helpers usable in spawned workers.
    from .preprocessing import prepare_dataset as prepare
    return prepare(config)


class WarmupDataset:
    def __init__(self, directory, verify_checksum=True):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("version") != CACHE_VERSION or self.manifest.get("stage") != "sft1_warmup":
            raise ValueError("Outdated or non-SFT-1 cache; set mode=prepare_only and run run_sft1.py again")
        if verify_checksum and file_hash(directory / "records.jsonl") != self.manifest["records_sha256"]:
            raise ValueError("Prepared records checksum failed")
        with (directory / "records.jsonl").open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records or len(self.records) != self.manifest["samples"]:
            raise ValueError("Prepared sample count differs from manifest")

    def validate_sources(self, config):
        if self.manifest.get("interaction_tokens") != config.interaction_tokens:
            raise ValueError("Interaction token count changed; run prepare again")
        if (self.manifest.get("image_policy") != IMAGE_POLICY
                or self.manifest.get("shrink_overlong_images") != config.shrink_overlong_images):
            raise ValueError("Image resize policy changed; run prepare again")
        if self.manifest.get("skip_overlong_samples") != config.skip_overlong_samples:
            raise ValueError("Overlength exclusion policy changed; run prepare again")
        if self.manifest.get("max_seq_length") != config.max_seq_length:
            raise ValueError("Sequence length budget changed; run prepare again to reconsider skipped/resized records")
        if _source_inventory(config.sources) != self.manifest["source_info"]:
            raise ValueError("Data sources or field mappings changed; run prepare again")
        if checkpoint_inventory(config.model_path)["metadata"] != self.manifest["checkpoint_metadata"]:
            raise ValueError("Checkpoint/processor metadata changed; run prepare again")
        if processor_settings(config) != self.manifest["process_settings"]:
            raise ValueError("Image pixel limits changed; run prepare again")
        if _processor_identity(load_processor(config)) != self.manifest["processor_identity"]:
            raise ValueError("Processor behavior changed; run prepare again")
        if self.manifest["length_max"] > config.max_seq_length:
            raise ValueError("Prepared trajectory exceeds max_seq_length; truncation is forbidden")
        for path, expected in self.manifest["image_stats"].items():
            stat = Path(path).stat()
            if [stat.st_size, stat.st_mtime_ns] != expected:
                raise ValueError(f"Training image changed; run prepare again: {path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if index < -1:
            raise IndexError("Only -1 is the distributed dummy sample sentinel")
        return {**self.records[max(index, 0)], "dummy": index == -1}


def encode_image_groups(processor, pictures, pixel_caps):
    """Preserve occurrence order when different samples use different budgets."""
    import torch
    groups = {}
    for index, cap in enumerate(pixel_caps):
        groups.setdefault(cap, []).append(index)
    pixels, grids = [None] * len(pictures), [None] * len(pictures)
    for cap, indices in groups.items():
        encoded = image_processor_with_cap(processor, cap)(
            images=[pictures[index] for index in indices], return_tensors="pt")
        offset = 0
        for local, index in enumerate(indices):
            grid = encoded["image_grid_thw"][local]
            count = int(grid.prod())
            pixels[index] = encoded["pixel_values"][offset:offset + count]
            grids[index] = grid
            offset += count
        if offset != len(encoded["pixel_values"]):
            raise ValueError("Image pixel/grid layout is inconsistent")
    return {"pixel_values": torch.cat(pixels), "image_grid_thw": torch.stack(grids)}


class WarmupCollator:
    def __init__(self, processor, model_config, manifest):
        self.processor, self.config = processor, model_config
        if manifest.get("processor_identity") != _processor_identity(processor):
            raise ValueError("Collator processor differs from the prepared cache")

    def __call__(self, records):
        import torch
        if not records:
            raise ValueError("Cannot collate an empty batch")
        size = max(len(r["input_ids"]) for r in records)
        ids = torch.full((len(records), size), self.processor.tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        positions = torch.ones(3, len(records), size, dtype=torch.long)
        batch_images, pictures, expected, unique, occurrences = [], [], [], {}, []
        pixel_caps = []
        feature_offset = 0
        for row, record in enumerate(records):
            length = len(record["input_ids"])
            ids[row, :length] = torch.tensor(record["input_ids"])
            if not record.get("dummy"):
                labels[row, :length] = torch.tensor(record["labels"])
            mask[row, :length] = True
            # Recompute from the fully inserted sequence, including every b_i.
            positions[:, row, :length] = position_ids_for_images(record["input_ids"], record["image_grid_thw"], self.config)
            image_row = []
            for item in record["images"]:
                path, grid = item["path"], item["grid"]
                cap = item.get("max_pixels")
                key = (path, cap)
                if key not in unique:
                    unique[key] = len(pictures)
                    pictures.append(load_training_image(path, self.processor.image_processor))
                    pixel_caps.append(cap)
                    expected.append(grid)
                elif expected[unique[key]] != grid:
                    raise ValueError("A repeated image has inconsistent cached grids")
                occurrences.append(unique[key])
                count = item["feature_end"] - item["feature_start"]
                image_row.append({**item, "feature_start": feature_offset, "feature_end": feature_offset + count})
                feature_offset += count
            batch_images.append(image_row)
        visual = encode_image_groups(self.processor.image_processor, pictures, pixel_caps) if pictures else {}
        if pictures:
            if visual["image_grid_thw"].tolist() != expected:
                raise ValueError("Image processor grids changed since preparation")
            offsets, offset = [], 0
            merge = self.processor.image_processor.merge_size ** 2
            for t, h, w in expected:
                count = t * h * w // merge
                offsets.append(range(offset, offset + count))
                offset += count
            visual["feature_indices"] = torch.tensor([i for occurrence in occurrences for i in offsets[occurrence]], dtype=torch.long)
        return {"input_ids": ids, "labels": labels, "attention_mask": mask, "position_ids": positions,
                "segments": [r["segments"] for r in records], "images": batch_images, "visual": visual,
                "ce_count": int(labels[:, 1:].ne(-100).sum()),
                "sample_count": sum(not r.get("dummy", False) for r in records),
                "source_lines": [{"source": r.get("source"), "line": r.get("source_line")} for r in records]}
