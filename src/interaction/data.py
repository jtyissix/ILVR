"""CoMT validation, deterministic token cache and image-only collation."""
import hashlib
import json
from pathlib import Path

TOKENS = {"pad": "<|latent_pad|>", "start": "<|latent_start|>", "end": "<|latent_end|>"}
CACHE_VERSION = 1


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_inventory(directory, hash_weights=False):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen2_5_vl":
        raise ValueError("Expected the official Qwen2.5-VL ILVR checkpoint")
    for name in ("tokenizer_config.json", "preprocessor_config.json", "tokenizer.json"):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / index_name
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            weights = sorted(set(mapping.values()))
            break
    else:
        weights = [name for name in ("model.safetensors", "pytorch_model.bin") if (directory / name).is_file()]
    if not weights:
        raise FileNotFoundError(f"No model weights in {directory}")
    records = {}
    for name in weights:
        path = directory / name
        stat = path.stat()
        if not stat.st_size:
            raise ValueError(f"Empty checkpoint shard: {path}")
        with path.open("rb") as handle:
            if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError(f"Git LFS pointer is not a downloaded weight shard: {path}")
        records[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if hash_weights:
            records[name]["sha256"] = file_hash(path)
    small = {p.name: file_hash(p) for p in sorted(directory.glob("*.json"))}
    return {"config": config, "files": records, "metadata": small,
            "fingerprint": stable_hash({"files": records, "metadata": small})}


def token_ids(processor, config):
    tokenizer = processor.tokenizer
    result = {}
    keys = {"pad": "latent_token_id", "start": "latent_start_id", "end": "latent_end_id"}
    for name, text in TOKENS.items():
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] != config.get(keys[name]):
            raise ValueError(f"Checkpoint/tokenizer disagreement for {text}: {encoded}")
        result[name] = encoded[0]
    if len(set(result.values())) != 3:
        raise ValueError("Latent token IDs must be distinct")
    if len(tokenizer) != config["vocab_size"]:
        raise ValueError("Tokenizer vocabulary and model weights must match; do not add tokens")
    return result


def resolve_image(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid image path: {value!r}")
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def normalize_sample(sample, image_root, check_helpers=True):
    if not isinstance(sample.get("text_input"), str):
        raise ValueError("text_input must be a string")
    paths = sample.get("image_input", [])
    paths = [paths] if isinstance(paths, str) else paths
    if not isinstance(paths, list):
        raise ValueError("image_input must be a path or list of paths")
    images = [str(resolve_image(image_root, value)) for value in paths]
    sequence = sample.get("sequence_plan")
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("sequence_plan must be a non-empty list")
    for step in sequence:
        if step.get("type") == "text":
            if not isinstance(step.get("content"), str):
                raise ValueError("Text step must have string content")
            if any(marker in step["content"] for marker in TOKENS.values()):
                raise ValueError("Raw rationale contains reserved latent delimiters")
        elif step.get("type") == "latent":
            if not step.get("helper_image"):
                raise ValueError("Latent step is missing helper_image")
            if check_helpers:
                resolve_image(image_root, step["helper_image"])
        else:
            raise ValueError(f"Unknown sequence step: {step.get('type')}")
    if not any(s["type"] == "text" and s["content"].strip() for s in sequence):
        raise ValueError("A training trajectory must contain supervised text")
    return images, sequence


def user_message(sample, paths):
    return {"role": "user", "content": [
        *({"type": "image", "image": p} for p in paths),
        {"type": "text", "text": sample["text_input"]},
    ]}


def find_segments(ids, assistant_start, special, latent_ids, expected_steps=8):
    segments, cursor, local_start = [], assistant_start, assistant_start
    while cursor < len(ids):
        token = ids[cursor]
        if token == latent_ids["start"]:
            end = cursor + expected_steps + 1
            if end >= len(ids) or ids[end] != latent_ids["end"]:
                raise ValueError(f"Malformed latent boundary at position {cursor}")
            pads = list(range(cursor + 1, end))
            if any(ids[p] != latent_ids["pad"] for p in pads):
                raise ValueError("Latent span must consist only of latent_pad placeholders")
            text = [i for i in range(local_start, cursor) if ids[i] not in special]
            segments.append({"start": cursor, "end": end, "pads": pads, "text": text})
            cursor = local_start = end + 1
        elif token in {latent_ids["pad"], latent_ids["end"]}:
            raise ValueError(f"Unmatched latent token at {cursor}")
        else:
            cursor += 1
    return segments


def expand_record(record, interaction_steps, latent_ids, special):
    """Canonical cache always contains 8 visual slots; expansion is per experiment."""
    ids, labels = [], []
    for token, label in zip(record["input_ids"], record["labels"]):
        ids.append(token)
        labels.append(label)
        if token == latent_ids["start"]:
            ids.extend([latent_ids["pad"]] * interaction_steps)
            labels.extend([-100] * interaction_steps)
    return {**record, "input_ids": ids, "labels": labels,
            "segments": find_segments(ids, record["assistant_start"], special, latent_ids, 8 + interaction_steps)}


def prepare_sample(sample, processor, model_config, image_root, max_seq_length, check_helpers=True):
    from PIL import Image
    paths, sequence = normalize_sample(sample, image_root, check_helpers)
    latent_ids = token_ids(processor, model_config)
    user = user_message(sample, paths)
    assistant = {"role": "assistant", "content": [
        {"type": "text", "text": step["content"] if step["type"] == "text" else
         TOKENS["start"] + TOKENS["pad"] * 8 + TOKENS["end"]}
        for step in sequence
    ]}
    prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template([user, assistant], tokenize=False, add_generation_prompt=False)
    if not full_text.startswith(prompt):
        raise ValueError("Chat template does not produce a consistent assistant prefix")
    pictures = []
    for path in paths:
        with Image.open(path) as im:
            pictures.append(im.convert("RGB"))
    batch = processor(text=[full_text], images=pictures or None, return_tensors="pt")
    ids = batch["input_ids"][0].tolist()
    grids = batch["image_grid_thw"].tolist() if "image_grid_thw" in batch else []
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    pieces = prompt.split(image_token)
    if len(pieces) != len(grids) + 1:
        raise ValueError("Image placeholder count does not match input images")
    merge = processor.image_processor.merge_size ** 2
    expanded_prompt = pieces[0]
    for grid, tail in zip(grids, pieces[1:]):
        expanded_prompt += image_token * (grid[0] * grid[1] * grid[2] // merge) + tail
    prefix = processor.tokenizer.encode(expanded_prompt, add_special_tokens=False)
    if ids[:len(prefix)] != prefix:
        raise ValueError("Assistant label boundary differs from tokenized prompt")
    special = set(processor.tokenizer.all_special_ids)
    segments = find_segments(ids, len(prefix), special, latent_ids)
    # Reserve one interaction slot per segment even when preparing a baseline cache.
    if len(ids) + len(segments) > max_seq_length:
        raise ValueError(f"Overlength trajectory: {len(ids) + len(segments)} > {max_seq_length}; no truncation performed")
    labels = [-100 if i < len(prefix) or token == latent_ids["pad"] else token for i, token in enumerate(ids)]
    if not any(x != -100 for x in labels[1:]):
        raise ValueError("No CE targets after masking")
    return {"input_ids": ids, "labels": labels, "assistant_start": len(prefix), "segments": segments,
            "images": paths, "image_grid_thw": grids, "answer": sample.get("original_final_answer"),
            "length": len(ids) + len(segments)}


def prepare_dataset(data_path, image_root, model_path, prepared_dir, max_seq_length=32768):
    from transformers import AutoProcessor
    inventory = checkpoint_inventory(model_path)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    latent_ids = token_ids(processor, inventory["config"])
    records, errors, image_stats = [], [], {}
    with Path(data_path).open(encoding="utf-8") as handle:
        for line, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                record = prepare_sample(json.loads(raw), processor, inventory["config"], image_root, max_seq_length)
                record["source_line"] = line
                records.append(record)
                for path in record["images"]:
                    stat = Path(path).stat()
                    image_stats[path] = [stat.st_size, stat.st_mtime_ns]
            except Exception as exc:
                errors.append({"line": line, "error": str(exc)})
    output = Path(prepared_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / "validation_errors.json", errors)
    if errors or not records:
        raise ValueError(f"Preparation failed: {len(errors)} invalid rows; see {output / 'validation_errors.json'}")
    cache_file = output / "records.jsonl"
    tmp = cache_file.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(cache_file)
    lengths = sorted(r["length"] for r in records)
    manifest = {"version": CACHE_VERSION, "source_sha256": file_hash(data_path),
                "records_sha256": file_hash(cache_file), "checkpoint_metadata": inventory["metadata"],
                "image_root": str(Path(image_root).resolve()), "max_seq_length": max_seq_length,
                "token_ids": latent_ids, "special_ids": processor.tokenizer.all_special_ids,
                "image_stats": image_stats, "samples": len(records),
                "length_p50": lengths[len(lengths) // 2], "length_p95": lengths[min(len(lengths)-1, int(len(lengths)*.95))],
                "length_max": max(lengths), "latent_segments": sum(len(r["segments"]) for r in records)}
    json_write(output / "manifest.json", manifest)
    return manifest


class PreparedDataset:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["version"] != CACHE_VERSION:
            raise ValueError("Outdated prepared cache; run prepare again")
        if file_hash(directory / "records.jsonl") != self.manifest["records_sha256"]:
            raise ValueError("Prepared records checksum failed")
        with (directory / "records.jsonl").open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]

    def validate_sources(self, config):
        if file_hash(config.data_path) != self.manifest["source_sha256"]:
            raise ValueError("TRAIN.jsonl changed; run prepare again")
        if str(Path(config.image_root).resolve()) != self.manifest["image_root"]:
            raise ValueError("image_root changed; run prepare on this server")
        if checkpoint_inventory(config.model_path)["metadata"] != self.manifest["checkpoint_metadata"]:
            raise ValueError("Checkpoint/processor metadata changed; run prepare again")
        if self.manifest["length_max"] > config.max_seq_length:
            raise ValueError("Prepared trajectory exceeds configured max_seq_length")
        for path, expected in self.manifest["image_stats"].items():
            stat = Path(path).stat()
            if [stat.st_size, stat.st_mtime_ns] != expected:
                raise ValueError(f"Input image changed; prepare again: {path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        # -1 is a masked padding example, used only to balance the last distributed window.
        return {**self.records[max(index, 0)], "dummy": index < 0}


def position_ids_for_images(ids, grids, config):
    """Image-only Qwen mRoPE, on CPU during collation. No model weights required."""
    import torch
    image_token = config["image_token_id"]
    merge = config["vision_config"]["spatial_merge_size"]
    parts, cursor, grid_index, offset = [], 0, 0, 0
    while cursor < len(ids):
        if ids[cursor] != image_token:
            end = cursor + 1
            while end < len(ids) and ids[end] != image_token:
                end += 1
            parts.append(torch.arange(offset, offset + end-cursor).expand(3, -1))
            offset += end-cursor
            cursor = end
        else:
            if grid_index >= len(grids):
                raise ValueError("More image tokens than image grids")
            t, h, w = grids[grid_index]
            h, w = h // merge, w // merge
            count = t * h * w
            if ids[cursor:cursor+count] != [image_token] * count:
                raise ValueError("Image token count/grid mismatch")
            # Still images have zero temporal spacing, matching Qwen get_rope_index.
            time = torch.zeros(count, dtype=torch.long)
            height = torch.arange(h).view(1, h, 1).expand(t, h, w).flatten()
            width = torch.arange(w).view(1, 1, w).expand(t, h, w).flatten()
            parts.append(torch.stack((time, height, width)) + offset)
            offset += max(h, w)
            cursor += count
            grid_index += 1
    if grid_index != len(grids):
        raise ValueError("Unused image grids")
    return torch.cat(parts, dim=1)


def tensorize(records, pad_id, config):
    import torch
    length = max(len(r["input_ids"]) for r in records)
    ids = torch.full((len(records), length), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    positions = torch.ones(3, len(records), length, dtype=torch.long)
    events = {}
    for row, r in enumerate(records):
        n = len(r["input_ids"])
        ids[row, :n] = torch.tensor(r["input_ids"])
        if not r.get("dummy"):
            labels[row, :n] = torch.tensor(r["labels"])
        mask[row, :n] = True
        positions[:, row, :n] = position_ids_for_images(r["input_ids"], r["image_grid_thw"], config)
        for segment, item in enumerate(r["segments"]):
            for step, pos in enumerate(item["pads"]):
                events.setdefault(pos, []).append((row, segment, step))
    return {"input_ids": ids, "labels": labels, "attention_mask": mask, "position_ids": positions,
            "latent_positions": events, "segments": [r["segments"] for r in records]}


class Collator:
    def __init__(self, processor, model_config, interaction_steps, manifest):
        self.processor = processor
        self.config = model_config
        self.steps = interaction_steps
        self.latent_ids = manifest["token_ids"]
        self.special = set(manifest["special_ids"])

    def __call__(self, records):
        import torch
        from PIL import Image
        reference = tensorize(records, self.processor.tokenizer.pad_token_id, self.config)
        student_records = [expand_record(r, self.steps, self.latent_ids, self.special) for r in records]
        student = tensorize(student_records, self.processor.tokenizer.pad_token_id, self.config)
        images, expected, unique, occurrences = [], [], {}, []
        for r in records:
            for path, grid in zip(r["images"], r["image_grid_thw"]):
                if path not in unique:
                    unique[path] = len(images)
                    with Image.open(path) as im:
                        images.append(im.convert("RGB"))
                    expected.append(grid)
                elif expected[unique[path]] != grid:
                    raise ValueError("One input image has inconsistent cached grids")
                occurrences.append(unique[path])
        visual = self.processor.image_processor(images=images, return_tensors="pt") if images else {}
        if images and visual["image_grid_thw"].tolist() != expected:
            raise ValueError("Image processor grids changed since prepare")
        if images:
            offsets, offset = [], 0
            merge = self.processor.image_processor.merge_size ** 2
            for t, h, w in expected:
                count = t*h*w//merge
                offsets.append(list(range(offset, offset+count)))
                offset += count
            visual["feature_indices"] = torch.tensor([i for occurrence in occurrences for i in offsets[occurrence]], dtype=torch.long)
        return {"reference": reference, "student": student, "visual": dict(visual),
                "ce_count": int(student["labels"][:, 1:].ne(-100).sum()),
                "sample_count": sum(not r.get("dummy", False) for r in records),
                "source_lines": [r["source_line"] for r in records]}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate official CoMT and cache tokens; no model GPU needed")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--prepared_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=32768)
    args = parser.parse_args()
    manifest = prepare_dataset(**vars(args))
    print(json.dumps({key: manifest[key] for key in ("samples", "latent_segments", "length_p50", "length_p95",
                                                    "length_max", "source_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
