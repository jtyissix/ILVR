"""Offline, frozen SFT-I action-state targets for SFT-II.

The coordinator and all GPU workers finish before student training starts. They restore
the *complete* warm-up trajectory, including its image adapter and earlier
interaction tokens, and selects the state after consuming each original action.
No teacher model, teacher KV cache, or original action text is needed in the
student process.  Cache tensors use safetensors; no pickle checkpoint is read.
"""
import argparse
from contextlib import contextmanager
from dataclasses import fields
import hashlib
import heapq
import json
import multiprocessing
import os
from pathlib import Path
import time


CACHE_FORMAT = "sft2-action-teacher-v1"
TARGET_DTYPE = "float32"


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def _writer_lock(directory):
    """An OS lock releases on process exit, so interrupted jobs can resume."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".prepare.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another process is preparing this teacher cache") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another process is preparing this teacher cache") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _get_dataset(config, dataset):
    if dataset is None:
        from .data import Stage2Dataset
        dataset = Stage2Dataset(config.prepared_dir)
    return dataset


def _bundle_files(config):
    root = Path(config.warmup_path).resolve()
    required = ("warmup_recipe.json", "interaction_config.json", "interaction.safetensors",
                "warmup_adapter_config.json", "warmup_adapter.safetensors", "backbone/config.json")
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(f"A complete SFT-I export is required for the action teacher: {root / name}")
    files = sorted(path.resolve() for path in root.rglob("*") if path.is_file()
                   and path.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja", ".tiktoken"})
    if not any(path.suffix == ".safetensors" and path.parent == root / "backbone" for path in files):
        raise FileNotFoundError("SFT-I backbone must contain safetensors model weights")
    return files


def _sources(config, dataset):
    paths = set(_bundle_files(config))
    paths.update((Path(config.prepared_dir) / name).resolve() for name in ("records.jsonl", "manifest.json"))
    project = Path(__file__).resolve().parents[2]
    for name in ("src/sft2/teacher.py", "src/sft2/data.py", "src/sft2/tokens.py", "src/sft1/model.py", "src/sft1/modules.py",
                 "src/sft1/data.py", "src/interaction/execution.py",
                 "transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
                 "transformers/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py"):
        paths.add((project / name).resolve())
    for record in dataset.records:
        for image in record["teacher_record"].get("images", []):
            paths.add(Path(image["path"]).resolve())
    return sorted(paths)


def _settings(config):
    from .data import TOKENIZATION_POLICY
    return {"format": CACHE_FORMAT, "target": "last_layer_after_original_action_last_token",
            "tokenization_policy": TOKENIZATION_POLICY,
            "dtype": TARGET_DTYPE, "execution": "incremental", "activation_checkpointing": False,
            "attention_backend": config.attention_backend,
            "min_image_pixels": config.min_image_pixels, "max_image_pixels": config.max_image_pixels}


def _source_inventory(config, dataset, verify_hashes):
    result = []
    for path in _sources(config, dataset):
        stat = path.stat()
        item = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if verify_hashes:
            item["sha256"] = _file_hash(path)
            after = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Teacher source changed while hashing: {path}")
        result.append(item)
    return result


def _record_key(record):
    return _hash({"record_id": record["record_id"], "teacher_record": record["teacher_record"],
                  "actions": [{"teacher_position": action["teacher_position"],
                               "original_text": action.get("original_text")}
                              for action in record["actions"]]})


def _record_map(dataset):
    result = {}
    for record in dataset.records:
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Teacher cache requires a nonempty string record_id")
        if record_id in result:
            raise ValueError(f"Duplicate teacher record_id: {record_id}")
        result[record_id] = record
    return result


def _relative_file(identity, record_id):
    key = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    return f"generations/{identity}/{key[:2]}/{key}.safetensors"


def _read_manifest(directory):
    manifest = _json_read(Path(directory) / "manifest.json")
    signature = manifest.get("manifest_sha256")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("format") != CACHE_FORMAT or signature != _hash(body):
        raise ValueError("Teacher cache manifest is incomplete, corrupt, or has an unsupported format")
    if manifest.get("identity") != _hash({"settings": manifest.get("settings"), "sources": manifest.get("sources")}):
        raise ValueError("Teacher cache source identity does not match its manifest")
    return manifest


def _validate_entry(directory, manifest, record_id, entry, record=None, read_tensor=True):
    relative = _relative_file(manifest["identity"], record_id)
    if entry.get("file") != relative or entry.get("record_id") != record_id:
        raise ValueError("Teacher cache tensor path or record ownership is invalid")
    if record is not None and (entry.get("record_key") != _record_key(record)
                              or entry.get("shape") != [len(record["actions"]), manifest["hidden_size"]]):
        raise ValueError("Teacher cache action positions, original text, or trajectory changed")
    path = Path(directory) / relative
    if not path.is_file() or path.stat().st_size != entry.get("size"):
        raise ValueError(f"Teacher target is missing or incomplete: {path}")
    if not read_tensor:
        return None
    if _file_hash(path) != entry.get("sha256"):
        raise ValueError(f"Teacher target checksum failed: {path}")
    from safetensors import safe_open
    import torch
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if list(handle.keys()) != ["action_states"]:
            raise ValueError("Teacher tensor file contains unexpected keys")
        expected = {"identity": manifest["identity"], "record_id": record_id,
                    "record_key": entry["record_key"], "format": CACHE_FORMAT}
        if metadata != expected:
            raise ValueError("Teacher tensor metadata does not match its record")
        value = handle.get_tensor("action_states")
    if (list(value.shape) != entry["shape"] or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())):
        raise ValueError("Teacher target has the wrong shape/dtype or nonfinite values")
    return value


def validate_teacher_cache(config, dataset=None, verify_hashes=True):
    """Verify all content before launch; ranks may check metadata after that.

``verify_hashes=False`` is only a fast startup check following the coordinator's
complete validation. Individual tensors still undergo checksum verification when loaded.
    """
    dataset = _get_dataset(config, dataset)
    manifest = _read_manifest(config.teacher_cache_dir)
    if manifest["settings"] != _settings(config):
        raise ValueError("Teacher execution settings changed; regenerate the teacher cache")
    inventory = _source_inventory(config, dataset, verify_hashes)
    expected = manifest["sources"] if verify_hashes else [
        {k: v for k, v in item.items() if k != "sha256"} for item in manifest["sources"]]
    if inventory != expected:
        raise ValueError("SFT-I weights, prepared data, or images changed; regenerate the teacher cache")
    records = _record_map(dataset)
    if set(manifest["records"]) != set(records) or manifest.get("samples") != len(records):
        raise ValueError("Teacher cache does not cover exactly the prepared records")
    for record_id, record in records.items():
        _validate_entry(config.teacher_cache_dir, manifest, record_id, manifest["records"][record_id],
                        record=record, read_tensor=verify_hashes)
    return manifest


def load_teacher_targets(cache_dir, record_id, manifest=None):
    """Return detached CPU float32 [number_of_actions, hidden_size] targets."""
    manifest = manifest if manifest is not None else _read_manifest(cache_dir)
    entry = manifest["records"].get(record_id)
    if entry is None:
        raise ValueError(f"Teacher target has not been prepared for record {record_id}")
    return _validate_entry(cache_dir, manifest, record_id, entry).detach()


class TeacherTargetReader:
    """Parse the immutable launch-validated manifest once per data worker."""
    def __init__(self, cache_dir):
        self.cache_dir = str(cache_dir)
        self.manifest = _read_manifest(cache_dir)

    def __call__(self, record_id):
        return load_teacher_targets(self.cache_dir, record_id, self.manifest)


def teacher_action_targets(teacher, batch, action_rows, image_features=None):
    """Select h[position], not h[position-1]: the last action token is consumed."""
    import torch
    if len(action_rows) != len(batch["input_ids"]):
        raise ValueError("Teacher action ownership must match batch rows")
    teacher.eval()
    with torch.inference_mode():
        hidden = teacher.trajectory(batch, image_features).hidden
        results = []
        for row, actions in enumerate(action_rows):
            positions = [action["teacher_position"] for action in actions]
            if (positions != sorted(set(positions))
                    or any(isinstance(p, bool) or not isinstance(p, int) or p < 0
                           or p >= hidden.shape[1] or not bool(batch["attention_mask"][row, p])
                           or int(batch["labels"][row, p]) == -100 for p in positions)):
                raise ValueError("Action teacher positions must be ordered, unique, supervised text positions")
            index = torch.tensor(positions, device=hidden.device, dtype=torch.long)
            results.append(hidden[row].index_select(0, index).float().cpu().contiguous())
        return results


def _load_teacher(config, device_index=0):
    import torch
    from safetensors.torch import load_file
    from src.interaction.model import load_full_model
    from src.sft1.config import WarmupConfig
    from src.sft1.data import WarmupCollator, _processor_identity, load_processor
    from src.sft1.model import WarmupModel
    from src.sft1.modules import IMAGE_INDEX_ENCODING, POSITION_ENCODING

    if not torch.cuda.is_available():
        raise RuntimeError("Each offline SFT-I teacher worker requires a CUDA GPU; prepare_only remains CPU-only")
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    root = Path(config.warmup_path)
    recipe = _json_read(root / "warmup_recipe.json")
    interaction = _json_read(root / "interaction_config.json")
    adapter = _json_read(root / "warmup_adapter_config.json")
    if (recipe.get("format") != "sft1-warmup-v1" or recipe.get("bundle_type") != "warmup_training_components"
            or interaction.get("format") != "sft1-interaction-v1"
            or adapter.get("format") != "sft1-image-adapter-v1"
            or interaction.get("position_encoding") != POSITION_ENCODING
            or adapter.get("position_encoding") != POSITION_ENCODING
            or interaction.get("image_index_encoding") != IMAGE_INDEX_ENCODING):
        raise ValueError("Teacher requires a complete supported SFT-I warm-up export")
    allowed = {field.name for field in fields(WarmupConfig)}
    unknown = set(recipe["config"]) - allowed
    if unknown:
        raise ValueError(f"Unsupported warm-up teacher recipe fields: {sorted(unknown)}")
    warmup = WarmupConfig(**recipe["config"])
    warmup.model_path = str(root / "backbone")
    warmup.attention_backend = config.attention_backend
    warmup.activation_checkpointing = False
    warmup.execution = "incremental"
    warmup.min_image_pixels = config.min_image_pixels
    warmup.max_image_pixels = config.max_image_pixels
    warmup.validate()
    full = load_full_model(warmup.model_path, warmup.attention_backend)
    teacher = WarmupModel(full.model, full.lm_head, full.config, warmup)
    expected_interaction = {"hidden_size": full.config.hidden_size, "dim": warmup.fusion_dim,
                            "heads": warmup.fusion_heads, "ffn_dim": warmup.fusion_ffn_dim,
                            "score_dim": warmup.fusion_score_dim, "image_branch_enabled": warmup.image_branch_enabled,
                            "slots": warmup.adapter_slots, "interaction_tokens": warmup.interaction_tokens}
    expected_adapter = {"hidden_size": full.config.hidden_size, "slots": warmup.adapter_slots,
                        "dim": warmup.fusion_dim, "heads": warmup.fusion_heads,
                        "ffn_dim": warmup.fusion_ffn_dim, "layers": warmup.adapter_layers}
    if interaction.get("kwargs") != expected_interaction or adapter.get("kwargs") != expected_adapter:
        raise ValueError("Teacher module configuration disagrees with the saved warm-up recipe")
    teacher.interaction.load_state_dict(load_file(str(root / "interaction.safetensors"), device="cpu"), strict=True)
    teacher.warmup_adapter.load_state_dict(load_file(str(root / "warmup_adapter.safetensors"), device="cpu"), strict=True)
    visual = full.visual.requires_grad_(False).eval().to(device)
    teacher.requires_grad_(False).eval().to(device=device, dtype=torch.bfloat16)
    del full
    processor = load_processor(warmup)
    collator = WarmupCollator(processor, teacher.model_config.to_dict(),
                             {"processor_identity": _processor_identity(processor)})
    return teacher, visual, collator, device


def _device_batch(batch, device):
    import torch
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _device_batch(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_device_batch(value, device) for value in batch]
    return batch


def _extract_record(components, record):
    import torch
    teacher, visual, collator, device = components
    batch = _device_batch(collator([record["teacher_record"]]), device)
    with torch.inference_mode():
        inputs = batch.get("visual")
        features = None
        if inputs:
            features = visual(inputs["pixel_values"].to(dtype=next(visual.parameters()).dtype),
                              grid_thw=inputs["image_grid_thw"])
            if inputs.get("feature_indices") is not None:
                features = features.index_select(0, inputs["feature_indices"])
        return teacher_action_targets(teacher, batch, [record["actions"]], features)[0]


def _progress_entries(path):
    """Ignore only an unfinished final append left by an interrupted process."""
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    entries = {}
    for index, line in enumerate(lines):
        try:
            item = json.loads(line)
            entries[item["record_id"]] = item
        except (json.JSONDecodeError, KeyError, TypeError):
            if index != len(lines) - 1:
                raise ValueError("Teacher cache progress index is corrupt before its last record")
    return entries


def _write_records(config, records, manifest, progress_file, device_index=0):
    """One writer per shard; each durable journal entry follows an atomic tensor."""
    import torch
    from safetensors.torch import save_file
    directory = Path(config.teacher_cache_dir)
    identity, hidden_size = manifest["identity"], manifest["hidden_size"]
    components, entries = None, {}
    last_log = time.monotonic()
    try:
        with Path(progress_file).open("a", encoding="utf-8") as journal:
            for record in records:
                record_id = record["record_id"]
                if not record["actions"]:
                    targets = torch.empty((0, hidden_size), dtype=torch.float32)
                else:
                    if components is None:
                        components = _load_teacher(config, device_index=device_index)
                    targets = _extract_record(components, record).detach().float().cpu().contiguous()
                if list(targets.shape) != [len(record["actions"]), hidden_size] or not bool(torch.isfinite(targets).all()):
                    raise ValueError(f"Invalid teacher action states for {record_id}")
                relative = _relative_file(identity, record_id)
                target = directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".tmp")
                metadata = {"identity": identity, "record_id": record_id,
                            "record_key": _record_key(record), "format": CACHE_FORMAT}
                save_file({"action_states": targets}, str(temporary), metadata=metadata)
                os.replace(temporary, target)
                entry = {"record_id": record_id, "record_key": metadata["record_key"], "file": relative,
                         "shape": list(targets.shape), "size": target.stat().st_size, "sha256": _file_hash(target)}
                journal.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                entries[record_id] = entry
                if time.monotonic() - last_log >= getattr(config, "prepare_log_seconds", 5):
                    print(json.dumps({"stage": "sft2_action_teacher", "worker": device_index,
                                      "completed": len(entries), "samples": len(records)}), flush=True)
                    last_log = time.monotonic()
    finally:
        del components
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    return entries


def _shard_records(records, workers):
    """Greedy length-balanced shards, with one complete trajectory per worker."""
    shards = [[] for _ in range(min(workers, len(records)))]
    loads = [(0, index) for index in range(len(shards))]
    def cost(record):
        teacher = record["teacher_record"]
        length = max(1, teacher.get("length", len(teacher["input_ids"])))
        return length * length
    for record in sorted(records, key=lambda r: (-cost(r), r["record_id"])):
        load, index = heapq.heappop(loads)
        # Do not replicate the unused student trajectory into GPU workers.
        shards[index].append({key: record[key] for key in ("record_id", "teacher_record", "actions")})
        heapq.heappush(loads, (load + cost(record), index))
    return shards


def _cache_worker(config, records, manifest, progress_file, device_index):
    # Spawned workers do not create any distributed process group or collectives.
    import torch
    torch.set_num_threads(1)
    _write_records(config, records, manifest, progress_file, device_index)


def _run_parallel(config, shards, manifest, generation):
    """Wait for successful exits; kill and join all siblings on any failure."""
    context = multiprocessing.get_context("spawn")
    processes = []
    paths = []
    # Workers need only tensor metadata, never the full source/checkpoint index.
    metadata = {key: manifest[key] for key in ("identity", "hidden_size")}
    try:
        for device_index, shard in enumerate(shards):
            path = generation / f"progress-rank-{device_index}.jsonl"
            paths.append(path)
            process = context.Process(target=_cache_worker,
                                      args=(config, shard, metadata, str(path), device_index),
                                      name=f"sft2-teacher-gpu-{device_index}")
            process.start()
            processes.append(process)
        while True:
            failures = [(p.name, p.exitcode) for p in processes if p.exitcode not in (None, 0)]
            if failures:
                raise RuntimeError(f"Teacher GPU worker failed: {failures}; student training was not started")
            if all(p.exitcode == 0 for p in processes):
                break
            # Short joins keep failure detection responsive without busy polling.
            for process in processes:
                if process.exitcode is None:
                    process.join(timeout=0.2)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
            process.close()
    entries = {}
    for shard, path in zip(shards, paths):
        result = _progress_entries(path)
        if set(result) != {record["record_id"] for record in shard} or set(result) & set(entries):
            raise ValueError("Teacher worker journal does not exactly match its assigned shard")
        for record in shard:
            _validate_entry(config.teacher_cache_dir, manifest, record["record_id"],
                            result[record["record_id"]], record=record)
        entries.update(result)
    return entries


def prepare_teacher_cache(config, dataset=None, workers=1):
    """Coordinate resumable GPU shards, then publish one complete cache manifest."""
    import torch
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("Teacher workers must be a positive integer")
    dataset = _get_dataset(config, dataset)
    directory = Path(config.teacher_cache_dir)
    with _writer_lock(directory):
        records = _record_map(dataset)
        sources = _source_inventory(config, dataset, verify_hashes=True)
        settings = _settings(config)
        identity = _hash({"settings": settings, "sources": sources})
        hidden_size = int(_json_read(Path(config.warmup_path) / "backbone" / "config.json")["hidden_size"])
        manifest = {"format": CACHE_FORMAT, "identity": identity, "settings": settings, "sources": sources,
                    "hidden_size": hidden_size, "samples": len(records), "records": {}}
        generation = directory / "generations" / identity
        generation.mkdir(parents=True, exist_ok=True)
        progress_file = generation / "progress.jsonl"
        completed = _progress_entries(progress_file)
        shard_journals = sorted(generation.glob("progress-rank-*.jsonl"))
        for path in shard_journals:
            completed.update(_progress_entries(path))
        # A completed manifest can recover its tensors even if the progress file was lost.
        if (directory / "manifest.json").exists():
            try:
                old = _read_manifest(directory)
                if old["identity"] == identity:
                    # A newer repaired tensor may already be in a worker's
                    # journal even when a later worker failed before publishing.
                    for record_id, entry in old["records"].items():
                        completed.setdefault(record_id, entry)
            except (ValueError, KeyError, json.JSONDecodeError):
                pass
        valid = {}
        for record_id, record in records.items():
            if record_id in completed:
                try:
                    _validate_entry(directory, manifest, record_id, completed[record_id], record=record)
                    valid[record_id] = completed[record_id]
                except (ValueError, OSError, RuntimeError, KeyError):
                    pass  # Recompute a damaged/missing entry; never trust its old target.
        # Replace the journal before appending, removing any interrupted final line.
        temporary = progress_file.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for entry in valid.values():
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, progress_file)
        # Consolidation above makes changing GPU count safe after interruption.
        for path in shard_journals:
            path.write_text("", encoding="utf-8")
        pending = [record for record_id, record in records.items() if record_id not in valid]
        manifest["records"].update(valid)
        if workers == 1:
            manifest["records"].update(_write_records(config, pending, manifest, progress_file))
        else:
            empty = [record for record in pending if not record["actions"]]
            manifest["records"].update(_write_records(config, empty, manifest, progress_file))
            shards = _shard_records([record for record in pending if record["actions"]], workers)
            if shards:
                if len(shards) > torch.cuda.device_count():
                    raise RuntimeError(f"Teacher needs {len(shards)} visible GPUs, found {torch.cuda.device_count()}")
                print(json.dumps({"stage": "sft2_action_teacher", "workers": len(shards),
                                  "pending_per_worker": [len(shard) for shard in shards],
                                  "reused": len(valid)}), flush=True)
                manifest["records"].update(_run_parallel(config, shards, manifest, generation))
        if set(manifest["records"]) != set(records):
            raise ValueError("Teacher preparation did not cover all records; no manifest was published")
        # Detect an image/checkpoint rewrite during the GPU preparation run.
        if _source_inventory(config, dataset, verify_hashes=False) != [
                {k: v for k, v in item.items() if k != "sha256"} for item in sources]:
            raise ValueError("Teacher source files changed during preparation; no cache manifest was published")
        manifest["manifest_sha256"] = _hash(manifest)
        _atomic_json(directory / "manifest.json", manifest)
        print(json.dumps({"stage": "sft2_action_teacher", "complete": True, "samples": len(records),
                          "identity": identity, "reused": len(valid)}), flush=True)
        return manifest


def main(argv=None):
    from .config import DEFAULT_CONFIG, Stage2Config
    parser = argparse.ArgumentParser(description="Prepare frozen warm-up action-state targets before SFT-II training")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--workers", type=int, help="GPU inference processes; defaults to the training GPU count")
    args = parser.parse_args(argv)
    config = Stage2Config.load(args.config)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Run the offline teacher once before torchrun, never once per distributed rank")
    if config.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpu_ids))
    import torch
    workers = args.workers if args.workers is not None else (config.num_gpus or torch.cuda.device_count() or 1)
    prepare_teacher_cache(config, workers=workers)


if __name__ == "__main__":
    main()
