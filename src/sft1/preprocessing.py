"""Bounded CPU preprocessing with deterministic streaming output and live reports."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
import json
import multiprocessing
import os
from pathlib import Path
import time

from ..interaction.data import checkpoint_inventory, file_hash, json_write
from .data import (CACHE_VERSION, IMAGE_POLICY, OverlengthTrajectoryError, _field, _iter_rows, _processor_identity,
                   _source_inventory, load_processor, prepare_sample, processor_settings,
                   validate_tokenizer_capacity)


_WORKER = None
_MISSING = object()


def exclusion_reason(sample, source):
    """Only explicitly configured exclusions; malformed selected records fail."""
    if not isinstance(sample, dict):
        return None
    if (_field(sample, source["plan_field"], _MISSING) is _MISSING
            and source.get("skip_missing_plan", False)):
        return f"missing rewritten field {source['plan_field']}"
    if source.get("skip_empty_answer", False):
        field = source.get("answer_field", "original_final_answer")
        value = _field(sample, field, _MISSING)
        if value is None or isinstance(value, str) and not value.strip():
            return f"empty authoritative answer field {field}"
    return None


def _initialize_worker(config, model_config):
    global _WORKER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    processor = load_processor(config)
    validate_tokenizer_capacity(processor, model_config)
    _WORKER = (processor, config, model_config)


def _process_one(task, processor, config, model_config):
    source, line, raw = task
    result = {"source": source["name"], "line": line}
    try:
        sample = json.loads(raw) if isinstance(raw, str) else raw
        reason = exclusion_reason(sample, source)
        if reason:
            return {**result, "status": "skipped", "reason": reason}
        record = prepare_sample(sample, source, processor, model_config, config.max_seq_length,
                                config.interaction_tokens,
                                config.shrink_overlong_images)
        record.update(source=source["name"], source_line=line)
        stats = {}
        for image in record["images"]:
            stat = Path(image["path"]).stat()
            stats[image["path"]] = [stat.st_size, stat.st_mtime_ns]
        return {**result, "status": "ok", "record": record, "image_stats": stats}
    except OverlengthTrajectoryError as exc:
        if config.skip_overlong_samples:
            return {**result, "status": "skipped", "reason": "overlength trajectory",
                    "length": exc.length, "max_seq_length": exc.limit}
        return {**result, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        return {**result, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _worker_task(task):
    return _process_one(task, *_WORKER)


def resize_report(record, max_seq_length):
    """Keep per-occurrence geometry and token accounting, without full token arrays."""
    keys = ("path", "kind", "image_id", "original_size", "initial_processed_size", "processed_size",
            "original_visual_tokens", "visual_tokens", "max_pixels")
    return {"source": record["source"], "line": record["source_line"],
            "original_length": record["original_length"], "length": record["length"],
            "max_seq_length": max_seq_length, "non_visual_tokens": record["non_visual_tokens"],
            "original_visual_tokens": record["original_visual_tokens"], "visual_tokens": record["visual_tokens"],
            "images": [{key: image[key] for key in keys} for image in record["images"]]}


def _tasks(sources):
    for source in sources:
        path = Path(source["path"])
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8-sig") as handle:
                for line, raw in enumerate(handle, 1):
                    if raw.strip():
                        yield source, line, raw
        else:
            for line, sample in _iter_rows(source):
                yield source, line, sample


def _row_count(source):
    if Path(source["path"]).suffix.lower() == ".jsonl":
        with Path(source["path"]).open(encoding="utf-8-sig") as handle:
            return sum(bool(line.strip()) for line in handle)
    return sum(1 for _ in _iter_rows(source))


def _parallel_results(tasks, executor, capacity, interval):
    """Keep both pending work and out-of-order results within one fixed window."""
    pending, ready = {}, {}
    iterator = iter(tasks)
    next_submit = next_emit = 0
    exhausted = False
    while pending or ready or not exhausted:
        while not exhausted and len(pending) + len(ready) < capacity:
            try:
                task = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending[executor.submit(_worker_task, task)] = next_submit
            next_submit += 1
        while next_emit in ready:
            yield ready.pop(next_emit)
            next_emit += 1
        if not pending:
            continue
        done, _ = wait(pending, timeout=interval, return_when=FIRST_COMPLETED)
        if not done:
            yield None  # Heartbeat even while every worker is decoding a large image.
        for future in done:
            ready[pending.pop(future)] = future.result()


def prepare_dataset(config):
    output = Path(config.prepared_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    counts = {source["name"]: {"total": 0, "processed": 0, "accepted": 0,
                               "skipped": 0, "errors": 0, "image_budget_reduced": 0}
              for source in config.sources}
    errors, lengths, image_stats = [], [], {}
    processed = skipped = reduced = helpers = total = 0
    current = None
    last_report = -float("inf")
    records_handle = None

    def report(status, force=False):
        nonlocal last_report
        now = time.monotonic()
        if not force and now - last_report < config.prepare_log_seconds:
            return
        elapsed = now - started
        speed = processed / elapsed if elapsed else 0
        remaining = (total - processed) / speed if speed and total else None
        progress = {"status": status, "processed": processed, "total": total,
                    "accepted": len(lengths), "skipped": skipped, "errors": len(errors),
                    "image_budget_reduced": reduced, "workers": config.prepare_workers,
                    "elapsed_seconds": round(elapsed, 1), "records_per_second": round(speed, 3),
                    "eta_seconds": round(remaining, 1) if remaining is not None else None,
                    "last_record": current, "sources": counts}
        if records_handle is not None:
            records_handle.flush()
        json_write(output / "progress.json", progress)
        json_write(output / "validation_errors.json", errors)
        eta = f"{remaining / 60:.1f}m" if remaining is not None else "--"
        location = f" last={current['source']}:{current['line']}" if current else ""
        print(f"[prepare:{status}] {processed}/{total} ok={len(lengths)} skipped={skipped} "
              f"errors={len(errors)} resized={reduced} workers={config.prepare_workers} "
              f"elapsed={elapsed / 60:.1f}m {speed:.2f} rows/s ETA={eta}{location}", flush=True)
        last_report = now

    report("indexing", True)
    try:
        for source in config.sources:
            print(f"[prepare] Counting {source['name']}: {source['path']}", flush=True)
            counts[source["name"]]["total"] = _row_count(source)
            total += counts[source["name"]]["total"]
            report("indexing", True)
        print("[prepare] Verifying source files, checkpoint metadata and tokenizer...", flush=True)
        source_info = _source_inventory(config.sources)
        inventory = checkpoint_inventory(config.model_path)
        processor = load_processor(config)
        validate_tokenizer_capacity(processor, inventory["config"])
        report("processing", True)
        partial = output / "records.partial.jsonl"
        with ExitStack() as stack:
            records_handle = stack.enter_context(partial.open("w", encoding="utf-8", newline="\n"))
            error_log = stack.enter_context((output / "validation_errors.jsonl").open("w", encoding="utf-8"))
            skip_log = stack.enter_context((output / "skipped_records.jsonl").open("w", encoding="utf-8"))
            resize_log = stack.enter_context((output / "resized_records.jsonl").open("w", encoding="utf-8"))
            tasks = _tasks(source_info)
            if config.prepare_workers == 1:
                results = (_process_one(task, processor, config, inventory["config"]) for task in tasks)
            else:
                executor = stack.enter_context(ProcessPoolExecutor(
                    max_workers=config.prepare_workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_worker, initargs=(config, inventory["config"])))
                results = _parallel_results(tasks, executor, config.prepare_workers * 2, config.prepare_log_seconds)
            for result in results:
                if result is None:
                    report("processing")
                    continue
                processed += 1
                current = {key: result[key] for key in ("source", "line")}
                source_count = counts[result["source"]]
                source_count["processed"] += 1
                if result["status"] == "ok":
                    record = result["record"]
                    records_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    lengths.append(record["length"])
                    helpers += len(record["segments"])
                    image_stats.update(result["image_stats"])
                    source_count["accepted"] += 1
                    adjusted = int(record["image_budget_reduced"])
                    reduced += adjusted
                    source_count["image_budget_reduced"] += adjusted
                    if adjusted:
                        resize_log.write(json.dumps(resize_report(record, config.max_seq_length), ensure_ascii=False) + "\n")
                        resize_log.flush()
                        if source_count["image_budget_reduced"] == 1:
                            print(f"[prepare:resize] {result['source']}:{result['line']} "
                                  f"length={record['original_length']}->{record['length']} "
                                  f"visual_tokens={record['original_visual_tokens']}->{record['visual_tokens']}; "
                                  "per-image sizes: resized_records.jsonl", flush=True)
                elif result["status"] == "skipped":
                    skipped += 1
                    source_count["skipped"] += 1
                    skip_log.write(json.dumps(result, ensure_ascii=False) + "\n")
                    skip_log.flush()
                else:
                    errors.append({**current, "error": result["error"]})
                    source_count["errors"] += 1
                    error_log.write(json.dumps(errors[-1], ensure_ascii=False) + "\n")
                    error_log.flush()
                    if len(errors) <= 5:
                        print(f"[prepare:error] {result['source']}:{result['line']} {result['error']}", flush=True)
                report("processing")
                if source_count["processed"] == source_count["total"]:
                    report("processing", True)
            for name, count in counts.items():
                if not count["accepted"]:
                    error = {"source": name, "error": "Source produced no valid training records"}
                    errors.append(error)
                    error_log.write(json.dumps(error) + "\n")
            report("processing" if errors else "writing", True)
            records_handle = None
        if errors or not lengths:
            raise ValueError(f"Preparation failed: {len(errors)} errors; see {output / 'validation_errors.json'}. "
                             "Partial results and skipped records were retained for inspection.")
        cache = output / "records.jsonl"
        partial.replace(cache)
        lengths.sort()
        manifest = {"version": CACHE_VERSION, "stage": "sft1_warmup", "source_info": source_info,
                    "source_counts": {name: c["accepted"] for name, c in counts.items()},
                    "source_stats": counts, "skipped_samples": skipped, "image_budget_reduced": reduced,
                    "records_sha256": file_hash(cache), "checkpoint_metadata": inventory["metadata"],
                    "image_stats": image_stats, "process_settings": processor_settings(config),
                    "processor_identity": _processor_identity(processor), "image_policy": IMAGE_POLICY,
                    "shrink_overlong_images": config.shrink_overlong_images,
                    "skip_overlong_samples": config.skip_overlong_samples,
                    "max_seq_length": config.max_seq_length, "interaction_tokens": config.interaction_tokens,
                    "samples": len(lengths), "helper_segments": helpers,
                    "length_p50": lengths[len(lengths) // 2],
                    "length_p95": lengths[min(len(lengths) - 1, int(len(lengths) * .95))],
                    "length_max": max(lengths)}
        json_write(output / "manifest.json", manifest)
        report("complete", True)
        return manifest
    except BaseException as exc:
        records_handle = None
        if not errors:
            errors.append({"stage": "prepare", "error": f"{type(exc).__name__}: {exc}"})
        report("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", True)
        raise
