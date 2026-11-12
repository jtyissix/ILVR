"""Edit the globals below, then run: python eval/download.py (CPU only)."""
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ===================== 只修改此配置区 =====================
DATASET_DIR = "/home/fit/renju/WORK/jty/dataset/lm-eval/"
ONLY_DATASETS = ["all"]      # 例如 ['mmvp', 'hrbench', 'mathvista']；hrbench 展开为 4K + 8K
EXCLUDE_DATASETS = []        # 和 ONLY_DATASETS 必须恰好一个非空
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
HF_TOKEN = os.environ.get("HF_TOKEN")
DOWNLOAD_WORKERS = 4
EMMA_VARIANT = "full"        # full=2788；mini=400
SCIENCEQA_FULL = False       # False: 带图 test=2017；True: 全部 test=4241
PREPARE_ONLY = False         # True: 使用已有 raw 文件，完全不联网
# =========================================================

from eval.common import SCHEMA_VERSION, read_json, read_jsonl, select_benchmarks, write_json
from eval.data import normalize
from eval.registry import EXPECTED_COUNTS, source_for


def prepare_one(name, dataset_dir, *, endpoint=HF_ENDPOINT, token=HF_TOKEN, workers=4,
                emma_variant="full", scienceqa_full=False, prepare_only=False):
    import json
    from dataclasses import asdict
    source = source_for(name, emma_variant)
    root = Path(dataset_dir) / name
    raw, prepared = root / "raw", root / "prepared"
    settings = {"schema_version": SCHEMA_VERSION, "source": asdict(source),
                "emma_variant": emma_variant if name == "emma" else None,
                "scienceqa_full": scienceqa_full if name == "scienceqa" else None}
    # Round-trip tuples to the JSON representation used in saved manifests.
    settings = json.loads(json.dumps(settings))
    manifest_path = prepared / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        # Old schema/implementation metadata does not invalidate prepared data.
        previous = manifest.get("settings", {})
        if any(previous.get(key) != settings[key] for key in ("source", "emma_variant", "scienceqa_full")):
            raise ValueError(f"{name}: existing data uses another variant/revision; choose a new DATASET_DIR")
        print(f"{name}: already prepared, {manifest['count']} samples", flush=True)
        return manifest
    samples_path = prepared / "samples.jsonl"
    # The completed JSONL is atomically renamed before the manifest is written.
    # Recover this boundary without replaying image preparation or downloading.
    if samples_path.is_file():
        samples = list(read_jsonl(samples_path))
    else:
        samples = None
    if samples is None and not prepare_only:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=source.repo, repo_type="dataset", revision=source.revision,
                          allow_patterns=list(source.patterns) + ["README.md", "LICENSE*"],
                          local_dir=str(raw), endpoint=endpoint, token=token, max_workers=workers)
    elif samples is None and not raw.is_dir():
        raise FileNotFoundError(raw)
    prepared.mkdir(parents=True, exist_ok=True)
    progress_settings = prepared / "prepare_settings.json"
    if progress_settings.is_file():
        previous = read_json(progress_settings)
        if any(previous.get(key) != settings[key] for key in ("source", "emma_variant", "scienceqa_full")):
            raise ValueError(f"{name}: partial preparation uses another variant/revision; choose a new DATASET_DIR")
    write_json(progress_settings, settings)
    print(f"{name}: preparing official release", flush=True)
    if samples is None:
        samples = normalize(name, raw, prepared, emma_variant, scienceqa_full)
    expected = EXPECTED_COUNTS.get(name)
    if name == "emma" and emma_variant == "mini":
        expected = 400
    if name == "scienceqa" and scienceqa_full:
        expected = 4241
    if expected is not None and len(samples) != expected:
        raise ValueError(f"{name}: expected {expected}, got {len(samples)}; refusing an incomplete/incorrect split")
    if name == "mm_math" and len(samples) < 5700:
        raise ValueError("MM-Math release appears incomplete (<5700 QA rows)")
    tmp = prepared / "samples.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(prepared / "samples.jsonl")
    image_paths = sorted({p for s in samples for p in s["images"].values()})
    manifest = {"settings": settings, "count": len(samples), "unique_images": len(image_paths),
                "images": image_paths,
                "source_url": f"https://huggingface.co/datasets/{source.repo}/tree/{source.revision}"}
    write_json(manifest_path, manifest)  # Completion marker is written last.
    print(f"{name}: ready, {len(samples)} samples / {len(image_paths)} images", flush=True)
    return manifest


def main():
    selected = select_benchmarks(ONLY_DATASETS, EXCLUDE_DATASETS)
    failures = {}
    for name in selected:
        try:
            prepare_one(name, DATASET_DIR, endpoint=HF_ENDPOINT, token=HF_TOKEN, workers=DOWNLOAD_WORKERS,
                        emma_variant=EMMA_VARIANT, scienceqa_full=SCIENCEQA_FULL,
                        prepare_only=PREPARE_ONLY)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
            print(f"{name}: FAILED: {failures[name]}", flush=True)
    if failures:
        write_json(Path(DATASET_DIR) / "download_failures.json", failures)
        raise SystemExit("Some datasets failed. Correct the cause and rerun; completed datasets are reused.")


if __name__ == "__main__":
    main()
