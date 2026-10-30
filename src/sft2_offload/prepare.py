"""CPU preflight for the opt-in offload launcher, reusing the SFT-II cache format."""
import argparse
import os
from pathlib import Path
import uuid

from ..interaction.data import json_write, stable_hash
from ..sft1.prepare import file_stat
from ..sft2.checkpoint import warmup_fingerprint
from ..sft2.data import Stage2Dataset, prepare_dataset
from ..sft2.prepare import read_preflight as _read_preflight, warmup_stat_fingerprint
from .config import DEFAULT_CONFIG, Stage2Config, configure_deepspeed, validate_local_inputs
from .runtime import check_runtime


def read_preflight(config):
    try:
        return _read_preflight(config)
    except (RuntimeError, ValueError) as exc:
        raise type(exc)(str(exc).replace("run_sft2.py", "run_sft2_offload.py")) from exc


def prepare(config):
    validate_local_inputs(config)
    configure_deepspeed(config)
    output = Path(config.output_dir)
    if config.mode != "prepare_only" and (output / "checkpoints" / "latest").exists() and config.resume is None:
        raise ValueError("output_dir contains checkpoints; set resume or select another output_dir")
    output.mkdir(parents=True, exist_ok=True)
    # Existing records/manifest remain untouched, including in prepare_only mode.
    # Changed data/image/tokenization settings fail validation rather than silently
    # overwriting a cache that another original SFT-II job may be reading.
    if not (Path(config.prepared_dir) / "manifest.json").is_file():
        print("Preparing SFT-II structured student and original-text teacher trajectories...", flush=True)
        prepare_dataset(config)
    else:
        print(f"Validating prepared SFT-II cache: {config.prepared_dir}", flush=True)
    dataset = Stage2Dataset(config.prepared_dir)
    dataset.validate_sources(config)
    if config.mode != "prepare_only":
        check_runtime()
    fingerprint = warmup_fingerprint(config.warmup_path, output / "initial_warmup.json")
    receipt = {"format": "sft2-preflight-v1", "launch_id": os.environ.get("SFT2_PREFLIGHT_ID") or uuid.uuid4().hex,
               "config_hash": stable_hash(config.to_dict()), "dataset_hash": stable_hash(dataset.manifest),
               "records_stat": file_stat(Path(config.prepared_dir) / "records.jsonl"),
               "source_stats": {source["path"]: file_stat(source["path"]) for source in config.sources},
               "warmup_stat_fingerprint": warmup_stat_fingerprint(config), "warmup_fingerprint": fingerprint}
    json_write(output / "preflight.json", receipt)
    json_write(output / "train_config.json", config.to_dict())
    print(f"SFT-II offload CPU preflight passed: {len(dataset)} samples, "
          f"{dataset.manifest['actions']} actions, {dataset.manifest['helper_segments']} helpers.", flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    prepare(Stage2Config.load(parser.parse_args(argv).config))


if __name__ == "__main__":
    main()
