"""Single-process preflight before NCCL starts; invoked by run_sft1.py."""
import argparse
import json
import os
from pathlib import Path
import uuid

from ..interaction.data import checkpoint_inventory, json_write, stable_hash
from .checkpoint import source_fingerprint
from .config import DEFAULT_CONFIG, WarmupConfig, validate_local_inputs


def file_stat(path):
    value = Path(path).stat()
    return [value.st_size, value.st_mtime_ns]


def prepare(config):
    from .data import WarmupDataset, prepare_dataset
    from .train import configure_deepspeed
    validate_local_inputs(config)
    configure_deepspeed(config)
    output = Path(config.output_dir)
    if config.mode != "prepare_only" and (output / "checkpoints" / "latest").exists() and config.resume is None:
        raise ValueError("output_dir contains checkpoints; set resume in warmup.json or select another output_dir")
    output.mkdir(parents=True, exist_ok=True)
    if config.mode == "prepare_only" or not (Path(config.prepared_dir) / "manifest.json").is_file():
        print("Preparing and validating all SFT-I sources before distributed startup...", flush=True)
        prepare_dataset(config)
    else:
        print(f"Reusing prepared SFT-I cache: {config.prepared_dir} (no data preprocessing); validating cached files...",
              flush=True)
    dataset = WarmupDataset(config.prepared_dir)
    dataset.validate_sources(config)
    print("Verifying the initial local checkpoint fingerprint...", flush=True)
    source = source_fingerprint(config.model_path, output / "initial_backbone.json")
    receipt = {
        "format": "sft1-preflight-v1",
        "launch_id": os.environ.get("SFT1_PREFLIGHT_ID") or uuid.uuid4().hex,
        "config_hash": stable_hash(config.to_dict()),
        "dataset_hash": stable_hash(dataset.manifest),
        "records_stat": file_stat(Path(config.prepared_dir) / "records.jsonl"),
        "source_stats": {source["path"]: file_stat(source["path"]) for source in config.sources},
        "checkpoint_stat_fingerprint": checkpoint_inventory(config.model_path)["fingerprint"],
        "initial_backbone_fingerprint": source,
    }
    json_write(output / "preflight.json", receipt)
    json_write(output / "train_config.json", config.to_dict())
    print(f"SFT-I preflight passed: {len(dataset)} samples, {dataset.manifest['helper_segments']} helpers.", flush=True)
    return receipt


def read_preflight(config):
    """Workers only do cheap consistency checks; no source/image scan here."""
    path = Path(config.output_dir) / "preflight.json"
    launch_id = os.environ.get("SFT1_PREFLIGHT_ID")
    if not launch_id or not path.is_file():
        raise RuntimeError("Missing SFT-I preflight. Use python run_sft1.py so full validation finishes before NCCL starts.")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt.get("format") != "sft1-preflight-v1" or receipt.get("launch_id") != launch_id
            or receipt.get("config_hash") != stable_hash(config.to_dict())):
        raise ValueError("SFT-I preflight belongs to another launch/config; restart with python run_sft1.py")
    if (receipt["records_stat"] != file_stat(Path(config.prepared_dir) / "records.jsonl")
            or receipt["checkpoint_stat_fingerprint"] != checkpoint_inventory(config.model_path)["fingerprint"]
            or any(file_stat(name) != stat for name, stat in receipt["source_stats"].items())):
        raise ValueError("SFT-I inputs changed after preflight; restart with python run_sft1.py")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    prepare(WarmupConfig.load(args.config))


if __name__ == "__main__":
    main()
