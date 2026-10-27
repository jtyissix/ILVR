"""CPU-only structured preflight, completed before teacher caching or NCCL."""
import argparse
import json
import os
from pathlib import Path
import uuid

from ..interaction.data import checkpoint_inventory, json_write, stable_hash
from ..sft1.prepare import file_stat
from .config import DEFAULT_CONFIG, Stage2Config, validate_local_inputs


def warmup_stat_fingerprint(config):
    root = Path(config.warmup_path)
    names = ("interaction.safetensors", "interaction_config.json", "warmup_recipe.json",
             "warmup_adapter.safetensors", "warmup_adapter_config.json")
    return stable_hash({"backbone": checkpoint_inventory(config.model_path)["fingerprint"],
                        "components": {name: file_stat(root / name) for name in names}})


def prepare(config):
    from .checkpoint import warmup_fingerprint
    from .data import Stage2Dataset, prepare_dataset
    from .train import configure_deepspeed
    validate_local_inputs(config)
    configure_deepspeed(config)
    output = Path(config.output_dir)
    if config.mode != "prepare_only" and (output / "checkpoints" / "latest").exists() and config.resume is None:
        raise ValueError("output_dir contains checkpoints; set resume or select another output_dir")
    output.mkdir(parents=True, exist_ok=True)
    if config.mode == "prepare_only" or not (Path(config.prepared_dir) / "manifest.json").is_file():
        print("Preparing SFT-II structured student and original-text teacher trajectories...", flush=True)
        prepare_dataset(config)
    else:
        print(f"Validating prepared SFT-II cache: {config.prepared_dir}", flush=True)
    dataset = Stage2Dataset(config.prepared_dir)
    dataset.validate_sources(config)
    fingerprint = warmup_fingerprint(config.warmup_path, output / "initial_warmup.json")
    receipt = {"format": "sft2-preflight-v1", "launch_id": os.environ.get("SFT2_PREFLIGHT_ID") or uuid.uuid4().hex,
               "config_hash": stable_hash(config.to_dict()), "dataset_hash": stable_hash(dataset.manifest),
               "records_stat": file_stat(Path(config.prepared_dir) / "records.jsonl"),
               "source_stats": {source["path"]: file_stat(source["path"]) for source in config.sources},
               "warmup_stat_fingerprint": warmup_stat_fingerprint(config), "warmup_fingerprint": fingerprint}
    json_write(output / "preflight.json", receipt)
    json_write(output / "train_config.json", config.to_dict())
    print(f"SFT-II CPU preflight passed: {len(dataset)} samples, {dataset.manifest['actions']} actions, "
          f"{dataset.manifest['helper_segments']} helpers.", flush=True)
    return receipt


def read_preflight(config):
    path = Path(config.output_dir) / "preflight.json"
    launch_id = os.environ.get("SFT2_PREFLIGHT_ID")
    if not launch_id or not path.is_file():
        raise RuntimeError("Missing SFT-II preflight. Use python run_sft2.py before distributed startup.")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt.get("format") != "sft2-preflight-v1" or receipt.get("launch_id") != launch_id
            or receipt.get("config_hash") != stable_hash(config.to_dict())):
        raise ValueError("SFT-II preflight belongs to another launch/config; restart with run_sft2.py")
    if (receipt["records_stat"] != file_stat(Path(config.prepared_dir) / "records.jsonl")
            or receipt["warmup_stat_fingerprint"] != warmup_stat_fingerprint(config)
            or any(file_stat(name) != stat for name, stat in receipt["source_stats"].items())):
        raise ValueError("SFT-II inputs changed after preflight; restart with run_sft2.py")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    prepare(Stage2Config.load(parser.parse_args(argv).config))


if __name__ == "__main__":
    main()
