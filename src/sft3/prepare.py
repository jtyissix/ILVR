"""CPU preflight: verify SFT-II student cache and backbone, with no teacher run."""
import argparse
import json
import os
from pathlib import Path
import uuid

from ..interaction.data import checkpoint_inventory, json_write, stable_hash
from ..sft1.prepare import file_stat
from .config import DEFAULT_CONFIG, Stage3Config, validate_local_inputs


def input_stats(config):
    root = Path(config.prepared_dir)
    return {"records": file_stat(root / "records.jsonl"), "manifest": file_stat(root / "manifest.json"),
            "recipe": file_stat(Path(config.stage2_path) / "stage2_recipe.json"),
            "backbone": checkpoint_inventory(config.model_path)["fingerprint"]}


def prepare(config):
    from .checkpoint import source_fingerprint
    from .data import Stage3Dataset, load_processor
    from .train import configure_deepspeed
    validate_local_inputs(config)
    configure_deepspeed(config)
    output = Path(config.output_dir)
    if config.mode != "prepare_only" and (output / "checkpoints" / "latest").exists() and config.resume is None:
        raise ValueError("output_dir contains checkpoints; set resume or select another output_dir")
    output.mkdir(parents=True, exist_ok=True)
    processor = load_processor(config)
    dataset = Stage3Dataset(config.prepared_dir)
    dataset.validate(config, processor)
    fingerprint = source_fingerprint(config.model_path, output / "initial_stage2.json")
    receipt = {"format": "sft3-preflight-v1", "launch_id": os.environ.get("SFT3_PREFLIGHT_ID") or uuid.uuid4().hex,
               "config_hash": stable_hash(config.to_dict()), "dataset_hash": stable_hash(dataset.manifest),
               "input_stats": input_stats(config), "stage2_fingerprint": fingerprint}
    json_write(output / "preflight.json", receipt)
    json_write(output / "train_config.json", config.to_dict())
    print(f"SFT-III CPU preflight passed: {len(dataset)} samples; question images only; "
          f"collapse targets={config.selected_types}, coefficient={config.collapse_weight}.", flush=True)
    return receipt


def read_preflight(config):
    path = Path(config.output_dir) / "preflight.json"
    launch_id = os.environ.get("SFT3_PREFLIGHT_ID")
    if not launch_id or not path.is_file():
        raise RuntimeError("Missing SFT-III preflight. Use python run_sft3.py before distributed startup.")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt.get("format") != "sft3-preflight-v1" or receipt.get("launch_id") != launch_id
            or receipt.get("config_hash") != stable_hash(config.to_dict())
            or receipt.get("input_stats") != input_stats(config)):
        raise ValueError("SFT-III inputs/config changed after preflight; restart with run_sft3.py")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    prepare(Stage3Config.load(parser.parse_args(argv).config))


if __name__ == "__main__":
    main()
