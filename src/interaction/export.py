"""Offline CPU export from a complete DeepSpeed checkpoint; no reference loaded."""
import argparse
import json
from pathlib import Path
import torch
from .config import TrainConfig
from .data import json_write
from .model import load_full_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", help="Original CoMT checkpoint, supplies the frozen visual encoder")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    config = TrainConfig.load(args.config, {"model_path": args.model_path} if args.model_path else None)
    path = Path(args.checkpoint)
    if not (path / "complete.json").is_file():
        path = path / (path / "latest").read_text(encoding="utf-8").strip()
    state = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    from huggingface_hub import save_torch_state_dict
    from transformers import AutoProcessor
    from .checkpoint import source_fingerprint
    source = source_fingerprint(config.model_path, Path(args.output_dir).parent / "export_source.json")
    if source != state["reference_fingerprint"]:
        raise ValueError("Initial checkpoint changed; cannot reconstruct the frozen vision encoder")
    full = load_full_model(config.model_path, "sdpa")
    model_config, visual = full.config.to_dict(), full.visual
    del full
    lazy = get_fp32_state_dict_from_zero_checkpoint(str(path.parent), tag=path.name, lazy_mode=True)
    weights = {}
    for name, value in lazy.items():
        if name.startswith("fusion."):
            continue
        key = "model." + name[len("decoder."):] if name.startswith("decoder.") else name
        weights[key] = value.contiguous().to(torch.bfloat16)
    for name, value in visual.state_dict().items():
        weights["visual." + name] = value.contiguous()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_torch_state_dict(weights, str(output), safe_serialization=True, max_shard_size="4GB")
    model_config.update(stage="stage2", visual_latent_size=8, interaction_latent_size=config.interaction_steps,
                        latent_size=config.latent_steps, interaction_training_mode=config.mode, reference_fingerprint=source)
    json_write(output / "config.json", model_config)
    AutoProcessor.from_pretrained(config.model_path, local_files_only=True).save_pretrained(output)
    json_write(output / "interaction_recipe.json", config.to_dict())


if __name__ == "__main__":
    main()
