"""Complete SFT-I resume checkpoints and separable warm-up components."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..interaction.data import checkpoint_inventory, json_write, stable_hash
from ..interaction.distributed import rank, world


def source_fingerprint(path, cache_file):
    current = checkpoint_inventory(path)
    cache_file = Path(cache_file)
    if cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("stat_fingerprint") == current["fingerprint"]:
            return cached["content_fingerprint"]
    full = checkpoint_inventory(path, hash_weights=True)
    digest = stable_hash({"metadata": full["metadata"],
                          "weights": {name: value["sha256"] for name, value in full["files"].items()}})
    json_write(cache_file, {"stat_fingerprint": current["fingerprint"], "content_fingerprint": digest})
    return digest


def resume_signature(config, dataset_hash, source_hash):
    values = config.to_dict()
    # Runtime/logging/stop controls may change; training semantics and world size may not.
    for key in ("output_dir", "resume", "max_steps", "smoke_steps", "mode", "save_steps", "log_steps",
                "num_workers", "prefetch_factor", "model_path", "prepared_dir", "gpu_ids", "num_gpus",
                "prepare_workers", "prepare_log_seconds"):
        values.pop(key, None)
    values["deepspeed"] = json.loads(Path(config.deepspeed).read_text(encoding="utf-8"))
    return stable_hash({"format": "sft1-warmup-v1", "config": values, "dataset": dataset_hash,
                        "initial_backbone": source_hash, "world_size": world()})


def random_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def save_training(engine, root, epoch, next_window, signature, source_hash):
    root = Path(root)
    tag = f"global_step{engine.global_steps:08d}"
    state = {"format": "sft1-warmup-v1", "epoch": epoch, "next_window": next_window,
             "signature": signature, "initial_backbone_fingerprint": source_hash,
             "world_size": world(), "optimizer_steps": engine.global_steps}
    # All ranks save optimizer shards and their own RNG state at a full optimizer boundary.
    engine.save_checkpoint(str(root), tag=tag, client_state=state, save_latest=False)
    torch.save(random_state(), root / tag / f"rng_rank{rank()}.pt")
    if dist.is_initialized():
        dist.barrier()
    if rank() == 0:
        json_write(root / tag / "complete.json", state)
        temporary = root / "latest.tmp"
        temporary.write_text(tag, encoding="utf-8")
        temporary.replace(root / "latest")
    if dist.is_initialized():
        dist.barrier()
    return root / tag


def load_training(engine, path, signature, source_hash):
    path = Path(path)
    if not (path / "complete.json").is_file():
        latest = path / "latest"
        if not latest.is_file():
            raise FileNotFoundError(f"No complete SFT-I checkpoint or latest pointer in {path}")
        tag = latest.read_text(encoding="utf-8").strip()
        if not tag or Path(tag).name != tag:
            raise ValueError("Invalid checkpoint latest pointer")
        path = path / tag
    state = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    if (state.get("format") != "sft1-warmup-v1" or state.get("signature") != signature
            or state.get("initial_backbone_fingerprint") != source_hash or state.get("world_size") != world()):
        raise ValueError("Resume rejected: SFT-I configuration, data, world size, or initial backbone changed")
    rng_path = path / f"rng_rank{rank()}.pt"
    if not rng_path.is_file():
        raise FileNotFoundError(f"Incomplete SFT-I checkpoint: {rng_path}")
    loaded, restored = engine.load_checkpoint(str(path.parent), tag=path.name, load_module_strict=True,
                                              load_optimizer_states=True, load_lr_scheduler_states=True)
    if (loaded is None or restored is None
            or any(restored.get(key) != value for key, value in state.items())
            or engine.global_steps != state["optimizer_steps"]):
        raise ValueError("Incomplete or inconsistent DeepSpeed checkpoint")
    # Load only this job's trusted training checkpoint, never an arbitrary downloaded pickle.
    restore_random_state(torch.load(rng_path, map_location="cpu", weights_only=False))
    return state["epoch"], state["next_window"]


def _cpu_state(module):
    # ZeRO-1 retains complete model parameters. Clone tied storage for safetensors.
    return {name: tensor.detach().cpu().contiguous().clone() for name, tensor in module.state_dict().items()}


def export_warmup(engine, visual, processor, output, config, source_hash):
    """Export all components. This is an SFT-I bundle, not a latent inference model."""
    result = [None]
    if rank() == 0:
        try:
            from huggingface_hub import save_torch_state_dict
            from safetensors.torch import save_file
            from .modules import IMAGE_INDEX_ENCODING, POSITION_ENCODING
            student = engine.module
            if any(hasattr(parameter, "ds_id") for parameter in student.parameters()):
                raise ValueError("SFT-I export supports replicated ZeRO-1 weights, not ZeRO-3 partitions")
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            backbone = output / "backbone"
            backbone.mkdir(exist_ok=True)
            state = {"model." + name: value for name, value in _cpu_state(student.decoder).items()}
            state.update({"lm_head." + name: value for name, value in _cpu_state(student.lm_head).items()})
            state.update({"visual." + name: value for name, value in _cpu_state(visual).items()})
            save_torch_state_dict(state, str(backbone), safe_serialization=True, max_shard_size="4GB")
            del state
            model_config = student.model_config.to_dict()
            # Do not inherit the old 8+1 latent generation recipe from a source export.
            for key in ("visual_latent_size", "interaction_latent_size", "latent_size", "interaction_training_mode",
                        "reference_fingerprint"):
                model_config.pop(key, None)
            model_config.update({"stage": "stage1", "sft1_warmup_backbone": True,
                                 "architectures": ["Qwen2_5_VLForConditionalGeneration"], "torch_dtype": "bfloat16"})
            json_write(backbone / "config.json", model_config)
            processor.save_pretrained(backbone)
            save_file(_cpu_state(student.interaction), str(output / "interaction.safetensors"))
            save_file(_cpu_state(student.warmup_adapter), str(output / "warmup_adapter.safetensors"))
            hidden_size = student.model_config.hidden_size
            json_write(output / "interaction_config.json", {
                "format": "sft1-interaction-v1", "class": "src.sft1.modules.InteractionCore",
                "kwargs": {"hidden_size": hidden_size, "dim": config.fusion_dim,
                           "heads": config.fusion_heads, "ffn_dim": config.fusion_ffn_dim,
                           "score_dim": config.fusion_score_dim, "image_branch_enabled": config.image_branch_enabled,
                           "slots": config.adapter_slots, "interaction_tokens": config.interaction_tokens},
                "slots": config.adapter_slots, "position_encoding": POSITION_ENCODING,
                "image_index_encoding": IMAGE_INDEX_ENCODING, "shared_input_projection": "visual_projection",
                "input_interface": f"[batch,{config.adapter_slots},hidden_size] -> shared W_Z -> ordered concatenation MLP",
                "output_interface": f"[batch,{config.interaction_tokens},hidden_size] ordered interaction embeddings",
                "state_dict_prefix": "", "file": "interaction.safetensors"})
            json_write(output / "warmup_adapter_config.json", {
                "format": "sft1-image-adapter-v1", "class": "src.sft1.modules.WarmupImageAdapter",
                "kwargs": {"hidden_size": hidden_size, "slots": config.adapter_slots,
                           "dim": config.fusion_dim, "heads": config.fusion_heads,
                           "ffn_dim": config.fusion_ffn_dim, "layers": config.adapter_layers},
                "position_encoding": POSITION_ENCODING, "state_dict_prefix": "",
                "file": "warmup_adapter.safetensors"})
            json_write(output / "warmup_recipe.json", {
                "format": "sft1-warmup-v1", "config": config.to_dict(),
                "initial_backbone_fingerprint": source_hash,
                "bundle_type": "warmup_training_components", "latent_inference_ready": False,
                "backbone": "backbone", "interaction": "interaction.safetensors",
                "warmup_adapter": "warmup_adapter.safetensors",
                "transfer": f"Remove warmup_adapter; feed {config.adapter_slots} autoregressive latent vectors into interaction.visual_projection (W_Z). "
                            "Keep backbone and interaction parameters; continue training for latent alignment."})
            result[0] = {"ok": True}
        except Exception as exc:
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise RuntimeError(f"SFT-I export failed: {result[0]['error']}")
