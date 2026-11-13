"""Complete SFT-III optimizer/RNG resumes and backbone-only exports."""
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
                "prepare_workers", "prepare_log_seconds", "stage2_path"):
        values.pop(key, None)
    values["deepspeed"] = json.loads(Path(config.deepspeed).read_text(encoding="utf-8"))
    return stable_hash({"format": "sft3-relaxation-v1", "config": values, "dataset": dataset_hash,
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
    state = {"format": "sft3-relaxation-v1", "epoch": epoch, "next_window": next_window,
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
            raise FileNotFoundError(f"No complete SFT-III checkpoint or latest pointer in {path}")
        tag = latest.read_text(encoding="utf-8").strip()
        if not tag or Path(tag).name != tag or tag in {".", ".."}:
            raise ValueError("Invalid checkpoint latest pointer")
        path = path / tag
    state = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    if (state.get("format") != "sft3-relaxation-v1" or state.get("signature") != signature
            or state.get("initial_backbone_fingerprint") != source_hash or state.get("world_size") != world()):
        raise ValueError("Resume rejected: SFT-III configuration (including DeepSpeed/ZeRO stage), data, "
                         "world size, or initial backbone changed. If only the ZeRO stage changed, "
                         "reuse the existing prepared data with a new optimizer run.")
    rng_path = path / f"rng_rank{rank()}.pt"
    if not rng_path.is_file():
        raise FileNotFoundError(f"Incomplete SFT-III checkpoint: {rng_path}")
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
    # ZeRO-1 and ZeRO-2 retain complete model parameters. Clone tied storage for safetensors.
    return {name: tensor.detach().cpu().contiguous().clone() for name, tensor in module.state_dict().items()}


def export_stage3(engine, visual, processor, output, config, source_hash):
    """Export a helper-free backbone; latent scheduling still needs this runtime."""
    from ..sft1.train import primary_call
    from ..sft2.tokens import BOUNDARY_POLICY, BOUNDARY_TOKENS

    def save():
        from huggingface_hub import save_torch_state_dict
        student, directory = engine.module, Path(output)
        if any(hasattr(p, "ds_id") for p in student.parameters()):
            raise ValueError("SFT-III export supports replicated ZeRO-1/2 weights, not ZeRO-3")
        if any(not name.startswith(("decoder.", "lm_head.")) for name, _ in student.named_parameters()):
            raise ValueError("SFT-III export must contain only backbone parameters")
        backbone = directory / "backbone"
        backbone.mkdir(parents=True, exist_ok=True)
        state = {"model." + n: v for n, v in _cpu_state(student.decoder).items()}
        state.update({"lm_head." + n: v for n, v in _cpu_state(student.lm_head).items()})
        state.update({"visual." + n: v for n, v in _cpu_state(visual).items()})
        save_torch_state_dict(state, str(backbone), safe_serialization=True, max_shard_size="4GB")
        del state
        metadata = student.model_config.to_dict()
        for key in ("visual_latent_size", "interaction_latent_size", "latent_size", "interaction_training_mode",
                    "reference_fingerprint", "sft1_warmup_backbone", "sft2_helper_conditioned_backbone"):
            metadata.pop(key, None)
        # The vendored stage3 forward implements a different legacy latent loop.
        # Keep its plain backbone mode and declare our 4/3/16 runtime explicitly.
        metadata.update(stage="stage1", sft3_latent_relaxation_backbone=True,
                        architectures=["Qwen2_5_VLForConditionalGeneration"], torch_dtype="bfloat16")
        json_write(backbone / "config.json", metadata)
        processor.save_pretrained(backbone)
        json_write(directory / "stage3_recipe.json", {
            "format": "sft3-relaxation-v1", "config": config.to_dict(), "stage2_fingerprint": source_hash,
            "backbone": "backbone", "boundary_policy": BOUNDARY_POLICY, "boundary_tokens": list(BOUNDARY_TOKENS),
            "latent_boundary_ids": [metadata["latent_start_id"], metadata["latent_end_id"]],
            "sequence": "text -> <|latent_start|> -> text_latents[4] -> interaction_latents[3] -> visual_latents[16] -> <|latent_end|> -> text",
            "latent_input": "previous final-layer hidden state, without detach, for all three types",
            "requires_helper_images": False, "requires_teacher": False, "requires_sequence_plan": True,
            "standalone_hf_generation_ready": False,
            "trainable_components": ["decoder", "lm_head"], "frozen_components": ["visual"],
            "collapse": {"targets": config.selected_types, "weight": config.collapse_weight,
                         "margin": config.collapse_margin, "alpha": config.collapse_alpha,
                         "epsilon": config.collapse_epsilon, "learnable_loss_parameters": False,
                         "normalization": "global pair mean per type, then mean over selected types"},
            "inference_note": "Use the SFT-III continuous-state runtime and block layout. "
                              "Plain HF generate does not schedule the 4/3/16 latent blocks."})
    primary_call(save)


