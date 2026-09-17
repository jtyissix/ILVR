"""Complete distributed checkpoints and standalone HF inference exports."""
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .data import checkpoint_inventory, json_write, stable_hash
from .distributed import rank, world


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
    for key in ("output_dir", "resume", "max_steps", "profile_steps", "save_steps", "log_steps", "num_workers",
                "prefetch_factor", "model_path", "reference_path", "data_path", "image_root", "prepared_dir"):
        values.pop(key, None)
    # Include resolved DeepSpeed settings, not only its potentially mutable pathname.
    values["deepspeed"] = json.loads(Path(config.deepspeed).read_text(encoding="utf-8"))
    return stable_hash({"config": values, "dataset": dataset_hash, "source": source_hash, "world_size": world()})


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
    state = {"epoch": epoch, "next_window": next_window, "signature": signature,
             "reference_fingerprint": source_hash, "world_size": world(), "optimizer_steps": engine.global_steps}
    # Every rank must enter DeepSpeed saving, including ZeRO-2.
    engine.save_checkpoint(str(root), tag=tag, client_state=state, save_latest=False)
    torch.save(random_state(), root / tag / f"rng_rank{rank()}.pt")
    if dist.is_initialized():
        dist.barrier()
    if rank() == 0:
        json_write(root / tag / "complete.json", state)
        (root / "latest").write_text(tag, encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
    return root / tag


def load_training(engine, path, signature, source_hash):
    path = Path(path)
    if not (path / "complete.json").is_file():
        latest = path / "latest"
        if not latest.is_file():
            raise FileNotFoundError(f"No complete checkpoint or latest pointer in {path}")
        path = path / latest.read_text(encoding="utf-8").strip()
    state = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    if state["signature"] != signature or state["reference_fingerprint"] != source_hash:
        raise ValueError("Resume rejected: experiment, data, world size, or frozen reference changed")
    loaded, restored = engine.load_checkpoint(str(path.parent), tag=path.name, load_module_strict=True,
                                              load_optimizer_states=True, load_lr_scheduler_states=True)
    # DeepSpeed strips reserved fields such as global_steps from returned client_state.
    if (loaded is None or restored is None
            or any(restored.get(key) != value for key, value in state.items())
            or engine.global_steps != state["optimizer_steps"]):
        raise ValueError("Incomplete or inconsistent DeepSpeed checkpoint")
    # Only load RNG pickle from checkpoints produced by this training program.
    restore_random_state(torch.load(path / f"rng_rank{rank()}.pt", map_location="cpu", weights_only=False))
    return state["epoch"], state["next_window"]


def export_inference(engine, visual, processor, output, config, source_hash):
    """Gather parameters only for export; reference/fusion are never exported."""
    from contextlib import nullcontext
    from huggingface_hub import save_torch_state_dict
    student = engine.module
    state = {} if rank() == 0 else None
    for name, param in student.named_parameters():
        if name.startswith("fusion."):
            continue
        if hasattr(param, "ds_id"):
            import deepspeed
            context = deepspeed.zero.GatheredParameters([param], modifier_rank=None)
        else:
            context = nullcontext()
        with context:
            if rank() == 0:
                target = "model." + name[len("decoder."):] if name.startswith("decoder.") else name
                state[target] = param.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().clone()
    if rank() == 0:
        for name, value in visual.state_dict().items():
            state["visual." + name] = value.detach().cpu().contiguous()
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        save_torch_state_dict(state, str(output), safe_serialization=True, max_shard_size="4GB")
        model_config = student.model_config.to_dict()
        model_config.update({"stage": "stage2", "visual_latent_size": config.visual_latent_size,
                             "interaction_latent_size": config.interaction_steps, "latent_size": config.latent_steps,
                             "interaction_training_mode": config.mode, "reference_fingerprint": source_hash,
                             "architectures": ["Qwen2_5_VLForConditionalGeneration"], "torch_dtype": "bfloat16"})
        json_write(output / "config.json", model_config)
        processor.save_pretrained(output)
        json_write(output / "interaction_recipe.json", config.to_dict())
    if dist.is_initialized():
        dist.barrier()
