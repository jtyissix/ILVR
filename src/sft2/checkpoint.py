"""Complete SFT-II resumes and explicitly helper-conditioned component exports."""
import json
from pathlib import Path

import torch
import torch.distributed as dist

from ..interaction.data import checkpoint_inventory, file_hash, json_write, stable_hash
from ..interaction.distributed import rank, world
from ..sft1.checkpoint import random_state, restore_random_state, source_fingerprint, _cpu_state
from .tokens import BOUNDARY_POLICY, BOUNDARY_TOKENS


FORMAT = "sft2-helper-conditioned-boundaries-v2"


def warmup_fingerprint(warmup_path, cache_file):
    """Fingerprint every component needed by the student or frozen stage-I teacher."""
    root, cache_file = Path(warmup_path), Path(cache_file)
    files = [root / name for name in ("interaction.safetensors", "interaction_config.json",
             "warmup_adapter.safetensors", "warmup_adapter_config.json", "warmup_recipe.json")]
    inventory = checkpoint_inventory(root / "backbone")
    stats = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in files}
    stat_hash = stable_hash({"backbone": inventory["fingerprint"], "components": stats})
    if cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("stat_fingerprint") == stat_hash:
            return cached["content_fingerprint"]
    backbone = source_fingerprint(root / "backbone", cache_file.with_name(cache_file.stem + "_backbone.json"))
    digest = stable_hash({"backbone": backbone, "components": {p.name: file_hash(p) for p in files}})
    json_write(cache_file, {"stat_fingerprint": stat_hash, "content_fingerprint": digest})
    return digest


def resume_signature(config, dataset_hash, warmup_hash, teacher_hash):
    values = config.to_dict()
    for key in ("output_dir", "resume", "max_steps", "smoke_steps", "mode", "save_steps", "log_steps",
                "num_workers", "prefetch_factor", "prepared_dir", "gpu_ids", "num_gpus",
                "prepare_workers", "prepare_log_seconds", "warmup_path", "teacher_cache_dir"):
        values.pop(key, None)
    values["deepspeed"] = json.loads(Path(config.deepspeed).read_text(encoding="utf-8"))
    return stable_hash({"format": FORMAT, "config": values, "dataset": dataset_hash,
                        "warmup": warmup_hash, "teacher": teacher_hash, "world_size": world()})


def save_training(engine, root, epoch, next_window, signature, warmup_hash, teacher_hash):
    root = Path(root)
    tag = f"global_step{engine.global_steps:08d}"
    state = {"format": FORMAT, "epoch": epoch, "next_window": next_window, "signature": signature,
             "warmup_fingerprint": warmup_hash, "teacher_fingerprint": teacher_hash,
             "world_size": world(), "optimizer_steps": engine.global_steps}
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


def load_training(engine, path, signature, warmup_hash, teacher_hash):
    path = Path(path)
    if not (path / "complete.json").is_file():
        latest = path / "latest"
        if not latest.is_file():
            raise FileNotFoundError(f"No complete SFT-II checkpoint or latest pointer in {path}")
        tag = latest.read_text(encoding="utf-8").strip()
        if not tag or Path(tag).name != tag or tag in {".", ".."}:
            raise ValueError("Invalid checkpoint latest pointer")
        path = path / tag
    state = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    if (state.get("format") != FORMAT or state.get("signature") != signature
            or state.get("warmup_fingerprint") != warmup_hash
            or state.get("teacher_fingerprint") != teacher_hash or state.get("world_size") != world()):
        raise ValueError("Resume rejected: SFT-II data, warmup/teacher weights, training semantics, "
                         "DeepSpeed configuration or world size changed")
    rng_path = path / f"rng_rank{rank()}.pt"
    if not rng_path.is_file():
        raise FileNotFoundError(f"Incomplete SFT-II checkpoint: {rng_path}")
    loaded, restored = engine.load_checkpoint(str(path.parent), tag=path.name, load_module_strict=True,
                                              load_optimizer_states=True, load_lr_scheduler_states=True)
    if (loaded is None or restored is None or any(restored.get(k) != v for k, v in state.items())
            or engine.global_steps != state["optimizer_steps"]):
        raise ValueError("Incomplete or inconsistent DeepSpeed checkpoint")
    restore_random_state(torch.load(rng_path, map_location="cpu", weights_only=False))
    return state["epoch"], state["next_window"]


def export_stage2(engine, visual, processor, output, config, warmup_hash, teacher_hash):
    """Save all trained components; never advertise a helper-free HF generate model."""
    result = [None]
    if rank() == 0:
        try:
            from huggingface_hub import save_torch_state_dict
            from safetensors.torch import save_file
            student, output = engine.module, Path(output)
            if any(hasattr(p, "ds_id") for p in student.parameters()):
                raise ValueError("SFT-II export supports replicated ZeRO-1/2 weights, not ZeRO-3")
            output.mkdir(parents=True, exist_ok=True)
            backbone = output / "backbone"
            backbone.mkdir(exist_ok=True)
            state = {"model." + n: v for n, v in _cpu_state(student.decoder).items()}
            state.update({"lm_head." + n: v for n, v in _cpu_state(student.lm_head).items()})
            state.update({"visual." + n: v for n, v in _cpu_state(visual).items()})
            save_torch_state_dict(state, str(backbone), safe_serialization=True, max_shard_size="4GB")
            del state
            model_config = student.model_config.to_dict()
            for key in ("visual_latent_size", "interaction_latent_size", "latent_size",
                        "interaction_training_mode", "reference_fingerprint", "sft1_warmup_backbone"):
                model_config.pop(key, None)
            model_config.update(stage="stage1", sft2_helper_conditioned_backbone=True,
                                architectures=["Qwen2_5_VLForConditionalGeneration"], torch_dtype="bfloat16")
            json_write(backbone / "config.json", model_config)
            processor.save_pretrained(backbone)
            save_file(_cpu_state(student.interaction), str(output / "interaction.safetensors"))
            interaction_config = json.loads((Path(config.warmup_path) / "interaction_config.json").read_text(encoding="utf-8"))
            interaction_config.update(format="sft2-interaction-v1", file="interaction.safetensors")
            json_write(output / "interaction_config.json", interaction_config)
            auxiliary = output / "auxiliary_decoder"
            auxiliary.mkdir(exist_ok=True)
            state = {"model." + n: v for n, v in _cpu_state(student.aux_decoder).items()}
            state.update({"lm_head." + n: v for n, v in _cpu_state(student.aux_lm_head).items()})
            save_torch_state_dict(state, str(auxiliary), safe_serialization=True, max_shard_size="4GB")
            del state
            json_write(auxiliary / "config.json", {"format": "sft2-auxiliary-decoder-v1",
                       "source_model_config": student.model_config.to_dict(),
                       "input_interface": "four action latent vectors, then student token embeddings",
                       "contains_vision_encoder": False, "standalone_hf_model": False})
            json_write(output / "stage2_recipe.json", {
                "format": FORMAT, "config": config.to_dict(), "warmup_fingerprint": warmup_hash,
                "teacher_fingerprint": teacher_hash, "backbone": "backbone",
                "interaction": "interaction.safetensors", "auxiliary_decoder": "auxiliary_decoder",
                "sequence": "analysis -> <|latent_start|> -> action_latents[4] -> interaction[3] -> visual_latents[16] -> <|latent_end|> -> text",
                "boundary_policy": BOUNDARY_POLICY, "boundary_tokens": list(BOUNDARY_TOKENS),
                "latent_boundary_ids": [model_config.get("latent_start_id"), model_config.get("latent_end_id")],
                "helper_prior": "ground-truth helper post-merger features pooled to 16 ordered vectors",
                "requires_helper_images": True, "requires_sequence_plan": True,
                "helper_free_generation_ready": False, "standalone_hf_generation_ready": False,
                "warmup_adapter_in_student": False,
                "inference_note": "Use the SFT-II trajectory with the matching helper image and event plan; "
                                  "plain HF generate does not implement this conditional recipe."})
            result[0] = {"ok": True}
        except Exception as exc:
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise RuntimeError(f"SFT-II export failed: {result[0]['error']}")
