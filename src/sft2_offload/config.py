"""Separate launch defaults and strict validation for optimizer-only CPU offload."""
import json

from ..interaction.distributed import world
from ..sft2.config import PROJECT_ROOT, Stage2Config, validate_local_inputs


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sft2_offload" / "latent.json"


def configure_deepspeed(config):
    with open(config.deepspeed, encoding="utf-8") as handle:
        ds = json.load(handle)
    zero = ds.get("zero_optimization", {})
    offload = zero.get("offload_optimizer", {})
    if zero.get("stage") != 2:
        raise ValueError("SFT-II offload requires ZeRO-2")
    if not isinstance(offload, dict) or offload.get("device") != "cpu":
        raise ValueError("SFT-II offload requires offload_optimizer.device=cpu")
    if not isinstance(offload.get("pin_memory", False), bool):
        raise ValueError("offload_optimizer.pin_memory must be boolean")
    if set(offload) - {"device", "pin_memory"}:
        raise ValueError("Only CPU device and pin_memory are supported offload_optimizer options")
    if (zero.get("overlap_comm") is not False or zero.get("reduce_scatter") is not True
            or zero.get("contiguous_gradients") is not True):
        raise ValueError("SFT-II offload requires overlap_comm=false, reduce_scatter=true, contiguous_gradients=true")
    if any(zero.get(key) for key in ("offload_param", "zenflow", "round_robin_gradients", "fp16_master_weights_and_gradients")):
        raise ValueError("Parameter offload, ZenFlow, round-robin gradients and low-precision master weights are unsupported")
    if ds.get("zenflow") or ds.get("deepcompile"):
        raise ValueError("ZenFlow and DeepCompile are unsupported")
    if ds.get("torch_autocast", {}).get("enabled") or ds.get("fp16", {}).get("enabled"):
        raise ValueError("SFT-II offload uses BF16 without FP16 or torch_autocast")
    if not ds.get("bf16", {}).get("enabled") or ds.get("communication_data_type", "bf16") != "bf16":
        raise ValueError("SFT-II offload requires BF16 parameters and communication")
    # Fixed-order hooks consume param.grad, not a separately cast grad_accum.
    if ds.get("data_types", {}).get("grad_accum_dtype", "bf16") not in ("bf16", "bfloat16"):
        raise ValueError("SFT-II ordered offload requires BF16 gradient accumulation")
    if "optimizer" in ds or "scheduler" in ds:
        raise ValueError("Optimizer and scheduler are controlled by latent.json, not DeepSpeed JSON")
    ds.update(train_micro_batch_size_per_gpu=config.micro_batch_size,
              gradient_accumulation_steps=config.gradient_accumulation_steps,
              train_batch_size=config.micro_batch_size * config.gradient_accumulation_steps * world(),
              zero_force_ds_cpu_optimizer=True)
    return ds
