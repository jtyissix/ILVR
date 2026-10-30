"""Instance-local ordered ZeRO-2 adapter for optimizer CPU offload.

The original SFT-I/SFT-II adapter intentionally supports GPU optimizer state
only. This opt-in adapter retains native CPU partition copies, clipping, Adam
updates and epilogues, and fixes DeepSpeed 0.18.0's missing intermediate CPU
accumulation writeback. No installed DeepSpeed file or original trainer changes.
"""
from __future__ import annotations

import hashlib
import inspect
import json

import torch
import torch.distributed as dist

from ..sft1.zero2 import OrderedGradientReduction, check_zero2_version
from ..sft2.train import ordered_parameters


def persist_offload_accumulation(optimizer, param):
    """Persist an intermediate sum after native offload accumulation.

    In DeepSpeed 0.18.0, ``async_accumulate_grad_in_cpu_via_gpu`` writes the
    first microbatch to CPU, but later microbatches only add that CPU value on
    GPU. With GAS > 2 the next microbatch consequently reuses a stale CPU sum.
    Write back intermediate sums before the native epilogue clears gradients.
    The native caller copies the boundary to its FP32 partition, and native
    accumulation copies the first microbatch, so neither needs another copy.

    Full-parameter CPU buffers here are native behavior. Their local partition
    offsets are handled subsequently by native ``copy_grads_in_partition``.
    Only the standard FP32-master path is supported. All transfers use the
    current stream, with communication overlap disabled by the installer.
    """
    if optimizer.micro_step_id <= 0 or optimizer.is_gradient_accumulation_boundary:
        return
    if optimizer.fp16_master_weights_and_gradients:
        raise RuntimeError("SFT-II offload requires FP32 master parameters and gradients")
    param_id = optimizer.get_param_id(param)
    gradient = optimizer.get_param_gradient_attribute(param)
    accumulated = optimizer.accumulated_grads_in_cpu[param_id]
    if (gradient is None or gradient.numel() != param.numel()
            or accumulated.device.type != "cpu" or accumulated.numel() != param.numel()
            or accumulated.dtype != gradient.dtype):
        raise RuntimeError("Unexpected native CPU offload accumulation buffer")
    accumulated.copy_(gradient.detach().view(-1), non_blocking=True)


def install_ordered_offload(engine):
    """Install only on this native ZeRO-2 CPUAdam engine, after DS initialize."""
    if hasattr(engine, "_sft2_offload_ordered_zero2"):
        return engine._sft2_offload_ordered_zero2
    if hasattr(engine, "_sft1_ordered_zero2"):
        raise RuntimeError("The GPU-state ZeRO-2 adapter cannot be combined with CPU offload")
    if int(engine.zero_optimization_stage()) != 2:
        raise RuntimeError("SFT-II optimizer offload requires ZeRO stage 2")

    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    check_zero2_version(deepspeed.__version__)
    optimizer = engine.optimizer
    if type(optimizer) is not DeepSpeedZeroOptimizer:
        raise RuntimeError("SFT-II offload requires the native DeepSpeedZeroOptimizer")
    if type(optimizer.optimizer) is not DeepSpeedCPUAdam:
        raise RuntimeError("SFT-II offload requires DeepSpeedCPUAdam")
    if (not optimizer.cpu_offload or not optimizer.partition_gradients
            or optimizer.overlap_comm or not optimizer.contiguous_gradients
            or not optimizer.reduce_scatter or optimizer.zenflow
            or optimizer.use_grad_accum_attribute or optimizer.has_moe_layers
            or optimizer.round_robin_gradients or optimizer.fp16_master_weights_and_gradients
            or optimizer.sequence_parallel_size != 1 or optimizer.model_parallel_world_size != 1):
        raise RuntimeError("Unsupported ordered CPU offload settings; use configs/sft2_offload/zero2.json")
    for name, arguments in (
            ("reduce_ready_partitions_and_remove_grads", ("param", "i")),
            ("independent_gradient_partition_epilogue", ()),
            ("async_accumulate_grad_in_cpu_via_gpu", ("param",))):
        if tuple(inspect.signature(getattr(optimizer, name)).parameters) != arguments:
            raise RuntimeError(f"DeepSpeed's {name} API changed; revalidate SFT-II CPU offload")

    masters = optimizer.single_partition_of_fp32_groups
    if len(masters) != len(optimizer.bit16_groups) or any(
            p.device.type != "cpu" or p.dtype != torch.float32
            or p.grad is None or p.grad.device.type != "cpu"
            or p.grad.dtype != torch.float32 or p.grad.shape != p.shape for p in masters):
        raise RuntimeError("SFT-II offload requires CPU FP32 master parameters and gradient partitions")
    entries = ordered_parameters(engine.module, optimizer.bit16_groups)
    dtypes = {param.dtype for _, param, _ in entries}
    if len(dtypes) != 1 or not dtypes.issubset({torch.bfloat16, torch.float32}):
        raise RuntimeError("SFT-II offload requires BF16 parameters (FP32 is supported for acceptance tests)")
    if any(getattr(param, "use_muon", False) for _, param, _ in entries):
        raise RuntimeError("SFT-II offload does not support Muon transforms")
    if any(optimizer.get_param_comm_dtype(param) != param.dtype for _, param, _ in entries):
        raise RuntimeError("DeepSpeed 0.18.0 oversized buckets require matching parameter/communication dtypes")

    descriptor = [(name, list(param.shape), str(param.dtype), group, param.param_idx_in_group)
                  for name, param, group in entries]
    digest = hashlib.sha256(json.dumps(descriptor).encode()).hexdigest()
    if dist.is_initialized():
        signatures = [None] * dist.get_world_size(group=optimizer.dp_process_group)
        dist.all_gather_object(signatures, digest, group=optimizer.dp_process_group)
        if len(set(signatures)) != 1:
            raise RuntimeError("SFT-II CPU offload reduction order differs across ranks")

    reducer = OrderedGradientReduction(entries, optimizer.reduce_ready_partitions_and_remove_grads)
    reducer.order_sha256 = digest
    native_backward = engine.backward
    native_epilogue = optimizer.independent_gradient_partition_epilogue
    native_accumulate = optimizer.async_accumulate_grad_in_cpu_via_gpu

    def accumulate(param):
        native_accumulate(param)
        persist_offload_accumulation(optimizer, param)

    def epilogue():
        reducer.finish()
        # CPU copies and accumulation belong to native offload. In particular,
        # never invoke the original adapter's GPU-only finish_gradient_partition.
        return native_epilogue()

    def backward(loss, retain_graph=False, scale_wrt_gas=True):
        if retain_graph:
            raise ValueError("SFT-II offload performs one backward per complete trajectory")
        reducer.begin()
        result = native_backward(loss, retain_graph=False, scale_wrt_gas=scale_wrt_gas)
        if reducer.active:
            raise RuntimeError("DeepSpeed skipped the CPU offload gradient partition epilogue")
        return result

    optimizer.async_accumulate_grad_in_cpu_via_gpu = accumulate
    optimizer.reduce_ready_partitions_and_remove_grads = reducer.ready
    optimizer.independent_gradient_partition_epilogue = epilogue
    engine.backward = backward
    engine._sft2_offload_ordered_zero2 = reducer
    return reducer
