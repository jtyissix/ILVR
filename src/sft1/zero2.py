"""Ordered, streaming ZeRO-2 reduction for unequal SFT-I autograd graphs.

DeepSpeed 0.18.0 fills IPG buckets in leaf-hook order, which can differ between
ranks when the number of helper images differs. Queue *references* to ready
parameters, and feed the native reducer in a shared, static order. Native ZeRO
still partitions/clears gradients during backward; a local epilogue also backports
the upstream gradient accumulation fix. No full-gradient flattening, extra gradient copy, reference
forward, or change to the loss/graph is introduced here.

This deliberately targets the pinned DeepSpeed version, not an undocumented
cross-version API. Keep tests/test_sft1_deepspeed.py as the distributed acceptance
test when upgrading it.
"""
from __future__ import annotations

import hashlib
import inspect
import json

import torch.distributed as dist


SUPPORTED_DEEPSPEED = "0.18.0"


def check_zero2_version(version):
    if version.split("+")[0] != SUPPORTED_DEEPSPEED:
        raise RuntimeError(
            f"SFT-I ordered ZeRO-2 requires deepspeed=={SUPPORTED_DEEPSPEED}; found {version}. "
            "Install the version pinned in requirements-interaction.txt. Prepared data is reusable.")


def ordered_parameters(model, groups):
    """Prefer backward decoder order; hold only the smaller branches until later.

    Ordering by optimizer groups is unsuitable: LR/decay groups interleave every
    decoder layer. A tied input/output embedding is deliberately placed last.
    """
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    indexed = {id(param): (group_index, param) for group_index, group in enumerate(groups) for param in group}
    if len(indexed) != sum(len(group) for group in groups) or set(indexed) != {id(p) for _, p in named}:
        raise RuntimeError("ZeRO-2 optimizer parameters must match all trainable SFT-I parameters exactly")
    embedding = getattr(getattr(model, "decoder", None), "embed_tokens", None)
    embedding_ids = {id(p) for p in embedding.parameters()} if embedding is not None else set()

    def key(item):
        index, (name, param) = item
        if id(param) in embedding_ids:
            return (4, 0, -index)
        if name.startswith("lm_head."):
            return (0, 0, -index)
        if name.startswith("decoder.norm."):
            return (1, 0, -index)
        if name.startswith("decoder.layers."):
            return (2, -int(name.split(".")[2]), -index)
        return (3, 0, -index)

    return [(name, param, indexed[id(param)][0]) for _, (name, param) in sorted(enumerate(named), key=key)]


def finish_gradient_partition(optimizer):
    """Finish native buckets and correctly retain *local* accumulated gradients.

    DeepSpeed 0.18.0's non-offload epilogue tests averaged_gradients even though
    that dictionary is populated only at the GAS boundary. Consequently it
    overwrites all_grad_tensors on every microbatch. The upstream 0.18.3 fix
    tests all_grad_tensors instead. Keep that fix here for the pinned 0.18.0
    environment, without editing the user's installed package.

    Native bucket reduction, ownership, partition slicing, buffer release and
    step/checkpoint code are retained. Offload/overlap/Muon are not supported.
    See DeepSpeed v0.18.3, independent_gradient_partition_epilogue (Apache-2.0).
    """
    optimizer.reduce_ipg_grads()
    for index in range(len(optimizer.params_already_reduced)):
        optimizer.params_already_reduced[index] = False
    for group, params in enumerate(optimizer.params_in_partition):
        current = optimizer.get_all_grad_tensors(params, dtype=optimizer.gradient_accumulation_dtype)
        accumulated = optimizer.all_grad_tensors.get(group)
        if accumulated is None:
            optimizer.all_grad_tensors[group] = current
        else:
            if len(accumulated) != len(current):
                raise RuntimeError("ZeRO-2 local gradient partition changed during accumulation")
            for previous, gradient in zip(accumulated, current):
                previous.add_(gradient)
        if optimizer.is_gradient_accumulation_boundary:
            optimizer.averaged_gradients[group] = optimizer.get_flat_partition(
                params, optimizer.first_offset[group], optimizer.partition_size[group],
                dtype=optimizer.gradient_accumulation_dtype,
                device=optimizer.single_partition_of_fp32_groups[group].device,
                param_group_idx=group, return_tensor_list=True)
            optimizer.all_grad_tensors[group] = None
    optimizer._release_ipg_buffers()
    optimizer.zero_grad(set_to_none=True)


class OrderedGradientReduction:
    """Single-backward queue; native ZeRO owns gradients and all collectives."""

    def __init__(self, entries, reduce_ready):
        self.entries = entries
        self.reduce_ready = reduce_ready
        self.positions = {id(param): i for i, (_, param, _) in enumerate(entries)}
        self.active = False
        self.peak_pending_bytes = 0
        self.completed_backwards = 0

    def begin(self):
        if self.active:
            raise RuntimeError("Nested or unfinished ordered ZeRO-2 backward")
        self.active = True
        self.cursor = 0
        self.seen = set()
        self.pending = set()
        self.pending_bytes = 0
        self.peak_pending_bytes = 0

    def ready(self, param, group_index):
        if not self.active:
            raise RuntimeError("Use engine.backward for SFT-I ordered ZeRO-2")
        position = self.positions.get(id(param))
        if position is None or self.entries[position][2] != group_index:
            raise RuntimeError("Unexpected parameter/group in ordered ZeRO-2")
        if position in self.seen:
            raise RuntimeError("A ZeRO-2 parameter became ready twice in one backward; use non-reentrant checkpointing")
        if param.grad is None or param.grad.is_sparse or not getattr(param, "ds_grad_is_ready", True):
            raise RuntimeError("Ordered ZeRO-2 requires a final dense gradient for every trainable parameter")
        self.seen.add(position)
        self.pending.add(position)
        self.pending_bytes += param.numel() * param.grad.element_size()
        while self.cursor in self.pending:
            _, current, group = self.entries[self.cursor]
            self.pending_bytes -= current.numel() * current.grad.element_size()
            self.pending.remove(self.cursor)
            self.reduce_ready(current, group)
            self.cursor += 1
        self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)

    def finish(self):
        if not self.active:
            raise RuntimeError("ZeRO-2 gradient epilogue ran without a backward")
        if self.cursor != len(self.entries):
            missing = [name for i, (name, _, _) in enumerate(self.entries) if i not in self.seen]
            raise RuntimeError(f"Incomplete ZeRO-2 backward; missing gradients: {missing[:12]}. "
                               "Keep the warm-up model's explicit zero-gradient branch dependencies.")
        self.active = False
        self.completed_backwards += 1


def install_ordered_zero2(engine):
    """Install once after DS initialize (also on a fresh engine before resume)."""
    if int(engine.zero_optimization_stage()) != 2:
        return None
    if hasattr(engine, "_sft1_ordered_zero2"):
        return engine._sft1_ordered_zero2
    import deepspeed
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    check_zero2_version(deepspeed.__version__)
    optimizer = engine.optimizer
    if type(optimizer) is not DeepSpeedZeroOptimizer:
        raise RuntimeError("Ordered ZeRO-2 requires the native DeepSpeedZeroOptimizer")
    if (not optimizer.partition_gradients or optimizer.overlap_comm or not optimizer.contiguous_gradients
            or not optimizer.reduce_scatter or optimizer.cpu_offload or optimizer.zenflow
            or optimizer.use_grad_accum_attribute or optimizer.has_moe_layers or optimizer.round_robin_gradients
            or optimizer.sequence_parallel_size != 1 or optimizer.model_parallel_world_size != 1):
        raise RuntimeError("Unsupported ordered ZeRO-2 optimizer settings; use configs/sft1/zero2.json")
    for method, arguments in (("reduce_ready_partitions_and_remove_grads", ("param", "i")),
                              ("independent_gradient_partition_epilogue", ())):
        if tuple(inspect.signature(getattr(optimizer, method)).parameters) != arguments:
            raise RuntimeError(f"DeepSpeed's {method} API changed; revalidate the ordered ZeRO-2 adapter")
    entries = ordered_parameters(engine.module, optimizer.bit16_groups)
    if any(getattr(param, "use_muon", False) for _, param, _ in entries):
        raise RuntimeError("SFT-I ordered ZeRO-2 does not support Muon parameter transforms")
    if len({param.dtype for _, param, _ in entries}) != 1:
        raise RuntimeError("SFT-I ordered ZeRO-2 requires one trainable parameter dtype")
    if any(optimizer.get_param_comm_dtype(param) != param.dtype for _, param, _ in entries):
        # 0.18.0 keys oversized-parameter storage by param.dtype but reads it by
        # comm_dtype. Qwen's embedding/head are larger than the reduction bucket.
        raise RuntimeError("DeepSpeed 0.18.0 requires matching parameter/communication dtypes for oversized buckets")
    # Native rank ownership also depends on parameter position *within* its LR/
    # decay group, even though we drain those groups in interleaved decoder order.
    descriptor = [(name, list(param.shape), str(param.dtype), group, param.param_idx_in_group)
                  for name, param, group in entries]
    digest = hashlib.sha256(json.dumps(descriptor).encode()).hexdigest()
    if dist.is_initialized():
        signatures = [None] * dist.get_world_size(group=optimizer.dp_process_group)
        dist.all_gather_object(signatures, digest, group=optimizer.dp_process_group)
        if len(set(signatures)) != 1:
            raise RuntimeError("SFT-I ZeRO-2 parameter reduction order differs across ranks")
    reducer = OrderedGradientReduction(entries, optimizer.reduce_ready_partitions_and_remove_grads)
    reducer.order_sha256 = digest
    native_backward = engine.backward

    def epilogue():
        # Assert completeness before the native final bucket flush. Every microbatch
        # must reach here, including non-boundary accumulation and masked dummies.
        reducer.finish()
        return finish_gradient_partition(optimizer)

    def backward(loss, retain_graph=False, scale_wrt_gas=True):
        if retain_graph:
            raise ValueError("SFT-I performs exactly one backward per complete trajectory")
        reducer.begin()
        result = native_backward(loss, retain_graph=False, scale_wrt_gas=scale_wrt_gas)
        if reducer.active:
            raise RuntimeError("DeepSpeed skipped the ZeRO-2 gradient partition epilogue")
        return result

    optimizer.reduce_ready_partitions_and_remove_grads = reducer.ready
    optimizer.independent_gradient_partition_epilogue = epilogue
    engine.backward = backward
    engine._sft1_ordered_zero2 = reducer
    return reducer
