"""Bounded-memory native ZeRO-2 resharding, with exact tensor verification."""
from collections import OrderedDict
import copy
import math
from pathlib import Path

import torch

CHUNK = 1 << 20
MODEL_FILE = "mp_rank_00_model_states.pt"


def optimizer_file(rank, bf16=False):
    return f"{'bf16_' if bf16 else ''}zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"


def optimizer_path(directory, rank):
    paths = [Path(directory) / optimizer_file(rank, precision) for precision in (False, True)]
    matches = [path for path in paths if path.is_file()]
    require(len(matches) == 1, f"Missing or ambiguous optimizer shard for rank {rank}: {directory}")
    return matches[0]


def load(path):
    # These are the user's trusted training checkpoints, including DS objects.
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(left, right, label):
    if isinstance(left, torch.Tensor):
        require(isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape, f"{label}: tensor metadata differs")
        a, b = left.reshape(-1), right.reshape(-1)
        for start in range(0, a.numel(), CHUNK):
            require(torch.equal(a[start:start+CHUNK], b[start:start+CHUNK]),
                    f"{label}: tensor values differ at chunk {start}")
    elif isinstance(left, dict):
        require(isinstance(right, dict) and left.keys() == right.keys(), f"{label}: keys differ")
        for key in left:
            equal(left[key], right[key], f"{label}.{key}")
    elif isinstance(left, (list, tuple)):
        require(isinstance(right, (list, tuple)) and len(left) == len(right), f"{label}: lengths differ")
        for i, (a, b) in enumerate(zip(left, right)):
            equal(a, b, f"{label}.{i}")
    elif hasattr(left, "__dict__"):
        require(type(left) is type(right), f"{label}: types differ")
        equal(vars(left), vars(right), label)
    else:
        require(left == right, f"{label}: {left!r} != {right!r}")


def finite(value, label, nonnegative=False):
    flat = value.detach().reshape(-1)
    for start in range(0, flat.numel(), CHUNK):
        part = flat[start:start+CHUNK]
        require(bool(torch.isfinite(part).all()), f"{label}: NaN/Inf")
        if nonnegative:
            require(bool((part >= 0).all()), f"{label}: negative second moment")


def partition_size(numel, world):
    # DS 0.18.0 aligns each flat group to 2 * DP; this is NOT ceil(numel/DP).
    return math.ceil(numel / (2 * world)) * 2


def mapping(shapes, size, rank, factory):
    result, offset = OrderedDict(), 0
    lo, hi = rank * size, (rank + 1) * size
    for name, shape in shapes.items():
        end = offset + math.prod(shape)
        a, b = max(lo, offset), min(hi, end)
        if b > a:
            result[name] = factory(numel=b-a, start=a-lo)
        offset = end
    return result


def fragment_type(shards):
    for shard in shards:
        for group in shard["optimizer_state_dict"]["param_slice_mappings"]:
            for value in group.values():
                return type(value)
    raise ValueError("No parameter slice mappings")


def validate_source(model, shards, world, step):
    """Verify every source master/moment, ordering, padding and model cast."""
    require(model["dp_world_size"] == world and model["mp_world_size"] == 1, "Expected pure data parallel ZeRO-2")
    require(model["ds_version"].split("+")[0] == "0.18.0", "Revalidate resharding for this DeepSpeed version")
    require(model["global_steps"] == step and model.get("skipped_steps", 0) == 0, "Invalid/skipped optimizer steps")
    shapes = model["param_shapes"]
    sizes = [sum(math.prod(shape) for shape in group.values()) for group in shapes]
    require(all(sizes), "Empty optimizer groups are unsupported")
    require(len(shards) == world, "Missing source rank")
    template = shards[0]["optimizer_state_dict"]
    factory = fragment_type(shards)
    base_groups = template["base_optimizer_state"]["param_groups"]
    require(len(base_groups) == len(sizes), "Group count mismatch")
    ids = [group["params"][0] for group in base_groups]
    require(all(len(group["params"]) == 1 for group in base_groups) and len(set(ids)) == len(ids),
            "Expected one flat FP32 master per optimizer group")
    for name, tensor in model["module"].items():
        if isinstance(tensor, torch.Tensor):
            finite(tensor, "model." + name)
    for rank, shard in enumerate(shards):
        state = shard["optimizer_state_dict"]
        require(int(state["zero_stage"]) == 2, "Only ZeRO-2 checkpoints are supported")
        require(state["partition_count"] == [world] * len(sizes), "Source partition count mismatch")
        require(not state["overflow"] and not state["dynamic_loss_scale"], "Overflow/FP16 checkpoints unsupported")
        require(set(state["base_optimizer_state"]["state"]) == set(ids), "Missing Adam state")
        equal(state["base_optimizer_state"]["param_groups"], base_groups, f"rank{rank}.groups")
        equal(shard["ds_config"], shards[0]["ds_config"], f"rank{rank}.config")
        for key in ("clip_grad", "ds_version"):
            equal(state[key], template[key], f"rank{rank}.{key}")
        equal(vars(state["loss_scaler"]), vars(template["loss_scaler"]), "loss_scaler")
        require(len(state["single_partition_of_fp32_groups"]) == len(sizes), "Master group count mismatch")
        for i, (numel, group_shapes, pid) in enumerate(zip(sizes, shapes, ids)):
            size = partition_size(numel, world)
            valid = min(size, max(0, numel - rank * size))
            require(state["group_paddings"][i] == size - valid, "Source padding mismatch")
            equal(state["param_slice_mappings"][i], mapping(group_shapes, size, rank, factory), "parameter order")
            master = state["single_partition_of_fp32_groups"][i]
            require(master.dtype == torch.float32 and master.shape == (valid,), "Invalid FP32 master shape/dtype")
            finite(master, f"rank{rank}.master.{i}")
            adam = state["base_optimizer_state"]["state"][pid]
            require(set(adam) == {"step", "exp_avg", "exp_avg_sq"}, "Unsupported Adam state fields")
            require(float(adam["step"]) == step, "Adam step does not match global_steps")
            equal(adam["step"], template["base_optimizer_state"]["state"][pid]["step"], "Adam step")
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = adam[key]
                require(tensor.dtype == torch.float32 and tensor.shape == (size,), "Invalid FP32 Adam partition")
                finite(tensor, f"rank{rank}.{key}.{i}", nonnegative=key == "exp_avg_sq")
            # Verify the param-name/order mapping against the actual BF16 model.
            offset = 0
            for name, shape in group_shapes.items():
                length = math.prod(shape)
                require(name in model["module"] and tuple(model["module"][name].shape) == tuple(shape),
                        f"Missing/wrong model parameter: {name}")
                lo, hi = max(offset, rank*size), min(offset+length, rank*size+valid)
                for start in range(lo, hi, CHUNK):
                    end = min(start + CHUNK, hi)
                    reference = model["module"][name].reshape(-1)[start-offset:end-offset]
                    equal(master[start-rank*size:end-rank*size].to(reference.dtype), reference,
                          f"master/model.{name}")
                offset += length
    return sizes


def source_slices(shards, group, field, numel, start, end):
    old_size = partition_size(numel, len(shards))
    for rank, shard in enumerate(shards):
        lo, hi = max(start, rank*old_size), min(end, (rank+1)*old_size, numel)
        if hi <= lo:
            continue
        state = shard["optimizer_state_dict"]
        if field == "master":
            tensor = state["single_partition_of_fp32_groups"][group]
        else:
            pid = state["base_optimizer_state"]["param_groups"][group]["params"][0]
            tensor = state["base_optimizer_state"]["state"][pid][field]
        yield lo-start, tensor[lo-rank*old_size:hi-rank*old_size]


def reshard_rank(shards, shapes, sizes, rank, target_world, ds_config):
    """Create only one new rank in memory; source shards remain mmap-backed."""
    first = shards[0]
    old = first["optimizer_state_dict"]
    result = {k: copy.deepcopy(v) for k, v in first.items() if k != "optimizer_state_dict"}
    result["ds_config"] = ds_config
    state = {k: copy.deepcopy(v) for k, v in old.items()
             if k not in {"base_optimizer_state", "single_partition_of_fp32_groups", "param_slice_mappings"}}
    result["optimizer_state_dict"] = state
    state["partition_count"] = [target_world] * len(sizes)
    state["group_paddings"], state["param_slice_mappings"], state["single_partition_of_fp32_groups"] = [], [], []
    state["base_optimizer_state"] = {"param_groups": copy.deepcopy(old["base_optimizer_state"]["param_groups"]), "state": {}}
    factory = fragment_type(shards)
    for i, numel in enumerate(sizes):
        size = partition_size(numel, target_world)
        valid = min(size, max(0, numel-rank*size))
        state["group_paddings"].append(size-valid)
        state["param_slice_mappings"].append(mapping(shapes[i], size, rank, factory))
        pid = state["base_optimizer_state"]["param_groups"][i]["params"][0]
        adam = {"step": copy.deepcopy(old["base_optimizer_state"]["state"][pid]["step"])}
        for field in ("master", "exp_avg", "exp_avg_sq"):
            tensor = torch.zeros(valid if field == "master" else size, dtype=torch.float32)
            for offset, source in source_slices(shards, i, field, numel, rank*size, rank*size+valid):
                tensor[offset:offset+source.numel()].copy_(source)
            if field == "master":
                state["single_partition_of_fp32_groups"].append(tensor)
            else:
                adam[field] = tensor
        state["base_optimizer_state"]["state"][pid] = adam
    return result


def verify_rank(path, sources, sizes, rank, target_world, expected):
    """Read saved bytes back and compare ALL valid elements, not a sample/hash."""
    saved = load(path)
    equal(saved, expected, f"rank{rank}.serialized")
    state = saved["optimizer_state_dict"]
    for i, numel in enumerate(sizes):
        size = partition_size(numel, target_world)
        valid = min(size, max(0, numel-rank*size))
        pid = state["base_optimizer_state"]["param_groups"][i]["params"][0]
        for field in ("master", "exp_avg", "exp_avg_sq"):
            tensor = (state["single_partition_of_fp32_groups"][i] if field == "master"
                      else state["base_optimizer_state"]["state"][pid][field])
            for offset, source in source_slices(sources, i, field, numel, rank*size, rank*size+valid):
                equal(tensor[offset:offset+source.numel()], source, f"rank{rank}.{field}.{i}")
            if field != "master":
                require(bool((tensor[valid:] == 0).all()), "New padding must be zero")
