"""Opt-in Linux/CUDA multi-rank correctness test; does not download a checkpoint.

torchrun --standalone --nproc_per_node=2 --module tests.test_interaction_deepspeed --zero_stage 2
Repeat with --zero_stage 3 and --attention_backend flash_attention_2 on the server.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
import torch.distributed as dist
from src.interaction.checkpoint import load_training, save_training
from src.interaction.config import TrainConfig
from src.interaction.distributed import align_zero3, normalized_loss, rank, sum_scalar, to_device, world
from src.interaction.model import InteractionStudent, reference_contexts
from tests.interaction_fixtures import TinyDecoder, latent_batch, simple_batch


def make_model(config, device):
    torch.manual_seed(42)
    return InteractionStudent(TinyDecoder(), torch.nn.Linear(24, 96),
                              SimpleNamespace(hidden_size=24, image_token_id=80), config).to(device=device, dtype=torch.bfloat16)


def initialize(config, stage, device):
    import deepspeed
    student = make_model(config, device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1/(step+1))
    ds = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 2, "train_batch_size": 2*world(),
          "bf16": {"enabled": True}, "zero_optimization": {"stage": stage},
          "zero_allow_untested_optimizer": True, "steps_per_print": 1000000}
    engine, _, _, _ = deepspeed.initialize(model=student, optimizer=optimizer, lr_scheduler=scheduler,
                                          config=ds, dist_init_required=False)
    return engine


def full_parameters(engine):
    from contextlib import nullcontext
    import deepspeed
    weights = {}
    for name, parameter in engine.module.named_parameters():
        context = deepspeed.zero.GatheredParameters([parameter]) if hasattr(parameter, "ds_id") else nullcontext()
        with context:
            weights[name] = parameter.detach().float().cpu().clone()
    return weights


def one_step(engine, reference, stage, device, backend):
    # Ranks have different sequence lengths, latent locations/counts and CE counts.
    batches = [latent_batch(1) if rank() % 2 == 0 else simple_batch(12), simple_batch(9+rank())]
    batches[0]["labels"][:, :2+rank()] = -100
    if rank() == world()-1:
        batches[1]["labels"][:] = -100  # tail-window dummy
    total = sum_scalar(sum(int(b["labels"][:, 1:].ne(-100).sum()) for b in batches), device)
    losses = []
    for index, batch in enumerate(batches):
        batch = to_device(batch, device)
        ref_batch = to_device(latent_batch(), device) if index == 0 and rank() % 2 == 0 else batch
        context = reference_contexts(reference, ref_batch, None, 80, backend)
        events, count = align_zero3(batch, 0, device) if stage == 3 else (None, None)
        loss = engine(batch, contexts=context, events=events, padded_ce_count=count)
        losses.append(float(loss.detach()))
        engine.backward(normalized_loss(loss, total, 2, world()))
        engine.step()
    return losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero_stage", type=int, choices=[2, 3], default=2)
    parser.add_argument("--attention_backend", choices=["sdpa", "flash_attention_2"], default="sdpa")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("cuda", torch.cuda.current_device())
    import deepspeed
    deepspeed.init_distributed("nccl")
    config = TrainConfig(fusion_dim=16, fusion_heads=2, fusion_ffn_dim=32, attention_backend=args.attention_backend,
                         activation_checkpointing=True, ce_chunk_size=3)
    torch.manual_seed(17)
    reference = TinyDecoder().requires_grad_(False).eval().to(device=device, dtype=torch.bfloat16)
    frozen = {n: p.detach().clone() for n, p in reference.named_parameters()}
    engine = initialize(config, args.zero_stage, device)
    before = full_parameters(engine)
    one_step(engine, reference, args.zero_stage, device, args.attention_backend)
    after = full_parameters(engine)
    assert any(not torch.equal(before[n], after[n]) for n in after if n.startswith("fusion."))
    assert all(torch.isfinite(value).all() for value in after.values())
    for name, param in reference.named_parameters():
        assert torch.equal(param, frozen[name]) and param.grad is None
    directory = [tempfile.mkdtemp(prefix="ilvr-ds-smoke-") if rank() == 0 else None]
    dist.broadcast_object_list(directory, 0)
    saved = save_training(engine, directory[0], 1, 0, "smoke-signature", "frozen-source")
    expected_losses = one_step(engine, reference, args.zero_stage, device, args.attention_backend)
    expected = full_parameters(engine)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    restored = initialize(config, args.zero_stage, device)
    assert load_training(restored, saved, "smoke-signature", "frozen-source") == (1, 0)
    actual_losses = one_step(restored, reference, args.zero_stage, device, args.attention_backend)
    actual = full_parameters(restored)
    torch.testing.assert_close(torch.tensor(actual_losses), torch.tensor(expected_losses), rtol=1e-3, atol=1e-3)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-3, atol=1e-3, msg=name)
    # All ranks must agree after the synchronized optimizer update.
    checksum = torch.tensor(sum(float(v.sum()) for v in actual.values()), device=device)
    gathered = [torch.empty_like(checksum) for _ in range(world())]
    dist.all_gather(gathered, checksum)
    assert all(torch.equal(gathered[0], item) for item in gathered)
    if rank() == 0:
        print(json.dumps({"status": "passed", "zero_stage": args.zero_stage, "world_size": world(),
                          "backend": args.attention_backend, "checkpoint": str(saved)}))


if __name__ == "__main__":
    main()
