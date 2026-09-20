"""Standalone continuous-latent generation, with per-segment exact step counts."""
import time
import torch
from .execution import DecoderExecutor


@torch.no_grad()
def generate_continuous(decoder, head, prompt_embeds, position_ids, token_ids, eos_ids,
                        latent_steps, max_new_tokens=1024, backend="flash_attention_2",
                        progress_callback=None, progress_interval_seconds=30):
    """Single example; shard examples across GPUs for evaluation.

    Hidden output of latent_start is the INPUT of the first continuous step.
    No LM head or embedding lookup is used at continuous positions.
    """
    if prompt_embeds.shape[0] != 1:
        raise ValueError("Generation accepts one example per GPU")
    if latent_steps <= 0 or max_new_tokens <= 0:
        raise ValueError("latent_steps and max_new_tokens must be positive")
    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive")
    started = last_progress = time.perf_counter()
    if progress_callback:
        progress_callback(stage="prefill", generated_tokens=0, elapsed_seconds=0.0)
    executor = DecoderExecutor(decoder, backend, activation_checkpointing=False)
    device = prompt_embeds.device
    mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.bool, device=device)
    hidden, cache = executor.chunk(prompt_embeds, mask, position_ids)
    last = hidden[:, -1:]
    del hidden
    next_position = int(position_ids.max()) + 1
    if progress_callback:
        progress_callback(stage="decoding", generated_tokens=0, elapsed_seconds=round(time.perf_counter() - started, 1))
    remaining, need_end, current_count = 0, False, 0
    generated, segment_counts = [], []
    stopped_on_eos = False
    for _ in range(max_new_tokens):
        if remaining:
            token = token_ids["pad"]
            value = last
            remaining -= 1
            current_count += 1
            need_end = remaining == 0
        elif need_end:
            token = token_ids["end"]
            value = decoder.embed_tokens(torch.tensor([[token]], device=device))
            segment_counts.append(current_count)
            current_count, need_end = 0, False
        else:
            logits = head(last[:, -1]).float()
            logits[:, token_ids["pad"]] = -torch.inf
            logits[:, token_ids["end"]] = -torch.inf
            token = int(logits.argmax(-1))
            if token in eos_ids:
                generated.append(token)
                stopped_on_eos = True
                break
            value = decoder.embed_tokens(torch.tensor([[token]], device=device))
            if token == token_ids["start"]:
                remaining = latent_steps
        generated.append(token)
        mask = torch.cat((mask, mask.new_ones(1, 1)), -1)
        pos = torch.full((3, 1, 1), next_position, device=device, dtype=torch.long)
        last, cache = executor.chunk(value, mask, pos, cache)
        next_position += 1
        if progress_callback:
            now = time.perf_counter()
            if now - last_progress >= progress_interval_seconds:
                progress_callback(stage="decoding", generated_tokens=len(generated),
                                  latent_segments=len(segment_counts), elapsed_seconds=round(now - started, 1))
                last_progress = now
    if progress_callback:
        progress_callback(stage="generated", generated_tokens=len(generated),
                          latent_segments=len(segment_counts), elapsed_seconds=round(time.perf_counter() - started, 1))
    return {"token_ids": generated, "segment_steps": segment_counts,
            "unfinished_latent_steps": current_count, "ended_mid_latent": bool(remaining or need_end),
            "stopped_on_eos": stopped_on_eos, "hit_token_limit": not stopped_on_eos}
