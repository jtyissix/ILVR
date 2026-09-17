"""Differentiable incremental Qwen decoder; no mutable HF cache or latent detaches.

Attention uses the existing Qwen projections, norms and MLP. Each checkpointed
layer returns only its NEW K/V. Cache histories are tuples of immutable chunks;
concatenation is INSIDE checkpoint replay, avoiding a saved full-prefix copy at
every latent step (quadratic KV memory).
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), -1)


def multimodal_rope(q, k, cos, sin, sections):
    # Qwen's three coordinate axes use cyclic slices of the doubled section list.
    cos = torch.cat([part[i % 3] for i, part in enumerate(cos.split(sections * 2, dim=-1))], -1).unsqueeze(1)
    sin = torch.cat([part[i % 3] for i, part in enumerate(sin.split(sections * 2, dim=-1))], -1).unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


@dataclass
class AttentionLayout:
    mask: torch.Tensor
    past_length: int
    query_length: int
    q_indices: torch.Tensor | None = None
    k_indices: torch.Tensor | None = None
    cu_q: torch.Tensor | None = None
    cu_k: torch.Tensor | None = None
    max_q: int = 0
    max_k: int = 0

    @classmethod
    def make(cls, mask, past_length, query_length, backend):
        obj = cls(mask.bool(), past_length, query_length)
        if backend == "flash_attention_2":
            q_mask = obj.mask[:, past_length:past_length + query_length]
            q_lengths = q_mask.sum(-1, dtype=torch.int32)
            active = q_lengths > 0
            k_mask = obj.mask & active[:, None]
            k_lengths = k_mask.sum(-1, dtype=torch.int32)
            obj.q_indices = q_mask.flatten().nonzero().flatten()
            obj.k_indices = k_mask.flatten().nonzero().flatten()
            obj.max_q, obj.max_k = int(q_lengths.max()), int(k_lengths.max())
            # Do not pass zero-query examples to a varlen CUDA kernel.
            q_lengths, k_lengths = q_lengths[active], k_lengths[active]
            zero = q_lengths.new_zeros(1)
            obj.cu_q = torch.cat((zero, q_lengths.cumsum(0, dtype=torch.int32)))
            obj.cu_k = torch.cat((zero, k_lengths.cumsum(0, dtype=torch.int32)))
        return obj


def attend(q, k, v, layout, backend):
    """q/k/v [B,heads,length,D], including GQA; masks support right-padded training."""
    if backend == "flash_attention_2":
        if layout.max_q == 0:
            # ZeRO-3 can align a chunk beyond every local sample's valid tail.
            return q * 0.0 + (k.sum() + v.sum()) * 0.0
        from flash_attn import flash_attn_varlen_func
        b, heads, length, dim = q.shape
        q_flat = q.transpose(1, 2).reshape(-1, heads, dim)
        k_flat = k.transpose(1, 2).reshape(-1, k.shape[1], dim)
        v_flat = v.transpose(1, 2).reshape_as(k_flat)
        out = flash_attn_varlen_func(
            q_flat.index_select(0, layout.q_indices), k_flat.index_select(0, layout.k_indices),
            v_flat.index_select(0, layout.k_indices), layout.cu_q, layout.cu_k,
            layout.max_q, layout.max_k, dropout_p=0.0, causal=True,
        )
        return q_flat.new_zeros(q_flat.shape).index_copy(0, layout.q_indices, out).view(b, length, heads, dim).transpose(1, 2)
    repeat = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
    q_pos = torch.arange(layout.past_length, layout.past_length + q.shape[2], device=q.device)
    k_pos = torch.arange(k.shape[2], device=q.device)
    allowed = (k_pos[None, :] <= q_pos[:, None])[None, None] & layout.mask[:, None, None, :]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
    return out * layout.mask[:, None, layout.past_length:layout.past_length + q.shape[2], None]


class DecoderExecutor:
    def __init__(self, decoder, backend="flash_attention_2", activation_checkpointing=True):
        self.decoder = decoder
        self.backend = backend
        self.activation_checkpointing = activation_checkpointing
        if getattr(decoder.config, "use_sliding_window", False):
            raise ValueError("Sliding-window checkpoints are not supported by this full-attention recipe")
        if getattr(decoder.config, "attention_dropout", 0.0) != 0:
            raise ValueError("Controlled reference/recompute equivalence requires attention_dropout=0")

    def _layer(self, layer, hidden, past_k, past_v, cos, sin, layout):
        residual = hidden
        x = layer.input_layernorm(hidden)
        attn = layer.self_attn
        b, length, _ = x.shape
        q = attn.q_proj(x).view(b, length, -1, attn.head_dim).transpose(1, 2)
        new_k = attn.k_proj(x).view(b, length, -1, attn.head_dim).transpose(1, 2)
        new_v = attn.v_proj(x).view(b, length, -1, attn.head_dim).transpose(1, 2)
        q, new_k = multimodal_rope(q, new_k, cos, sin, attn.rope_scaling["mrope_section"])
        k = torch.cat((*past_k, new_k), dim=2)
        v = torch.cat((*past_v, new_v), dim=2)
        a = attend(q, k, v, layout, self.backend).transpose(1, 2).reshape(b, length, -1)
        x = residual + attn.o_proj(a)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return x, new_k, new_v

    def chunk(self, embeds, attention_mask, position_ids, cache=None):
        past_length = 0 if cache is None else sum(piece.shape[2] for piece in cache[0][0])
        layout = AttentionLayout.make(attention_mask, past_length, embeds.shape[1], self.backend)
        cos, sin = self.decoder.rotary_emb(embeds, position_ids)
        hidden, next_cache = embeds, []
        for index, layer in enumerate(self.decoder.layers):
            if cache is None:
                past_k, past_v = (), ()
            else:
                past_k, past_v = cache[index]
            # Bind layer/layout NOW: checkpoint executes again after the loop has advanced.
            def forward(x, c, s, *history, layer=layer, layout=layout, count=len(past_k)):
                return self._layer(layer, x, history[:count], history[count:], c, s, layout)
            if self.activation_checkpointing and torch.is_grad_enabled():
                hidden, new_k, new_v = checkpoint(forward, hidden, cos, sin, *past_k, *past_v, use_reentrant=False)
            else:
                hidden, new_k, new_v = forward(hidden, cos, sin, *past_k, *past_v)
            next_cache.append(((*past_k, new_k), (*past_v, new_v)))
        return self.decoder.norm(hidden), tuple(next_cache)


@dataclass
class Trajectory:
    hidden: torch.Tensor
    latents: dict


def run_trajectory(executor, embeds, attention_mask, position_ids, latent_positions,
                   injected=None, execution="incremental", events=None):
    """Position -> [(batch row, segment index, within-segment index)] on CPU.

    A union of positions across ranks may be passed as events for ZeRO-3, keeping
    every rank's parameter-gather call order identical. Injected values are keyed
    by (row, segment); only the FIRST latent position receives an injection.
    """
    injected = injected or {}
    length = embeds.shape[1]
    align_graph = events is not None
    events = sorted(latent_positions) if events is None else sorted(events)
    chunks, consumed, states = [], [], {}
    cache, cursor, previous = None, 0, None
    block_dependency = None

    def process(value, start, end):
        nonlocal cache, block_dependency
        if align_graph and block_dependency is not None:
            # Keep the same backward dependency between chunks on ZeRO-3 ranks,
            # including ranks whose current event is text or padding.
            value = value + block_dependency
        if execution == "recompute":
            prefix = torch.cat(consumed + [value], dim=1)
            h, _ = executor.chunk(prefix, attention_mask[:, :end], position_ids[:, :, :end])
            consumed.append(value)
            h = h[:, start:end]
        else:
            h, cache = executor.chunk(value, attention_mask[:, :end], position_ids[:, :, start:end], cache)
        if align_graph:
            block_dependency = h.sum() * 0.0
        return h

    for pos in events:
        if pos <= 0 or pos >= length:
            raise ValueError(f"Invalid continuous position {pos} for length {length}")
        if cursor < pos:
            text = process(embeds[:, cursor:pos], cursor, pos)
            chunks.append(text)
            previous = text[:, -1]
        active_rows = [False] * embeds.shape[0]
        row_values = [previous[row] for row in range(embeds.shape[0])]
        for row, segment, step in latent_positions.get(pos, []):
            value = injected[(row, segment)] if step == 0 and (row, segment) in injected else previous[row]
            row_values[row] = value
            active_rows[row] = True
            states[(row, segment, step)] = value
        # A tensor selection preserves the recurrence graph for all ranks, even
        # when every local row is text at a globally aligned latent position.
        active = torch.tensor(active_rows, dtype=torch.bool, device=embeds.device)
        value = torch.where(active[:, None], torch.stack(row_values), embeds[:, pos])[:, None]
        h = process(value, pos, pos + 1)
        chunks.append(h)
        previous, cursor = h[:, -1], pos + 1
    if cursor < length:
        chunks.append(process(embeds[:, cursor:], cursor, length))
    return Trajectory(torch.cat(chunks, dim=1), states)


def causal_ce_sum(hidden, labels, head, chunk_size=256, checkpoint_head=True, padded_count=None):
    """Only materialize logits for supervised next-token predictions.

    Head recomputation prevents all CE chunks' vocabulary logits remaining live.
    padded_count synchronizes head calls across ZeRO-3 ranks, without adding loss.
    """
    positions = labels[:, 1:].ne(-100).nonzero(as_tuple=False)
    selected = hidden[positions[:, 0], positions[:, 1]]
    targets = labels[positions[:, 0], positions[:, 1] + 1]
    count = max(len(targets), padded_count or 0, 1)
    if count > len(targets):
        selected = torch.cat((selected, hidden.new_zeros(count - len(targets), hidden.shape[-1])), 0)
        targets = torch.cat((targets, targets.new_full((count - len(targets),), -100)), 0)
    loss = hidden.sum() * 0.0
    for start in range(0, count, chunk_size):
        def linear_ce(h, y):
            return F.cross_entropy(head(h).float(), y, ignore_index=-100, reduction="sum")
        h, y = selected[start:start + chunk_size], targets[start:start + chunk_size]
        if checkpoint_head and torch.is_grad_enabled():
            loss = loss + checkpoint(linear_ce, h, y, use_reentrant=False)
        else:
            loss = loss + linear_ce(h, y)
    return loss
