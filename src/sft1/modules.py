"""Reusable interaction core and the removable original-image warm-up adapter.

The core consumes a configured number of ordered H-dimensional vectors, regardless of
whether they came from the warm-up adapter or a later autoregressive stage.
The adapter learns one image summary per slot; CE alone does not align those
summaries with future autoregressive latents.
"""
from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..interaction.fusion import RMSNorm, SwiGLU
from .config import WarmupConfig


VISUAL_SLOTS = WarmupConfig.adapter_slots
POSITION_ENCODING = "sinusoidal_xy_split_base10000_raw_merged_grid_v1"
IMAGE_INDEX_ENCODING = "sinusoidal_zero_based_image_index_base10000_v1"


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def _check_dimensions(hidden_size, dim, heads, ffn_dim):
    for name, value in (("hidden_size", hidden_size), ("dim", dim),
                        ("heads", heads), ("ffn_dim", ffn_dim)):
        _positive_integer(name, value)
    if dim % heads:
        raise ValueError("dim must be divisible by heads")
    if dim < 2:
        raise ValueError("dim must be at least 2 for two-dimensional positions")


def _sinusoidal(values: Tensor, dim: int) -> Tensor:
    """Interleaved sin/cos encoding, computed in FP32, including odd widths."""
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, dim, 2, device=values.device, dtype=torch.float32)
        / dim
    )
    phase = values.float().unsqueeze(-1) * frequencies
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)[..., :dim]


class _SpatialEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.projection = nn.Linear(dim, dim, bias=False)

    def forward(self, coordinates):
        split = self.dim // 2
        encoded = torch.cat((_sinusoidal(coordinates[..., 0], split),
                             _sinusoidal(coordinates[..., 1], self.dim - split)), -1)
        return self.projection(encoded.to(self.projection.weight.dtype))


class _ImageIndexEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.projection = nn.Linear(dim, dim, bias=False)

    def forward(self, image_index, device):
        encoded = _sinusoidal(torch.tensor(image_index, device=device), self.dim)
        return self.projection(encoded.to(self.projection.weight.dtype))


class _Attention(nn.Module):
    """Attention for any query length, with defined empty/all-masked behavior."""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, query, memory, valid):
        batch, queries, dim = query.shape
        length = memory.shape[1]
        # Mask before projection so even nonfinite padding cannot leak through V.
        memory = memory.masked_fill(~valid.unsqueeze(-1), 0)
        q = self.q(query)
        k, v = self.k(memory), self.v(memory)
        # Preserve a zero-gradient dependency for an empty local text span.
        result = q * 0 + k[..., :0].sum() + v[..., :0].sum()
        active = valid.any(-1).nonzero(as_tuple=True)[0]
        if active.numel() and queries:
            q_heads = q[active].reshape(-1, queries, self.heads, dim // self.heads).transpose(1, 2)
            k_heads = k[active].reshape(-1, length, self.heads, dim // self.heads).transpose(1, 2)
            v_heads = v[active].reshape(-1, length, self.heads, dim // self.heads).transpose(1, 2)
            attended = F.scaled_dot_product_attention(
                q_heads, k_heads, v_heads,
                attn_mask=valid[active, None, None, :], dropout_p=0.0,
            ).transpose(1, 2).reshape(-1, queries, dim)
            result = result.index_copy(0, active, attended)
        return self.out(result)


def _valid_mask(mask, shape, device, name):
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}; got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise ValueError(f"{name} must have boolean dtype")
    if mask.device != device:
        raise ValueError(f"{name} must be on the same device as its features")
    return mask


@dataclass
class ImageMemory:
    """One question image, with processor-order post-merger grid coordinates.

    ``image_index`` is its zero-based occurrence in the original image_input
    list. Repeated paths retain distinct occurrences. Helper images must never
    be supplied here. A mask is optional because unpadded per-image arrays are
    the usual input representation.
    """
    features: Tensor
    coordinates: Tensor
    image_index: int
    mask: Optional[Tensor] = None


class InteractionCore(nn.Module):
    """Ordered concat-MLP -> local text -> latent vectors + question images."""
    def __init__(self, hidden_size, dim=1024, heads=8, ffn_dim=4096,
                 score_dim=256, image_branch_enabled=True, slots=VISUAL_SLOTS,
                 interaction_tokens=WarmupConfig.interaction_tokens):
        super().__init__()
        _check_dimensions(hidden_size, dim, heads, ffn_dim)
        _positive_integer("score_dim", score_dim)
        _positive_integer("slots", slots)
        _positive_integer("interaction_tokens", interaction_tokens)
        if not isinstance(image_branch_enabled, bool):
            raise ValueError("image_branch_enabled must be boolean")
        self.hidden_size = hidden_size
        self.dim = dim
        self.slots = slots
        self.interaction_tokens = interaction_tokens
        self.image_branch_enabled = image_branch_enabled
        self.visual_projection = nn.Linear(hidden_size, dim, bias=False)
        nn.init.xavier_uniform_(self.visual_projection.weight)
        self.query_mlp = nn.Sequential(
            nn.Linear(slots * dim, dim, bias=True), nn.SiLU(),
            nn.Linear(dim, interaction_tokens * dim, bias=True),
        )
        for layer in self.query_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.text_projection = nn.Linear(hidden_size, dim, bias=False)
        self.text_query_norm, self.text_memory_norm = RMSNorm(dim), RMSNorm(dim)
        self.text_attention = _Attention(dim, heads)
        self.text_ffn_norm = RMSNorm(dim)
        self.text_ffn = SwiGLU(dim, ffn_dim)
        self.latent_query_norm, self.latent_memory_norm = RMSNorm(dim), RMSNorm(dim)
        self.latent_attention = _Attention(dim, heads)
        self.final_ffn_norm = RMSNorm(dim)
        self.final_ffn = SwiGLU(dim, ffn_dim)
        self.output_norm = RMSNorm(dim)
        self.output_projection = nn.Linear(dim, hidden_size, bias=False)
        # Keep the shared parameter initialization independent of branch mode.
        self.alpha_text = nn.Parameter(torch.tensor(0.1))
        self.beta_text = nn.Parameter(torch.tensor(0.1))
        self.alpha_latent = nn.Parameter(torch.tensor(0.1))
        self.beta_final = nn.Parameter(torch.tensor(0.1))
        self.alpha_image = nn.Parameter(torch.tensor(0.1))
        self.image_projection = nn.Linear(hidden_size, dim, bias=False)
        self.image_index_encoding = _ImageIndexEncoding(dim)
        self.spatial_encoding = _SpatialEncoding(dim)
        self.image_query_norm, self.image_memory_norm = RMSNorm(dim), RMSNorm(dim)
        self.image_attention = _Attention(dim, heads)
        self.score_query_norm, self.score_read_norm = RMSNorm(dim), RMSNorm(dim)
        self.score_query = nn.Linear(dim, score_dim, bias=False)
        self.score_read = nn.Linear(dim, score_dim, bias=False)
        self.score_index = nn.Linear(dim, score_dim, bias=False)
        self.score_output = nn.Linear(score_dim, 1, bias=False)
        self.gate_query_norm = RMSNorm(dim)
        self.gate_latent_norm = RMSNorm(dim)
        self.gate_image_norm = RMSNorm(dim)
        self.image_gate = nn.Linear(3 * dim, 1, bias=True)
        nn.init.zeros_(self.image_gate.weight)
        nn.init.constant_(self.image_gate.bias, math.log(0.1 / 0.9))

    def build_query(self, latents):
        if latents.ndim != 3 or latents.shape[1:] != (self.slots, self.hidden_size):
            raise ValueError(f"latents must have shape [N,{self.slots},{self.hidden_size}]")
        projected = self.visual_projection(latents)
        # Slot order is the adapter query order (warm-up) / generation order (later).
        query = self.query_mlp(projected.reshape(projected.shape[0], self.slots * self.dim))
        return query.reshape(latents.shape[0], self.interaction_tokens, self.dim), projected

    def _read_images(self, query, images):
        reads = []
        for row, sample_images in enumerate(images):
            per_image, index_encodings = [], []
            for memory in sample_images:
                if not isinstance(memory, ImageMemory):
                    raise ValueError("images must contain lists of ImageMemory")
                features, coordinates = memory.features, memory.coordinates
                if features.ndim != 2 or features.shape[1] != self.hidden_size:
                    raise ValueError(f"image features must have shape [L,{self.hidden_size}]")
                if coordinates.shape != (features.shape[0], 2):
                    raise ValueError("image coordinates must have shape [L,2]")
                if features.device != query.device or coordinates.device != query.device:
                    raise ValueError("image features and coordinates must share the query device")
                if isinstance(memory.image_index, bool) or not isinstance(memory.image_index, int) or memory.image_index < 0:
                    raise ValueError("image_index must be a nonnegative integer")
                valid = _valid_mask(memory.mask, features.shape[:1], features.device, "image mask")
                # Empty pictures are excluded from the image-level softmax.
                if not valid.any():
                    continue
                features = features.detach().masked_fill(~valid[:, None], 0)
                coordinates = coordinates.masked_fill(~valid[:, None], 0)
                index_encoding = self.image_index_encoding(memory.image_index, query.device)
                projected = (self.image_projection(features) + index_encoding
                             + self.spatial_encoding(coordinates))
                per_image.append(self.image_attention(
                    self.image_query_norm(query[row:row + 1]),
                    self.image_memory_norm(projected.unsqueeze(0)), valid.unsqueeze(0),
                )[0])
                index_encodings.append(index_encoding)
            if not per_image:
                reads.append(torch.zeros_like(query[row]))
                continue
            image_reads = torch.stack(per_image)
            image_indices = torch.stack(index_encodings)
            scores = self.score_output(F.silu(
                self.score_query(self.score_query_norm(query[row]))
                + self.score_read(self.score_read_norm(image_reads))
                + self.score_index(image_indices)[:, None]
            )).squeeze(-1)
            weights = scores.float().softmax(dim=0).to(image_reads.dtype)
            reads.append((weights[..., None] * image_reads).sum(0))
        return torch.stack(reads) if reads else torch.zeros_like(query)

    def forward(self, text, latents, text_mask, images=None):
        """Return [N,K,H]; each ordered query reads its own sample's text/images."""
        query, projected_latents = self.build_query(latents)
        if text.ndim != 3 or text.shape[0] != latents.shape[0] or text.shape[-1] != self.hidden_size:
            raise ValueError(f"text must have shape [N,T,{self.hidden_size}]")
        if text.device != latents.device:
            raise ValueError("text and latents must share a device")
        valid = _valid_mask(text_mask, text.shape[:2], text.device, "text_mask")
        text_memory = self.text_projection(text.masked_fill(~valid.unsqueeze(-1), 0))
        query = query + self.alpha_text * self.text_attention(
            self.text_query_norm(query), self.text_memory_norm(text_memory), valid,
        )
        query = query + self.beta_text * self.text_ffn(self.text_ffn_norm(query))
        latent_read = self.latent_attention(
            self.latent_query_norm(query), self.latent_memory_norm(projected_latents),
            torch.ones(projected_latents.shape[:2], dtype=torch.bool, device=latents.device),
        )
        fused = query + self.alpha_latent * latent_read
        if self.image_branch_enabled:
            if images is None:
                images = [[] for _ in range(text.shape[0])]
            if len(images) != text.shape[0]:
                raise ValueError("images must have one image list per sample/segment")
            image_read = self._read_images(query, images)
            gate = torch.sigmoid(self.image_gate(torch.cat((
                self.gate_query_norm(query), self.gate_latent_norm(latent_read),
                self.gate_image_norm(image_read),
            ), -1)))
            fused = fused + self.alpha_image * gate * image_read
        fused = fused + self.beta_final * self.final_ffn(self.final_ffn_norm(fused))
        return self.output_projection(self.output_norm(fused))


class _ResamplerLayer(nn.Module):
    def __init__(self, dim, heads, ffn_dim):
        super().__init__()
        self.cross_query_norm, self.cross_memory_norm = RMSNorm(dim), RMSNorm(dim)
        self.cross_attention = _Attention(dim, heads)
        self.self_norm = RMSNorm(dim)
        self.self_attention = _Attention(dim, heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)

    def forward(self, query, memory, valid):
        query = query + self.cross_attention(
            self.cross_query_norm(query), self.cross_memory_norm(memory), valid,
        )
        normalized = self.self_norm(query)
        query = query + self.self_attention(
            normalized, normalized,
            torch.ones(query.shape[:2], dtype=torch.bool, device=query.device),
        )
        return query + self.ffn(self.ffn_norm(query))


class WarmupImageAdapter(nn.Module):
    """Two-layer resampler; removable without changing InteractionCore weights."""
    def __init__(self, hidden_size, dim=1024, heads=8, ffn_dim=4096, layers=2, slots=VISUAL_SLOTS):
        super().__init__()
        _check_dimensions(hidden_size, dim, heads, ffn_dim)
        _positive_integer("layers", layers)
        _positive_integer("slots", slots)
        self.hidden_size, self.slots = hidden_size, slots
        self.input_projection = nn.Linear(hidden_size, dim, bias=False)
        self.spatial_encoding = _SpatialEncoding(dim)
        self.queries = nn.Parameter(torch.empty(slots, dim))
        nn.init.normal_(self.queries, std=dim ** -0.5)
        self.layers = nn.ModuleList([_ResamplerLayer(dim, heads, ffn_dim) for _ in range(layers)])
        self.output_norm = RMSNorm(dim)
        self.output_projection = nn.Linear(dim, hidden_size, bias=False)

    def forward(self, features, coordinates, mask=None):
        if features.ndim != 3 or features.shape[-1] != self.hidden_size:
            raise ValueError(f"features must have shape [N,L,{self.hidden_size}]")
        if coordinates.shape != (*features.shape[:2], 2):
            raise ValueError("coordinates must have shape [N,L,2]")
        if features.device != coordinates.device:
            raise ValueError("features and coordinates must share a device")
        valid = _valid_mask(mask, features.shape[:2], features.device, "mask")
        # E_vis and merger are frozen; this stop-gradient does not freeze W_V.
        features = features.detach().masked_fill(~valid.unsqueeze(-1), 0)
        coordinates = coordinates.masked_fill(~valid.unsqueeze(-1), 0)
        memory = self.input_projection(features) + self.spatial_encoding(coordinates)
        query = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        for layer in self.layers:
            query = layer(query, memory, valid)
        return self.output_projection(self.output_norm(query))
