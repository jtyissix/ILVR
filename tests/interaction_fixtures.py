"""Small differentiable decoder matching the Qwen execution interface."""
from types import SimpleNamespace
import torch
from torch import nn
from src.interaction.fusion import RMSNorm, SwiGLU


class TinyRotary(nn.Module):
    def forward(self, x, position_ids):
        angles = position_ids.float()[..., None] * torch.arange(1, 5, device=x.device)[None, None, None] / 100
        angles = torch.cat((angles, angles), -1)
        return angles.cos().to(x.dtype), angles.sin().to(x.dtype)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 8
        self.rope_scaling = {"mrope_section": [1, 1, 2]}
        self.q_proj = nn.Linear(24, 24)
        self.k_proj = nn.Linear(24, 8)
        self.v_proj = nn.Linear(24, 8)
        self.o_proj = nn.Linear(24, 24, bias=False)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(24)
        self.post_attention_layernorm = RMSNorm(24)
        self.self_attn = TinyAttention()
        self.mlp = SwiGLU(24, 40)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_sliding_window=False, attention_dropout=0)
        self.embed_tokens = nn.Embedding(96, 24)
        self.layers = nn.ModuleList([TinyLayer(), TinyLayer()])
        self.norm = RMSNorm(24)
        self.rotary_emb = TinyRotary()


def simple_batch(length=16):
    ids = torch.arange(1, length+1)[None]
    positions = torch.arange(length)[None, None].expand(3, 1, -1)
    return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids, dtype=torch.bool),
            "position_ids": positions, "latent_positions": {}, "segments": [[]]}


def latent_batch(extra=0):
    ids = torch.tensor([[3, 4, 5, 70] + [71]*(8+extra) + [72, 6, 7, 8]])
    length = ids.shape[1]
    pads = list(range(4, 12+extra))
    labels = ids.clone()
    labels[:, :2] = -100
    labels[:, pads] = -100
    return {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids, dtype=torch.bool),
            "position_ids": torch.arange(length)[None, None].expand(3, 1, -1),
            "latent_positions": {p: [(0, 0, i)] for i, p in enumerate(pads)},
            "segments": [[{"start": 3, "end": 12+extra, "pads": pads, "text": [2]}]]}
