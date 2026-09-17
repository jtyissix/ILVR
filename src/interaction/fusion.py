"""Visual -> text -> visual round-trip fusion (design equations 13--23)."""
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, query, context, valid):
        n, length, dim = context.shape
        q = self.q(query).view(n, 1, self.heads, -1).transpose(1, 2)
        k = self.k(context).view(n, length, self.heads, -1).transpose(1, 2)
        v = self.v(context).view(n, length, self.heads, -1).transpose(1, 2)
        # A dummy valid entry makes an empty local text span well-defined on all backends.
        present = valid.any(-1)
        safe = valid.clone()
        safe[:, 0] |= ~present
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=safe[:, None, None, :], dropout_p=0.0)
        a = self.out(a.transpose(1, 2).reshape(n, 1, dim))
        return a * present[:, None, None].to(a.dtype)


class RoundTripFusion(nn.Module):
    def __init__(self, hidden_size, dim=1024, heads=8, ffn_dim=4096):
        super().__init__()
        self.text_projection = nn.Linear(hidden_size, dim, bias=False)
        self.visual_projection = nn.Linear(hidden_size, dim, bias=False)
        self.query_init = nn.Linear(2 * dim, dim, bias=False)
        self.text_attention = CrossAttention(dim, heads)
        self.visual_attention = CrossAttention(dim, heads)
        self.text_ffn = SwiGLU(dim, ffn_dim)
        self.visual_ffn = SwiGLU(dim, ffn_dim)
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(7)])
        self.scales = nn.Parameter(torch.full((4,), 0.1))
        self.output_projection = nn.Linear(dim, hidden_size, bias=False)

    def forward(self, text, visual, text_mask):
        """text [N,M,H], visual [N,8,H]; N packs segments across samples."""
        if text.shape[1] == 0:
            text = text.new_zeros(text.shape[0], 1, text.shape[-1])
            text_mask = torch.zeros(text.shape[:2], dtype=torch.bool, device=text.device)
        t = self.text_projection(text)
        z = self.visual_projection(visual)
        stats = z.float()
        mean = stats.mean(1)
        std = ((stats - mean[:, None]).square().mean(1) + 1e-6).sqrt()
        q = self.query_init(torch.cat((mean, std), -1).to(z.dtype))[:, None]
        q = q + self.scales[0] * self.text_attention(self.norms[0](q), self.norms[1](t), text_mask)
        q = q + self.scales[1] * self.text_ffn(self.norms[2](q))
        visual_mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        q = q + self.scales[2] * self.visual_attention(self.norms[3](q), self.norms[4](z), visual_mask)
        q = q + self.scales[3] * self.visual_ffn(self.norms[5](q))
        return self.output_projection(self.norms[6](q))[:, 0]
