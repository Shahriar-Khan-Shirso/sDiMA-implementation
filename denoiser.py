import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
import math
import copy

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 480


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device) / (half - 1))
        args  = t[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ScoreEstimator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=512,
                 n_layers=6, n_heads=8, use_grad_ckpt=True):
        super().__init__()
        self.latent_dim    = latent_dim
        self.hidden_dim    = hidden_dim
        self.use_grad_ckpt = use_grad_ckpt

        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)

        self.time_emb  = SinusoidalTimeEmb(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        self.time_gates = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1, batch_first=True, norm_first=True
            ) for _ in range(n_layers)
        ])

        self.skip_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.deep_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_t, t, mask, x_self_cond=None):
        if x_self_cond is None:
            x_self_cond = torch.zeros_like(x_t)

        if self.training:
            x_t.requires_grad_(True)

        h = torch.cat([x_t, x_self_cond], dim=-1)
        h = self.input_proj(h)

        t_emb = F.silu(self.time_proj(self.time_emb(t)))
        t_emb = t_emb.unsqueeze(1)

        src_key_padding_mask = ~mask

        shallow_out = None
        deep_out    = None
        n = len(self.blocks)

        for i, (block, gate) in enumerate(zip(self.blocks, self.time_gates)):
            h = h + gate(t_emb)

            if self.use_grad_ckpt and self.training:
                h = grad_ckpt(block, h, None, src_key_padding_mask, use_reentrant=False)
            else:
                h = block(h, src_key_padding_mask=src_key_padding_mask)

            if i == n // 2 - 1:
                shallow_out = h
            if i == n - 2:
                deep_out = h

        if shallow_out is not None and deep_out is not None:
            h = self.skip_proj(torch.cat([h, shallow_out], dim=-1))
            h = self.deep_proj(torch.cat([h, deep_out],   dim=-1))

        out = self.out_proj(self.out_norm(h))
        out = out * mask.unsqueeze(-1).float()
        return out


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model  = model
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    def update(self):
        with torch.no_grad():
            for sp, mp in zip(self.shadow.parameters(), self.model.parameters()):
                sp.data.copy_(self.decay * sp.data + (1 - self.decay) * mp.data)

    def get_model(self):
        return self.shadow
