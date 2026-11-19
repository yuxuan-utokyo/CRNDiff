# moved from cc/experiments/EXP21_strengthen_ours_joint/run_exp21.py::AttnX0Posterior / _cosine_sched.
# changes (item by item):
#   1. `DEV = O.DEVICE`(EXP21 takes it from EXP08/ours.py) → directly in this file
#      `DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
#      checked: EXP08/ours.py::DEVICE is an equivalent definition of this line; numerical behaviour identical.
#   2. only move the two objects AttnX0Posterior and _cosine_sched (needed by D3PM / DDPM / OURS),
#      variants not needed in this round, such as ResMLPX0Posterior, were not moved.
# class and function bodies unchanged.
"""OURS / D3PM / DDPM shared cross-gene attention trunk."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttnX0Posterior(nn.Module):
    def __init__(self, nmax, dim, d_model=128, nheads=4, nlayers=3, input_scale=None):
        super().__init__()
        self.dim = dim; self.nmax = nmax
        if input_scale is None:
            input_scale = np.ones(dim, dtype=np.float64)
        self.register_buffer("input_scale", torch.tensor(input_scale, dtype=torch.float32))
        self.register_buffer("gene_ids", torch.arange(dim))
        self.count_proj = nn.Linear(2, d_model)           # [scaled count, log1p-normalized count] -> token
        self.gene_emb = nn.Embedding(dim, d_model)          # learned per-gene positional embedding
        self.time_mlp = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nheads, dim_feedforward=4 * d_model,
                                         batch_first=True, activation="gelu", dropout=0.0)
        self.transformer = nn.TransformerEncoder(enc, nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, nmax + 1)
        self._lognmax = math.log1p(nmax)

    def forward(self, n_t, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        sc = (n_t * self.input_scale).unsqueeze(-1)                       # [B,dim,1]
        lg = (torch.log1p(n_t) / self._lognmax).unsqueeze(-1)            # [B,dim,1]
        tok = self.count_proj(torch.cat([sc, lg], -1))                   # [B,dim,d]
        tok = tok + self.gene_emb(self.gene_ids)[None]                   # + per-gene id
        tok = tok + self.time_mlp(t)[:, None, :]                         # + time
        h = self.norm(self.transformer(tok))                            # [B,dim,d]
        return self.head(h)                                             # [B,dim,nmax+1]

    @torch.no_grad()
    def post(self, n_t, t):
        return F.softmax(self.forward(n_t, t), dim=2)


def _cosine_sched(opt, steps, warmup):
    def f(s):
        if s < warmup:
            return (s + 1) / max(warmup, 1)
        p = (s - warmup) / max(steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)
