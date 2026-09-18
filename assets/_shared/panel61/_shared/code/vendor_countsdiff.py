# 搬自 cc/experiments/EXP08_toy_countsdiff/countsdiff.py（整文件），改动：仅在文件顶部加入本行注释；
# 其余逐字未动（p_schedule / w_weight / sample_countsdiff / CountsDiffNet 全部原样）。
# EXP134 中以 `import countsdiff as C` 使用。只依赖 math / numpy / torch。
"""Faithful CountsDiff reproduction (arXiv 2604.03779, §3 + Alg 1/2).

Forward (pure-death, binomial thinning), per coordinate:
    q(x_t | x_0) = Binomial(x_0, p(t)),   p(t) = cos^2(pi t / 2),  p(0)=1, p(1)=0.

Network predicts y_t = x_0 - x_t (counts dropped so far):
    yhat = softplus(NN_theta(x_t, t))      # small JOINT MLP over all 10 dims

Loss (eq 6), cosine schedule weight w(t) = (pi/2) sin(pi t) (= NLL, paper B.5):
    E_{t~U[0,1]} [ w(t) * ( yhat - y_t * log yhat ) ]

Reverse (Algorithm 2, with attrition eta), from x_1 = 0, t: 1 -> 0:
    sigma_max = min(1, (1 - p(s)) / p(t))
    sigma     = eta * sigma_max
    beta      = (p(s) - (1 - sigma) p(t)) / (1 - p(t))
    yhat_clip = random_round(yhat) = floor(yhat) + Bernoulli(frac)
    b ~ Binomial(yhat_clip, beta)
    n ~ Binomial(x_t, 1 - sigma)
    x_s = b + n

Unconditional generation -> no CFG. Attrition eta is the ONLY design knob, swept on
val (including eta=0 = pure-birth). Training = Algorithm 1.

Architecture is chosen identical to the codex MomentumNet (in_dim=11, hidden=128,
depth=3, SiLU, out_dim=10) so the parameter count is an EXACT match (35,850 params)
to our method's x0/momentum net (within the required +-10%).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_DIM = 10

# matches codex MomentumNet (run_momentum_refbatch_training.py:45-66)
HIDDEN = 128
DEPTH = 3


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def p_schedule(t: float | np.ndarray | torch.Tensor):
    """p(t) = cos^2(pi t / 2);  p(0)=1, p(1)=0."""
    if isinstance(t, torch.Tensor):
        return torch.cos(math.pi * t / 2.0) ** 2
    return np.cos(math.pi * np.asarray(t, dtype=np.float64) / 2.0) ** 2


def w_weight(t: torch.Tensor) -> torch.Tensor:
    """Loss weight w(t) = (pi/2) sin(pi t)  (cosine schedule == NLL)."""
    return (math.pi / 2.0) * torch.sin(math.pi * t)


# --------------------------------------------------------------------------- #
# Network: yhat = softplus(NN(x_t, t))
# --------------------------------------------------------------------------- #
class CountsDiffNet(nn.Module):
    def __init__(self, dim: int = N_DIM, hidden: int = HIDDEN, depth: int = DEPTH,
                 input_scale: np.ndarray | None = None) -> None:
        super().__init__()
        self.dim = dim
        # Per-dim input featurization scale (constant; does NOT change param count
        # or the method -- just keeps the heavy-tail input O(1) for stable training).
        if input_scale is None:
            input_scale = np.ones(dim, dtype=np.float64)
        self.register_buffer("input_scale", torch.tensor(input_scale, dtype=torch.float32))
        layers: list[nn.Module] = []
        in_dim = dim + 1
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: [B, dim] (float counts), t: [B] or [B,1] in [0,1]
        if t.dim() == 1:
            t = t.unsqueeze(1)
        xs = x_t * self.input_scale  # featurization only
        h = self.net(torch.cat([xs, t], dim=1))
        return nn.functional.softplus(h)  # yhat > 0


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Training (Algorithm 1)
# --------------------------------------------------------------------------- #
def train_countsdiff(
    train_counts: np.ndarray,
    *,
    steps: int = 12000,
    batch_size: int = 1024,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    seed: int = 20260625,
    input_scale: np.ndarray | None = None,
    val_counts: np.ndarray | None = None,
    log_every: int = 0,
    log_fn=None,
) -> tuple[CountsDiffNet, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)

    x0_all = torch.tensor(train_counts, dtype=torch.float32)  # [N, dim]
    n_train = x0_all.shape[0]
    dim = x0_all.shape[1]

    model = CountsDiffNet(dim=dim, input_scale=input_scale).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_curve: list[float] = []
    val_curve: list[float] = []
    eps = 1e-6
    for step in range(steps):
        idx = torch.randint(0, n_train, (batch_size,), generator=g)
        x0 = x0_all[idx].to(DEVICE)                       # [B, dim]
        t = torch.rand(batch_size, generator=g).to(DEVICE)  # U[0,1]
        p_t = p_schedule(t).unsqueeze(1)                  # [B,1]
        # forward binomial thinning per coordinate: x_t ~ Bin(x0, p(t))
        x_t = torch.binomial(x0, p_t.expand_as(x0))       # [B, dim]
        y_t = x0 - x_t                                    # target counts dropped
        yhat = model(x_t, t)                              # [B, dim] > 0
        # eq6 integrand: w(t) * sum_dims ( yhat - y_t * log yhat )
        per = yhat - y_t * torch.log(yhat.clamp_min(eps))
        loss = (w_weight(t).unsqueeze(1) * per).sum(dim=1).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_curve.append(float(loss.detach().cpu()))

        if log_every and val_counts is not None and (step + 1) % log_every == 0:
            vl = _val_loss(model, val_counts, seed=seed + 7)
            val_curve.append((step + 1, vl))
            if log_fn:
                log_fn(f"    step {step+1}/{steps} train_loss={loss_curve[-1]:.4f} val_loss={vl:.4f}")

    info = {
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "n_params": count_params(model),
        "final_train_loss": float(np.mean(loss_curve[-50:])),
        "val_curve": val_curve,
    }
    return model, info


@torch.no_grad()
def _val_loss(model: CountsDiffNet, val_counts: np.ndarray, *, seed: int, n: int = 4096) -> float:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x0_all = torch.tensor(val_counts, dtype=torch.float32)
    idx = torch.randint(0, x0_all.shape[0], (min(n, x0_all.shape[0]),), generator=g)
    x0 = x0_all[idx].to(DEVICE)
    t = torch.rand(x0.shape[0], generator=g).to(DEVICE)
    p_t = p_schedule(t).unsqueeze(1)
    x_t = torch.binomial(x0, p_t.expand_as(x0))
    y_t = x0 - x_t
    yhat = model(x_t, t)
    per = yhat - y_t * torch.log(yhat.clamp_min(1e-6))
    return float((w_weight(t).unsqueeze(1) * per).sum(dim=1).mean().cpu())


# --------------------------------------------------------------------------- #
# Reverse sampling (Algorithm 2, with attrition)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_countsdiff(
    model: CountsDiffNet,
    *,
    n_samples: int,
    eta: float,
    n_steps: int = 200,
    seed: int = 20260801,
    return_traj: bool = False,
) -> np.ndarray:
    """Generate counts by reversing from x_1 = 0 (Algorithm 2).

    Time grid: uniform t_k from 1 down to 0 with (n_steps+1) edges.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    edges = torch.linspace(1.0, 0.0, n_steps + 1, device=DEVICE)  # t = 1 ... 0
    x = torch.zeros(n_samples, model.dim, device=DEVICE)          # x_1 = 0

    traj = [x.clone().cpu().numpy()] if return_traj else None
    for k in range(n_steps):
        t = float(edges[k].item())      # current (larger) time
        s = float(edges[k + 1].item())  # next (smaller) time
        p_t = float(p_schedule(t))
        p_s = float(p_schedule(s))

        # sigma_max = min(1, (1-p_s)/p_t) ; guard p_t=0 -> +inf -> 1
        if p_t <= 0.0:
            sigma_max = 1.0
        else:
            sigma_max = min(1.0, (1.0 - p_s) / p_t)
        sigma = eta * sigma_max
        denom = (1.0 - p_t)
        if denom <= 0.0:
            # p_t = 1 only at t=0 (final edge); loop won't reach it as a "from" node
            beta = 0.0
        else:
            beta = (p_s - (1.0 - sigma) * p_t) / denom
        beta = min(max(beta, 0.0), 1.0)
        keep = 1.0 - sigma  # Bin(x_t, 1-sigma)

        t_vec = torch.full((n_samples,), t, device=DEVICE)
        yhat = model(x, t_vec)  # [B, dim] > 0
        # randomized rounding: floor + Bernoulli(frac) -- prevents 0-collapse
        floor = torch.floor(yhat)
        frac = yhat - floor
        yhat_clip = floor + torch.bernoulli(frac, generator=g)
        yhat_clip = yhat_clip.clamp_min(0.0)

        b = torch.binomial(yhat_clip, torch.full_like(yhat_clip, beta))
        n_keep = torch.binomial(x, torch.full_like(x, keep))
        x = b + n_keep
        if return_traj:
            traj.append(x.clone().cpu().numpy())

    out = x.cpu().numpy().astype(np.int64)
    if return_traj:
        return out, np.stack(traj, axis=0)
    return out
