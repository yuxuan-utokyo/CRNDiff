# 搬自 cc/experiments/EXP08_toy_countsdiff/ours.py（整文件），改动：仅在文件顶部加入本行注释；
# 其余每一个字节逐字未动（KernelBank / make_learned_post_fn / sample_bridge / reverse_grid /
# _categorical / DEVICE / T_MAX / T_FLOOR 全部原样）。EXP134 中以 `import ours as O` 使用。
# 该文件只依赖 math / functools / typing / numpy / torch —— 无任何 cc/experiments/ 依赖。
"""OURS — joint immigration-death x0-posterior + 3 samplers (bridge / bridge+attrition / bare-tau) + oracle.

Sampling MATH is replicated VERBATIM from the validated engine
`cc/experiments/traj_viz/run_traj_viz.py` (kmat / rho_matrix / momentum_from_postrows /
tau_leap step / bridge step / prior). The only change is VECTORIZATION across the 40k
samples and 10 coordinates (the reference is scalar per-trajectory and far too slow);
every formula, the 1e-300 floors, and the geomspace(t_max,1e-2,K)+[0] reverse grid are
unchanged. A scalar-vs-vectorized equivalence check lives in `validate_vec_vs_scalar`.

Model: trunk identical to CountsDiff (in 11 -> hidden 128 x depth 3, SiLU); the only
difference is the output head — per-coordinate categorical over {0..NMAX} (NMAX from data
range + buffer). Forward = immigration-death q_t(n|n0)=Bin(n0,e^-t) * Pois(V(1-e^-t)),
stationary Pois(V), per coordinate; V_d = per-dim train mean.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_DIM = 10
HIDDEN = 128
DEPTH = 3
T_FLOOR = 1e-2          # reverse-grid floor (DESIGN: geomspace(t_max,1e-2,K)+[0])
T_MAX = 8.0             # forward horizon: e^-8 * max_n0 ~ negligible -> reaches Pois(V)


# --------------------------------------------------------------------------- #
# Exact forward kernel (VERBATIM from run_traj_viz.py:67-84, NMAX parameterized)
# --------------------------------------------------------------------------- #
def _logfact_arr(nmax: int) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, nmax + 2)))])


def kmat(tau: float, V: float, nmax: int, logfact: np.ndarray) -> np.ndarray:
    """K[n0,n] = q_tau(n|n0) = Bin(n0,e^-tau) (*) Pois(V(1-e^-tau)), 0..nmax. VERBATIM."""
    p = math.exp(-tau); lam = V * (1.0 - math.exp(-tau))
    js = np.arange(nmax + 1)
    pois = np.exp(-lam + js * math.log(lam + 1e-300) - logfact[js])
    K = np.zeros((nmax + 1, nmax + 1))
    for n0 in range(nmax + 1):
        ks = np.arange(n0 + 1)
        lb = (logfact[n0] - logfact[ks] - logfact[n0 - ks]
              + ks * math.log(p + 1e-300) + (n0 - ks) * math.log(1 - p + 1e-300))
        K[n0] = np.convolve(np.exp(lb), pois)[:nmax + 1]
    return K


def rho_matrix(K: np.ndarray) -> np.ndarray:
    """rho[n0,n] = q(n+1|n0)/q(n|n0). VERBATIM."""
    num = K[:, 1:]; den = np.clip(K[:, :-1], 1e-300, None)
    return num / den


def pois_prior(V: float, nmax: int, logfact: np.ndarray) -> np.ndarray:
    js = np.arange(nmax + 1)
    pr = np.exp(-V + js * math.log(V + 1e-300) - logfact[js])
    return pr / pr.sum()


# --------------------------------------------------------------------------- #
# Model: joint x0-posterior (trunk == CountsDiff; head = per-coord categorical)
# --------------------------------------------------------------------------- #
class X0PosteriorJoint(nn.Module):
    def __init__(self, nmax: int, dim: int = N_DIM, hidden: int = HIDDEN, depth: int = DEPTH,
                 input_scale: np.ndarray | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.nmax = nmax
        if input_scale is None:
            input_scale = np.ones(dim, dtype=np.float64)
        self.register_buffer("input_scale", torch.tensor(input_scale, dtype=torch.float32))
        layers: list[nn.Module] = []
        in_dim = dim + 1
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.SiLU())
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, dim * (nmax + 1))

    def forward(self, n_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        xs = n_t * self.input_scale
        h = self.trunk(torch.cat([xs, t], dim=1))
        logits = self.head(h).view(-1, self.dim, self.nmax + 1)
        return logits  # [B, dim, nmax+1]

    @torch.no_grad()
    def post(self, n_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(n_t, t), dim=2)  # [B, dim, nmax+1]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- #
# Training (x0-posterior NLL; immigration-death forward)
# --------------------------------------------------------------------------- #
def train_ours(
    train_counts: np.ndarray,
    *,
    V: np.ndarray,
    nmax: int,
    steps: int = 12000,
    batch_size: int = 1024,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    seed: int = 20260625,
    input_scale: np.ndarray | None = None,
    val_counts: np.ndarray | None = None,
    t_max: float = T_MAX,
    t_floor: float = T_FLOOR,
    log_every: int = 0,
    log_fn=None,
) -> tuple[X0PosteriorJoint, dict[str, Any]]:
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x0_all = torch.tensor(train_counts, dtype=torch.float32)
    n_train = x0_all.shape[0]
    dim = x0_all.shape[1]
    Vt = torch.tensor(V, dtype=torch.float32, device=DEVICE)  # [dim]

    model = X0PosteriorJoint(nmax, dim=dim, input_scale=input_scale).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    log_tmax, log_tfloor = math.log(t_max), math.log(t_floor)

    loss_curve: list[float] = []
    val_curve: list = []
    for step in range(steps):
        idx = torch.randint(0, n_train, (batch_size,), generator=g)
        x0 = x0_all[idx].to(DEVICE)                                  # [B, dim]
        # log-uniform t in [t_floor, t_max]
        u = torch.rand(batch_size, generator=g).to(DEVICE)
        t = torch.exp(log_tfloor + u * (log_tmax - log_tfloor))     # [B]
        r = torch.exp(-t).unsqueeze(1)                              # [B,1]
        lam = Vt.unsqueeze(0) * (1.0 - r)                          # [B,dim]
        # immigration-death forward: Bin(n0,e^-t) + Pois(V(1-e^-t))
        n_t = torch.binomial(x0, r.expand_as(x0)) + torch.poisson(lam)
        n_t = n_t.clamp(0, nmax)
        logits = model(n_t, t)                                      # [B,dim,nmax+1]
        target = x0.long().clamp(0, nmax)                          # [B,dim]
        loss = F.cross_entropy(logits.reshape(-1, nmax + 1), target.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        loss_curve.append(float(loss.detach().cpu()))
        if log_every and val_counts is not None and (step + 1) % log_every == 0:
            vl = _val_nll(model, val_counts, V, nmax, seed=seed + 7, t_max=t_max, t_floor=t_floor)
            val_curve.append((step + 1, vl))
            if log_fn:
                log_fn(f"    step {step+1}/{steps} train_nll={loss_curve[-1]:.4f} val_nll={vl:.4f}")

    info = {"steps": steps, "batch_size": batch_size, "lr": lr, "seed": seed,
            "nmax": nmax, "n_params": count_params(model),
            "final_train_nll": float(np.mean(loss_curve[-50:])), "val_curve": val_curve}
    return model, info


@torch.no_grad()
def _val_nll(model, val_counts, V, nmax, *, seed, n=4096, t_max=T_MAX, t_floor=T_FLOOR):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x0_all = torch.tensor(val_counts, dtype=torch.float32)
    idx = torch.randint(0, x0_all.shape[0], (min(n, x0_all.shape[0]),), generator=g)
    x0 = x0_all[idx].to(DEVICE)
    Vt = torch.tensor(V, dtype=torch.float32, device=DEVICE)
    u = torch.rand(x0.shape[0], generator=g).to(DEVICE)
    t = torch.exp(math.log(t_floor) + u * (math.log(t_max) - math.log(t_floor)))
    r = torch.exp(-t).unsqueeze(1)
    n_t = (torch.binomial(x0, r.expand_as(x0)) + torch.poisson(Vt.unsqueeze(0) * (1 - r))).clamp(0, nmax)
    logits = model(n_t, t)
    target = x0.long().clamp(0, nmax)
    return float(F.cross_entropy(logits.reshape(-1, nmax + 1), target.reshape(-1)).cpu())


# --------------------------------------------------------------------------- #
# Reverse grid (DESIGN: geomspace(t_max,1e-2,K)+[0], NEVER uniform-to-0)
# --------------------------------------------------------------------------- #
def reverse_grid(K: int, t_max: float = T_MAX, t_floor: float = T_FLOOR) -> np.ndarray:
    return np.concatenate([np.geomspace(t_max, t_floor, K), [0.0]])


# --------------------------------------------------------------------------- #
# Kernel cache per dataset (V vector fixed)
# --------------------------------------------------------------------------- #
class KernelBank:
    """Precompute/caches kmat & rho per coordinate over the reverse grid. Memory-bounded:
    stores Kgrid (kmat at each tgrid time), Kdelta (kmat at each dt), Rgrid (rho), per coord."""

    def __init__(self, V: np.ndarray, tgrid: np.ndarray, nmax: int):
        self.V = np.asarray(V, dtype=np.float64)
        self.dim = len(self.V)
        self.tgrid = np.asarray(tgrid, dtype=np.float64)
        self.nmax = nmax
        self.logfact = _logfact_arr(nmax)
        self._cache: dict[tuple, np.ndarray] = {}
        self.K = len(tgrid)
        # per-step kernels
        self.Kgrid = [[self._k(self.tgrid[k], self.V[d]) for d in range(self.dim)] for k in range(self.K)]
        self.Rgrid = [[rho_matrix(self.Kgrid[k][d]) for d in range(self.dim)] for k in range(self.K)]
        self.Kdelta = [[self._k(self.tgrid[k] - self.tgrid[k + 1], self.V[d]) for d in range(self.dim)]
                       for k in range(self.K - 1)]
        self.prior = np.stack([pois_prior(self.V[d], nmax, self.logfact) for d in range(self.dim)])  # [dim,nmax+1]

    def _k(self, tau: float, V: float) -> np.ndarray:
        key = (round(float(tau), 10), round(float(V), 8))
        if key not in self._cache:
            self._cache[key] = kmat(max(tau, 0.0), V, self.nmax, self.logfact)
        return self._cache[key]


# --------------------------------------------------------------------------- #
# Vectorized samplers. post_fn(n_t[B,dim], t) -> post[B,dim,nmax+1] (numpy)
# --------------------------------------------------------------------------- #
def _categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Robust categorical over last axis for unnormalized nonneg weights (VERBATIM _sample_idx
    logic: nan->0, clip>=0, cumsum/searchsorted, tot<=0 -> 0). Vectorized over rows."""
    p = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    c = np.cumsum(p, axis=1)
    tot = c[:, -1]
    u = rng.random(len(p)) * tot
    out = np.array([np.searchsorted(c[i], u[i]) for i in range(len(p))])
    out[~np.isfinite(tot) | (tot <= 0)] = 0
    return out.astype(np.int64)


def _categorical_torch(probs_t: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Fast categorical for [B, K] nonneg weights via torch.multinomial; tot<=0 -> 0."""
    p = torch.nan_to_num(probs_t, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    tot = p.sum(dim=1)
    bad = ~torch.isfinite(tot) | (tot <= 0)
    p[bad] = 0.0
    p[bad, 0] = 1.0
    out = torch.multinomial(p, 1, generator=gen).squeeze(1)
    return out


def sample_bare_tau(post_fn, bank: KernelBank, *, n_samples: int, seed: int, churn: float = 0.0) -> np.ndarray:
    """bare tau-leap on momentum p from posterior: lam=(n+1)e^-p(n), mu=V e^{p(n-1)}. VERBATIM math.

    The momentum FUNCTION p_t(n) re-queries the posterior conditioned on state n (reference
    Pmom[k,n] from post_fn(alln,t)). So p(n-1) for coord d uses the posterior conditioned on
    the state with coord d decremented -> one extra net eval per coord (decremented input).
    """
    rng = np.random.default_rng(seed)
    dim, nmax, tgrid = bank.dim, bank.nmax, bank.tgrid
    n = np.stack([_categorical(np.tile(bank.prior[d], (n_samples, 1)), rng) for d in range(dim)], axis=1)
    Vt = bank.V
    for k in range(len(tgrid) - 1):
        t = tgrid[k]; dt = tgrid[k] - tgrid[k + 1]
        post_here = post_fn(n.astype(np.float32), float(t))            # posterior | current n
        for d in range(dim):
            R = bank.Rgrid[k][d]
            nd = np.clip(n[:, d], 0, nmax)
            # p(n): e^{-p}=sum_n0 R[n0,min(n,nmax-1)] post(n0|n_t=n)
            col_here = np.clip(nd, 0, nmax - 1)
            epm_here = np.einsum("bn,bn->b", post_here[:, d, :], R[:, col_here].T)
            p_here = -np.log(np.maximum(epm_here, 1e-300))
            # p(n-1): posterior conditioned on coord d decremented (re-query)
            n_dec = n.copy(); n_dec[:, d] = np.clip(n[:, d] - 1, 0, nmax)
            post_dec = post_fn(n_dec.astype(np.float32), float(t))
            col_dn = np.clip(nd - 1, 0, nmax - 1)
            epm_dn = np.einsum("bn,bn->b", post_dec[:, d, :], R[:, col_dn].T)
            p_dn = -np.log(np.maximum(epm_dn, 1e-300))
            lam = (nd + 1) * np.exp(-p_here)
            mu = np.where(nd > 0, Vt[d] * np.exp(p_dn), 0.0)
            up = rng.poisson(np.maximum(lam * dt, 0.0))
            dn = rng.poisson(np.maximum(mu * dt, 0.0))
            n[:, d] = np.clip(nd + up - dn, 0, nmax)
    return n.astype(np.int64)


def sample_bridge(post_fn, bank: KernelBank, *, n_samples: int, seed: int, churn: float = 0.0) -> np.ndarray:
    """bridge: n0~q_theta then n_s ~ Kgrid[k+1][n0,:] * Kdelta[k][:,n] (VERBATIM bridge weights).
    churn>0 (bridge+attrition): after the bridge step, re-equilibrate a fraction toward the
    stationary Pois(V): n <- Bin(n, 1-churn) + Pois(V*churn) (immigration-death churn)."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    dim, nmax, tgrid = bank.dim, bank.nmax, bank.tgrid
    n = np.stack([_categorical(np.tile(bank.prior[d], (n_samples, 1)), rng) for d in range(dim)], axis=1)
    for k in range(len(tgrid) - 1):
        t = tgrid[k]
        post = post_fn(n.astype(np.float32), float(t))  # [B,dim,nmax+1]
        for d in range(dim):
            # sample n0 ~ posterior
            n0 = _categorical_torch(torch.tensor(post[:, d, :], device=DEVICE), gen).cpu().numpy()
            Knext = bank.Kgrid[k + 1][d]           # [nmax+1,nmax+1] rows=n0
            Kdel = bank.Kdelta[k][d]               # [nmax+1,nmax+1] K[n_s, n]?  K[a,b]=q(b|a)
            # br[n_s] = Knext[n0, n_s] * Kdel[n_s, n_current]   (VERBATIM: Kgrid[k+1][n0,:]*Kdelta[k][:,n])
            row = Knext[n0, :]                     # [B,nmax+1] over n_s
            col = Kdel[:, np.clip(n[:, d], 0, nmax)].T  # [B,nmax+1] over n_s
            br = row * col
            ns = _categorical_torch(torch.tensor(br, device=DEVICE), gen).cpu().numpy()
            n[:, d] = ns
        if churn > 0.0:
            keep = rng.binomial(n, 1.0 - churn)
            imm = rng.poisson(bank.V[None, :] * churn * np.ones_like(n))
            n = np.clip(keep + imm, 0, nmax)
    return n.astype(np.int64)


# --------------------------------------------------------------------------- #
# Oracle posterior (empirical-Bayes from train hist; per-coord) — VERBATIM oracle_post_rows
# --------------------------------------------------------------------------- #
def make_oracle_post_fn(train_counts: np.ndarray, bank: KernelBank):
    """Returns post_fn(n_t,t)->post using Pemp(n0)*K[n0,n_t], per coordinate.
    Uses the kernel at the QUERIED t (looked up from bank by nearest tgrid index)."""
    dim, nmax = bank.dim, bank.nmax
    Pemp = np.stack([np.bincount(np.clip(train_counts[:, d], 0, nmax), minlength=nmax + 1).astype(np.float64)
                     for d in range(dim)])
    Pemp = Pemp / Pemp.sum(axis=1, keepdims=True)  # [dim,nmax+1]
    tgrid = bank.tgrid

    def post_fn(n_t: np.ndarray, t: float) -> np.ndarray:
        k = int(np.argmin(np.abs(tgrid - t)))
        B = n_t.shape[0]
        out = np.zeros((B, dim, nmax + 1))
        for d in range(dim):
            K = bank.Kgrid[k][d]                       # [n0, n]
            nd = np.clip(n_t[:, d].astype(int), 0, nmax)
            w = Pemp[d][None, :] * K[:, nd].T          # [B, n0]
            s = w.sum(axis=1, keepdims=True)
            out[:, d, :] = np.where(s > 0, w / np.maximum(s, 1e-300), Pemp[d][None, :])
        return out
    return post_fn


def make_learned_post_fn(model: X0PosteriorJoint):
    @torch.no_grad()
    def post_fn(n_t: np.ndarray, t: float) -> np.ndarray:
        nt = torch.tensor(n_t, dtype=torch.float32, device=DEVICE)
        tt = torch.full((nt.shape[0],), float(t), device=DEVICE)
        return model.post(nt, tt).cpu().numpy()
    return post_fn
