# -*- coding: utf-8 -*-
"""Verified numerical primitives for the 2-D toy (S=2, C=64).
Purpose: the numerical primitives (log-domain Poisson, the forward immigration-death kernel with
three self-checking identities, the reverse time grid, feature construction, a torch MLP and
row-wise categorical sampling). All of it is a function-level copy of the verified primitives of
the 10-dimensional toy package, whose body is byte-identical to the original experiment module
(sha256 recorded in PROVENANCE.json). Only S_DIMS was changed to 2 and the 10-dimensional data
model and import side effects were removed. The arithmetic of the sampling helpers is unchanged,

Function-level copies from the 10D verified module; the arithmetic of every function kept here
is unchanged. Removed relative to the source: the 10D data model (Toy10, PAIRS, TYPE_NAMES,
TYPE_PRIORS), the module-import mkdir side effect, and the 10D data loader. The 2-D data model
lives in closed-form pmf tables (data/data2d_ring_two_curves_pmf.npz); see toy2d/metrics.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# -- caliber constants (identical to the 10D verified module) --
EPS_T = 0.01          # T_O = 1/2 ln(kappa2 / EPS_T)
F_BASE = 0.004        # reverse grid t_1 = F_BASE * T_O
K_REV = 32            # reverse steps (NFE)   [work order 189 section 0: 64 -> 32]
CF_MULT = 2           # forward grid = CF_MULT * C (headroom for the truncated-renormalised kernel)
S_DIMS = 2            # the 2-D toy: dim 0, dim 1
LN2 = math.log(2.0)


# =====================================================================
# Poisson primitives (all in log domain)
# =====================================================================
def pois_logpmf(lam: float, nmax: int) -> np.ndarray:
    """log Pois(lam) on 0..nmax; lam<=0 gives delta_0 (-inf for zero mass)."""
    if lam <= 0:
        out = np.full(nmax + 1, -np.inf)
        out[0] = 0.0
        return out
    k = np.arange(nmax + 1, dtype=np.float64)
    lgk = np.cumsum(np.concatenate([[0.0], np.log(k[1:])]))
    return -lam + k * math.log(lam) - lgk


def pois_pmf(lam: float, nmax: int) -> np.ndarray:
    if lam <= 0:
        out = np.zeros(nmax + 1)
        out[0] = 1.0
        return out
    return np.exp(pois_logpmf(lam, nmax))


def trunc_pois_logpmf(lam: float, C: int) -> np.ndarray:
    """Truncated to 0..C-1 and renormalised (the data model's atomic level pmf)."""
    lp = pois_logpmf(lam, C - 1)
    m = lp.max()
    lz = m + math.log(np.exp(lp - m).sum())
    return lp - lz


def trunc_pois_pmf(lam: float, C: int) -> np.ndarray:
    return np.exp(trunc_pois_logpmf(lam, C))


def pois_tail_beyond(lam: float, C: int) -> float:
    """Mass of the bare Pois(lam) at >= C (how much the truncation hurts)."""
    return float(max(0.0, 1.0 - pois_pmf(lam, C - 1).sum()))


def binom_pmf(n: int, p: float, nmax: int) -> np.ndarray:
    out = np.zeros(nmax + 1)
    if p >= 1.0:
        if n <= nmax:
            out[n] = 1.0
        return out
    if p <= 0.0:
        out[0] = 1.0
        return out
    m = np.arange(0, min(n, nmax) + 1)
    lg = np.cumsum(np.concatenate([[0.0], np.log(np.arange(1, n + 1))])) if n > 0 else np.zeros(1)
    out[m] = np.exp(lg[n] - lg[m] - lg[n - m] + m * math.log(p) + (n - m) * math.log1p(-p))
    return out


# =====================================================================
# Forward kernel (immigration-death)
# =====================================================================
TRUNC_TOL = 1e-12
_KCACHE: dict = {}
_MAXLEAK = 0.0


def kmat(t: float, V: float, C: int) -> np.ndarray:
    """q_t(m|n), shape [n, m] on 0..C-1. t=0 is the identity. Low-half row truncation
    > 1e-12 raises."""
    global _MAXLEAK
    if t <= 0.0:
        return np.eye(C)
    key = (round(t, 12), round(V, 12), C)
    if key in _KCACHE:
        return _KCACHE[key]
    b = math.exp(-t)
    P = pois_pmf(V * (1.0 - b), C - 1)
    K = np.stack([np.convolve(binom_pmf(n, b, C - 1), P)[:C] for n in range(C)])
    leak = 1.0 - K.sum(1)
    _MAXLEAK = max(_MAXLEAK, float(leak.max()))
    lo = float(leak[: C // 2 + 1].max())
    if lo > TRUNC_TOL:
        raise RuntimeError(f"kmat low-half truncation {lo:.2e} > {TRUNC_TOL:.0e} (C={C}, V={V:g}, t={t:g})")
    K = K / K.sum(1, keepdims=True)
    _KCACHE[key] = K
    return K


def self_check(V: float, C: int) -> dict:
    """Semigroup / stationary / row-sum identities; any failure stops."""
    semi = float(np.abs(kmat(0.7, V, C) @ kmat(1.3, V, C) - kmat(2.0, V, C)).max())
    pi = trunc_pois_pmf(V, C)
    stat = float(np.abs(pi @ kmat(3.0, V, C) - pi).max())
    rows = float(np.abs(kmat(1.0, V, C).sum(1) - 1).max())
    if semi > 1e-9 or stat > 1e-9 or rows > 1e-12:
        raise RuntimeError(f"forward-kernel self-check FAILED: semigroup={semi:.2e} "
                           f"stationary={stat:.2e} rowsum={rows:.2e}")
    return {"semigroup": semi, "stationary": stat, "rowsum": rows,
            "max_row_leak_before_renorm": _MAXLEAK}


def tgrid(T: float, K: int = K_REV, f_base: float = F_BASE) -> np.ndarray:
    """t_0 = 0, the rest log-equidistant up to T."""
    return np.concatenate([[0.0], np.linspace(f_base * T, T, K)])


# =====================================================================
# Features + network (torch, float64)
# =====================================================================
NFEAT_PER_GENE = 4
NFEAT_GLOBAL = 4


def nfeat(S: int = S_DIMS, onehot: int = 0) -> int:
    return NFEAT_PER_GENE * S + NFEAT_GLOBAL + S * onehot


def feats(nt: np.ndarray, t: np.ndarray, C: int, T_O: float,
          onehot: int = 0) -> np.ndarray:
    """[N,S] int + [N] t -> [N, 4S+4 (+S*onehot)] float64 (per-dim u, u^2, sqrt(u), log1p;
    global beta group; optional per-dim one-hot for per-cell resolution at small t)."""
    nt = np.asarray(nt, np.float64)
    t = np.asarray(t, np.float64)
    u = nt / (C - 1)
    b = np.exp(-t)
    per = np.concatenate([u, u * u, np.sqrt(u), np.log1p(nt) / math.log1p(C)], axis=1)
    glob = np.stack([b, b * b, np.sqrt(b), t / T_O], axis=1)
    parts = [per, glob]
    if onehot > 0:
        N, S = nt.shape
        oh = np.zeros((N, S, onehot))
        idx = np.clip(nt.astype(np.int64), 0, onehot - 1)
        oh[np.arange(N)[:, None], np.arange(S)[None, :], idx] = 1.0
        parts.append(oh.reshape(N, S * onehot))
    return np.concatenate(parts, axis=1)


def build_net(S: int, C: int, hidden: int = 512, seed: int = 0, onehot: int = 0):
    """2-hidden-layer tanh MLP -> [N, S, C] logits. torch float64."""
    import torch
    torch.manual_seed(seed)
    net = _Net(S, C, hidden, onehot)
    return net.double()


def _torch():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False   # red line: tf32 off
    torch.backends.cudnn.allow_tf32 = False
    return torch


def _make_net_class():
    import torch

    class Net2D(torch.nn.Module):
        def __init__(self, S, C, hidden, onehot=0):
            super().__init__()
            self.S, self.C, self.onehot = S, C, onehot
            self.l1 = torch.nn.Linear(nfeat(S, onehot), hidden)
            self.l2 = torch.nn.Linear(hidden, hidden)
            self.l3 = torch.nn.Linear(hidden, S * C)

        def forward(self, x):
            h = torch.tanh(self.l1(x))
            h = torch.tanh(self.l2(h))
            return self.l3(h).view(-1, self.S, self.C)

    return Net2D


_NET_CLS = None


def _Net(S, C, hidden, onehot=0):
    global _NET_CLS
    if _NET_CLS is None:
        _torch()
        _NET_CLS = _make_net_class()
    return _NET_CLS(S, C, hidden, onehot)


# =====================================================================
# Sampling helpers (bitwise determinism depends on these being the ONLY copy)
# =====================================================================
def draw_rows(rng, P: np.ndarray) -> np.ndarray:
    """Row-wise categorical. P [N, C] non-negative (renormalised internally)."""
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
    cdf = np.cumsum(P, 1)
    r = rng.random((len(P), 1))
    return (cdf < r).sum(1).clip(0, P.shape[1] - 1)


def draw_iid(rng, pmf: np.ndarray, n: int) -> np.ndarray:
    cdf = np.cumsum(pmf)
    return np.searchsorted(cdf, rng.random(n) * cdf[-1], side="right").clip(0, len(pmf) - 1)


def softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def write_json(path: Path, obj) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
