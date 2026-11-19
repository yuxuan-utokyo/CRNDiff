# -*- coding: utf-8 -*-
"""EXP174 · **oracle posterior** sampling. From `sample_hvg2k.py` copy, replacing only `p = net.post(...)` one place.

    python Baseline/OURS/code/sample_oracle_hvg2k.py --stage gate     # pass two gates first
    python Baseline/OURS/code/sample_oracle_hvg2k.py --stage sample --ks 256

**`sample_hvg2k.py` / `model_hvg2k.py` / `bank_dedup.py` not a single byte changed** (red line).
this file is a new copy; the network is **not loaded at all**.

## question to answer (EXP174 section 0)

the variance deficit of low-expression genes: is it (A) the learned x0 posterior not being spread enough, or (B) the bridge reverse chain itself
discretization loss at low counts? Replace the learned posterior with an **analytic per-gene Bayesian posterior**, everything else unchanged:
low-expression bin back to 1.0 -> (A); still 0.85 -> (B).

## oracle posterior (section 1)

    p_oracle[b,d,c]  ∝  K_{t_k}^{(d)}[c, n_t[b,d]]  ·  π_d[c]

* `K` reuse the kernels already in the bank, **no new computation**.
* `π_d` = empirical count distribution of gene d on **train**, clipped to NMAX, plus `1e-8` then renormalize.

### ★ index convention: verified myself, take **columns** (consistent with the main-line reading)

`vendor_ours.kmat` docstring and implementation both hard-code

    K[n0, n] = q_tau(n | n0) = Bin(n0, e^-tau) (*) Pois(V(1-e^-tau))

(`K[n0] = np.convolve(binom(n0,·), pois)`, **the row index is n0**).
`DedupBank` in `Kgrid[k]` stores `layer(tau)` ****not transposed**, so
`Kgrid[k][d][c, n_t]` is `P(n_t | n_0=c)` — the oracle needs **column n_t**.

side evidence: the same `DedupBank` in `Kdelta_T` stores `layer(t).transpose(1, 2)`,
while the sampler uses for it `index_select(0, n_t)` — taking rows of the transpose = taking **columns** of the original matrix,
is exactly the second factor of the bridge `K_Δ(n_s -> n_t)` needed. The two conventions are consistent.

implemented with `U.transpose(1,2)` then by `[vi, n_t]` advanced indexing, obtained in one go `[B, dim, C]`,
and in the batched branch `Knext.U[vi_next, n0]` is the same indexing pattern.

## dtype

bank is float64, so the oracle `p` is **float64** (the network version is float32).
this **raises precision rather than lowering it**, so it does not violate the rule that float64 must not be downcast to float32.
the number and shapes of the two multinomial draws are unchanged; the RNG consumption order is kept verbatim.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
import config_hvg2k as CFG                              # noqa: E402
import model_hvg2k as M                                 # noqa: E402
from bank_dedup import DedupBank                        # noqa: E402

VO = importlib.import_module("panel61_shared.vendor_ours")
LOG = ROOT / "RUN_LOG.md"
DEV = M.DEV
TAG = "_ORACLE"


def rl(m):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"- `{time.strftime('%H:%M:%S')}` {m}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- the following three functions are copied verbatim from sample_hvg2k.py ----
def _cat(p, gen):
    """`vendor_ours._categorical_torch` verbatim semantics."""
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    tot = p.sum(dim=1)
    bad = ~torch.isfinite(tot) | (tot <= 0)
    if bad.any():
        p = p.clone(); p[bad] = 0.0; p[bad, 0] = 1.0
    return torch.multinomial(p, 1, generator=gen).squeeze(1)


def _slot_v_idx(slot, dev, _cache={}):
    key = id(slot)
    t = _cache.get(key)
    if t is None or t.device != dev:
        t = torch.as_tensor(slot.g2v, dtype=torch.long, device=dev)
        _cache[key] = t
    return t


# ---- added: oracle posterior ----
def emp_prior(train, nmax, dev):
    """π_d[c] = empirical count distribution of gene d on **train**, clipped to nmax, +1e-8 then renormalized."""
    X = np.clip(np.asarray(train), 0, nmax).astype(np.int64)
    dim = X.shape[1]
    P = np.zeros((dim, nmax + 1), np.float64)
    for d in range(dim):
        P[d] = np.bincount(X[:, d], minlength=nmax + 1)[:nmax + 1]
    P /= P.sum(1, keepdims=True)
    P += 1e-8                                    # Laplace smoothing, to avoid zero probabilities locking out categories forever
    P /= P.sum(1, keepdims=True)
    return torch.as_tensor(P, dtype=torch.float64, device=dev)


def _slot_UT(slot, _cache={}):
    """`U[v][n0, n]` -> `UT[v][n, n0]`, for `UT[vi, n_t]` taken in one go `P(n_t | n_0=·)`."""
    key = id(slot)
    t = _cache.get(key)
    if t is None:
        t = slot.U.transpose(1, 2).contiguous()
        _cache.clear()                           # keep only the current layer; 31.6 MB is not worth accumulating
        _cache[key] = t
    return t


def oracle_post(bank, k, n_t, pi, nmax):
    """p[b,d,c] ∝ K_{t_k}^{(d)}[c, n_t[b,d]] · pi_d[c]. **Unnormalized** (normalization is left to the verbatim downstream block)."""
    slot = bank.Kgrid[k]
    UT = _slot_UT(slot)                                        # [n_V, C, C] float64
    n, dim = n_t.shape
    vi = _slot_v_idx(slot, n_t.device)[None, :].expand(n, dim)
    p = UT[vi, n_t.clamp(0, nmax)]                             # [B, dim, C] = P(n_t|n_0=c)
    p = p * pi[None, :, :]                                     # × empirical prior
    return p


@torch.no_grad()
def sample_chunk_oracle(bank, pi, dmax_t, n, seed, nmax, batched: bool = True):
    """and `sample_hvg2k.sample_chunk` verbatim identical, **only `p = net.post(nt, tt)` replaced by the oracle**."""
    dev = bank.device
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    dim, tgrid = bank.dim, bank.tgrid
    prior = bank.prior.cpu().numpy()
    cdf = np.cumsum(prior, axis=1)
    u = rng.random((n, dim))
    n_t = torch.as_tensor(
        np.stack([np.searchsorted(cdf[d], u[:, d]) for d in range(dim)], 1).clip(0, nmax),
        dtype=torch.long, device=dev)
    for k in range(len(tgrid) - 1):
        # ---- the only replacement point: network posterior -> oracle posterior (nothing else changes)----
        p = oracle_post(bank, k, n_t, pi, nmax)                  # [B, dim, C] float64
        p = p.masked_fill(torch.arange(p.shape[2], device=dev)[None, None, :]
                          > dmax_t[None, :, None], 0.0)          # A4 posterior-support truncation
        s = p.sum(-1, keepdim=True)
        p = torch.where(s > 0, p / torch.clamp_min(s, 1e-30), p)
        Knext, KdelT = bank.Kgrid[k + 1], bank.Kdelta_T[k]
        C = p.shape[2]
        n0 = _cat(p.reshape(n * dim, C), gen).view(n, dim)       # first large multinomial
        del p
        vi_next = _slot_v_idx(Knext, dev)[None, :].expand(n, dim)
        vi_delt = _slot_v_idx(KdelT, dev)[None, :].expand(n, dim)
        row = Knext.U[vi_next, n0]                               # [B, dim, C] float64
        row.mul_(KdelT.U[vi_delt, n_t.clamp(0, nmax)])           # in-place multiply
        n_t = _cat(row.reshape(n * dim, C), gen).view(n, dim)    # second
        del row, n0
    return n_t.cpu().numpy().astype(np.int64)


# =============================== gates (section 1)=============================== #
@torch.no_grad()
def gates(bank, pi, dmax_t, train, nmax, n_probe=4000, seed=0):
    """gate 1: at the last grid point (t->0), argmax_c p_oracle == n_t with hit rate > 0.99.
       gate 2: each row of p sums to 1, no NaN/Inf."""
    dev = bank.device
    k_last = len(bank.tgrid) - 2                 # the smallest used in the loop tau
    tau = float(bank.tgrid[k_last])
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(train), 64, replace=False)
    n_t = torch.as_tensor(np.clip(np.asarray(train)[np.sort(rows)], 0, nmax),
                          dtype=torch.long, device=dev)
    p = oracle_post(bank, k_last, n_t, pi, nmax)
    p = p.masked_fill(torch.arange(p.shape[2], device=dev)[None, None, :]
                      > dmax_t[None, :, None], 0.0)
    s = p.sum(-1, keepdim=True)
    p = torch.where(s > 0, p / torch.clamp_min(s, 1e-30), p)

    # gate 2
    rs = p.sum(-1)
    g2 = {"tau": tau, "n_rows": int(rs.numel()),
          "max_abs_sum_minus_1": float((rs - 1.0).abs().max()),
          "n_nan": int(torch.isnan(p).sum()), "n_inf": int(torch.isinf(p).sum())}
    # gate 1: judged only where the A4 cut did not remove the true value (c=n_t must still be in the support)
    am = p.argmax(-1)
    keep = (n_t <= dmax_t[None, :])
    hit = ((am == n_t) & keep).sum().item()
    tot = int(keep.sum().item())
    g1 = {"tau": tau, "n_checked": tot, "n_hit": int(hit),
          "hit_rate": hit / max(tot, 1),
          "n_excluded_by_A4": int((~keep).sum().item())}
    del p
    return g1, g2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gate", "sample"], default="gate")
    ap.add_argument("--ks", type=int, nargs="+", default=[256])
    ap.add_argument("--samps", type=int, nargs="+", default=[20260901])
    ap.add_argument("--n-gen", type=int, default=5000)
    ap.add_argument("--cell-chunk", type=int, default=64)
    a = ap.parse_args()

    d, meta = M.load_data(("train",))
    tr = d["train"]
    der = M.derive(tr)
    dim, nmax = tr.shape[1], CFG.NMAX
    T = M.horizon()
    logfact = VO._logfact_arr(nmax)
    dmax_t = torch.as_tensor(der["dmax"], dtype=torch.long, device=DEV)
    (ROOT / "Baseline" / "OURS" / "results").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pi = emp_prior(tr, nmax, DEV)
    rl(f"### EXP174 oracle {a.stage} · K={a.ks} n_gen={a.n_gen} chunk={a.cell_chunk} "
       f"T={T:.4f} NMAX={nmax} dim={dim}; the empirical prior π is computed from **train**,"
       f"clip {nmax} + 1e-8 normalized, {time.time()-t0:.1f}s; **the network is not loaded at all**")

    out = {"T": T, "NMAX": nmax, "dim": dim, "n_gen": a.n_gen,
           "cell_chunk": a.cell_chunk, "batched": True, "oracle": True, "cells": {}}
    for K in a.ks:
        gf, nfe = M.build_grid(K, T)
        t0 = time.time()
        bank = DedupBank(der["V"], gf, nmax,
                         lambda tau, v, nm: VO.kmat(tau, v, nm, logfact),
                         lambda v, nm: VO.pois_prior(v, nm, logfact), DEV)
        g = bank.verify(lambda tau, v, nm: VO.kmat(tau, v, nm, logfact), der["V"])
        rl(f"  K={K} bank {time.time()-t0:.1f}s  deduplication gate bit_exact={g['bit_exact']} "
           f"({g['n_checked']} spot checks)  n_V={g['n_V']}")
        if not g["bit_exact"]:
            rl(f"  [STOP] K={K} deduplication gate failed")
            return 2

        g1, g2 = gates(bank, pi, dmax_t, tr, nmax)
        rl(f"  **gate 1** (t->0,tau={g1['tau']:.4f}) argmax hit rate "
           f"**{g1['hit_rate']:.5f}**({g1['n_hit']}/{g1['n_checked']}, trigger line >0.99;"
           f"A4 truncation excluded {g1['n_excluded_by_A4']} )")
        rl(f"  **gate 2** maximum deviation of row sums from 1 **{g2['max_abs_sum_minus_1']:.3e}**,"
           f"NaN {g2['n_nan']},Inf {g2['n_inf']}")
        ok = (g1["hit_rate"] > 0.99 and g2["max_abs_sum_minus_1"] < 1e-9
              and g2["n_nan"] == 0 and g2["n_inf"] == 0)
        out.setdefault("gates", {})[f"K{K}"] = {"gate1": g1, "gate2": g2, "pass": bool(ok)}
        if not ok:
            rl("  **[STOP] gate failed; stop per the section 7 stop condition and report to the main line**")
            (HERE.parents[1] / "results" / f"oracle_gate_K{K}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=1, default=float),
                encoding="utf-8")
            return 3
        rl("  all gates passed")
        if a.stage == "gate":
            del bank; gc.collect(); torch.cuda.empty_cache(); continue

        for sp in a.samps:
            t1 = time.time()
            parts = []
            for ci, lo in enumerate(range(0, a.n_gen, a.cell_chunk)):
                b = min(a.cell_chunk, a.n_gen - lo)
                parts.append(sample_chunk_oracle(bank, pi, dmax_t, b,
                                                 int(sp) * 1000 + ci, nmax))
            gen = np.concatenate(parts, 0)
            wt = time.time() - t1
            f = (ROOT / "Baseline" / "OURS" / "results" /
                 f"linspace_T_O_K{K}_sORACLE_samp{sp}"
                 f"_batched1_chunk{a.cell_chunk}_n{a.n_gen}{TAG}.npy")
            if f.exists():
                raise SystemExit(f"[ABORT] name clash, refusing to overwrite:{f.name}")
            np.save(f, gen.astype(np.int16))
            out["cells"][f"K{K}|samp{sp}"] = {
                "K": K, "nfe": nfe, "samp_seed": int(sp), "walltime_s": wt,
                "s_per_step": wt / max(nfe, 1), "gen_file": f.name,
                "cell_chunk": a.cell_chunk,
                "peak_alloc_gb": float(torch.cuda.max_memory_allocated() / 2 ** 30),
                "mean": float(gen.mean()), "max": int(gen.max()),
                "zerofrac": float((gen == 0).mean())}
            rl(f"  K={K} samp{sp} **{wt/60:.1f}min({wt/nfe:.3f} s/step)** -> {f.name}  "
               f"mean={gen.mean():.4f} max={gen.max()} zeros={(gen==0).mean():.4f}"
               f"  (real train mean={tr.mean():.4f} max={tr.max()} "
               f"zeros={(tr==0).mean():.4f})")
        del bank; gc.collect(); torch.cuda.empty_cache()

    lg = (ROOT / "Baseline" / "OURS" / "results" /
          f"oracle_log_K{'-'.join(str(k) for k in a.ks)}_{a.stage}.json")
    if lg.exists():
        raise SystemExit(f"[ABORT] name clash, refusing to overwrite:{lg.name}")
    lg.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                  encoding="utf-8")
    rl(f"### EXP174 oracle {a.stage} done -> {lg.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
