# -*- coding: utf-8 -*-
"""EXP176 · sampling for the three grid arms (A=T8 / B=T_O / D=T_Oplus jump).

    python Baseline/OURS/code/sample_hvg2k_toplus.py --ckpt <Tm8.pt> --horizon T8
    python Baseline/OURS/code/sample_hvg2k_toplus.py --ckpt <Tm8.pt> --horizon T_O
    python Baseline/OURS/code/sample_hvg2k_toplus.py --ckpt <Tm8.pt> --horizon T_Oplus

from `sample_hvg2k.py` **verbatim copy**, replacing only the grid construction (`M.build_grid` -> `grid_variant`),
plus one `--horizon` switch and file name suffix.
**`sample_hvg2k.py` / `model_hvg2k.py` / `bank_dedup.py` not a single byte changed** (red line).

## three grids (EXP176 3)

    A  T8       linspace(8.0 -> F_BASE, K) + [0]            NFE = K
    B  T_O      linspace(T_O -> F_BASE, K) + [0]            NFE = K       <- control variable
    D  T_Oplus  [8.0] + linspace(T_O -> F_BASE, K) + [0]    NFE = K + 1   <- jump

**B grid shares a bit-identical source with the old run1c (arm C)**: both call `M.build_grid(K, M.horizon())`,
the same function with the same arguments, not a rewrite.
**D the middle segment is bit-identical to B**: D simply concatenates one in front of the grid of B 8.0,
the two arms differ only by one extra leading step: a clean single-variable control.

conventions follow the toy verbatim `exp139_step3.py::build_grid(horizon="T_Oplus")`:
start with one `T_PLUS_START = 8.0`, middle segment unchanged, 0 appended at the end.

NFE always use the **measured value** (D is 257, not 256) in file names, json and table axes.
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


T_PLUS_START = 8.0          # EXP176 starting point of the jump arm (same-name constant as toy exp139_step3)


def grid_variant(K, horizon):
    """the one place that is replaced."""
    if horizon == "T8":
        return M.build_grid(K, T_PLUS_START)
    gf, nfe = M.build_grid(K, M.horizon())              # B: shares a bit-identical source with arm C
    if horizon == "T_Oplus":
        gf = np.concatenate([[T_PLUS_START], gf])       # D: one extra step in front of B
        nfe = int(len(gf) - 1)
    return gf, nfe


def rl(m):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"- `{time.strftime('%H:%M:%S')}` {m}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _cat(p, gen):
    """`vendor_ours._categorical_torch` verbatim semantics."""
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    tot = p.sum(dim=1)
    bad = ~torch.isfinite(tot) | (tot <= 0)
    if bad.any():
        p = p.clone(); p[bad] = 0.0; p[bad, 0] = 1.0
    return torch.multinomial(p, 1, generator=gen).squeeze(1)


def _slot_v_idx(slot, dev, _cache={}):
    """`_Slot.g2v`(CPU int list) -> GPU long tensor, for advanced indexing in the batched path.

    read-only `slot.g2v`, **without changing `bank_dedup.py` a single byte** (red line). Cached by id,
    each slot on the whole ladder is built only once.
    """
    key = id(slot)
    t = _cache.get(key)
    if t is None or t.device != dev:
        t = torch.as_tensor(slot.g2v, dtype=torch.long, device=dev)
        _cache[key] = t
    return t


@torch.no_grad()
def sample_chunk(net, bank, dmax_t, n, seed, nmax, batched: bool = False):
    """the full reverse chain of one block of cells.

    `batched=False`(default): structure identical to `ours_fast_v2.sample_bridge_v2(batched=False)` consistent:
        2·dim multinomials per gene. At dim=2000 that is 4000 kernel launches per reverse step,
        `n_gen=20000` + `cell_chunk=500` is 40 blocks -> **160k per reverse step**,
        measured 69 s/step, whereas linear extrapolation in dimension from the 61gene 0.215 s/step would give only 7.05 s/step.
        the extra tenfold is all kernel launch latency.

    `batched=True`(EXP160, **class B**): verbatim as `ours_fast_v2.sample_bridge_v2` of
        batched branch: compresses 2·dim calls into **2 calls** `[B*dim, C]` large multinomial.
        changes the consumption order of the torch generator, so it **cannot** be bit-identical to batched=False,
        must pass the distribution-equivalence gate (`_diag/t1_batched_gate.py`).
        The main line lifted the EXP157 ban on 2026-08-10: experiment X is a new experiment with no published numbers to match.

    ★ numerical precision unchanged: the bank is still float64,`row*col` done in float64,
      `p` still the float32 network output. batched only changes **how batching is done**, not any dtype.
    """
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
        t = float(tgrid[k])
        nt = n_t.to(torch.float32)
        tt = torch.full((n,), t, device=dev)
        p = net.post(nt, tt)                                     # [B, dim, C] float32
        p = p.masked_fill(torch.arange(p.shape[2], device=dev)[None, None, :]
                          > dmax_t[None, :, None], 0.0)          # A4 posterior-support truncation
        s = p.sum(-1, keepdim=True)
        p = torch.where(s > 0, p / torch.clamp_min(s, 1e-30), p)
        Knext, KdelT = bank.Kgrid[k + 1], bank.Kdelta_T[k]
        if not batched:
            for d in range(dim):
                n0 = _cat(p[:, d, :], gen)
                row = Knext[d].index_select(0, n0)
                col = KdelT[d].index_select(0, n_t[:, d].clamp(0, nmax))
                n_t[:, d] = _cat(row * col, gen)
        else:
            C = p.shape[2]
            n0 = _cat(p.reshape(n * dim, C), gen).view(n, dim)   # first large multinomial
            del p
            vi_next = _slot_v_idx(Knext, dev)[None, :].expand(n, dim)
            vi_delt = _slot_v_idx(KdelT, dev)[None, :].expand(n, dim)
            row = Knext.U[vi_next, n0]                           # [B, dim, C] float64
            row.mul_(KdelT.U[vi_delt, n_t.clamp(0, nmax)])       # in-place multiply, saves a copy 4 GB
            n_t = _cat(row.reshape(n * dim, C), gen).view(n, dim)  # second
            del row, n0
            continue
        del p
    return n_t.cpu().numpy().astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=CFG.K_LADDER_PILOT)
    ap.add_argument("--samps", type=int, nargs="+", default=CFG.SAMP_SEEDS[:1])
    ap.add_argument("--n-gen", type=int, default=CFG.N_GEN)
    ap.add_argument("--cell-chunk", type=int, default=500)
    ap.add_argument("--batched", action="store_true",
                    help="EXP160: 2 large multinomials instead of 2·dim (class B; must pass the distribution-equivalence gate)")
    ap.add_argument("--tag", default="", help="appended to the file name, to avoid clashing with existing outputs")
    ap.add_argument("--horizon", choices=["T8", "T_O", "T_Oplus"], default="T_O",
                    help="A=T8 throughout T_m / B=T_O control / D=T_Oplus jump (NFE=K+1)")
    a = ap.parse_args()

    d, meta = M.load_data(("train",))
    tr = d["train"]
    der = M.derive(tr)
    dim, nmax = tr.shape[1], CFG.NMAX
    T = M.horizon()
    logfact = VO._logfact_arr(nmax)
    net = M.build_net(dim, der["iscale"]).to(DEV)
    net.load_state_dict(torch.load(a.ckpt, map_location=DEV, weights_only=False))
    net.eval()
    dmax_t = torch.as_tensor(der["dmax"], dtype=torch.long, device=DEV)
    (ROOT / "Baseline" / "OURS" / "results").mkdir(parents=True, exist_ok=True)

    rl(f"### EXP176 sampling starts **horizon={a.horizon}** ckpt={Path(a.ckpt).name} "
       f"K={a.ks} samp={a.samps} n_gen={a.n_gen} cell_chunk={a.cell_chunk} "
       f"T_O={T:.4f} NMAX={nmax} dim={dim}")
    out = {"ckpt": Path(a.ckpt).name, "T": T, "NMAX": nmax, "dim": dim,
           "n_gen": a.n_gen, "cell_chunk": a.cell_chunk, "batched": bool(a.batched),
           "cells": {}}
    for K in a.ks:
        gf, nfe = grid_variant(K, a.horizon)      # <- the only replacement point
        t0 = time.time()
        bank = DedupBank(der["V"], gf, nmax,
                         lambda tau, v, nm: VO.kmat(tau, v, nm, logfact),
                         lambda v, nm: VO.pois_prior(v, nm, logfact), DEV)
        g = bank.verify(lambda tau, v, nm: VO.kmat(tau, v, nm, logfact), der["V"])
        t_bank = time.time() - t0
        rl(f"  K={K} horizon={a.horizon} **measured NFE={nfe}** "
           f"grid[{gf[0]:.4f}, {gf[1]:.4f}, ..., {gf[-2]:.4f}, {gf[-1]:.4f}] "
           f"len={len(gf)}; bank {t_bank:.1f}s deduplication gate bit_exact={g['bit_exact']} "
           f"({g['n_checked']} spot checks) n_V={g['n_V']}")
        if not g["bit_exact"]:
            rl(f"  [SKIP] K={K} deduplication gate failed, skipped")
            del bank; gc.collect(); torch.cuda.empty_cache(); continue
        for sp in a.samps:
            t1 = time.time()
            parts = []
            for ci, lo in enumerate(range(0, a.n_gen, a.cell_chunk)):
                b = min(a.cell_chunk, a.n_gen - lo)
                parts.append(sample_chunk(net, bank, dmax_t, b, int(sp) * 1000 + ci, nmax,
                                          batched=a.batched))
            gen = np.concatenate(parts, 0)
            wt = time.time() - t1
            # ★ `batched` and `cell_chunk` both go into the file name: both change the RNG consumption order and are **part of the definition**,
            #   outputs with different values must coexist and must not overwrite each other (red line: never overwrite an existing .npy).
            suf = ((f"_batched{int(a.batched)}_chunk{a.cell_chunk}" if a.batched else "")
                   + f"_{a.horizon}_nfe{nfe}" + a.tag)
            f = (ROOT / "Baseline" / "OURS" / "results" /
                 f"linspace_T_O_K{K}_s{Path(a.ckpt).stem}_samp{sp}{suf}.npy")
            if f.exists():
                raise SystemExit(f"[ABORT] name clash, refusing to overwrite:{f.name}")
            np.save(f, gen.astype(np.int16))
            out["cells"][f"K{K}|samp{sp}"] = {
                "K": K, "nfe": nfe, "samp_seed": int(sp), "walltime_s": wt,
                "s_per_step": wt / max(nfe, 1), "gen_file": f.name,
                "batched": bool(a.batched), "cell_chunk": a.cell_chunk,
                "peak_alloc_gb": float(torch.cuda.max_memory_allocated() / 2 ** 30)
                if torch.cuda.is_available() else 0.0,
                "mean": float(gen.mean()), "max": int(gen.max()),
                "zerofrac": float((gen == 0).mean())}
            rl(f"  K={K} samp{sp} {wt/60:.1f}min ({wt/nfe:.3f} s/steps) -> {f.name}  "
               f"mean={gen.mean():.4f} max={gen.max()} zeros={(gen==0).mean():.4f}"
               f"  (real train mean={tr.mean():.4f} max={tr.max()} zeros={(tr==0).mean():.4f})")
        del bank; gc.collect(); torch.cuda.empty_cache()
    # ★ 2026-08-10 fix: the log name **must include K**. It used to include only batched/chunk, so under the same chunk
    #   names clash from the second level on and are blocked by no-overwrite at the end of main(), rc=1:
    #   **the samples were actually all saved**, but the chain log marks these levels as failed, and aggregation would skip them as missing.
    ktag = "K" + "-".join(f"{k:03d}" for k in a.ks)
    # ★ EXP172 fix: the log name **does not include ckpt**, so resampling the same (K, chunk, tag) with another training seed
    #   names collide: the sample .npy is already written, but SystemExit happens at the last step and the chain records this level as rc=1.
    #   seed reproduction runs three ckpts at the same K, so this pitfall is guaranteed; hence the ckpt stem is added to the log name.
    #   **only the log file name; never touches any numerical path.**
    lg = ROOT / "Baseline" / "OURS" / "results" / (
        f"sample_log_{ktag}_s{Path(a.ckpt).stem}_{a.horizon}"
        f"{'_chunk%d_batched1' % a.cell_chunk if a.batched else ''}{a.tag}.json")
    if lg.exists():
        raise SystemExit(f"[ABORT] name clash, refusing to overwrite:{lg.name}")
    lg.write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    rl(f"### hvg2k sampling done, saving {len(out['cells'])} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
