# -*- coding: utf-8 -*-
"""EXP178 · at large t, shrink the posterior toward the empirical marginal (**no training**).

    python Baseline/OURS/code/sample_hvg2k_shrink.py --ckpt <run1c.pt> --ks 32 --s 0.1

from `sample_hvg2k.py` **verbatim copy**, only in `p = net.post(nt, tt)` after,
**A4 one mixing line inserted **before the truncation**; nothing else changed.
**`sample_hvg2k.py` / `model_hvg2k.py` / `bank_dedup.py` not a single byte changed** (red line).

## fix

    p_used(n0 | n_t) = (1 - lam_d(t)) * p_net + lam_d(t) * pi_d

`pi_d` = EXP174/177 the empirical marginal used (train histogram, clip NMAX + 1e-8 normalized),
**no learning needed; counted directly**.

    lam(t, d) = (1 - beta_t) * exp(-beta_t * nbar_d / s),   beta_t = e^{-t}

(form after the main-line correction of 2026-08-13.) Each of the two factors handles one thing:

* `(1 - beta_t)` ensures **lam->0 as t->0**. At t=0.01 it is only 0.00995, so **for any s,
  lam <= 0.01**; cross-gene conditioning of the network at small t is fully preserved; EXP177 measured a real advantage there, which must not be touched.
* `exp(-beta_t * nbar_d / s)` controls **per-gene differences at large t**. At t=T_O `1-beta_t = 0.9945`,
  the second term is about 1 for rare genes -> lam≈0.99; common genes are still held down by the second term at moderate t.

> old form `exp(-beta_t*nbar_d/s)` used alone it is wrong: rare genes `beta_t*nbar_d≈0.06`,
> t->0 lam **does not converge to 0** (s>=0.1 still shrinks by 0.62~0.95 at t=0.01),
> replaced exactly the end that should have been kept. The root cause was using `beta_t*nbar_d` treated as information, while as t->0 the observation **is**
> n_0, the information is full however rare the gene. Replaced per the main-line correction.

`--s 0` means **lam is identically 0**, i.e. the status-quo control.

## dtype: mixing done in **float32**

in the current sampling path `p` was already float32 (`sample_hvg2k.py` comment says so); the float64 one is
`row = Knext.U[...]` that branch. The red line is **do not downcast anything that is now float64 to float32**,
not [everything must be float64]. So `pi` **cast from float64 to float32** before `p` mixing,
**precision state bit-identical to the status quo** (minimum pi 1e-8, representable in float32), temporaries halved too.

`--s 0` so it follows the same numerical path as the main-table K=32 output.
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
from bank_dedup_v2 import DedupBankV2                   # noqa: E402  (only used with --bank v2)

VO = importlib.import_module("panel61_shared.vendor_ours")
# ★ EXP179: E1/E2/F1/F2 need shrinkage and a non-default grid at the same time. The grid logic is **not rewritten**,
#   import directly from that EXP176 file `grid_variant` — so that A/B/D of EXP176 and
#   EXP179 E1/E2/F1/F2 use **the same function**; the grids share a bit-identical source and cannot be misaligned by a copy error.
from sample_hvg2k_toplus import T_PLUS_START, grid_variant   # noqa: E402,F401
LOG = ROOT / "RUN_LOG.md"
DEV = M.DEV


def emp_prior(train, nmax):
    """pi_d[c]: empirical count distribution on train, clipped to nmax, +1e-8 then renormalized (identical to EXP174/177)."""
    Xc = np.clip(np.asarray(train), 0, nmax).astype(np.int64)
    P = np.stack([np.bincount(Xc[:, d], minlength=nmax + 1)[:nmax + 1]
                  for d in range(Xc.shape[1])]).astype(np.float64)
    P /= P.sum(1, keepdims=True)
    P += 1e-8
    P /= P.sum(1, keepdims=True)
    return P


def lam_of(t, nbar, s):
    """lam(t,d) = (1 - beta_t) * exp(-beta_t * nbar_d / s), beta_t = e^{-t}.

    s<=0 -> constantly 0 (status-quo control). Form after the main-line correction of 2026-08-13; see the file header.
    """
    nbar = np.asarray(nbar, np.float64)
    if s is None or s <= 0:
        return np.zeros_like(nbar)
    beta = float(np.exp(-float(t)))
    return (1.0 - beta) * np.exp(-beta * nbar / float(s))


_GATE = {"max_rowsum_dev": 0.0}


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
def sample_chunk(net, bank, dmax_t, n, seed, nmax, batched: bool = False,
                 pi=None, nbar=None, s=0.0):
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
        # ===== EXP178 the only insertion point: shrink toward the empirical marginal (before the A4 cut)=====
        # mixing runs in **float32**: p is already float32, pi is cast down; precision state bit-identical to the status quo.
        if pi is not None and s and s > 0:
            lam = torch.as_tensor(lam_of(t, nbar, s), dtype=torch.float32,
                                  device=dev)[None, :, None]     # [1, dim, 1]
            p = (1.0 - lam) * p + lam * pi[None, :, :]
        # ================================================================
        p = p.masked_fill(torch.arange(p.shape[2], device=dev)[None, None, :]
                          > dmax_t[None, :, None], 0.0)          # A4 posterior-support truncation
        ssum = p.sum(-1, keepdim=True)
        p = torch.where(ssum > 0, p / torch.clamp_min(ssum, 1e-30), p)
        # gate (section 3): after mixing + renormalization, row sums must be 1, no NaN/Inf
        # threshold 1e-5: the rounding floor of summing 513 terms in float32 is about 1e-6; 1e-9 is the float64 standard,
        # would raise false alarms on the float32 path. The measured maximum is also written to RUN_LOG, not just"pass".
        _rs = float((p.sum(-1) - 1.0).abs().max())
        if not (_rs < 1e-5) or bool(torch.isnan(p).any()) or bool(torch.isinf(p).any()):
            raise SystemExit(f"[STOP] EXP178 normalization gate failed: row sums deviate by {_rs:.3e},"
                             f"NaN={int(torch.isnan(p).sum())} Inf={int(torch.isinf(p).sum())}")
        _GATE["max_rowsum_dev"] = max(_GATE["max_rowsum_dev"], _rs)
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
    ap.add_argument("--bank", choices=["v1", "v2"], default="v1",
                    help="EXP180: v2 additionally deduplicates **exactly by tau** on top of V**"
                         "(Kdelta_T K-1 layers -> 5 layers); K=512 only fits this way. Default v1 = old behaviour.")
    ap.add_argument("--mmap-dir", default=None,
                    help="EXP180: the host_resident Kgrid is moved to an on-disk memmap."
                         "K=1024 pinning 30 GB on a machine with 41 GB free gives CUDA OOM,"
                         "pageable memory would blow the CPU heap; memmap catches it.")
    ap.add_argument("--host-resident", action="store_true",
                    help="EXP180: Kgrid resident on CPU, uploaded layer by layer. Needed for K=1024 (GPU would need 30.3 GB).")
    ap.add_argument("--horizon", choices=["T8", "T_O", "T_Oplus"], default="T_O",
                    help="EXP179: the same grids as A/B/D of EXP176. NFE of T_Oplus = K+1.")
    ap.add_argument("--s", type=float, default=0.0,
                    help="lam = exp(-e^{-t}·nbar_d/s). **--s 0 means lam is constantly 0 (status-quo control)**."
                         "§2 hard-coded scan range: 0, 0.01, 0.03, 0.1, 0.3, 1.0; do not add other forms.")
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

    nbar = np.asarray(tr, np.float64).mean(0)
    pi_t = torch.as_tensor(emp_prior(tr, nmax), dtype=torch.float32, device=DEV)
    det = 1 - (np.asarray(tr) == 0).mean(0)
    rare = det < 0.05
    t_hi = T_PLUS_START if a.horizon in ('T8', 'T_Oplus') else T
    l_hi, l_lo = lam_of(t_hi, nbar, a.s), lam_of(CFG.F_BASE, nbar, a.s)
    rl(f"### shrinkage sampling **s={a.s} horizon={a.horizon}** ckpt={Path(a.ckpt).name} K={a.ks} "
       f"samp={a.samps} n_gen={a.n_gen} cell_chunk={a.cell_chunk} T={T:.4f} "
       f"NMAX={nmax} dim={dim}")
    rl(f"  **lam median**: t={t_hi:.4f}(start) all {np.median(l_hi):.4f} / "
       f"rare (det<5%,n={int(rare.sum())}) {np.median(l_hi[rare]):.4f};"
       f"t=F_BASE({CFG.F_BASE}) all {np.median(l_lo):.4f} / "
       f"rare {np.median(l_lo[rare]):.4f}")
    out = {"ckpt": Path(a.ckpt).name, "T": T, "NMAX": nmax, "dim": dim,
           "n_gen": a.n_gen, "cell_chunk": a.cell_chunk, "batched": bool(a.batched),
           "cells": {}}
    for K in a.ks:
        gf, nfe = grid_variant(K, a.horizon)      # EXP179: the same function as EXP176
        t0 = time.time()
        if a.bank == "v2":
            bank = DedupBankV2(der["V"], gf, nmax,
                               lambda tau, v, nm: VO.kmat(tau, v, nm, logfact),
                               lambda v, nm: VO.pois_prior(v, nm, logfact), DEV,
                               host_resident=a.host_resident,
                               mmap_dir=a.mmap_dir)
        else:
            bank = DedupBank(der["V"], gf, nmax,
                             lambda tau, v, nm: VO.kmat(tau, v, nm, logfact),
                             lambda v, nm: VO.pois_prior(v, nm, logfact), DEV)
        g = bank.verify(lambda tau, v, nm: VO.kmat(tau, v, nm, logfact), der["V"])
        t_bank = time.time() - t0
        rl(f"  K={K} bank {t_bank:.1f}s  deduplication gate bit_exact={g['bit_exact']} "
           f"({g['n_checked']} spot checks)  n_V={g['n_V']}")
        if not g["bit_exact"]:
            rl(f"  [SKIP] K={K} deduplication gate failed, skipped")
            del bank; gc.collect(); torch.cuda.empty_cache(); continue
        for sp in a.samps:
            t1 = time.time()
            parts = []
            for ci, lo in enumerate(range(0, a.n_gen, a.cell_chunk)):
                b = min(a.cell_chunk, a.n_gen - lo)
                parts.append(sample_chunk(net, bank, dmax_t, b, int(sp) * 1000 + ci, nmax,
                                          batched=a.batched, pi=pi_t, nbar=nbar, s=a.s))
            gen = np.concatenate(parts, 0)
            wt = time.time() - t1
            # ★ `batched` and `cell_chunk` both go into the file name: both change the RNG consumption order and are **part of the definition**,
            #   outputs with different values must coexist and must not overwrite each other (red line: never overwrite an existing .npy).
            suf = ((f"_batched{int(a.batched)}_chunk{a.cell_chunk}" if a.batched else "")
                   + f"_{a.horizon}_nfe{nfe}_SHRINK_s{a.s:g}"
                   + ("" if a.bank == "v1" else
                      f"_bank{a.bank}{'hr' if a.host_resident else ''}") + a.tag)
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
        f"sample_log_{ktag}_s{Path(a.ckpt).stem}_{a.horizon}_SHRINK_s{a.s:g}"
        f"{'' if a.bank == 'v1' else '_bank' + a.bank + ('hr' if a.host_resident else '')}"
        f"{'_chunk%d_batched1' % a.cell_chunk if a.batched else ''}{a.tag}.json")
    if lg.exists():
        raise SystemExit(f"[ABORT] name clash, refusing to overwrite:{lg.name}")
    lg.write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    rl(f"### EXP178 s={a.s} sampling done, saving {len(out['cells'])} cells; **normalization gate** maximum deviation of row sums from 1 {_GATE['max_rowsum_dev']:.3e}(trigger line <1e-9)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
