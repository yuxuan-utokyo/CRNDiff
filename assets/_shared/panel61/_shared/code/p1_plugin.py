"""P1 — zero-network plug-in on the 61-gene panel.

Replace the learned x0-posterior with

    Q_hat(n0 | n_t) ∝ p_emp^train(n0) q_t(n_t | n0)        (= O1, per gene)

keeping EVERYTHING else identical to the main-table OURS sampler (same KernelBank, same
bridge, same grid construction), and compare per K. This says what the network actually
buys on the panel, as opposed to on the toy where it bought coupling (corr_err .035 vs
.084) but nothing on the marginal metrics.

Grid: `linspace @ T_O`, with T_O = T_MP = 4.3144 (OURS/model.py::T_MP), F_BASE = 0.01 --
i.e. the same `arm="linspace"` branch main-table OURS exposes.

The posterior maths is `vendor_ours.make_oracle_post_fn` (read-only reuse), but that
implementation is numpy and loops genes, which at 61 genes x 1232 reverse steps would not
finish. `_gpu_oracle_post_fn` below is a GPU port; `validate_vs_reference()` asserts it
agrees with the numpy original and its result is stored with the run.

NEW FILE under scrna/_shared/. Reads existing artefacts; writes only its own outputs.
Does NOT modify vendor_ours*.py or any published arm directory.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRNA = Path(__file__).resolve().parents[1]
for _p in (str(SCRNA), str(SCRNA / "OURS")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _shared import vendor_ours as O                # noqa: E402
from _shared import vendor_ours_fast as OF          # noqa: E402
from _shared.backbone import DEV                    # noqa: E402
from _shared.data import load_panel                 # noqa: E402
from _shared.evaluate_common import evaluate_array  # noqa: E402
from _shared.seeds import METRICS4, NMAX, N_GEN, SAMP_SEEDS   # noqa: E402

OUT_DIR = SCRNA / "_shared" / "p1_plugin"
KS = [4, 16, 64, 128, 1024]


def _gpu_oracle_post_fn(train, bank, device=DEV):
    """GPU port of vendor_ours.make_oracle_post_fn (O1, per gene)."""
    dim, nmax = bank.dim, bank.nmax
    P = np.stack([np.bincount(np.clip(train[:, d], 0, nmax), minlength=nmax + 1)
                  .astype(np.float64) for d in range(dim)])
    P = P / P.sum(axis=1, keepdims=True)
    Pt = torch.as_tensor(P, device=device, dtype=torch.float64)          # [dim, N]
    tgrid = bank.tgrid
    cache: dict[int, torch.Tensor] = {}

    def post_fn(n_t: np.ndarray, t: float) -> np.ndarray:
        k = int(np.argmin(np.abs(tgrid - t)))
        if k not in cache:
            cache.clear()
            cache[k] = torch.as_tensor(np.stack(bank.Kgrid[k]), device=device,
                                       dtype=torch.float64)              # [dim, n0, n]
        Kt = cache[k]
        nd = torch.as_tensor(np.clip(n_t.astype(np.int64), 0, nmax), device=device)
        out = torch.empty(nd.shape[0], dim, nmax + 1, device=device, dtype=torch.float64)
        for d in range(dim):
            w = Pt[d].unsqueeze(0) * Kt[d].t()[nd[:, d]]                 # [B, n0]
            s = w.sum(1, keepdim=True)
            out[:, d, :] = torch.where(s > 0, w / s.clamp_min(1e-300),
                                       Pt[d].unsqueeze(0).expand_as(w))
        return out.float().cpu().numpy()
    return post_fn


def validate_vs_reference(train, bank, n=64, seed=7) -> dict:
    ref = O.make_oracle_post_fn(train, bank)
    gpu = _gpu_oracle_post_fn(train, bank)
    rng = np.random.default_rng(seed)
    nt = np.clip(train[rng.choice(len(train), n, replace=False)], 0, bank.nmax).astype(np.float32)
    worst = 0.0
    for k in (0, len(bank.tgrid) // 2, len(bank.tgrid) - 2):
        t = float(bank.tgrid[k])
        worst = max(worst, float(np.max(np.abs(ref(nt, t) - gpu(nt, t)))))
    return {"worst_max_abs_err": worst, "passed": worst < 1e-5, "n_probe": n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    ap.add_argument("--n-gen", type=int, default=N_GEN)
    ap.add_argument("--samp-seeds", default=",".join(str(s) for s in SAMP_SEEDS))
    ap.add_argument("--grid", choices=["linspace", "wake-sigma"], default="linspace",
                    help="wake-sigma @ T=8 is the MAIN-TABLE grid; running the plug-in on "
                         "it too is the adversarial check that the linspace@T_O choice is "
                         "not what is handicapping the plug-in")
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import model as M      # OURS/model.py (grid construction, StreamBank thresholds)
    p = load_panel()
    T_O = a.T if a.T is not None else (M.T8 if a.grid == "wake-sigma" else M.T_MP)
    res = {"what": "zero-network plug-in (O1) on the 61-gene panel",
           "grid": f"{a.grid} @ T = {T_O}", "F_BASE": M.F_BASE,
           "n_gen": a.n_gen, "NMAX": int(NMAX), "network_calls": 0,
           "posterior": "Q(n0|n_t) prop p_emp^train(n0) q_t(n_t|n0)  (per gene, O1)",
           "prior_source": "TRAIN empirical marginal only (no val, no test)",
           "cells": {}}

    print(f"panel dim={p.dim} NMAX={NMAX} T_O={T_O} n_gen={a.n_gen}")
    for K in [int(x) for x in a.ks.split(",")]:
        if a.grid == "wake-sigma":
            sig = M.spectrum(p.train)
            grid = M.wake_grid(sig, (1 + sig) ** 2 / sig, K, M.F_BASE, T_O)
        else:
            grid = np.linspace(T_O, M.F_BASE, K)
        ok, why = M.grid_shape_gate(grid, K, M.F_BASE, T_O)
        if not ok:
            print(f"[SKIP] K={K} grid gate: {why}")
            res["cells"][f"K{K}"] = {"skipped": why}
            continue
        gf, nfe = M.gfull_nfe(grid)
        bank = O.KernelBank(np.asarray(p.Vg, np.float64), gf, NMAX)
        if K == 4:
            res["gpu_oracle_validation"] = validate_vs_reference(p.train, bank)
            v = res["gpu_oracle_validation"]
            print(f"  [validate] GPU O1 vs numpy reference: worst |err| = "
                  f"{v['worst_max_abs_err']:.3e} -> {'OK' if v['passed'] else 'FAIL'}")
            if not v["passed"]:
                print("[ABORT] GPU oracle does not match the reference")
                return 2
        fb = (M.StreamBank(bank, DEV) if M._fast_gb(nfe, p.dim) > M.FAST_DEV_GB_MAX
              else OF.FastBank(bank, DEV))
        pf = _gpu_oracle_post_fn(p.train, bank)
        for ss in [int(s) for s in a.samp_seeds.split(",")]:
            t0 = time.time()
            gen = OF.sample_bridge_fast(pf, fb, n_samples=a.n_gen, seed=ss)
            wt = time.time() - t0
            np.save(OUT_DIR / f"plugin_{a.grid}_T{T_O}_K{K}_samp{ss}.npy",
                    gen.astype(np.int16))
            m = evaluate_array(gen)
            res["cells"][f"K{K}_samp{ss}"] = {
                "K": K, "nfe_reverse_steps": int(nfe), "nfe_network_calls": 0,
                "samp_seed": ss, "walltime_s": wt, "gen_max": int(gen.max()),
                "main4": {k: m[k] for k in METRICS4}, "metrics": m}
            print(f"  K={K:>5d} samp={ss} nfe_steps={nfe} netNFE=0 "
                  f"resid={m['resid_corr_sqrt2']:.4f} "
                  f"corr_err={m['mean_abs_pairwise_corr_err']:.4f} "
                  f"FD={m['FD_frechet']:.4f} gen_max={int(gen.max())} ({wt:.0f}s)")
            (SCRNA / "_shared" / f"p1_plugin{a.tag}.json").write_text(
                json.dumps(res, indent=1), encoding="utf-8")
        del bank, fb, pf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (SCRNA / "_shared" / f"p1_plugin{a.tag}.json").write_text(json.dumps(res, indent=1),
                                                      encoding="utf-8")
    print("wrote _shared/p1_plugin.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
