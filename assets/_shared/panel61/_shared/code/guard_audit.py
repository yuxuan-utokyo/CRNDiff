"""AUDIT: does the degeneracy guard fire on the MAIN-TABLE scRNA path?

Context. `_shared/vendor_ours_fast.py::_cat` (and its reference twin
`vendor_ours.py::_categorical_torch`) force a weight row to state 0 whenever the row
sums to <= 0 or is non-finite. On the EXP138 toy this fires at the `bridge_weights`
call site and silently deletes the largest counts. The scRNA main table runs the SAME
guard through the SAME sampler, so the main-table OURS numbers have to be checked.

Two scRNA-specific aggravators vs the toy:
  * `pf_trunc` (A4 posterior-support truncation) zeroes posterior mass above each gene's
    data max and only renormalises when the remaining sum is > 0; a row whose mass sat
    entirely above the cap survives as all-zero and lands in the guard.
  * the bridge weights here are float64 (FastBank default), so any hit is NOT float32
    underflow.

This module MUTATES NOTHING. It wraps `_cat` at runtime for the duration of the audit
and restores it afterwards; no published file is edited and no checkpoint is retrained.

  python -m _shared.guard_audit --ks 4,16,64,128,1024 --n-gen 20000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRNA = Path(__file__).resolve().parents[1]
for p in (str(SCRNA), str(SCRNA / "OURS")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _shared import vendor_ours_fast as OF     # noqa: E402
from _shared.data import load_panel            # noqa: E402
from _shared.seeds import NMAX, SAMP_SEEDS, TRAIN_SEEDS  # noqa: E402


class Counter:
    """Wraps _cat. The sampler calls it strictly alternating (posterior, bridge) inside
    the per-gene loop, so call parity identifies the site; row-sum magnitude is recorded
    as an independent cross-check (posterior rows sum to ~1, bridge rows do not)."""

    def __init__(self, orig, dim: int):
        self.orig, self.dim = orig, dim
        self.calls = 0
        self.stats = {s: {"rows": 0, "bad": 0, "bad_per_gene": np.zeros(dim, dtype=np.int64)}
                      for s in ("x0_posterior", "bridge_weights")}
        self.rowsum_check = {s: [] for s in ("x0_posterior", "bridge_weights")}
        self._gene = 0

    def __call__(self, p, gen):
        site = "x0_posterior" if self.calls % 2 == 0 else "bridge_weights"
        if site == "x0_posterior":
            self._gene = (self._gene) % self.dim
        self.calls += 1
        q = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        tot = q.sum(dim=1)
        bad = ~torch.isfinite(tot) | (tot <= 0)
        nb = int(bad.sum().item())
        st = self.stats[site]
        st["rows"] += int(bad.numel())
        st["bad"] += nb
        if nb:
            st["bad_per_gene"][self._gene] += nb
        if len(self.rowsum_check[site]) < 5:
            self.rowsum_check[site].append(float(tot.median().item()))
        if site == "bridge_weights":
            self._gene += 1
        return self.orig(p, gen)

    def report(self) -> dict:
        out = {}
        tot_bad = tot_rows = 0
        for s, v in self.stats.items():
            out[s] = {"rows": int(v["rows"]), "fallbacks": int(v["bad"]),
                      "rate": v["bad"] / max(v["rows"], 1),
                      "bad_per_gene": v["bad_per_gene"].tolist() if v["bad"] else None,
                      "median_rowsum_sample": self.rowsum_check[s]}
            tot_bad += v["bad"]; tot_rows += v["rows"]
        out["overall"] = {"rows": int(tot_rows), "fallbacks": int(tot_bad),
                          "rate": tot_bad / max(tot_rows, 1), "any_triggered": tot_bad > 0}
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="4,16,64,128,1024")
    ap.add_argument("--n-gen", type=int, default=20000)
    ap.add_argument("--ckpt-seed", type=int, default=TRAIN_SEEDS[0])
    ap.add_argument("--samp-seed", type=int, default=SAMP_SEEDS[0])
    ap.add_argument("--grid", default="wake-sigma")
    ap.add_argument("--out", default=str(SCRNA / "_shared" / "guard_audit.json"))
    a = ap.parse_args(argv)

    import model as M   # OURS/model.py

    p = load_panel()
    ckpt = SCRNA / "OURS" / "ckpts" / f"s{a.ckpt_seed}.pt"
    net = M.load_ckpt(str(ckpt), p.dim, p.iscale)
    print(f"panel dim={p.dim} NMAX={NMAX} n_gen={a.n_gen} ckpt={ckpt.name} "
          f"grid={a.grid} samp_seed={a.samp_seed}")
    print(f"per-gene data max: min={int(np.min(p.dmax))} max={int(np.max(p.dmax))} "
          f"(A4 pf_trunc caps the posterior at these)")

    results = {"panel_dim": int(p.dim), "NMAX": int(NMAX), "n_gen": int(a.n_gen),
               "ckpt": ckpt.name, "samp_seed": int(a.samp_seed), "grid": a.grid,
               "bridge_dtype": "float64 (FastBank default)", "cells": {}}
    for K in [int(x) for x in a.ks.split(",")]:
        orig = OF._cat
        ctr = Counter(orig, p.dim)
        OF._cat = ctr
        t0 = time.time()
        try:
            gen, nfe, mode = M.sample_ours(net, p.train, p.Vg, p.dmax, K, a.samp_seed,
                                           a.n_gen, arm=a.grid, T=M.T8)
        except Exception as e:                                   # noqa: BLE001
            OF._cat = orig
            print(f"  [SKIP] K={K} reason={type(e).__name__}: {e}")
            results["cells"][f"K{K}"] = {"skipped": f"{type(e).__name__}: {e}"}
            continue
        finally:
            OF._cat = orig
        rep = ctr.report()
        rep.update({"K": K, "nfe": int(nfe), "mode": mode, "wall_sec": time.time() - t0,
                    "gen_max": int(gen.max()), "gen_zero_frac": float((gen == 0).mean()),
                    "gen_mean": float(gen.mean())})
        results["cells"][f"K{K}"] = rep
        o = rep["overall"]
        print(f"  K={K:>5d} nfe={nfe:>5d} mode={mode:<12s} "
              f"guard {o['fallbacks']}/{o['rows']} = {o['rate']:.3e}"
              f"{'   <<< NONZERO' if o['any_triggered'] else ''}  "
              f"[post {rep['x0_posterior']['fallbacks']} / bridge "
              f"{rep['bridge_weights']['fallbacks']}]  "
              f"gen_max={rep['gen_max']} zero={rep['gen_zero_frac']:.4f} "
              f"({rep['wall_sec']:.0f}s)")
        Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
