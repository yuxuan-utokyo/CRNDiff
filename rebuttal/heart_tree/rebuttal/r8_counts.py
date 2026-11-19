# -*- coding: utf-8 -*-
"""R8: count statistics of the HVG2K train and val splits, on raw counts, per type and overall.
Test split not read.

    cd Experiments/HVG && python -m rebuttal.r8_counts
"""
import json, os
os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets/legacy_root/_shared/data/all"
OUT = ROOT / "rebuttal" / "out" / "R8"; OUT.mkdir(parents=True, exist_ok=True)
Q = [1, 5, 25, 50, 75, 95, 99]

def stats(X):
    n = X.size; nz = int((X > 0).sum()); lib = X.sum(1).astype(np.float64)
    gm = X.mean(0, dtype=np.float64)
    e = {"zero": 1 - nz / n, "eq1": float((X == 1).sum()) / n, "le3": float((X <= 3).sum()) / n,
         "le5": float((X <= 5).sum()) / n, "gt20": float((X > 20).sum()) / n}
    return {"n_cells": int(X.shape[0]), "entries": e,
            "among_nonzero": {"eq1": float((X == 1).sum()) / nz, "le3": float(((X > 0) & (X <= 3)).sum()) / nz,
                              "le5": float(((X > 0) & (X <= 5)).sum()) / nz, "gt20": float((X > 20).sum()) / nz},
            "max": int(X.max()),
            "gene_mean_q": dict(zip(map(str, Q), np.percentile(gm, Q).tolist())),
            "gene_mean_share_lt_0p1": float((gm < 0.1).mean()), "gene_mean_share_lt_1": float((gm < 1).mean()),
            "gene_mean_share_gt_10": float((gm > 10).mean()),
            "lib_mean": float(lib.mean()), "lib_q": dict(zip(map(str, Q), np.percentile(lib, Q).tolist()))}

res, gm_all, lib_all = {}, {}, {}
for sp in ("train", "val"):
    X = np.load(DATA / f"{sp}.npy", mmap_mode="r"); y = np.load(DATA / f"{sp}_celltype.npy", allow_pickle=True).astype(str)
    X = np.asarray(X, dtype=np.int16)
    res[sp] = {"all": stats(X)}
    gm_all[sp], lib_all[sp] = X.mean(0, dtype=np.float64), X.sum(1)
    for t in sorted(set(y.tolist())):
        res[sp][t] = stats(X[y == t])
        print(sp, t, res[sp][t]["n_cells"], {k: round(v, 4) for k, v in res[sp][t]["entries"].items()}, "lib", round(res[sp][t]["lib_mean"], 0), flush=True)
    print(sp, "all", res[sp]["all"], flush=True)
    del X
(OUT / "r8_numbers.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"; C = {"train": "#2a78d6", "val": "#eb6834"}
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.5})
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.1))
bins = np.logspace(-6, 3, 55)
ax[0].hist(gm_all["train"], bins=bins, color=C["train"], label=None)
ax[0].set_xscale("log"); ax[0].set_xlabel("Mean count per gene, train (log scale)", color=MUTED); ax[0].set_ylabel("Number of genes", color=MUTED)
ax[0].axvline(1, color=INK, lw=0.8); ax[0].text(1.3, ax[0].get_ylim()[1] * 0.9, "mean = 1", fontsize=7, color=INK)
bl = np.logspace(1, 5.2, 55)
ax[1].hist(lib_all["train"], bins=bl, color=C["train"])
ax[1].set_xscale("log"); ax[1].set_xlabel("Total counts per cell in these 2,000 genes, train", color=MUTED); ax[1].set_ylabel("Number of cells", color=MUTED)
for a in ax:
    a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", lw=0.3, color=GRID); a.set_axisbelow(True)
fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.21, wspace=0.27)
fig.savefig(OUT / "R8_counts.png", dpi=300)
