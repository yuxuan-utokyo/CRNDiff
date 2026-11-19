# -*- coding: utf-8 -*-
"""R2 (z7bw: sparsity not directly evaluated): zero fraction vs mean, per-cell zeros, detected genes.
Raw counts: zeros and library size are count-scale quantities and log-CP10K leaves zeros unchanged.
Real reference = val cells of the target; floor = train cells of the target, thinned to the same n
(5 draws), scored against val. Test split not read.

    cd Experiments/HVG && python -m rebuttal.r2_sparsity
"""
import json, os, sys
os.environ["OMP_NUM_THREADS"] = os.environ.get("R_THREADS", "4")   # R12 CPU window: 1
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.dont_write_bytecode = True
from rebuttal.w2_runs import TARGETS, N_OUT, ARMS, COL, run_list, load_cells, real_cells
OUT = Path(__file__).resolve().parent / "out" / "R2"; OUT.mkdir(parents=True, exist_ok=True)
EDGES = np.logspace(-4, 2, 25)


def prof(x):
    gm = x.mean(0); gz = (x == 0).mean(0)
    cz = (x == 0).mean(1); det = (x > 0).sum(1); lib = x.sum(1)
    return gm, gz, cz, det, lib


def curve(gm, gz):
    idx = np.digitize(gm, EDGES)
    return np.array([gz[idx == i].mean() if (idx == i).sum() >= 5 else np.nan for i in range(len(EDGES) + 1)])


def compare(x, ref):
    gm, gz, cz, det, lib = prof(x); rgm, rgz, rcz, rdet, rlib = ref
    # genes unseen in the real target are kept: the generator should also leave them at zero
    c, rc = curve(gm, gz), curve(rgm, rgz)
    ok = np.isfinite(c) & np.isfinite(rc)
    return {"zero_overall": float((x == 0).mean()), "gene_zero_mae_vs_real": float(np.abs(gz - rgz).mean()),
            "gene_zero_pcc_vs_real": float(np.corrcoef(gz, rgz)[0, 1]),
            "curve_rmse_vs_real": float(np.sqrt(np.nanmean((c[ok] - rc[ok]) ** 2))),
            "cell_zero_mean": float(cz.mean()), "cell_zero_sd": float(cz.std(ddof=1)),
            "detected_mean": float(det.mean()), "detected_sd": float(det.std(ddof=1)),
            "detected_cv": float(det.std(ddof=1) / det.mean()),
            "lib_mean": float(lib.mean()), "lib_median": float(np.median(lib)), "lib_sd": float(lib.std(ddof=1))}, c


if __name__ != "__main__":
    TARGETS_RUN = []
else:
    TARGETS_RUN = TARGETS
res = {}; curves = {}
for t in TARGETS_RUN:
    xv = real_cells("val", t); xt = real_cells("train", t); ref = prof(xv)
    res[t] = {"real_val": {"n": len(xv), **compare(xv, ref)[0]}}; curves[t] = {"real": curve(ref[0], ref[1])}
    fl = []
    for s in range(5):
        i = np.random.default_rng(20261107 + s).choice(len(xt), min(len(xv), len(xt)), replace=False)
        fl.append(compare(xt[i], ref)[0])
    res[t]["floor_train_vs_val"] = {k: {"mean": float(np.mean([f[k] for f in fl])), "min": float(np.min([f[k] for f in fl])), "max": float(np.max([f[k] for f in fl]))} for k in fl[0]}
    rows = []
    for r in run_list(t):
        x = load_cells(r, t); st, c = compare(x, ref)
        rows.append({"arm": r["arm"], "train": r["train"], "samp": r["samp"], "n": len(x), **st})
        curves[t].setdefault(r["arm"], []).append(c)
    res[t]["runs"] = rows
    print(t, "real", {k: round(v, 4) for k, v in res[t]["real_val"].items()}, flush=True)
    print("  floor", {k: round(v["mean"], 4) for k, v in res[t]["floor_train_vs_val"].items()}, flush=True)
    for a in ARMS:
        rr = [q for q in rows if q["arm"] == a]
        print(" ", a, len(rr), {k: round(float(np.mean([q[k] for q in rr])), 4) for k in fl[0]}, flush=True)
if __name__ == "__main__":
    (OUT / "r2_numbers.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    np.savez(OUT / "r2_curves.npz", edges=EDGES, **{f"{t}__{k}": (np.nanmean(np.vstack(v), 0) if k != "real" else v) for t in TARGETS for k, v in curves[t].items()})
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.5})
    mid = np.r_[EDGES[0] / 1.5, np.sqrt(EDGES[:-1] * EDGES[1:]), EDGES[-1] * 1.5]
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 3.7), gridspec_kw={"height_ratios": [1.25, 1]})
    for j, t in enumerate(TARGETS):
        a = ax[0, j]
        for arm in ARMS:
            a.plot(mid, np.nanmean(np.vstack(curves[t][arm]), 0), color=COL[arm], lw=1.2, label=arm)
        a.plot(mid, curves[t]["real"], color=INK, lw=1.6, ls=(0, (3, 1.5)), label="Real val")
        a.set_xscale("log"); a.set_xlim(1e-3, 30); a.set_ylim(0, 1.02); a.set_title(f"{t}, n={N_OUT[t]:,}", fontsize=8, pad=3)
        a.set_xlabel("Mean count per gene (log scale)", color=MUTED, fontsize=7)
        b = ax[1, j]; v = res[t]["real_val"]
        b.axvline(v["detected_mean"], color=INK, lw=1, ls=(0, (3, 1.5)))
        for i, arm in enumerate(ARMS):
            rr = [q for q in res[t]["runs"] if q["arm"] == arm]; d = np.array([q["detected_mean"] for q in rr])
            b.plot([d.min(), d.max()], [i, i], color=COL[arm], lw=2, solid_capstyle="round"); b.plot(d.mean(), i, "o", ms=4.5, color=COL[arm], mec="white", mew=0.5)
        b.set_yticks(range(len(ARMS))); b.set_yticklabels(ARMS if j == 0 else []); b.invert_yaxis()
        b.set_xlabel("Detected genes per cell, mean", color=MUTED, fontsize=7)
        for q in (a, b):
            q.spines[["top", "right"]].set_visible(False); q.grid(lw=0.3, color=GRID); q.set_axisbelow(True)
    ax[0, 0].set_ylabel("Zero fraction, binned", color=MUTED)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.1, wspace=0.14, hspace=0.42)
    fig.savefig(OUT / "R2_sparsity.png", dpi=300)
