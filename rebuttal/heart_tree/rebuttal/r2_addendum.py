# -*- coding: utf-8 -*-
"""R2 addendum: train 20260626 row and per-gene Fano (variance / mean, raw counts, ddof=1, genes with
mean > 0) for every arm and target.

    cd Experiments/HVG && python -m rebuttal.r2_addendum
"""
import json, os, sys
os.environ["OMP_NUM_THREADS"] = os.environ.get("R_THREADS", "4")   # R12 CPU window: 1
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
from rebuttal.w2_runs import RUNS, TARGETS, ARMS, N_OUT, run_list, load_cells, real_cells, ours_stem
from rebuttal.r2_sparsity import compare, prof
OUT = Path(__file__).resolve().parent / "out" / "R2"


def fano(x):
    m = x.mean(0); v = x.var(0, ddof=1); k = m > 0
    return float(np.median(v[k] / m[k]))


def runs626(t):
    r = [dict(arm="OURS_626", train=20260626, samp=20260987, path=RUNS / "table2_seeds_s20260626" / f"{t}_s20260626" / ours_stem(t, 20260987))]
    for s in (20260988, 20260989):
        r.append(dict(arm="OURS_626", train=20260626, samp=s, path=RUNS / f"table2_ours_s20260626_samp{s}" / t / f"ours_{t}_train20260626_samp{s}.npy"))
    return r


res = {}
for t in TARGETS:
    xv = real_cells("val", t); xt = real_cells("train", t); ref = prof(xv)
    d = {"real_val_fano": fano(xv), "real_train_full_fano": fano(xt)}
    d["floor_fano"] = [fano(xt[np.random.default_rng(20261107 + s).choice(len(xt), min(len(xv), len(xt)), replace=False)]) for s in range(5)]
    rows = []
    for r in run_list(t) + runs626(t):
        x = load_cells(r, t); st = compare(x, ref)[0]
        rows.append({"arm": r["arm"], "train": r["train"], "samp": r["samp"], "fano": fano(x), **st})
    d["runs"] = rows; res[t] = d
    print(t, "real", round(d["real_val_fano"], 3), "floor", np.round(d["floor_fano"], 3).tolist())
    for a in ARMS + ["OURS_626"]:
        rr = [q for q in rows if q["arm"] == a]; f = [q["fano"] for q in rr]
        print(" ", a, len(rr), "fano", round(float(np.mean(f)), 3), round(min(f), 3), round(max(f), 3), {k: round(float(np.mean([q[k] for q in rr])), 3) for k in ("gene_zero_mae_vs_real", "curve_rmse_vs_real", "detected_mean", "detected_sd", "lib_mean")}, flush=True)
(OUT / "r2_addendum.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
