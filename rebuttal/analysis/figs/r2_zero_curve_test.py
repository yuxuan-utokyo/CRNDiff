#!/usr/bin/env python
r"""Zero fraction vs mean count, binned (R2 top row) on TEST for the paper. Not a new analysis: the curves are
re-derived with the R19 code (prof / curve / compare / EDGES / load_cells copied verbatim from scripts/r12_extra/r1r2_metrics.py da155341)
on the same runs R19 read (runs/r19/test/r1r2_crndiff_<T>_test.json, r1r2_baselines_test.json, r1r2_cd_k128_test.json), and the GATE is
that every run's curve_rmse_vs_real recomputed here equals the stored R19 test value bit for bit, so nothing new is read.
Layout: 3 panels (Endothelial, Myeloid, Neuronal), legend on top, "CRNDiff" (not OURS), CountsDiff = K128 (3 training seeds); line =
mean over a method's runs of the binned curve; real test cells dashed black. Colours: CRNDiff vermilion #D55E00 (the generated colour of
the PCA grid), the rest Okabe-Ito; the real-cell green of the PCA grid is not used for a method.
    python r2_zero_curve_test.py OUT_DIR
"""
import hashlib, json, os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "4"
from pathlib import Path
import numpy as np
W = Path("./work"); D = W / "data/hvg2k"; sys.path.insert(0, str(W / "scripts/figs/fig7_src"))
import style as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(W / "scripts/r12_extra/r1r2_metrics.py")[:8] == "da155341"
# ---- verbatim from r1r2_metrics.py (r2_sparsity.py / w2_runs.load_cells)
EDGES = np.logspace(-4, 2, 25)
def prof(x):
    gm = x.mean(0); gz = (x == 0).mean(0)
    cz = (x == 0).mean(1); det = (x > 0).sum(1); lib = x.sum(1)
    return gm, gz, cz, det, lib
def curve(gm, gz):
    idx = np.digitize(gm, EDGES)
    return np.array([gz[idx == i].mean() if (idx == i).sum() >= 5 else np.nan for i in range(len(EDGES) + 1)])
N_OUT = {"Endothelial": 10057, "Myeloid": 2500, "Neuronal": 2500}; SUBSAMPLE_SEED = 20261107
def load_cells(path, t, samp, train):
    x = np.load(path, mmap_mode="r"); n = N_OUT[t]
    if x.shape[0] > n:
        s = np.random.SeedSequence([SUBSAMPLE_SEED, samp, train]).generate_state(1)
        idx = np.sort(np.random.default_rng(int(s[0])).choice(x.shape[0], n, replace=False))
        return np.asarray(x[idx], dtype=np.float64)
    return np.asarray(x, dtype=np.float64)
# ---- runs exactly as R19 read them
T = ["Endothelial", "Myeloid", "Neuronal"]; R19 = W / "runs/r19/test"
J = {"crn": {t: json.load(open(R19 / f"r1r2_crndiff_{t}_test.json")) for t in T}, "base": json.load(open(R19 / "r1r2_baselines_test.json")), "cd": json.load(open(R19 / "r1r2_cd_k128_test.json"))}
for d in [*J["crn"].values(), J["base"], J["cd"]]: assert d["split"] == "test"
LAB = {"OURS": "CRNDiff", "CountsDiff": "CountsDiff (K128)", "CFGen": "CFGen", "MDLM": "MDLM", "scVI": "scVI", "scANVI": "scANVI"}
ORDER = ["OURS", "CountsDiff", "CFGen", "MDLM", "scVI", "scANVI"]
COL = {"OURS": "#D55E00", "CountsDiff": "#CC79A7", "CFGen": "#0072B2", "MDLM": "#56B4E9", "scVI": "#E69F00", "scANVI": "#7F7F7F"}
runs = lambda t: [r for r in J["crn"][t]["runs"] if r["target"] == t] + [r for r in J["base"]["runs"] if r["target"] == t] + [r for r in J["cd"]["runs"] if r["target"] == t]
X = np.load(D / "test.npy", mmap_mode="r"); Y = np.load(D / "test_celltype.npy", allow_pickle=True).astype(str)
assert sha(D / "test.npy")[:8] == "ad800326" and sha(D / "test_celltype.npy")[:8] == "851a3d3d"
CUR, RMSE, gate, NREAL = {}, {}, [], {}
for t in T:
    xv = np.asarray(X[np.flatnonzero(Y == t)], dtype=np.float64); gm, gz = prof(xv)[:2]; rc = curve(gm, gz); NREAL[t] = len(xv)
    CUR[(t, "REAL")] = rc
    for r in runs(t):
        x = load_cells(r["path"], t, r["seed"], r.get("train", 0)); c = curve(*prof(x)[:2]); ok = np.isfinite(c) & np.isfinite(rc)
        rm = float(np.sqrt(np.nanmean((c[ok] - rc[ok]) ** 2))); gate.append(abs(rm - r["r2"]["curve_rmse_vs_real"]))
        CUR.setdefault((t, r["method"]), []).append(c); RMSE.setdefault((t, r["method"]), []).append(r["r2"]["curve_rmse_vs_real"])
assert max(gate) == 0.0, f"curve RMSE differs from R19 test by {max(gate)}"
# ---- draw (top row only)
st.apply_rc()
xc = np.sqrt(EDGES[:-1] * EDGES[1:]); xc = np.concatenate([[EDGES[0] / np.sqrt(EDGES[1] / EDGES[0])], xc, [EDGES[-1] * np.sqrt(EDGES[1] / EDGES[0])]])
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True); fig.subplots_adjust(left=.075, right=.995, top=.74, bottom=.19, wspace=.08)
for ax, t in zip(axes, T):
    for m in ORDER:
        c = np.nanmean(np.array(CUR[(t, m)]), 0) if np.isfinite(np.array(CUR[(t, m)])).any() else None
        k = np.isfinite(c); ax.plot(xc[k], c[k], color=COL[m], lw=1.3, label=LAB[m], zorder=3 if m == "OURS" else 2)
    rc = CUR[(t, "REAL")]; k = np.isfinite(rc); ax.plot(xc[k], rc[k], color="black", lw=1.4, ls="--", label="Real test cells", zorder=4)
    ax.set_xscale("log"); ax.set_xlim(1e-3, 30); ax.set_ylim(0, 1.02); ax.set_title(f"{t} (test, n = {NREAL[t]:,})", fontsize=8, pad=3)
    ax.set_xlabel("Mean count per gene (log scale)", fontsize=7.5); ax.grid(alpha=.3, lw=.4)
axes[0].set_ylabel("Zero fraction, binned", fontsize=7.5)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=7, frameon=False, fontsize=7, bbox_to_anchor=(.53, 1.0), handlelength=1.6, columnspacing=1.1)
for e in ("pdf", "png"): fig.savefig(out / f"hvg_zero_fraction_curve_test.{e}", dpi=400 if e == "png" else 300, bbox_inches="tight", facecolor="#ffffff")
plt.close(fig)
rmse = {t: {LAB[m]: {"mean": float(np.mean(RMSE[(t, m)])), "sd": float(np.std(RMSE[(t, m)], ddof=1)), "n_runs": len(RMSE[(t, m)])} for m in ORDER} for t in T}
rec = {"gate": {"n_runs": len(gate), "max_abs_diff_vs_R19_test": max(gate), "passed": True}, "curve_rmse_test": rmse, "n_real_test": NREAL,
       "inputs": {p: sha(R19 / p)[:8] for p in ["r1r2_baselines_test.json", "r1r2_cd_k128_test.json"] + [f"r1r2_crndiff_{t}_test.json" for t in T]},
       "colours": {LAB[m]: COL[m] for m in ORDER}, "script_sha256": sha(__file__),
       "outputs": {p.name: sha(p) for p in sorted(out.glob("hvg_zero_fraction_curve_test.*")) if p.suffix in (".pdf", ".png")}}
json.dump(rec, open(out / "hvg_zero_fraction_curve_test.json", "w"), indent=1); print(json.dumps({k: rec[k] for k in ("gate", "outputs")}, indent=1))
with open(W / "runs/r19/READ_LOG.tsv", "a") as f:
    f.write(f"{time.strftime('%m-%d_%H:%M:%S')}\tredraw of R19 R2 zero-fraction curves (figure only; every run's curve RMSE bit-identical to R19 test; r2_zero_curve_test.py {sha(__file__)[:8]})\tresults/figs/hvg_zero_fraction_curve_test.png\t{rec['outputs']['hvg_zero_fraction_curve_test.png'][:8]}\n")
