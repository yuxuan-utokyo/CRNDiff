#!/usr/bin/env python
r"""R29 UMAP overlay.
Heart Fig. 7 (scripts/figs/fig7_rebuttal.py) layout and sizes: panel (a) real eval-split cells in grey with the target classes coloured,
then one panel per target with CRNDiff's generated cells (seed-1 array of the v3 manifest) over the grey real cells; legend on top,
"UMAP 1" / "UMAP 2", point sizes 0.6 / 0.8 / 1.2, rasterised. Changes: data (HLCA), 3 panels (plasma cell, ionocyte), page width x 3/4,
target colours Okabe-Ito vermilion / blue (the heart TYPE_COLORS belong to heart types).
Reducer settings copied from the heart reducer (fig7_src/fig_umap_panels.json frozen_basis: umap_expX.py, PCA50 -> UMAP, n_neighbors 30,
min_dist 0.3, random_state 0, n_components 2, default euclidean metric, fit once on all 45,145 heart val cells). Differences, both forced by
"fit on HLCA train only": the fit set is 45,145 HLCA TRAIN cells (the heart fit-set size; seed 20261112) instead of val cells, and the PCA50
input is the train-fit basis (runs/r29_30/hlca_pca50_train.npz). Cached (runs/r29_30/umap_train.joblib) so val and test share the reducer.
    python r29_umap_hlca.py val|test OUT_DIR      (test: once; logged)
"""
import hashlib, json, os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"): os.environ[v] = "8"
from pathlib import Path
import numpy as np
SPL, out = sys.argv[1], Path(sys.argv[2]); assert SPL in ("val", "test"); out.mkdir(parents=True, exist_ok=True)
W = Path("./work"); FIG = W / "scripts/figs"; sys.path.insert(0, str(FIG / "fig7_src"))
import style as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, umap
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUTF = out / f"hlca_umap_{SPL}.png"; assert not OUTF.exists(), f"{OUTF} exists"
if SPL == "test": assert (out / "hlca_umap_val.png").exists(), "val first"
P = np.load(W / "runs/r29_30/hlca_pca50_train.npz")
def pca50(c):
    x = np.asarray(c, dtype=np.float64); lib = x.sum(1, keepdims=True)
    return ((np.log1p(x / np.maximum(lib, 1) * 1e4)).astype(np.float32) - P["mean"]) @ P["components"].T
RF = W / "runs/r29_30/umap_train.joblib"
if not RF.exists():
    X = np.load(W / "data/r16_hlca_fit/train.npy", mmap_mode="r")
    ii = np.sort(np.random.default_rng(20261112).choice(X.shape[0], 45145, replace=False))
    red = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0, n_components=2, metric="euclidean").fit(pca50(X[ii]))
    joblib.dump(red, RF); json.dump({"n_fit": 45145, "subsample_seed": 20261112, "random_state": 0, "n_neighbors": 30, "min_dist": 0.3, "metric": "euclidean",
                                     "heart_reducer": "umap_expX.py: PCA50 -> UMAP, n_neighbors 30, min_dist 0.3, random_state 0, fit on all 45,145 heart val cells",
                                     "umap_version": umap.__version__, "basis_sha256": sha(W / "runs/r29_30/hlca_pca50_train.npz"), "sha256": sha(RF)},
                                    open(str(RF).replace(".joblib", ".json"), "w"), indent=1)
red = joblib.load(RF)
ED = W / ("data/r16_hlca_score" if SPL == "val" else "data/r16_hlca_test_view")
XE = np.load(ED / "val.npy", mmap_mode="r"); YE = np.load(ED / "val_celltype.npy", allow_pickle=True).astype(str)
ur = np.concatenate([red.transform(pca50(XE[i:i + 20000])) for i in range(0, len(YE), 20000)])
man = json.load(open(W / "runs/r24_26_3seed/manifest.json")); T2 = ["plasma cell", "ionocyte"]; COL = {"plasma cell": "#D55E00", "ionocyte": "#0072B2"}
gen = {}
for t in T2:
    e = [x for x in man if x["method"] == "CRNDiff" and x["target"] == t and x["seed"] == "gen20260625"]; assert len(e) == 1
    gen[t] = (red.transform(pca50(np.load(e[0]["path"]))), e[0]["path"].replace(str(W) + "/", ""))
st.apply_rc(); BG_REAL, BG_GEN = "#dcdbd5", "#e7e6e1"; FS_T, FS_LEG = 9, 9
fig, axes = plt.subplots(1, 3, figsize=(500.386 * 3 / 4 / 72, 158 / 72))
fig.subplots_adjust(left=0.05, right=0.995, top=0.74, bottom=0.10, wspace=0.07)
ax = axes[0]; ax.scatter(ur[:, 0], ur[:, 1], s=0.6, c=BG_REAL, linewidths=0, zorder=1, rasterized=True)
for t in T2:
    m = YE == t; ax.scatter(ur[m, 0], ur[m, 1], s=0.8, c=COL[t], linewidths=0, zorder=2, rasterized=True)
ax.set_title(f"(a) {'Test' if SPL == 'test' else 'Validation'} cells", fontsize=FS_T, pad=3); ax.set_ylabel("UMAP 2", fontsize=st.FS_LABEL)
for ax, t, L in zip(axes[1:], T2, "bc"):
    ax.scatter(ur[:, 0], ur[:, 1], s=0.6, c=BG_GEN, linewidths=0, zorder=1, rasterized=True)
    g = gen[t][0]; ax.scatter(g[:, 0], g[:, 1], s=1.2, c=COL[t], linewidths=0, zorder=3, rasterized=True)
    ax.set_title(f"({L}) {t}", fontsize=FS_T, pad=3)
xl = (ur[:, 0].min(), ur[:, 0].max()); yl = (ur[:, 1].min(), ur[:, 1].max()); px, py = 0.04 * (xl[1] - xl[0]), 0.04 * (yl[1] - yl[0])
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1", fontsize=st.FS_LABEL, labelpad=2)
    ax.set_xlim(xl[0] - px, xl[1] + px); ax.set_ylim(yl[0] - py, yl[1] + py)
    for s in ax.spines.values(): s.set_linewidth(0.6)
h = [plt.Line2D([0], [0], ls="none", marker="o", ms=6, color=COL[t]) for t in T2]
fig.legend(h, T2, loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.6, borderpad=0.2, fontsize=FS_LEG)
for e in ("pdf", "png"): fig.savefig(out / f"hlca_umap_{SPL}.{e}", dpi=600 if e == "png" else 300)
plt.close(fig)
rec = {"split": SPL, "reducer": json.load(open(str(RF).replace(".joblib", ".json"))), "n_eval": int(len(YE)), "n_target": {t: int((YE == t).sum()) for t in T2},
       "generated": {t: gen[t][1] for t in T2}, "script_sha256": sha(__file__), "outputs": {p.name: sha(p) for p in sorted(out.glob(f"hlca_umap_{SPL}.*"))}}
json.dump(rec, open(out / f"hlca_umap_{SPL}.json", "w"), indent=1); print(json.dumps({k: rec[k] for k in ("n_target", "outputs")}))
if SPL == "test":
    with open(W / "runs/r16/TEST_READ_LOG.tsv", "a") as f:
        f.write(f"{time.strftime('%m-%d_%H:%M:%S')}\tHLCA R29 UMAP overlay test (real test cells drawn; r29_umap_hlca.py {sha(__file__)[:8]})\truns/r29_30/fig/hlca_umap_test.png\t{sha(OUTF)[:8]}\n")
