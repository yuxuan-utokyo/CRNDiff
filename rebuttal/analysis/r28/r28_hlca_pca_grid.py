#!/usr/bin/env python
r"""R28: the heart PCA grid (scripts/figs/pca_grid_rebuttal.py, fig:hvg-pca-grid) redrawn on HLCA plasma cell +
ionocyte. Same drawing: colours, point sizes, alpha, rasterised, dpi, title / legend sizes from pca_src/pca_grid_coords_k128.json
(style_from_r12_pca_grid), CRNDiff bold and rightmost, legend above. Changes: data (HLCA), 2 rows x 5 methods (scVI, CFGen, scANVI,
CountsDiff K128, CRNDiff; no MDLM on HLCA), width scaled 6.6 in x 5/6.
Basis: the heart pca2 recipe (r1_bases_train.py: PCA(n_components=30, random_state=20261031) on log1p(CP10K) float64, first 2 components),
fit on ALL HLCA train cells (cached in runs/r28/hlca_pca2_train.npz). Background = every real eval-split cell (grey), target = real eval
target cells (green), generated = each method's seed-1 array from the v3 manifest (2,500 cells, all plotted).
    python r28_hlca_pca_grid.py val|test OUT_DIR      (test: once; logged)
"""
import hashlib, json, os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "8"
from pathlib import Path
import numpy as np
SPL, out = sys.argv[1], Path(sys.argv[2]); assert SPL in ("val", "test"); out.mkdir(parents=True, exist_ok=True)
W = Path("./work"); FIG = W / "scripts/figs"; sys.path.insert(0, str(FIG / "fig7_src"))
import style as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUTF = out / f"hlca_pca_grid_{SPL}.png"
assert not OUTF.exists(), f"{OUTF} exists; each split is drawn once"
if SPL == "test": assert (out / "hlca_pca_grid_val.png").exists(), "val first"
def cpm(x):
    x = np.asarray(x, dtype=np.float64); lib = x.sum(1, keepdims=True); return np.log1p(x / np.maximum(lib, 1) * 1e4)
BF = W / "runs/r28/hlca_pca2_train.npz"
if not BF.exists():
    from sklearn.decomposition import PCA
    X = np.load(W / "data/r16_hlca_fit/train.npy", mmap_mode="r")
    Z = np.concatenate([cpm(X[i:i + 20000]) for i in range(0, X.shape[0], 20000)])
    p = PCA(n_components=30, random_state=20261031).fit(Z)
    BF.parent.mkdir(parents=True, exist_ok=True)
    np.savez(BF, mean_=p.mean_, comp2=p.components_[:2], evr2=p.explained_variance_ratio_[:2], n_fit=len(Z)); del Z
B = np.load(BF); proj = lambda c: (cpm(c) - B["mean_"]) @ B["comp2"].T
ED = W / ("data/r16_hlca_score" if SPL == "val" else "data/r16_hlca_test_view")
XE = np.load(ED / "val.npy", mmap_mode="r"); YE = np.load(ED / "val_celltype.npy", allow_pickle=True).astype(str)
zr = np.concatenate([proj(XE[i:i + 20000]) for i in range(0, len(YE), 20000)])
man = json.load(open(W / "runs/r24_26_3seed/manifest.json"))
SEED1 = {"CRNDiff": "gen20260625", "CFGen": "samp20260987", "CountsDiff": "train20260625", "scVI": "train20261050", "scANVI": "train20261060"}
ROWS = ["plasma cell", "ionocyte"]; COLS = ["scVI", "CFGen", "scANVI", "CountsDiff", "CRNDiff"]
J = json.load(open(FIG / "pca_src/pca_grid_coords_k128.json")); sty = J["style_from_r12_pca_grid"]
st.apply_rc(); FS_T = sty["title_fs"]; FS_LEG = FS_T
W6, H3 = sty["figsize_6col"]
fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(W6 * len(COLS) / 6, H3 * len(ROWS) / 3 + 0.18), sharex=True, sharey=True)
fig.subplots_adjust(left=.048, right=.996, top=.84, bottom=.02, wspace=.05, hspace=.06)
used = {}
for i, ct in enumerate(ROWS):
    tgt = YE == ct
    for j, m in enumerate(COLS):
        ax = axes[i, j]
        e = [x for x in man if x["method"] == m and x["target"] == ct and x["seed"] == SEED1[m]]; assert len(e) == 1, (m, ct)
        g = proj(np.load(e[0]["path"])); used[f"{ct}|{m}"] = {"path": e[0]["path"].replace(str(W) + "/", ""), "n": len(g)}
        ax.scatter(zr[:, 0], zr[:, 1], s=sty["bg_s"], c=sty["COL_BG"], linewidths=0, zorder=1, rasterized=True)
        ax.scatter(zr[tgt, 0], zr[tgt, 1], s=sty["target_s"], c=sty["COL_TARGET"], linewidths=0, zorder=2, rasterized=True)
        ax.scatter(g[:, 0], g[:, 1], s=sty["gen_s"], c=sty["COL_GEN"], alpha=sty["gen_alpha"], linewidths=0, zorder=3, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_linewidth(.5)
        if i == 0: ax.set_title(m, fontsize=FS_T, pad=2, fontweight="bold" if m == "CRNDiff" else "normal")
        if j == 0: ax.set_ylabel(ct, fontsize=FS_T, labelpad=2)
h = [plt.Line2D([0], [0], ls="none", marker="o", ms=4.2, color=sty["COL_TARGET"]), plt.Line2D([0], [0], ls="none", marker="o", ms=4.2, color=sty["COL_GEN"])]
fig.legend(h, sty["legend_labels"], loc="upper center", bbox_to_anchor=(.52, 1.0), ncol=2, frameon=False, fontsize=FS_LEG, handletextpad=.3, columnspacing=2.0, borderpad=0.)
for e in ("pdf", "png"):
    fig.savefig(out / f"hlca_pca_grid_{SPL}.{e}", dpi=sty["dpi"], bbox_inches="tight", facecolor="#ffffff")
plt.close(fig)
rec = {"split": SPL, "basis": {"file": str(BF.relative_to(W)), "sha256": sha(BF), "evr_pc1_pc2": B["evr2"].tolist(), "n_fit": int(B["n_fit"])},
       "n_background": int(len(YE)), "n_target": {ct: int((YE == ct).sum()) for ct in ROWS}, "generated": used, "columns": COLS, "rows": ROWS,
       "style_source": "scripts/figs/pca_src/pca_grid_coords_k128.json style_from_r12_pca_grid", "style_py_sha256": sha(FIG / "fig7_src/style.py"),
       "script_sha256": sha(__file__), "outputs": {p.name: sha(p) for p in sorted(out.glob(f"hlca_pca_grid_{SPL}.*"))}}
json.dump(rec, open(out / f"hlca_pca_grid_{SPL}.json", "w"), indent=1); print(json.dumps({k: rec[k] for k in ("basis", "n_target", "outputs")}))
if SPL == "test":
    with open(W / "runs/r16/TEST_READ_LOG.tsv", "a") as f:
        f.write(f"{time.strftime('%m-%d_%H:%M:%S')}\tHLCA R28 PCA grid test (real test cells drawn; r28_hlca_pca_grid.py {sha(__file__)[:8]})\truns/r28/fig/hlca_pca_grid_test.png\t{sha(OUTF)[:8]}\n")
