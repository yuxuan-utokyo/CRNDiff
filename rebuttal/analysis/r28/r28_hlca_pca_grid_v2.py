#!/usr/bin/env python
r"""R28 v2: style-only redraw of the R28 grid. Same projection code and basis (71e296c7)
as r28_hlca_pca_grid.py (7d650db8); the coordinates are now SAVED (runs/r28/hlca_pca_grid_coords_<split>.npz). Gate: the v1 style is
re-rendered from these coordinates and its PNG must be byte-identical to the first draw (sha in hlca_pca_grid_<split>.json), which shows
the coordinates are the same. Then v2 style: real target cells above the generated cells (zorder 4), s 0.8 -> 1.2 with a 0.08 pt black
edge, generated alpha 0.5 -> 0.25 (same as the heart pca_grid_rebuttal_v2.py). Outputs hlca_pca_grid_<split>_v2.*; v1 kept.
v1: R28: the heart PCA grid (scripts/figs/pca_grid_rebuttal.py, fig:hvg-pca-grid) redrawn on HLCA plasma cell +
ionocyte. Same drawing: colours, point sizes, alpha, rasterised, dpi, title / legend sizes from pca_src/pca_grid_coords_k128.json
(style_from_r12_pca_grid), CRNDiff bold and rightmost, legend above. Changes: data (HLCA), 2 rows x 5 methods (scVI, CFGen, scANVI,
CountsDiff K128, CRNDiff; no MDLM on HLCA), width scaled 6.6 in x 5/6.
Basis: the heart pca2 recipe (r1_bases_train.py: PCA(n_components=30, random_state=20261031) on log1p(CP10K) float64, first 2 components),
fit on ALL HLCA train cells (cached in runs/r28/hlca_pca2_train.npz). Background = every real eval-split cell (grey), target = real eval
target cells (green), generated = each method's seed-1 array from the v3 manifest (2,500 cells, all plotted).
    python r28_hlca_pca_grid_v2.py val|test OUT_DIR      (test: logged as a style-only redraw)
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
OUTF = out / f"hlca_pca_grid_{SPL}_v2.png"
assert not OUTF.exists(), f"{OUTF} exists"
V1 = json.load(open(W / f"runs/r28/fig/hlca_pca_grid_{SPL}.json")); V1PNG = V1["outputs"][f"hlca_pca_grid_{SPL}.png"]
if SPL == "test": assert (out / "hlca_pca_grid_val_v2.png").exists(), "val first"
def cpm(x):
    x = np.asarray(x, dtype=np.float64); lib = x.sum(1, keepdims=True); return np.log1p(x / np.maximum(lib, 1) * 1e4)
BF = W / "runs/r28/hlca_pca2_train.npz"
assert BF.exists() and sha(BF) == V1["basis"]["sha256"]
B = np.load(BF); proj = lambda c: (cpm(c) - B["mean_"]) @ B["comp2"].T
ED = W / ("data/r16_hlca_score" if SPL == "val" else "data/r16_hlca_test_view")
XE = np.load(ED / "val.npy", mmap_mode="r"); YE = np.load(ED / "val_celltype.npy", allow_pickle=True).astype(str)
zr = np.concatenate([proj(XE[i:i + 20000]) for i in range(0, len(YE), 20000)])
man = json.load(open(W / "runs/r24_26_3seed/manifest.json"))
SEED1 = {"CRNDiff": "gen20260625", "CFGen": "samp20260987", "CountsDiff": "train20260625", "scVI": "train20261050", "scANVI": "train20261060"}
ROWS = ["plasma cell", "ionocyte"]; COLS = ["scVI", "CFGen", "scANVI", "CountsDiff", "CRNDiff"]
G = {}; used = {}
for ct in ROWS:
    for m in COLS:
        e = [x for x in man if x["method"] == m and x["target"] == ct and x["seed"] == SEED1[m]]; assert len(e) == 1, (m, ct)
        G[(ct, m)] = proj(np.load(e[0]["path"])); used[f"{ct}|{m}"] = {"path": e[0]["path"].replace(str(W) + "/", ""), "n": len(G[(ct, m)])}
assert used == V1["generated"], "generated arrays differ from the first draw"
CO = W / f"runs/r28/hlca_pca_grid_coords_{SPL}.npz"
np.savez(CO, real_xy=zr, real_type=YE, **{f"gen__{ct}__{m}": G[(ct, m)] for ct in ROWS for m in COLS})
J = json.load(open(FIG / "pca_src/pca_grid_coords_k128.json")); sty = J["style_from_r12_pca_grid"]
st.apply_rc(); FS_T = sty["title_fs"]; FS_LEG = FS_T
W6, H3 = sty["figsize_6col"]
def draw(stem, v2):
    tg = dict(s=1.2, edgecolors="black", linewidths=0.08, zorder=4) if v2 else dict(s=sty["target_s"], linewidths=0, zorder=2)
    ga = 0.25 if v2 else sty["gen_alpha"]
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(W6 * len(COLS) / 6, H3 * len(ROWS) / 3 + 0.18), sharex=True, sharey=True)
    fig.subplots_adjust(left=.048, right=.996, top=.84, bottom=.02, wspace=.05, hspace=.06)
    for i, ct in enumerate(ROWS):
        tgt = YE == ct
        for j, m in enumerate(COLS):
            ax = axes[i, j]; g = G[(ct, m)]
            ax.scatter(zr[:, 0], zr[:, 1], s=sty["bg_s"], c=sty["COL_BG"], linewidths=0, zorder=1, rasterized=True)
            ax.scatter(zr[tgt, 0], zr[tgt, 1], c=sty["COL_TARGET"], rasterized=True, **tg)
            ax.scatter(g[:, 0], g[:, 1], s=sty["gen_s"], c=sty["COL_GEN"], alpha=ga, linewidths=0, zorder=3, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values(): s_.set_linewidth(.5)
            if i == 0: ax.set_title(m, fontsize=FS_T, pad=2, fontweight="bold" if m == "CRNDiff" else "normal")
            if j == 0: ax.set_ylabel(ct, fontsize=FS_T, labelpad=2)
    if v2: h = [plt.Line2D([0], [0], ls="none", marker="o", ms=4.2, color=sty["COL_TARGET"], markeredgecolor="black", markeredgewidth=0.4), plt.Line2D([0], [0], ls="none", marker="o", ms=4.2, color=sty["COL_GEN"], alpha=0.6)]
    else: h = [plt.Line2D([0], [0], ls="none", marker="o", ms=4.2, color=sty["COL_TARGET"]), plt.Line2D([0], [0], ls="none", marker="o", ms=4.2, color=sty["COL_GEN"])]
    fig.legend(h, sty["legend_labels"], loc="upper center", bbox_to_anchor=(.52, 1.0), ncol=2, frameon=False, fontsize=FS_LEG, handletextpad=.3, columnspacing=2.0, borderpad=0.)
    for e in (("pdf", "png") if v2 else ("png",)):
        fig.savefig(stem.with_suffix("." + e), dpi=sty["dpi"], bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
import tempfile
with tempfile.TemporaryDirectory() as td:
    draw(Path(td) / "v1check", False); v1 = sha(Path(td) / "v1check.png")
assert v1 == V1PNG, f"v1 re-render {v1[:8]} != first draw {V1PNG[:8]}: coordinates or rendering differ"
draw(out / f"hlca_pca_grid_{SPL}_v2", True)
rec = {"split": SPL, "basis": V1["basis"], "coords": {"file": str(CO.relative_to(W)), "sha256": sha(CO)}, "gate_v1_png_identical": True, "v1_png_sha256": V1PNG,
       "v2_changes": {"target_s": 1.2, "target_edge_lw": 0.08, "target_zorder": 4, "gen_alpha": 0.25}, "generated": used, "script_sha256": sha(__file__),
       "outputs": {p.name: sha(p) for p in sorted(out.glob(f"hlca_pca_grid_{SPL}_v2.*")) if p.suffix in (".pdf", ".png")}}
json.dump(rec, open(out / f"hlca_pca_grid_{SPL}_v2.json", "w"), indent=1); print(json.dumps({k: rec[k] for k in ("gate_v1_png_identical", "coords", "outputs")}))
if SPL == "test":
    with open(W / "runs/r16/TEST_READ_LOG.tsv", "a") as f:
        f.write(f"{time.strftime('%m-%d_%H:%M:%S')}\tredraw of R28 (style only), same basis {V1['basis']['sha256'][:8]}; v1 re-render byte-identical; coords saved (r28_hlca_pca_grid_v2.py {sha(__file__)[:8]})\t{OUTF.resolve().relative_to(W)}\t{sha(OUTF)[:8]}\n")
