#!/usr/bin/env python
r"""PCA grid (fig:hvg-pca-grid) redraw for the revision.
Points: pca_grid_coords_k128.npz (Windows, r12_pca_grid_coords_k128.py db5682a9): real_xy / real_type (all val cells, frozen atlas PCA)
and gen__<row>__<column> (the 2,500 plotted generated cells; CountsDiff = K128 g1.0, train 20260625). Drawing copied from
r12_pca_grid.py draw() (colours, point sizes, alpha, 3.42 in height, 6.6 in width for 6 columns, rasterised, 400 dpi, bbox tight) with
the 18:50 changes only: column order scVI, CFGen, scANVI, MDLM, CountsDiff, CRNDiff (bold, rightmost); legend moved above the grid and
enlarged to the column-title scale. rc from the same style.py as Fig. 7. Nothing is projected or refit here.
    python pca_grid_rebuttal.py OUT_DIR
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; S = HERE / "pca_src"; sys.path.insert(0, str(HERE / "fig7_src"))
import style as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
st.apply_rc()
z = np.load(S / "pca_grid_coords_k128.npz", allow_pickle=False); J = json.load(open(S / "pca_grid_coords_k128.json")); sty = J["style_from_r12_pca_grid"]
zr, yt = z["real_xy"], z["real_type"].astype(str); ROWS = J["rows"]; COLS = ["scVI", "CFGen", "scANVI", "MDLM", "CountsDiff", "Ours"]
assert COLS == J["column_order_requested_18_50"], J["column_order_requested_18_50"]
FS_T = sty["title_fs"]; FS_LEG = FS_T                                         # legend at the column-title scale (was 6.4)
fig, axes = plt.subplots(3, len(COLS), figsize=(sty["figsize_6col"][0], sty["figsize_6col"][1] + 0.18), sharex=True, sharey=True)
fig.subplots_adjust(left=.048, right=.996, top=.885, bottom=.02, wspace=.05, hspace=.06)
for i, ct in enumerate(ROWS):
    tgt = yt == ct
    for j, m in enumerate(COLS):
        ax = axes[i, j]
        ax.scatter(zr[:, 0], zr[:, 1], s=sty["bg_s"], c=sty["COL_BG"], linewidths=0, zorder=1, rasterized=True)
        ax.scatter(zr[tgt, 0], zr[tgt, 1], s=sty["target_s"], c=sty["COL_TARGET"], linewidths=0, zorder=2, rasterized=True)
        g = z[f"gen__{ct}__{m}"]; assert g.shape == (J["n_panel"], 2)
        ax.scatter(g[:, 0], g[:, 1], s=sty["gen_s"], c=sty["COL_GEN"], alpha=sty["gen_alpha"], linewidths=0, zorder=3, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_linewidth(.5)                    # as draw(): top/right stay off via rc
        if i == 0: ax.set_title("CRNDiff" if m == "Ours" else m, fontsize=FS_T, pad=2, fontweight="bold" if m == "Ours" else "normal")
        if j == 0: ax.set_ylabel(ct, fontsize=FS_T, labelpad=2)
h = [plt.Line2D([0], [0], ls="none", marker="o", ms=4.2, color=sty["COL_TARGET"]), plt.Line2D([0], [0], ls="none", marker="o", ms=4.2, color=sty["COL_GEN"])]
fig.legend(h, sty["legend_labels"], loc="upper center", bbox_to_anchor=(.52, 1.0), ncol=2, frameon=False, fontsize=FS_LEG, handletextpad=.3, columnspacing=2.0, borderpad=0.)
for e in ("pdf", "png"):
    fig.savefig(out / f"hvg_pca_baselines_all_rebuttal.{e}", dpi=sty["dpi"], bbox_inches="tight", facecolor="#ffffff")
plt.close(fig)
import matplotlib.image as mpimg
ims = [(mpimg.imread(S / "fig_pca_baselines_all_v3.png"), "Paper (v3, 5 columns)"), (mpimg.imread(S / "hvg_pca_baselines_all_r12_countsdiff.png"), "Earlier rebuttal render (CountsDiff at per-target K, CRNDiff 5th)"),
       (mpimg.imread(out / "hvg_pca_baselines_all_rebuttal.png"), "This redraw (CountsDiff K128, CRNDiff rightmost, legend above)")]
f2, a2 = plt.subplots(3, 1, figsize=(7.2, 10.5))
for a, (im, ti) in zip(a2, ims): a.imshow(im); a.set_axis_off(); a.set_title(ti, fontsize=8, loc="left")
f2.tight_layout(); f2.savefig(out / "hvg_pca_baselines_all_old_vs_new.png", dpi=170); plt.close(f2)
rec = {"inputs": {p.name: sha(p) for p in sorted(S.iterdir()) if p.is_file()}, "style_py_sha256": sha(HERE / "fig7_src" / "style.py"), "script_sha256": sha(__file__),
       "columns": COLS, "rows": ROWS, "legend_fontsize": FS_LEG, "legend_marker_ms": 4.2, "outputs": {p.name: sha(p) for p in sorted(out.glob("hvg_pca_baselines_all_*")) if p.suffix in (".pdf", ".png")}}
json.dump(rec, open(out / "hvg_pca_baselines_all_rebuttal.json", "w"), indent=1); print(json.dumps({k: rec[k] for k in ("script_sha256", "columns", "outputs")}, indent=1))
