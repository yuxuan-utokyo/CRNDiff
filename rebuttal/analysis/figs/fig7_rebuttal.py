#!/usr/bin/env python
r"""Fig. 7 (hvg_umap_panels) redraw for the revision.
Points: fig_umap_panels.coords.npz (Sep 7 cache; real = 45,145 val cells in the frozen PCA50 -> UMAP reducer, <Type>_umap = 2,500
delivered cells per target); real-panel colours from val_celltype.npy. Palette, fonts, rc from the copied style.py (sha recorded).
Changes against the paper layer (figure7_source.pdf): Mesothelial is no longer coloured or in the legend (its cells stay in the grey
background like every non-target type); legend on top, enlarged for the column width; titles "(a) Test cells", "(b) Endothelial",
"(c) Myeloid", "(d) Neuronal"; axes "UMAP 1" / "UMAP 2"; no n, no "generated:", no letters below. Page 500.386 x 158 pt as the layer.
Nothing is projected or refit here.
    python fig7_rebuttal.py OUT_DIR
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
S = Path(__file__).resolve().parent / "fig7_src"; sys.path.insert(0, str(S))
import style as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
st.apply_rc()
z = np.load(S / "fig_umap_panels.coords.npz"); y = np.load(S / "val_celltype.npy", allow_pickle=True).astype(str)
ur = z["real"]; assert len(ur) == len(y) == 45145
T3 = ["Endothelial", "Myeloid", "Neuronal"]; BG_REAL, BG_GEN = "#dcdbd5", "#e7e6e1"     # fig_umap_panels.py COL_BG_REAL / COL_BG_GEN
FS_T, FS_LEG = 9, 9                                                                    # titles / legend, enlarged for column width
fig, axes = plt.subplots(1, 4, figsize=(500.386 / 72, 158 / 72))
fig.subplots_adjust(left=0.04, right=0.995, top=0.74, bottom=0.10, wspace=0.07)
ax = axes[0]
ax.scatter(ur[:, 0], ur[:, 1], s=0.6, c=BG_REAL, linewidths=0, zorder=1, rasterized=True)
for t in T3:
    m = y == t; ax.scatter(ur[m, 0], ur[m, 1], s=0.8, c=st.TYPE_COLORS[t], linewidths=0, zorder=2, rasterized=True)
ax.set_title("(a) Test cells", fontsize=FS_T, pad=3); ax.set_ylabel("UMAP 2", fontsize=st.FS_LABEL)
for ax, t, L in zip(axes[1:], T3, "bcd"):
    ax.scatter(ur[:, 0], ur[:, 1], s=0.6, c=BG_GEN, linewidths=0, zorder=1, rasterized=True)
    g = z[f"{t}_umap"]; ax.scatter(g[:, 0], g[:, 1], s=1.2, c=st.TYPE_COLORS[t], linewidths=0, zorder=3, rasterized=True)
    ax.set_title(f"({L}) {t}", fontsize=FS_T, pad=3)
xl = (ur[:, 0].min(), ur[:, 0].max()); yl = (ur[:, 1].min(), ur[:, 1].max()); px, py = 0.04 * (xl[1] - xl[0]), 0.04 * (yl[1] - yl[0])
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1", fontsize=st.FS_LABEL, labelpad=2)
    ax.set_xlim(xl[0] - px, xl[1] + px); ax.set_ylim(yl[0] - py, yl[1] + py)
    for s in ax.spines.values(): s.set_linewidth(0.6)
h = [plt.Line2D([0], [0], ls="none", marker="o", ms=6, color=st.TYPE_COLORS[t]) for t in T3]
fig.legend(h, T3, loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=3, handletextpad=0.3, columnspacing=1.6, borderpad=0.2, fontsize=FS_LEG)
for e in ("pdf", "png"):
    fig.savefig(out / f"hvg_umap_panels_rebuttal.{e}", dpi=600 if e == "png" else 300)
plt.close(fig)
# side by side with the paper render (150 dpi)
import matplotlib.image as mpimg
old = mpimg.imread(S / "figure7_source_render150dpi.png"); new = mpimg.imread(out / "hvg_umap_panels_rebuttal.png")
f2, a2 = plt.subplots(2, 1, figsize=(7.2, 4.9))
for a, im, ti in zip(a2, (old, new), ("Paper Fig. 7 (current layer, 150 dpi render)", "Rebuttal redraw (Mesothelial in the grey background, enlarged legend)")):
    a.imshow(im); a.set_axis_off(); a.set_title(ti, fontsize=8, loc="left")
f2.tight_layout(); f2.savefig(out / "hvg_umap_panels_old_vs_new.png", dpi=200); plt.close(f2)
rec = {"inputs": {p.name: sha(p) for p in sorted(S.iterdir()) if p.is_file()}, "script_sha256": sha(__file__),
       "outputs": {p.name: sha(p) for p in sorted(out.glob("hvg_umap_panels_*")) if p.suffix in (".pdf", ".png")},
       "real_cells_coloured": {t: int((y == t).sum()) for t in T3}, "mesothelial_in_background": int((y == "Mesothelial").sum()),
       "page_pt": [500.386, 158], "legend_fontsize": FS_LEG, "title_fontsize": FS_T}
json.dump(rec, open(out / "hvg_umap_panels_rebuttal.json", "w"), indent=1); print(json.dumps(rec, indent=1))
