# -*- coding: utf-8 -*-
"""R1 figures, drawn from rebuttal/out/R1/{r1_cells.npz,r1_numbers.json,r1_clumps.npz}.

    cd Experiments/HVG && python -m rebuttal.r1_figures
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "out" / "R1"
ARMS = ["OURS", "CFGen", "MDLM", "scVI", "scANVI"]
COL = dict(zip(ARMS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
RUN = {"OURS": "OURS_20260625_20260987", "CFGen": "CFGen_run1_20260931",
       "MDLM": "MDLM_20260625_20260931", "scVI": "scVI_20261050_20261051",
       "scANVI": "scANVI_20261060_20261061"}
INK, MUTED, BG = "#0b0b0b", "#52514e", "#e4e3de"
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.5, "font.family": "DejaVu Sans"})
z = np.load(OUT / "r1_cells.npz")
real, y = z["real_xy"], z["yval"].astype(str)
neu = real[y == "Neuronal"]
c = np.load(OUT / "r1_clumps.npz")
cj = json.loads((OUT / "r1_clumps.json").read_text(encoding="utf-8"))
C1 = [r for r in cj["clumps"] if r["label"] == "C1"][0]

# ---- Figure 1: overlay in the frozen UMAP, then zooms for OURS vs real
tr = np.load(OUT / "train_neuronal_umap.npy")
fig = plt.figure(figsize=(7.4, 3.75))
gs = fig.add_gridspec(2, 20, height_ratios=[1, 1.02], left=0.035, right=0.995, top=0.91, bottom=0.075,
                      hspace=0.28, wspace=0.0)
top = [fig.add_subplot(gs[0, 4 * i:4 * i + 4]) for i in range(5)]
for a, arm in zip(top, ARMS):
    a.scatter(real[:, 0], real[:, 1], s=0.5, c=BG, linewidths=0, rasterized=True)
    a.scatter(neu[:, 0], neu[:, 1], s=3.2, facecolors="none", edgecolors=INK, linewidths=0.25, rasterized=True)
    g = z[f"xy__{RUN[arm]}"]
    a.scatter(g[:, 0], g[:, 1], s=1.8, c=COL[arm], linewidths=0, rasterized=True)
    a.set_title(f"{arm} (n=2,500)", fontsize=8, pad=2, color=INK)
    a.set_xticks([]); a.set_yticks([])
top[0].set_ylabel("UMAP 2", color=MUTED)
ours = z[f"xy__{RUN['OURS']}"]
c0 = c["ours_xy"][c["clump"] == 0]
c1 = c["ours_xy"][c["clump"] == 1]
def zoom(a, pts, col, box, title, ring=False):
    a.scatter(real[:, 0], real[:, 1], s=2.5, c=BG, linewidths=0, rasterized=True)
    if ring:
        a.scatter(pts[:, 0], pts[:, 1], s=7, facecolors="none", edgecolors=INK, linewidths=0.4, rasterized=True)
    else:
        a.scatter(pts[:, 0], pts[:, 1], s=4, c=col, linewidths=0, rasterized=True)
    a.set_xlim(box[0], box[1]); a.set_ylim(box[2], box[3])
    a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=7, pad=2, color=INK)
def box(p, m=0.5):
    lo, hi = np.percentile(p, 0.5, axis=0), np.percentile(p, 99.5, axis=0)
    return lo[0] - m, hi[0] + m, lo[1] - m, hi[1] + m
b0, b1 = box(c0, 0.45), box(c1, 0.8)
ax4 = [fig.add_subplot(gs[1, 5 * i:5 * i + 5]) for i in range(4)]
zoom(ax4[0], ours, COL["OURS"], b0, "OURS, main neuronal region")
zoom(ax4[1], neu, None, b0, "Real val neuronal, main region", ring=True)
zoom(ax4[2], ours, COL["OURS"], b1, "OURS, group C1 (fibroblast region)")
zoom(ax4[3], neu, None, b1, "Real val neuronal, group C1 region", ring=True)
ax4[0].set_ylabel("UMAP 2", color=MUTED)
for a in top + ax4:
    a.set_xlabel("UMAP 1", color=MUTED, fontsize=6.5, labelpad=1)
    for sp in a.spines.values():
        sp.set_color(MUTED)
h = [plt.Line2D([0], [0], ls="none", marker="o", ms=3, mfc="none", mec=INK, mew=0.5, label="Real neuronal, val (n=396)"),
     plt.Line2D([0], [0], ls="none", marker="o", ms=3, color=BG, label="All real val cells")]
fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.01))
fig.savefig(OUT / "R1_umap_overlay.png", dpi=300); plt.close(fig)

# ---- Figure 2: neighbour fraction, per space
num = json.loads((OUT / "r1_numbers.json").read_text(encoding="utf-8"))
rows = num["rows"]
SP = [("pca50", "PCA, 50 PCs"), ("full", "All 2,000 genes"), ("pca2", "Atlas PCA, 2 PCs")]
fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.9), sharey=True)
for a, (s, nm) in zip(ax, SP):
    rv = [r for r in rows if r["space"] == s and r["arm"] == "real_val"][0]["mean_neuronal_frac"]
    rt = [r for r in rows if r["space"] == s and r["arm"] == "real_train"][0]["mean_neuronal_frac"]
    a.axvline(rv, color=INK, lw=0.8)
    a.axvline(rt, color=INK, lw=0.8, ls=(0, (3, 2)))
    for i, arm in enumerate(ARMS):
        v = np.array([r["mean_neuronal_frac"] for r in rows if r["space"] == s and r["arm"] == arm])
        a.plot([v.min(), v.max()], [i, i], color=COL[arm], lw=2, solid_capstyle="round")
        a.plot(v.mean(), i, "o", ms=5, color=COL[arm], mec="white", mew=0.6)
    a.set_yticks(range(5)); a.set_yticklabels(ARMS); a.invert_yaxis()
    a.set_xlim(0, 1); a.set_xticks([0, 0.25, 0.5, 0.75, 1]); a.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    a.set_xlabel("Mean neuronal share of 15 nearest real cells", fontsize=6.5, color=MUTED)
    a.set_title(nm, fontsize=8, pad=3)
    a.grid(axis="x", lw=0.3, color=BG); a.set_axisbelow(True)
    a.spines[["top", "right"]].set_visible(False)
h = [plt.Line2D([0], [0], color=INK, lw=0.8, label="Real val neuronal"),
     plt.Line2D([0], [0], color=INK, lw=0.8, ls=(0, (3, 2)), label="Real train neuronal")]
fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.03))
fig.subplots_adjust(left=0.075, right=0.99, top=0.8, bottom=0.2, wspace=0.08)
fig.savefig(OUT / "R1_neighbour_fraction.png", dpi=300); plt.close(fig)
