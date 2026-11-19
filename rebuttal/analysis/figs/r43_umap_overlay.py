#!/usr/bin/env python
r"""R43: redraw Fig. 7 (hvg_umap_panels) as a real + generated overlay, one row of three
panels (Endothelial / Myeloid / Neuronal), no "(a) Test cells" panel, no insets. Cells and coordinates are exactly those of the existing
Fig. 7 (scripts/figs/fig7_rebuttal.py): fig7_src/fig_umap_panels.coords.npz = the frozen PCA50 -> UMAP reducer (umap_reducer.pkl 00c3f9bd,
transform only, never refit) applied to the 45,145 HVG2K **val** cells (labels val_celltype.npy) and to the 2,500 delivered CRNDiff cells
per target (gen 20260625 / samp 20260987; Endothelial subsampled 10,057 -> 2,500 there, seed 20261031). Nothing is projected here.
Per panel, bottom to top: (1) all val cells, light grey; (2) the target's real val cells, light tint of the target colour with a white
edge; (3) the generated cells, dark shade of the target colour, solid. Endothelial has 10,057 real cells against 2,500 generated, so its
real layer is subsampled to 2,500 (seed SUB_SEED); Myeloid (2,302) and Neuronal (396) keep all real cells and all 2,500 generated cells.
Width 5.5 in (text width), titles 8.5 pt regular, axis labels 8 pt.
    python r43_umap_overlay.py OUT_DIR
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
S = Path(__file__).resolve().parent / "fig7_src"; sys.path.insert(0, str(S))
import style as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
st.apply_rc(); plt.rcParams["pdf.fonttype"] = 42
z = np.load(S / "fig_umap_panels.coords.npz"); y = np.load(S / "val_celltype.npy", allow_pickle=True).astype(str)
FJ = json.load(open(S / "fig_umap_panels.json")); ur = z["real"]; assert len(ur) == len(y) == 45145
T3 = ["Endothelial", "Myeloid", "Neuronal"]; SUB_SEED = 20261111; NMAX = 2500
mix = lambda c, w, t: tuple((1 - t) * np.array(to_rgb(c)) + t * np.array(to_rgb(w)))
TINT = {t: mix(st.TYPE_COLORS[t], "white", 0.50) for t in T3}; DARK = {t: mix(st.TYPE_COLORS[t], "black", 0.45) for t in T3}
BG = "#e3e2dc"
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05))
fig.subplots_adjust(left=0.045, right=0.995, top=0.80, bottom=0.085, wspace=0.05)
rng = np.random.default_rng(SUB_SEED); rec_n = {}
for ax, t, L in zip(axes, T3, "abc"):
    ax.scatter(ur[:, 0], ur[:, 1], s=0.5, c=BG, linewidths=0, zorder=1, rasterized=True)
    idx = np.flatnonzero(y == t); g = z[f"{t}_umap"]; n_real_all = len(idx)
    if len(idx) > NMAX: idx = np.sort(rng.choice(idx, NMAX, replace=False))
    r = ur[idx]
    ax.scatter(r[:, 0], r[:, 1], s=9, facecolors=[TINT[t]], edgecolors="white", linewidths=0.3, alpha=0.95, zorder=2, rasterized=True)
    ax.scatter(g[:, 0], g[:, 1], s=0.6, c=[DARK[t]], linewidths=0, alpha=0.75, zorder=3, rasterized=True)
    ax.set_title(f"({L}) {t}", fontsize=8.5, fontweight="normal", loc="center", pad=3)
    rec_n[t] = {"val_cells_background": int(len(ur)), "real_target_val_cells_total": n_real_all, "real_target_val_cells_plotted": int(len(idx)),
                "real_subsampled": bool(n_real_all > NMAX), "generated_plotted": int(len(g)),
                "generated_run": {k: FJ["delivered_arrays"][t][k] for k in ("path", "sha256", "n_source", "n_plotted", "subsampled")}}
xl = (ur[:, 0].min(), ur[:, 0].max()); yl = (ur[:, 1].min(), ur[:, 1].max()); px, py = 0.04 * (xl[1] - xl[0]), 0.04 * (yl[1] - yl[0])
for i, ax in enumerate(axes):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1", fontsize=8, labelpad=2)
    if i == 0: ax.set_ylabel("UMAP 2", fontsize=8, labelpad=2)
    ax.set_xlim(xl[0] - px, xl[1] + px); ax.set_ylim(yl[0] - py, yl[1] + py)
    for s_ in ax.spines.values(): s_.set_linewidth(0.6)
hr = tuple(plt.Line2D([0], [0], ls="none", marker="o", ms=5, mfc=TINT[t], mec="white", mew=0.4) for t in T3)
hg = tuple(plt.Line2D([0], [0], ls="none", marker="o", ms=3.2, mfc=DARK[t], mec=DARK[t]) for t in T3)
fig.legend([hr, hg], ["Real target cells", "Generated (CRNDiff)"], handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
           loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=2, fontsize=8, frameon=False, handlelength=3.0, columnspacing=2.0, borderpad=0.2)
for e in ("pdf", "png"): fig.savefig(out / f"hvg_umap_overlay_val.{e}", dpi=300)
plt.close(fig)
rec = {"what": "Fig. 7 redrawn as real + generated overlay (R43); same cells and coordinates as the existing Fig. 7",
       "split": "val (HVG2K val split, 45,145 cells): the background and the real target cells are VAL cells, not test; the existing Fig. 7 panel titled 'Test cells' also drew these val cells",
       "embedding": {"reducer_sha256": FJ["frozen_basis"]["reducer_sha256"], "refitted": False, "projected_here": False,
                     "coords_cache": "fig7_src/fig_umap_panels.coords.npz", "coords_cache_sha256": sha(S / "fig_umap_panels.coords.npz"),
                     "labels": "fig7_src/val_celltype.npy", "labels_sha256": sha(S / "val_celltype.npy")},
       "generated": "CRNDiff gen 20260625 x samp 20260987 (the runs of the existing Fig. 7 (b)-(d))",
       "real_subsample": {"rule": f"real target cells subsampled to {NMAX} only when more than {NMAX} (Endothelial); generated layer unchanged", "seed": SUB_SEED},
       "per_panel": rec_n, "colours": {t: {"base": st.TYPE_COLORS[t], "real_tint": "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in TINT[t]),
                                           "generated_dark": "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in DARK[t])} for t in T3},
       "size_in": [5.5, 2.05], "fonts_pt": {"title": 8.5, "axis_label": 8, "legend": 8},
       "style_py": sha(S / "style.py"), "script_sha256": sha(__file__),
       "outputs": {p.name: sha(p) for p in sorted(out.glob("hvg_umap_overlay_val.*")) if p.suffix in (".pdf", ".png")}}
json.dump(rec, open(out / "hvg_umap_overlay_val.json", "w"), indent=1); print(json.dumps({k: rec[k] for k in ("per_panel", "colours", "outputs")}, indent=1))
