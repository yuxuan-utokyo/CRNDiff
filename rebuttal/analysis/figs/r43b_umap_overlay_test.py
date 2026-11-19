#!/usr/bin/env python
r"""R43b: the R43 overlay (scripts/figs/r43_umap_overlay.py) on TEST cells, equal n per panel.
Test coordinates: the cached transform of all 45,163 HVG2K test cells through the frozen reducer (umap_reducer.pkl 00c3f9bd, transform
only) from from_windows/fig7_test/fig7_test_overlay.npz (Windows script 65b430d4; the cache behind hvg_umap_fig7d_test; SHA256SUMS
checked). No test cells are read here, only the cached coordinates and labels. Generated cells: the three runs of the existing Fig. 7 (b)-(d)
(gen 20260625 x samp 20260987), coordinates from fig7_src/fig_umap_panels.coords.npz (same reducer).
Gates: (1) the val-real reference in the test npz equals the Fig. 7 val coordinates (same embedding); (2) the Neuronal generated
coordinates in the test npz equal the Fig. 7 cache.
Equal n per panel: n = min(n_real_test, n_generated); real and generated are each subsampled to n without replacement, one generator
(seed 20261111) consumed per panel in the order Endothelial, Myeloid, Neuronal, real before generated.
Layout identical to R43: one row of three panels, grey test background, real target cells as a light tint with a white edge, generated
cells as a dark shade on top, 5.5 in wide, titles 8.5 pt, axis labels 8 pt, two legend entries.
    python r43b_umap_overlay_test.py OUT_DIR
"""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
W = Path("./work"); S = W / "scripts/figs/fig7_src"; D = W / "from_windows/fig7_test"; sys.path.insert(0, str(S))
import style as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
subprocess.run(["shasum", "-a", "256", "-c", "--quiet", "SHA256SUMS"], cwd=D, check=True)
st.apply_rc(); plt.rcParams["pdf.fonttype"] = 42
zt = np.load(D / "fig7_test_overlay.npz"); JT = json.load(open(D / "fig7_test_overlay.json"))
zc = np.load(S / "fig_umap_panels.coords.npz"); FJ = json.load(open(S / "fig_umap_panels.json"))
assert JT["reducer_gate"]["reducer_sha256"] == FJ["frozen_basis"]["reducer_sha256"]
g1 = float(np.abs(zt["xy_val_real_reference"] - zc["real"]).max()); g2 = float(np.abs(zt["xy_gen"] - zc["Neuronal_umap"]).max()); assert g1 == 0.0 and g2 == 0.0, (g1, g2)
xy, y = zt["xy_test"], zt["y_test"].astype(str); assert len(xy) == JT["n_test"] == 45163
T3 = ["Endothelial", "Myeloid", "Neuronal"]; SUB_SEED = 20261111
mix = lambda c, w, t: tuple((1 - t) * np.array(to_rgb(c)) + t * np.array(to_rgb(w)))
TINT = {t: mix(st.TYPE_COLORS[t], "white", 0.50) for t in T3}; DARK = {t: mix(st.TYPE_COLORS[t], "black", 0.45) for t in T3}
BG = "#e3e2dc"
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05))
fig.subplots_adjust(left=0.045, right=0.995, top=0.80, bottom=0.085, wspace=0.05)
rng = np.random.default_rng(SUB_SEED); rec_n = {}
for ax, t, L in zip(axes, T3, "abc"):
    ax.scatter(xy[:, 0], xy[:, 1], s=0.5, c=BG, linewidths=0, zorder=1, rasterized=True)
    ir = np.flatnonzero(y == t); G = zc[f"{t}_umap"]; n = min(len(ir), len(G))
    ir_s = np.sort(rng.choice(ir, n, replace=False)) if n < len(ir) else ir
    ig_s = np.sort(rng.choice(len(G), n, replace=False)) if n < len(G) else np.arange(len(G))
    r, g = xy[ir_s], G[ig_s]
    ax.scatter(r[:, 0], r[:, 1], s=9, facecolors=[TINT[t]], edgecolors="white", linewidths=0.3, alpha=0.95, zorder=2, rasterized=True)
    ax.scatter(g[:, 0], g[:, 1], s=0.6, c=[DARK[t]], linewidths=0, alpha=0.75, zorder=3, rasterized=True)
    ax.set_title(f"({L}) {t}", fontsize=8.5, fontweight="normal", loc="center", pad=3)
    rec_n[t] = {"test_cells_background": int(len(xy)), "real_target_test_cells_total": int(len(ir)), "generated_cells_total": int(len(G)),
                "n_each_plotted": int(n), "real_subsampled": bool(n < len(ir)), "generated_subsampled": bool(n < len(G)),
                "generated_run": {k: FJ["delivered_arrays"][t][k] for k in ("path", "sha256", "n_source", "n_plotted", "subsampled")}}
xl = (xy[:, 0].min(), xy[:, 0].max()); yl = (xy[:, 1].min(), xy[:, 1].max()); px, py = 0.04 * (xl[1] - xl[0]), 0.04 * (yl[1] - yl[0])
for i, ax in enumerate(axes):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1", fontsize=8, labelpad=2)
    if i == 0: ax.set_ylabel("UMAP 2", fontsize=8, labelpad=2)
    ax.set_xlim(xl[0] - px, xl[1] + px); ax.set_ylim(yl[0] - py, yl[1] + py)
    for s_ in ax.spines.values(): s_.set_linewidth(0.6)
hr = tuple(plt.Line2D([0], [0], ls="none", marker="o", ms=5, mfc=TINT[t], mec="white", mew=0.4) for t in T3)
hg = tuple(plt.Line2D([0], [0], ls="none", marker="o", ms=3.2, mfc=DARK[t], mec=DARK[t]) for t in T3)
fig.legend([hr, hg], ["Real target cells", "Generated (CRNDiff)"], handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
           loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=2, fontsize=8, frameon=False, handlelength=3.0, columnspacing=2.0, borderpad=0.2)
for e in ("pdf", "png"): fig.savefig(out / f"hvg_umap_overlay_test.{e}", dpi=300)
plt.close(fig)
rec = {"what": "Fig. 7 real + generated overlay on TEST cells, equal n per panel (R43b)",
       "split": "test (HVG2K test split, 45,163 cells): background and real target cells are TEST cells, from the cached frozen-reducer transform; no new test read",
       "embedding": {"reducer_sha256": FJ["frozen_basis"]["reducer_sha256"], "refitted": False, "projected_here": False,
                     "test_coords_cache": "from_windows/fig7_test/fig7_test_overlay.npz", "test_coords_cache_sha256": sha(D / "fig7_test_overlay.npz"),
                     "windows_script_sha256": JT["script_sha256"], "generated_coords_cache": "fig7_src/fig_umap_panels.coords.npz",
                     "generated_coords_cache_sha256": sha(S / "fig_umap_panels.coords.npz"),
                     "gates": {"val_reference_in_test_npz_vs_fig7_val_coords_max_abs": g1, "neuronal_generated_test_npz_vs_fig7_cache_max_abs": g2}},
       "generated": "CRNDiff gen 20260625 x samp 20260987 (the runs of the existing Fig. 7 (b)-(d))",
       "equal_n": {"rule": "n = min(n_real_test, n_generated) per panel; both subsampled to n without replacement; one generator, panels in order, real before generated", "seed": SUB_SEED},
       "per_panel": rec_n, "colours": {t: {"base": st.TYPE_COLORS[t], "real_tint": "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in TINT[t]),
                                           "generated_dark": "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in DARK[t])} for t in T3},
       "size_in": [5.5, 2.05], "fonts_pt": {"title": 8.5, "axis_label": 8, "legend": 8},
       "style_py": sha(S / "style.py"), "script_sha256": sha(__file__),
       "outputs": {p.name: sha(p) for p in sorted(out.glob("hvg_umap_overlay_test.*")) if p.suffix in (".pdf", ".png")}}
json.dump(rec, open(out / "hvg_umap_overlay_test.json", "w"), indent=1)
print(json.dumps({"gates": rec["embedding"]["gates"], "n": {t: (v["real_target_test_cells_total"], v["generated_cells_total"], v["n_each_plotted"]) for t, v in rec_n.items()}, "outputs": rec["outputs"]}, indent=1))
