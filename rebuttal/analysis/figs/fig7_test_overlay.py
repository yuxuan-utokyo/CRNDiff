#!/usr/bin/env python
r"""Fig. 7 overlay on TEST. Coordinates from Windows (from_windows/fig7_test/fig7_test_overlay.npz,
script 65b430d4): the frozen heart reducer behind Fig. 7d (umap_reducer.pkl 00c3f9bd, transform only; gate reducer.embedding_ ==
umap_xy_val_real.npy, max delta 0) applied to every TEST cell and to the Fig. 7d run (gen 20260625 / samp 20260987; gate vs the paper panel,
max delta 0). SAME EMBEDDING AS FIG. 7d; the reducer of the other Fig. 7 panels was never saved, so do not compare with them.
Drawing copied from scripts/figs/fig7_rebuttal.py (style.py rc, BG colours, point sizes, TYPE_COLORS, fonts, legend on top, UMAP axes);
2 panels: (a) test cells with Neuronal coloured, (b) the generated Neuronal cells over the grey test cells. The caption numbers are the
R19 test neighbour fractions of this same run (runs/r19/test/r1r2_crndiff_Neuronal_test.json).
    python fig7_test_overlay.py OUT_DIR
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
W = Path("./work"); S = W / "scripts/figs/fig7_src"; sys.path.insert(0, str(S))
import style as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
D = W / "from_windows/fig7_test"; z = np.load(D / "fig7_test_overlay.npz"); J = json.load(open(D / "fig7_test_overlay.json"))
assert J["reducer_gate"]["passed"] and J["gate_generated_vs_paper_panel_max_abs_delta"] == 0.0
xy, y, g = z["xy_test"], z["y_test"].astype(str), z["xy_gen"]; assert len(xy) == J["n_test"] and int((y == "Neuronal").sum()) == J["n_test_neuronal"]
R1 = json.load(open(W / "runs/r19/test/r1r2_crndiff_Neuronal_test.json"))
r = [x for x in R1["runs"] if int(x["train"]) == 20260625 and int(x["seed"]) == 20260987]; assert len(r) == 1
fr = {sp: r[0]["r1"][sp]["mean_neuronal_frac"] for sp in ("pca2", "pca50", "full")}
st.apply_rc(); BG_REAL, BG_GEN = "#dcdbd5", "#e7e6e1"; FS_T, FS_LEG = 9, 9; col = st.TYPE_COLORS["Neuronal"]
fig, axes = plt.subplots(1, 2, figsize=(500.386 / 2 / 72 * 1.05, 158 / 72))
fig.subplots_adjust(left=0.07, right=0.995, top=0.74, bottom=0.10, wspace=0.07)
ax = axes[0]; ax.scatter(xy[:, 0], xy[:, 1], s=0.6, c=BG_REAL, linewidths=0, zorder=1, rasterized=True)
m = y == "Neuronal"; ax.scatter(xy[m, 0], xy[m, 1], s=0.8, c=col, linewidths=0, zorder=2, rasterized=True)
ax.set_title("(a) Test cells", fontsize=FS_T, pad=3); ax.set_ylabel("UMAP 2", fontsize=st.FS_LABEL)
ax = axes[1]; ax.scatter(xy[:, 0], xy[:, 1], s=0.6, c=BG_GEN, linewidths=0, zorder=1, rasterized=True)
ax.scatter(g[:, 0], g[:, 1], s=1.2, c=col, linewidths=0, zorder=3, rasterized=True); ax.set_title("(b) Neuronal", fontsize=FS_T, pad=3)
xl = (xy[:, 0].min(), xy[:, 0].max()); yl = (xy[:, 1].min(), xy[:, 1].max()); px, py = 0.04 * (xl[1] - xl[0]), 0.04 * (yl[1] - yl[0])
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1", fontsize=st.FS_LABEL, labelpad=2)
    ax.set_xlim(xl[0] - px, xl[1] + px); ax.set_ylim(yl[0] - py, yl[1] + py)
    for s_ in ax.spines.values(): s_.set_linewidth(0.6)
fig.legend([plt.Line2D([0], [0], ls="none", marker="o", ms=6, color=col)], ["Neuronal"], loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=1, borderpad=0.2, fontsize=FS_LEG)
for e in ("pdf", "png"): fig.savefig(out / f"hvg_umap_fig7d_test.{e}", dpi=600 if e == "png" else 300)
plt.close(fig)
caption = ("Real and generated Neuronal cells on the held-out test split, in the same UMAP embedding as Fig. 7d (frozen reducer, transform only). "
           f"(a) all {len(y):,} test cells, Neuronal ({int(m.sum())}) coloured; (b) the 2,500 CRNDiff cells of the Fig. 7d run. "
           f"Share of Neuronal cells among each generated cell's 15 nearest test neighbours: {fr['pca2']:.3f} (2-D PCA), {fr['pca50']:.3f} (50-D PCA), {fr['full']:.3f} (all 2,000 genes).")
rec = {"caption": caption, "inputs": {"npz": sha(D / "fig7_test_overlay.npz"), "json": sha(D / "fig7_test_overlay.json"), "windows_script": J["script_sha256"],
       "reducer_sha256": J["reducer_gate"]["reducer_sha256"], "r1_test_json": sha(W / "runs/r19/test/r1r2_crndiff_Neuronal_test.json")},
       "neighbour_fraction_this_run": fr, "style_py": sha(S / "style.py"), "script_sha256": sha(__file__),
       "outputs": {p.name: sha(p) for p in sorted(out.glob("hvg_umap_fig7d_test.*")) if p.suffix in (".pdf", ".png")}}
json.dump(rec, open(out / "hvg_umap_fig7d_test.json", "w"), indent=1); print(json.dumps(rec, indent=1))
