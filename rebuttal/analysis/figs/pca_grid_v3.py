#!/usr/bin/env python
r"""PCA grids v3, heart + lung: drawing taken from the paper's Fig. 9 3x3 script
, stored coordinates only, nothing projected or refit.
Fig. 9 3x3 parameters (lines 81-88 there):   background s 0.35, COL_BG, no edge, zorder 1, rasterised
                                              real target  s 0.8,  COL_TARGET, no edge, zorder 2, rasterised
                                              generated    s 0.7,  COL_GEN, alpha 0.50, no edge, zorder 3, rasterised
The ONLY allowed change if green is hidden again: the generated alpha (GEN_ALPHA below). Everything else in each grid (layout, columns,
legend, fonts, dpi) is the v1 of that grid. Colours from fig_pca_baselines.py (COL_BG #dcdbd5, COL_TARGET #1baf7a, COL_GEN #D55E00).
    python pca_grid_v3.py OUT_DIR [GEN_ALPHA]
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
W = Path("./work"); HERE = W / "scripts/figs"; sys.path.insert(0, str(HERE / "fig7_src"))
import style as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); GEN_ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.50
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
F9 = W / "from_windows/fig9_src"; src = (F9 / "fig_ablation_pca_3x3.py").read_text(); pb = (F9 / "fig_pca_baselines.py").read_text()
for frag in ('s=.35, c=COL_BG, linewidths=0, zorder=1', 's=.8, c=COL_TARGET, linewidths=0, zorder=2', 's=.7, c=COL_GEN, alpha=.50, linewidths=0'):
    assert frag in src, frag                                   # the parameters below are the 3x3 script's, checked against its text
for frag in ('COL_BG = "#dcdbd5"', 'COL_TARGET = "#1baf7a"', 'COL_GEN = "#D55E00"'): assert frag in pb, frag
F9P = {"bg": dict(s=.35, c="#dcdbd5", linewidths=0, zorder=1, rasterized=True), "target": dict(s=.8, c="#1baf7a", linewidths=0, zorder=2, rasterized=True),
       "gen": dict(s=.7, c="#D55E00", alpha=GEN_ALPHA, linewidths=0, zorder=3, rasterized=True)}
st.apply_rc()
def grid(xy, yt, gens, rows, cols, title_of, bold, figsize, adjust, legend_ms, title_fs, dpi, stem):
    fig, axes = plt.subplots(len(rows), len(cols), figsize=figsize, sharex=True, sharey=True); fig.subplots_adjust(**adjust)
    for i, ct in enumerate(rows):
        tgt = yt == ct
        for j, m in enumerate(cols):
            ax = axes[i, j]; g = gens[(ct, m)]
            ax.scatter(xy[:, 0], xy[:, 1], **F9P["bg"]); ax.scatter(xy[tgt, 0], xy[tgt, 1], **F9P["target"]); ax.scatter(g[:, 0], g[:, 1], **F9P["gen"])
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values(): s_.set_linewidth(.5)
            if i == 0: ax.set_title(title_of(m), fontsize=title_fs, pad=2, fontweight="bold" if m == bold else "normal")
            if j == 0: ax.set_ylabel(ct, fontsize=title_fs, labelpad=2)
    h = [plt.Line2D([0], [0], ls="none", marker="o", ms=legend_ms, color="#1baf7a"), plt.Line2D([0], [0], ls="none", marker="o", ms=legend_ms, color="#D55E00")]
    fig.legend(h, ["Real target cells", "Generated cells"], loc="upper center", bbox_to_anchor=(.52, 1.0), ncol=2, frameon=False, fontsize=title_fs, handletextpad=.3, columnspacing=2.0, borderpad=0.)
    for e in ("pdf", "png"): fig.savefig(out / f"{stem}.{e}", dpi=dpi, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
# heart: stored coordinates of the rebuttal grid (pca_grid_rebuttal.py v1 layout)
S = HERE / "pca_src"; z = np.load(S / "pca_grid_coords_k128.npz"); J = json.load(open(S / "pca_grid_coords_k128.json")); sty = J["style_from_r12_pca_grid"]
COLS = ["scVI", "CFGen", "scANVI", "MDLM", "CountsDiff", "Ours"]
grid(z["real_xy"], z["real_type"].astype(str), {(ct, m): z[f"gen__{ct}__{m}"] for ct in J["rows"] for m in COLS}, J["rows"], COLS, lambda m: "CRNDiff" if m == "Ours" else m, "Ours",
     (sty["figsize_6col"][0], sty["figsize_6col"][1] + 0.18), dict(left=.048, right=.996, top=.885, bottom=.02, wspace=.05, hspace=.06), 4.2, sty["title_fs"], sty["dpi"], "hvg_pca_baselines_all_rebuttal_v3")
# lung: stored coordinates of the R28 test grid (saved by r28_hlca_pca_grid_v2.py; r28 v1 layout)
C = W / "runs/r28/hlca_pca_grid_coords_test.npz"; zc = np.load(C, allow_pickle=False); RL = ["plasma cell", "ionocyte"]; CL = ["scVI", "CFGen", "scANVI", "CountsDiff", "CRNDiff"]
grid(zc["real_xy"], zc["real_type"].astype(str), {(ct, m): zc[f"gen__{ct}__{m}"] for ct in RL for m in CL}, RL, CL, lambda m: m, "CRNDiff",
     (sty["figsize_6col"][0] * 5 / 6, sty["figsize_6col"][1] * 2 / 3 + 0.18), dict(left=.048, right=.996, top=.84, bottom=.02, wspace=.05, hspace=.06), 4.2, sty["title_fs"], sty["dpi"], "hlca_pca_grid_test_v3")
rec = {"fig9_source": {"script": "fig_ablation_pca_3x3.py", "sha256": sha(F9 / "fig_ablation_pca_3x3.py"), "colours_from": sha(F9 / "fig_pca_baselines.py")},
       "params": {k: {kk: vv for kk, vv in v.items()} for k, v in F9P.items()}, "gen_alpha_fig9": 0.50, "gen_alpha_used": GEN_ALPHA,
       "coords": {"heart": sha(S / "pca_grid_coords_k128.npz"), "lung_test": sha(C)}, "script_sha256": sha(__file__),
       "outputs": {p.name: sha(p) for p in sorted(out.glob("*_v3.*")) if p.suffix in (".pdf", ".png")}}
json.dump(rec, open(out / "pca_grid_v3.json", "w"), indent=1); print(json.dumps({k: rec[k] for k in ("gen_alpha_used", "outputs")}, indent=1))
