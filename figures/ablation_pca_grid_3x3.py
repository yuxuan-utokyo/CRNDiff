# -*- coding: utf-8 -*-
"""The ablation PCA grid in the palette of the baseline grid, with no text inside the panels.

The earlier ablation script is not edited and its figure is left in place. This file imports its
data-selection logic unchanged and changes only two things about the rendering: the delivered
points take the same orange as the baseline grid, so the two figures share one palette, and the
purity annotations are not drawn inside the panels. The numbers are still read from the
authoritative summary json and written to the sidecar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures import panels as H                      # noqa: E402
from figures.ablation_pca_base import (ARMS, SEEDS, TYPES3, arm_paths,   # noqa: E402
                                      purity_pooled, take_panel_pooled)
from figures.pca_grid_base import (COL_BG, COL_GEN,               # noqa: E402
                                       COL_TARGET)

# column titles follow the ablation table's arm order
TITLES = {"tilt_only": "Product tilt", "fk_only": "FK (untilted)", "tilt_fk": "Ours"}


def main() -> None:
    st = H.load_style()
    st.apply_rc()
    import matplotlib.pyplot as plt

    xval, yval = H.real_cells()
    pca = H.FrozenPCA()
    gate = pca.gate_against_archive(xval)
    if not gate["passed"]:
        raise SystemExit(f"frozen PCA gate failed: {gate}")
    zr = pca.project(xval)

    rng_real = np.random.default_rng(H.SUBSAMPLE_SEED)
    real_idx, real_rec = {}, {}
    for t in TYPES3:
        pos = np.flatnonzero(yval == t)
        k = min(pos.size, H.N_PANEL)
        sel = np.sort(rng_real.choice(pos.size, size=k, replace=False))
        real_idx[t] = pos[sel]
        real_rec[t] = {"n_total": int(pos.size), "n_plotted": int(k),
                       "subsample_seed": int(H.SUBSAMPLE_SEED),
                       "subsampled": bool(k < pos.size)}

    rng = np.random.default_rng(H.SUBSAMPLE_SEED)
    gen, arrays, numbers = {}, {}, {}
    for t in TYPES3:
        for key, _title, summary_name, arm_tpl in ARMS:
            arm = arm_tpl.format(t=t)
            paths = arm_paths(t, key)
            numbers[f"{t}/{key}"] = purity_pooled(summary_name, arm, paths)
            x, per_file, total = take_panel_pooled(paths, rng, H.N_PANEL)
            gen[(t, key)] = pca.project(x)
            arrays[f"{t}/{key}"] = {"pooled_over_seeds": list(SEEDS),
                                    "n_source_pooled": total,
                                    "n_plotted": int(x.shape[0]),
                                    "subsample_seed": int(H.SUBSAMPLE_SEED),
                                    "files": per_file}

    fig, axes = plt.subplots(3, 3, figsize=(5.5, 4.62), sharex=True, sharey=True)
    fig.subplots_adjust(left=.062, right=.995, top=.945, bottom=.075,
                        wspace=.05, hspace=.05)
    for i, t in enumerate(TYPES3):
        for j, (key, _t204, _s, _a) in enumerate(ARMS):
            ax = axes[i][j]
            ax.scatter(zr[:, 0], zr[:, 1], s=.35, c=COL_BG, linewidths=0, zorder=1,
                       rasterized=True)
            ri = real_idx[t]
            ax.scatter(zr[ri, 0], zr[ri, 1], s=.8, c=COL_TARGET, linewidths=0, zorder=2,
                       rasterized=True)
            g = gen[(t, key)]
            ax.scatter(g[:, 0], g[:, 1], s=.7, c=COL_GEN, alpha=.50, linewidths=0,
                       zorder=3, rasterized=True)
            if i == 0:
                ax.set_title(TITLES[key], fontsize=6.8, pad=2,
                             fontweight="bold" if key == "tilt_fk" else "normal")
            if j == 0:
                ax.set_ylabel(t, fontsize=6.8, labelpad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_linewidth(.5)

    h = [plt.Line2D([0], [0], ls="none", marker="o", ms=2.6, color=COL_BG),
         plt.Line2D([0], [0], ls="none", marker="o", ms=2.6, color=COL_TARGET),
         plt.Line2D([0], [0], ls="none", marker="o", ms=2.6, color=COL_GEN)]
    fig.legend(h, [f"All real cells (n={zr.shape[0]:,})",
                   "Real cells of the requested type (row)", "Delivered cells"],
               loc="lower center", bbox_to_anchor=(.5, -.010), ncol=3, frameon=False,
               fontsize=6.4, handletextpad=.3, columnspacing=1.4, borderpad=0.)

    H.save(fig, "fig_ablation_pca_3x3", {
        "figure": "ablation PCA 3x3 (rows = requests, cols = arms)", "ticket": 210,
        "rows": TYPES3, "cols": [TITLES[k] for k, _, _, _ in ARMS],
        "col_keys": [k for k, _, _, _ in ARMS],
        "relation_to_204": {
            "source_script": "figures/fig_ablation_pca.py (UNCHANGED, sha256 "
                             + H.sha256(ROOT / "figures" / "ablation_pca_base.py") + ")",
            "imported": ["ARMS", "SEEDS", "TYPES3", "arm_paths", "purity_pooled",
                         "take_panel_pooled"],
            "changed_here": ["delivered-cell colour: #c3a6e0 (204) -> " + COL_GEN
                             + " (the PCA grid's colour, per ticket 210 block 3)",
                             "the per-panel purity inset text is NOT drawn"],
            "cell_selection_identical": "same SUBSAMPLE_SEED, same row-major consumption, "
                                        "same three-seed pooling -- the points plotted are "
                                        "the same ones 204 plots"},
        "palette": {"all_real_cells": COL_BG, "requested_type": COL_TARGET,
                    "delivered": COL_GEN,
                    "source": "figures/fig_pca_baselines.py (same three layers)"},
        "n_panel": H.N_PANEL, "subsample_seed": int(H.SUBSAMPLE_SEED),
        "zero_training": True, "zero_sampling": True, "zero_rescoring": True,
        "pca_gate": gate,
        "real_cells": real_rec, "arrays": arrays,
        "purity_numbers_not_drawn": {
            "why": "ticket 210 block 3 asks for no inset text",
            "what": "relative purity = purity_matched_n / canonical_ceiling, three seeds "
                    "pooled; kept here so the figure stays auditable",
            "per_panel": numbers},
    })
    H.copy_pdf_to_paper("fig_ablation_pca_3x3", "hvg_ablation_pca_3x3.pdf")


if __name__ == "__main__":
    main()
