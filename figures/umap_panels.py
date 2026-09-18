# -*- coding: utf-8 -*-
"""Real against delivered cells on the frozen UMAP, four panels.

The layout follows the archived embedding figure: one row of four, shared axes, a legend on top
and the panel letters below; the real panel carries every type. The coordinate system is the
frozen reducer fitted once on the validation set, and it is only ever used to TRANSFORM, never
to refit.

One thing has to be said plainly: these are NOT the coordinates of the archived embedding
figure. That one was fitted separately, so the two are not comparable point by point.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures import panels as H                      # noqa: E402

COL_BG_REAL = "#dcdbd5"
COL_BG_GEN = "#e7e6e1"
TYPES_REAL = H.TYPES3 + ["Mesothelial"]      # the real panel carries every type


def main() -> None:
    st = H.load_style()
    st.apply_rc()
    import matplotlib.pyplot as plt

    _xval, yval = H.real_cells()
    um = H.FrozenUMAP()
    gate = um.gate_against_archive()
    gate["NOT_the_archived_figure7_embedding"][
        "max_abs_delta_between_the_two_real_embeddings"] = H.umap_vs_figure7_delta()
    ur = um.real_xy

    rng = np.random.default_rng(H.SUBSAMPLE_SEED)
    gen, arrays = {}, {}
    for t in H.TYPES3:                    # the same generator consumed in the archived order
        p = H.DELIVERED[t]
        x, idx = H.take_panel(p, rng)
        t0 = time.time()
        gen[t] = um.project(x)
        print(f"[umap] projected {t}: n={len(idx)}  ({time.time() - t0:.0f}s)", flush=True)
        arrays[t] = H.array_record(p, len(idx), idx, H.SUBSAMPLE_SEED)

    np.savez_compressed(H.FIG / "fig_umap_panels.coords.npz",
                        real=ur, **{f"{t}_umap": gen[t] for t in H.TYPES3})

    fig, axes = plt.subplots(1, 4, figsize=(st.WIDE_W, 2.46))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.79, bottom=0.185, wspace=0.08)

    ax = axes[0]
    ax.scatter(ur[:, 0], ur[:, 1], s=0.6, c=COL_BG_REAL, linewidths=0, zorder=1,
               rasterized=True)
    for t in TYPES_REAL:
        m = yval == t
        ax.scatter(ur[m, 0], ur[m, 1], s=0.8, c=st.TYPE_COLORS[t], linewidths=0,
                   zorder=2, label=t, rasterized=True)
    ax.set_title(f"Real cells  (n={ur.shape[0]:,})", fontsize=st.FS_MIN, pad=2)
    ax.set_ylabel("UMAP 2")

    for ax, t in zip(axes[1:], H.TYPES3):
        ax.scatter(ur[:, 0], ur[:, 1], s=0.6, c=COL_BG_GEN, linewidths=0, zorder=1,
                   rasterized=True)
        g = gen[t]
        ax.scatter(g[:, 0], g[:, 1], s=1.2, c=st.TYPE_COLORS[t], linewidths=0,
                   zorder=3, rasterized=True)
        ax.set_title(f"generated: {t}  (n={len(g):,})", fontsize=st.FS_MIN, pad=2)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        for s in ax.spines.values():
            s.set_linewidth(0.5)
    h = [plt.Line2D([0], [0], ls="none", marker="o", ms=3, color=st.TYPE_COLORS[t])
         for t in TYPES_REAL]
    fig.legend(h, TYPES_REAL, loc="lower center", bbox_to_anchor=(0.53, 0.855),
               ncol=len(TYPES_REAL), handletextpad=0.25, columnspacing=1.2, borderpad=0.0,
               fontsize=st.FS_MIN - 1)
    for n, ax in enumerate(axes):
        bb = ax.get_position()
        fig.text((bb.x0 + bb.x1) / 2, 0.018, f"({'abcd'[n]})", ha="center", va="bottom",
                 fontsize=st.FS_LABEL, color=st.INK)

    counts = {t: int((yval == t).sum()) for t in TYPES_REAL}
    sidecar = {
        "figure": "F3 fig_umap_panels", "ticket": 180,
        "claim": "real vs delivered in the frozen UMAP, layout of the archived Figure 7",
        "zero_sampling": True, "zero_fitting": True,
        "frozen_basis": gate,
        "real_background": {
            "coords_file": str(H.UMAP_REAL_XY), "coords_sha256": H.sha256(H.UMAP_REAL_XY),
            "labels_file": str(H.DATA / "val_celltype.npy"),
            "labels_sha256": H.sha256(H.DATA / "val_celltype.npy"),
            "labels_note": "true held-out cell types (val_celltype.npy), the same labels the "
                           "archived Figure 7 coloured by -- NOT umap_expX's "
                           "classifier-predicted labels",
            "n_plotted": int(ur.shape[0]), "subsampled": False},
        "delivered_arrays": arrays,
        "projection": "cpm_log1p -> (x - pca50.mean) @ pca50.components.T -> "
                      "reducer.transform  (umap_expX.py::embed, verbatim)",
        "coords_cache": str(H.FIG / "fig_umap_panels.coords.npz"),
        "subsample": {"seed": H.SUBSAMPLE_SEED, "n_per_panel": H.N_PANEL,
                      "rule": "sorted(rng.choice(N, min(2500, N), replace=False)); ONE "
                              "generator consumed in the order " + ", ".join(H.TYPES3)},
        "palette": {t: st.TYPE_COLORS[t] for t in TYPES_REAL},
        "palette_source": str(H.VIZ / "style.py"),
        "palette_drift_gate": "style.assert_palette passed",
        "numbers_on_the_figure": {
            "real_panel_n": int(ur.shape[0]),
            **{f"generated_{t}_n": int(len(gen[t])) for t in H.TYPES3}},
        "numbers_source": "point counts of the plotted arrays themselves; no metric is "
                          "printed on this figure",
        "real_cells_per_type_drawn": counts,
        "mesothelial_note": "Mesothelial (n=%d) appears among the real types but does not "
                            "enter the purity columns; the caption says so, as before"
                            % counts["Mesothelial"],
        "not_drawn": {"Ventricular_Cardiomyocyte": "the ticket s1 F3 asks for the three "
                      "target types plus Mesothelial. The archived Figure 7 also drew "
                      "Ventricular_Cardiomyocyte; its registry colour is still asserted "
                      "here but no point is drawn in it."},
    }
    H.save(fig, "fig_umap_panels", sidecar)
    H.copy_pdf_to_paper("fig_umap_panels", "hvg_umap_panels.pdf")


if __name__ == "__main__":
    main()
