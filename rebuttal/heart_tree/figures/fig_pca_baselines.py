# -*- coding: utf-8 -*-
"""four-way comparison figure for the main text: on the rare request (Neuronal) real + three baselines + ours, frozen atlas PCA.

difference from fig_pca_panels.py: that figure only draws the deliveries of our three requests (for illustration); this one, in **the same
puts the four samplers side by side on the request**; this is the figure for direct comparison with baselines. Neuronal is chosen because of the three requests only it
can separate the methods (on Endothelial none of the four is bad); the full 3-request x 4-method grid goes to the appendix; the main text does not cherry-pick.

how to read: grey = all real cells (background); green = real Neuronal cells, i.e. **where the request should land**;
orange = cells actually delivered by this sampler. Orange on top of green = correct targeting.

data source
  three baselines: the source registered in out/summary/e21_table1_ours.json (absolute Windows path,
            (this script translates by mount point), the same arrays read by fig_dispersion.py;
  ours:    hvgfig.DELIVERED["Neuronal"](the run of the paper at the time), **not** the one in e21
            archived path: fig_dispersion.py makes the same replacement in the same place.
  PCA basis: the frozen frozen_atlas_pca_basis.npz, apply only, pass the archive gate, never refit.

No numbers in the figure: purity is in the main-text table, and repeating it only adds another number to keep in sync.

    python -m figures.fig_pca_baselines
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures import hvgfig as H                      # noqa: E402

TYPE = "Neuronal"
SEED = 20261109                                      # a subsampling stream dedicated to this figure, not sharing that of F9b
N_PANEL = 2500
BASELINES = ["scVI", "CFGen", "scANVI"]
REGISTRY = ROOT / "out" / "summary" / "e21_table1_ours.json"

COL_BG = "#dcdbd5"                                   # all real cells
COL_TARGET = "#1baf7a"                               # real Neuronal = target region (the Neuronal colour in style)
COL_GEN = "#D55E00"                                  # delivered (as in the toy figure: orange = sampler particles)


def win_to_mount(p: str) -> Path:
    """e21 stores absolute Windows paths; re-rooted at the mount point using the part after crndiff/."""
    s = str(p).replace("\\", "/")
    i = s.find("crndiff/")
    if i < 0:
        raise SystemExit(f"path not under the connected folder: {p}")
    return Path.home() / "mnt" / "crndiff" / s[i + len("crndiff/"):]


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
    tgt = yval == TYPE

    rows = json.loads(REGISTRY.read_text(encoding="utf-8"))["types"][TYPE]["rows"]
    srcs = {m: win_to_mount(rows[m]["source"]) for m in BASELINES}
    srcs["Ours"] = H.DELIVERED[TYPE]                 # current run, not the e21 archived path
    for m, p in srcs.items():
        if not p.is_file():
            raise SystemExit(f"missing delivery for {m}: {p}")

    rng = np.random.default_rng(SEED)
    gen, rec = {}, {}
    for m in BASELINES + ["Ours"]:
        x, idx = H.take_panel(srcs[m], rng, n=N_PANEL)
        gen[m] = pca.project(x)
        rec[m] = H.array_record(srcs[m], len(idx), idx, SEED)

    order = ["Real cells"] + BASELINES + ["Ours"]
    fig, axes = plt.subplots(1, 5, figsize=(5.5, 1.42), sharex=True, sharey=True)
    fig.subplots_adjust(left=.012, right=.998, top=.845, bottom=.075, wspace=.05)

    for ax, m in zip(axes, order):
        ax.scatter(zr[:, 0], zr[:, 1], s=.45, c=COL_BG, linewidths=0, zorder=1,
                   rasterized=True)
        if m != "Real cells":
            g = gen[m]
            ax.scatter(g[:, 0], g[:, 1], s=.9, c=COL_GEN, alpha=.75, linewidths=0,
                       zorder=2, rasterized=True)
        # target region drawn on top: otherwise the green points of the baseline panels are buried under 2500 orange points,
        # the reader can no longer judge whether the orange lands correctly
        ax.scatter(zr[tgt, 0], zr[tgt, 1], s=1.1, c=COL_TARGET, linewidths=0, zorder=4,
                   rasterized=True)
        ax.set_title(m, fontsize=6.6, pad=2,
                     fontweight="bold" if m == "Ours" else "normal")
        ax.set_xticks([]); ax.set_yticks([])
        if m == "Real cells":
            ax.set_xlabel("PC 1", fontsize=6.0, labelpad=1)
            ax.set_ylabel("PC 2", fontsize=6.0, labelpad=1)
        for s_ in ax.spines.values():
            s_.set_linewidth(.5)

    h = [plt.Line2D([0], [0], ls="none", marker="o", ms=2.6, color=COL_TARGET),
         plt.Line2D([0], [0], ls="none", marker="o", ms=2.6, color=COL_GEN)]
    fig.legend(h, [f"Real {TYPE.lower()} cells", "Delivered cells"],
               loc="lower center", bbox_to_anchor=(.5, -.055), ncol=2, frameon=False,
               fontsize=6.2, handletextpad=.3, columnspacing=1.6, borderpad=0.)

    H.save(fig, "fig_pca_baselines", {
        "figure": "PCA, four samplers on the rare request",
        "type": TYPE, "n_panel": N_PANEL, "subsample_seed": SEED,
        "zero_training": True, "zero_sampling": True,
        "pca_gate": gate, "arrays": rec,
        "note": "Ours uses hvgfig.DELIVERED (current run), not the e21 archived path; "
                "fig_dispersion.py makes the same swap.",
    })
    H.copy_pdf_to_paper("fig_pca_baselines", "hvg_pca_baselines.pdf")


def grid() -> None:
    """full grid for the appendix: three requests x four samplers, same frozen PCA basis.

    the main text has no PCA (only dispersion there); this figure lays out all three requests:
    Endothelial the four methods differ little; Myeloid starts to separate, Neuronal separates the most.
    give everything first, so nothing is dropped"pick only the request that separates"pretext.
    """
    st = H.load_style()
    st.apply_rc()
    import matplotlib.pyplot as plt

    xval, yval = H.real_cells()
    pca = H.FrozenPCA()
    gate = pca.gate_against_archive(xval)
    if not gate["passed"]:
        raise SystemExit(f"frozen PCA gate failed: {gate}")
    zr = pca.project(xval)
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))["types"]

    cols = BASELINES + ["Ours"]
    rng = np.random.default_rng(SEED)
    gen, rec = {}, {}
    for ct in H.TYPES3:                       # row order fixed; rng consumed in row-column order
        rows = reg[ct]["rows"]
        for m in cols:
            src = H.DELIVERED[ct] if m == "Ours" else win_to_mount(rows[m]["source"])
            if not src.is_file():
                raise SystemExit(f"missing delivery for {ct}/{m}: {src}")
            x, idx = H.take_panel(src, rng, n=N_PANEL)
            gen[(ct, m)] = pca.project(x)
            rec[f"{ct}/{m}"] = H.array_record(src, len(idx), idx, SEED)

    fig, axes = plt.subplots(3, 4, figsize=(5.5, 4.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=.055, right=.995, top=.945, bottom=.055,
                        wspace=.05, hspace=.06)
    for i, ct in enumerate(H.TYPES3):
        tgt = yval == ct
        for j, m in enumerate(cols):
            ax = axes[i, j]
            ax.scatter(zr[:, 0], zr[:, 1], s=.35, c=COL_BG, linewidths=0, zorder=1,
                       rasterized=True)
            # target region below, deliveries semi-transparent on top. The single-row figure draws green on top because Neuronal
            # only 396 real cells, which do not cover the orange; here Endothelial has over ten thousand, so green goes on top
            # would bury the orange of the whole row, so instead alpha lets the green underneath show through.
            ax.scatter(zr[tgt, 0], zr[tgt, 1], s=.8, c=COL_TARGET, linewidths=0,
                       zorder=2, rasterized=True)
            g = gen[(ct, m)]
            ax.scatter(g[:, 0], g[:, 1], s=.7, c=COL_GEN, alpha=.50, linewidths=0,
                       zorder=3, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_linewidth(.5)
            if i == 0:
                ax.set_title(m, fontsize=6.8, pad=2,
                             fontweight="bold" if m == "Ours" else "normal")
            if j == 0:
                ax.set_ylabel(ct, fontsize=6.8, labelpad=2)

    h = [plt.Line2D([0], [0], ls="none", marker="o", ms=2.6, color=COL_TARGET),
         plt.Line2D([0], [0], ls="none", marker="o", ms=2.6, color=COL_GEN)]
    fig.legend(h, ["Real cells of the requested type", "Delivered cells"],
               loc="lower center", bbox_to_anchor=(.5, -.018), ncol=2, frameon=False,
               fontsize=6.4, handletextpad=.3, columnspacing=1.6, borderpad=0.)

    H.save(fig, "fig_pca_baselines_all", {
        "figure": "PCA, four samplers x three requests (appendix)",
        "types": H.TYPES3, "samplers": cols,
        "n_panel": N_PANEL, "subsample_seed": SEED,
        "zero_training": True, "zero_sampling": True,
        "pca_gate": gate, "arrays": rec,
        "note": "Ours uses hvgfig.DELIVERED (current runs), not the e21 archived paths; "
                "fig_dispersion.py makes the same swap. Every panel is subsampled to "
                "2,500 cells so panel densities are comparable across rows.",
    })
    H.copy_pdf_to_paper("fig_pca_baselines_all", "hvg_pca_baselines_all.pdf")


if __name__ == "__main__":
    if "--grid" in sys.argv:
        grid()
    else:
        main()
