# -*- coding: utf-8 -*-
"""The four-way comparison on the atlas PCA basis: real cells, three baselines and our sampler.

Unlike the demonstration panels, this puts four samplers on the SAME request so they can be
compared directly. The rare request is chosen because it is the only one of the three that
separates the methods at all; the full request-by-method grid goes to the appendix, so the main
text is not selecting on the result.

How to read it: grey is all real cells as background, green the real cells of the requested
type, which is where a delivery should land, and orange what the sampler actually delivered.
Orange covering green means the request was met.

Sources: the baselines come from the paths registered in the scoring summary, our arm from the
delivered arrays of this round. Nothing is resampled, rescored or refitted here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures import panels as H                      # noqa: E402

TYPE = "Neuronal"
SEED = 20261109                                      # a subsampling stream reserved for this figure
N_PANEL = 2500
BASELINES = ["scVI", "CFGen", "scANVI"]
REGISTRY = ROOT / "out" / "summary" / "e21_table1_ours.json"

COL_BG = "#dcdbd5"                                   # all real cells
COL_TARGET = "#1baf7a"                               # the real cells of the request: where a delivery should land
COL_GEN = "#D55E00"                                  # delivered cells; orange means the sampler's particles, as in the toy figures


def win_to_mount(p: str) -> Path:
    """The scoring summary stores Windows absolute paths; reattach them to this repository."""
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
    srcs["Ours"] = H.DELIVERED[TYPE]                 # this round's run, not the archived path
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
        # the target layer is drawn on top: otherwise the baseline panels' green is buried under the
        # orange and a reader cannot judge whether the orange landed correctly
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
    """The full appendix grid: three requests by four samplers on one frozen PCA basis.

        The point of this figure is to show all three requests: on the abundant one the four methods
        barely differ, on the next they begin to separate, and on the rare one they separate most.
        Showing all of them is what keeps the main text from being accused of picking the one that separates.
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
    for ct in H.TYPES3:                       # the row order is fixed and the generator is consumed in row-column order
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
            # here the target layer is below and the translucent delivery above. The single-row figure puts
            # green on top because the rare request has few real cells; here the abundant request has over
            # ten thousand, which would bury the orange, so the order is reversed and alpha lets green show
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
