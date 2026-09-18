# -*- coding: utf-8 -*-
"""Figure 1: the deliveries of the two particle arms on the requested diagonal.

Reads results/toy/figdata.npz: the delivered samples and ancestor ids of the untilted FK arm and
of the tilted FK arm, plus the training mixture used as a grey background.

The product-tilt arm is not drawn. It is a proposal adaptation rather than a sampler in its own
right; its numbers stay in the toy table.

Marker area is proportional to the number of delivered descendants of an initial ancestor.
Scatter layers are rasterised, so the figure must be saved at a high dpi or the layer is blurred.

    python scripts/reproduce/figure_01_toy_delivered.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
TOY_DATA = REPO / "assets" / "toy" / "data"
TOY_RUNS = REPO / "results" / "toy" / "runs"
TOY_FIG  = REPO / "results" / "toy"
FIG_OUT  = REPO / "out" / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)
OUT = FIG_OUT
D = np.load(TOY_FIG / "figdata.npz", allow_pickle=False)
META = json.loads((TOY_FIG / "figmeta.json").read_text(encoding="utf-8"))
BG = D["bg"].astype(float)
C = 64
VERM = "#D55E00"
NAME = {"tilt_off_interval8": "FK, untilted", "tilt_on_interval8": "Tilted FK"}
ARMS = ["tilt_off_interval8", "tilt_on_interval8"]


def _bg(ax, s=1.6, alpha=.055):
    ax.scatter(BG[:, 0], BG[:, 1], s=s, c="#8f8f8f", alpha=alpha,
               linewidths=0, zorder=1, rasterized=True)


def _families(arm):
    X = D[f"{arm}_out_X"].astype(float)
    anc = D[f"{arm}_out_anc"]
    u, inv = np.unique(anc, return_inverse=True)
    fam = np.bincount(inv)
    first = np.zeros(len(u), int)
    for j, i in enumerate(inv):
        first[i] = j
    return X[first], fam


def main():
    plt.rcParams.update({"font.size": 8, "axes.linewidth": .6,
                         "xtick.major.width": .6, "ytick.major.width": .6,
                         "xtick.major.size": 2.2, "ytick.major.size": 2.2,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.76), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.995, bottom=.205, top=.87, wspace=.10)
    for ax, arm in zip(axes, ARMS):
        _bg(ax)
        Q, fam = _families(arm)
        ax.scatter(Q[:, 0], Q[:, 1], s=1.1 * fam, c=VERM, alpha=.55,
                   linewidths=0, zorder=3)
        ax.set_title(NAME[arm], fontsize=7.8, pad=3)
        ax.set_aspect("equal"); ax.set_xlim(-1, C); ax.set_ylim(-1, C)
        ax.set_xticks([0, 30, 60]); ax.set_yticks([0, 30, 60])
        ax.set_xlabel("Dim 0", labelpad=1.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        print(f"  {NAME[arm]:14s} ancestors {len(fam):4d}  max family {fam.max():3d}")
    axes[0].set_ylabel("Dim 1", labelpad=1.5)
    fig.savefig(OUT / "figure_01_toy_delivered.pdf", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
