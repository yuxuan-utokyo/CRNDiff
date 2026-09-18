# -*- coding: utf-8 -*-
"""Draw the remaining toy figures from results/toy/figdata.npz.

Run build_toy_figure_data.py first, then this script. Outputs go to out/figures/.

    figure_toy_delivered_three_arms.pdf   all three arms' deliveries
    figure_toy_resample_marginal.pdf      the first coordinate's marginal before and after the
                                          terminal resampling, against the exact request marginal
    figure_toy_reverse_chain.pdf          the reverse chain step by step, with the resampling
                                          events boxed in red

Conventions: orange is the sampler's particles and grey the training data, whose density shows
through by transparency. A red box marks a step where resampling fired, around the cloud it
acted on; the delivery is a result rather than an event and is not boxed. Scatter layers are
rasterised, so every save passes an explicit high dpi.

    python scripts/reproduce/figure_toy_chain_and_resampling.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

# Okabe-Ito
BLUE, VERM, GREEN, GREY, RED = "#0072B2", "#D55E00", "#009E73", "#9a9a9a", "#D0021B"
BEF, AFT = "#8FBEDD", "#E2703A"                    # before and after, in the marginal figure
NAME = {"tilt_only": "Product tilt",
        "tilt_off_interval8": "FK, untilted",
        "tilt_on_interval8": "Tilted FK"}


def _bg(ax, s=1.4, alpha=.05):
    ax.scatter(BG[:, 0], BG[:, 1], s=s, c="#8f8f8f", alpha=alpha,
               linewidths=0, zorder=1, rasterized=True)


def _families(arm):
    """One marker per ancestor: the representative point and the family size."""
    X = D[f"{arm}_out_X"].astype(float)
    anc = D[f"{arm}_out_anc"]
    u, inv = np.unique(anc, return_inverse=True)
    fam = np.bincount(inv)
    first = np.zeros(len(u), int)
    for j, i in enumerate(inv):
        first[i] = j                               # use the last member of the family as its representative
    return X[first], fam


# ------------------------------------------------------------------ figure 1
def fig_dataset():
    """Render the construction from the frozen PMF using the dedicated script."""
    import runpy
    runpy.run_path(str(HERE / "figure_toy_dataset.py"), run_name="__main__")


# ------------------------------------------------------------------ figure 2
def fig_delivered():
    """The deliveries of the three arms. One marker is one distinct ancestor, its area proportional to the number of copies."""
    plt.rcParams.update({"font.size": 8, "axes.linewidth": .6,
                         "xtick.major.width": .6, "ytick.major.width": .6,
                         "xtick.major.size": 2.2, "ytick.major.size": 2.2,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    fig, axes = plt.subplots(1, 3, figsize=(3.55, 1.62), sharex=True, sharey=True)
    fig.subplots_adjust(left=.115, right=.995, bottom=.225, top=.845, wspace=.13)
    for ax, arm in zip(axes, META["arms"]):
        _bg(ax, s=1.6, alpha=.055)
        Q, fam = _families(arm)
        ax.scatter(Q[:, 0], Q[:, 1], s=1.1 * fam, c=VERM, alpha=.55,
                   linewidths=0, zorder=3)
        ax.set_title(NAME[arm], fontsize=7.8, pad=3)
        ax.set_aspect("equal"); ax.set_xlim(-1, C); ax.set_ylim(-1, C)
        ax.set_xticks([0, 30, 60]); ax.set_yticks([0, 30, 60])
        ax.set_xlabel("Dim 0", labelpad=1.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        print(f"  toy_delivered.pdf      {NAME[arm]:14s} ancestors {len(fam):4d}  "
              f"max family {fam.max():3d}")
    axes[0].set_ylabel("Dim 1", labelpad=1.5)
    fig.savefig(OUT / "figure_toy_delivered_three_arms.pdf", dpi=600)
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig_resample_dim0():
    """The first coordinate's marginal before and after the terminal resampling, against the

        exact request marginal. Note that this coordinate CANNOT show which diagonal was
        selected: the two diagonals have bitwise identical marginals by construction. So the
        figure shows only the marginal half of the work, which is exactly the wall: the tilt
    """
    tgt = D["target_dim0"]; g = np.arange(C)
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": .7,
                         "xtick.major.width": .7, "ytick.major.width": .7,
                         "xtick.major.size": 2.4, "ytick.major.size": 2.4,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.80), sharex=True, sharey=True)
    fig.subplots_adjust(left=.175, right=.985, bottom=.152, top=.815, hspace=.34)
    for ax, (arm, tag) in zip(axes, [("tilt_off_interval8", "(a) FK, untilted"),
                                     ("tilt_on_interval8", "(b) Tilted FK")]):
        pre = D[f"{arm}_pre_dim0"].astype(int); post = D[f"{arm}_post_dim0"].astype(int)
        hp = np.bincount(pre, minlength=C) / len(pre)
        ha = np.bincount(post, minlength=C) / len(post)
        ax.bar(g, hp, width=1.0, color=BEF, alpha=.95, linewidth=0,
               label="Before resample", zorder=2)
        ax.bar(g, ha, width=1.0, color=AFT, alpha=.72, linewidth=0,
               label="After resample", zorder=3)
        ax.step(np.append(g, C) - .5, np.append(tgt, tgt[-1]), where="post",
                color="black", lw=.9, ls=(0, (3, 2)), label="Request marginal", zorder=4)
        ax.set_title(tag, fontsize=8.5, pad=3)
        ax.set_xlim(-.5, C - .5); ax.set_ylim(0, .185)
        ax.set_xticks([0, 20, 40, 60]); ax.set_yticks([0, .05, .10, .15])
        ax.set_ylabel("Prob. mass", fontsize=8.5, labelpad=2)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        tv = lambda h: .5 * float(np.abs(h - tgt).sum())
        print(f"  toy_resample_dim0.pdf  {NAME[arm]:14s} TV before {tv(hp):.4f} "
              f"-> after {tv(ha):.4f}")
    axes[1].set_xlabel("Dim 0", fontsize=8.5, labelpad=1.5)
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(x) for x in ("Before resample", "After resample", "Request marginal")]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center",
               ncol=3, frameon=False, fontsize=6.8, handlelength=1.1,
               handletextpad=.4, columnspacing=1.0, borderaxespad=0,
               bbox_to_anchor=(.56, 1.0))
    fig.savefig(OUT / "figure_toy_resample_marginal.pdf", dpi=600)
    plt.close(fig)


# ------------------------------------------------------------------ figure 4
def fig_reverse_strip():
    """The reverse chain step by step. Rows are the arms, columns the steps and the delivery.

        Known limitation: the trajectory is recorded AFTER the resampling index is applied, so
        the two panels where resampling fired show the cloud AFTER the event and look tighter
        than the rows above and below. The last panel is the weighted cloud BEFORE the terminal
        resampling, which is a different quantity again. The caption must say so until a
    """
    KS = META["ks"]
    plt.rcParams.update({"font.size": 7.5, "axes.linewidth": .6,
                         "xtick.major.width": .6, "ytick.major.width": .6,
                         "xtick.major.size": 2.0, "ytick.major.size": 2.0,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    nc = len(KS) + 1
    fig, axes = plt.subplots(3, nc, figsize=(6.1, 3.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=.096, right=.997, bottom=.115, top=.895,
                        wspace=.13, hspace=.13)
    rng = np.random.default_rng(3)
    for i, arm in enumerate(META["arms"]):
        fired = set(META["fired"][arm]); is_fk = META["is_fk"][arm]
        for j in range(nc):
            ax = axes[i, j]
            _bg(ax)
            if j < len(KS):
                k = KS[j]
                red = (k in fired) or (k == 0 and is_fk)   # the last step is the terminal resampling
                X = D[f"{arm}_k{k}_X"].astype(float); w = D[f"{arm}_k{k}_w"]
                sel = rng.choice(len(X), size=3500, replace=False)
                ax.scatter(X[sel, 0], X[sel, 1], s=np.clip(3000.0 * w[sel], 1.3, 60.0),
                           c=VERM, alpha=.34, linewidths=0, zorder=3, rasterized=True)
                if i == 0:
                    ax.set_title(f"$k={k}$", fontsize=7.5, pad=3)
            else:
                red = False                                # the delivery is a result, not an event
                Q, fam = _families(arm)
                ax.scatter(Q[:, 0], Q[:, 1], s=1.1 * fam, c=VERM, alpha=.55,
                           linewidths=0, zorder=3)
                if i == 0:
                    ax.set_title("Delivered", fontsize=7.5, pad=3)
            ax.set_aspect("equal"); ax.set_xlim(-1, C); ax.set_ylim(-1, C)
            ax.set_xticks([0, 30, 60]); ax.set_yticks([0, 30, 60])
            if red:
                for s in ("top", "right", "bottom", "left"):
                    ax.spines[s].set_visible(True)
                    ax.spines[s].set_color(RED); ax.spines[s].set_linewidth(.9)
                ax.text(.965, .955, "RESAMPLE", transform=ax.transAxes,
                        ha="right", va="top", fontsize=4.6, color=RED,
                        fontweight="bold", zorder=6)
            else:
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
            if j == 0:
                ax.set_ylabel(NAME[arm] + "\nDim 1", fontsize=7.0,
                              labelpad=1.5, linespacing=1.5)
            if i == 2:
                ax.set_xlabel("Dim 0", fontsize=7.0, labelpad=1.0)
    fig.savefig(OUT / "figure_toy_reverse_chain.pdf", dpi=600)
    plt.close(fig)
    print("  toy_reverse_strip.pdf  fired: " +
          ", ".join(f"{NAME[a]}={META['fired'][a]}" for a in META["arms"]))


if __name__ == "__main__":
    print(f"[figures] {META['data_stem']}  seed {META['seed']}  -> {OUT}")
    fig_dataset()
    fig_delivered()
    fig_resample_dim0()
    fig_reverse_strip()
    print("[done]")
