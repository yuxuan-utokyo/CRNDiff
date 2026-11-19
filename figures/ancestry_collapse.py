# -*- coding: utf-8 -*-
"""Ancestry collapse along the reverse chain: one row of three panels, one per request.

The vertical axis is the surviving fraction of initial ancestors, on a LOG scale, because the
decay is geometric event by event and is close to a straight line there, which is what makes the
figure worth drawing. Tick labels stay decimal rather than scientific. Each panel carries one
line per arm, the mean over three seeds with a translucent standard-deviation band, and vertical
marks at the resampling events.

Every number is read from the ancestry summary json; none is transcribed. The upper row uses the
delivered-set ancestry, which is the convention the table reports.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures import panels as H                       # noqa: E402
from figures.style import C, FIG                      # noqa: E402

TYPES = ["Endothelial", "Myeloid", "Neuronal"]
# arm to an existing palette key; no colour is defined here
ARM_COLOR = {"a1": C["blue"], "fk_only": C["vermilion"]}
ARM_LABEL = {"a1": "Tilted FK sampler", "fk_only": "FK sampler (untilted)"}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    src = H.SUMMARY / "ancestry_trace.json"
    d = H.read_json(src)
    if not d["gates"]["all_runs_passed"]:
        raise SystemExit("refusing to draw: not every run passed the ticket 182 gates -- "
                         f"see {src}")
    by = {(c["target"], c["arm"]): c for c in d["cells"]}

    # the lower row of distinct-particle curves never fell below 0.9997 and carried no information,
    # so it was removed from the figure; the quantity is still in the summary json and the sidecar
    fig, axes1 = plt.subplots(1, 3, figsize=(7.4, 2.5), sharex=True)
    axes = np.empty((2, 3), dtype=object)
    axes[0, :] = axes1
    fig.subplots_adjust(left=0.088, right=0.995, top=0.795, bottom=0.20,
                        wspace=0.16)

    numbers = {}
    for j, t in enumerate(TYPES):
        for arm in ("a1", "fk_only"):
            c = by.get((t, arm))
            if c is None:
                raise SystemExit(f"missing cell {t}/{arm} in {src.name}")
            cur = c["curves"]
            x = np.asarray(cur["step"], dtype=float) + 1.0      # reverse steps, one-based
            col = ARM_COLOR[arm]
            for row, (mkey, ax) in enumerate((("anc_frac_delivered", axes[0, j]),)):
                m = np.asarray(cur[f"{mkey}_mean"])
                s = np.asarray(cur[f"{mkey}_sd"])
                ax.fill_between(x, m - s, m + s, color=col, alpha=0.18, linewidth=0,
                                zorder=2)
                ax.plot(x, m, lw=1.4, color=col, zorder=3,
                        label=ARM_LABEL[arm] if (row == 0 and j == 0) else None)
            # resampling events are marked with short vertical lines. The three seeds do not always fire at
            # the same steps, so only steps where ALL THREE fired are drawn; the union and the per-seed sets are in the sidecar
            per_seed = [set(s) for s in c["resample_steps_per_seed"]]
            fired_all = sorted(set.intersection(*per_seed)) if per_seed else []
            fired_any = sorted(set.union(*per_seed)) if per_seed else []
            for k in fired_all:
                axes[0, j].axvline(k + 1, color=col, ls=(0, (1.6, 1.8)), lw=0.55,
                                   alpha=0.45, zorder=1)
            numbers[f"{t}/{arm}"] = {
                "n_seeds": c["n_seeds"], "seeds": c["seeds"], "M": c["M"], "N": c["N"],
                "resample_steps_marked_fired_in_all_seeds": fired_all,
                "resample_steps_fired_in_any_seed": fired_any,
                "resample_steps_per_seed": c["resample_steps_per_seed"],
                "first_firing_step_per_seed": [min(x) + 1 for x in c["resample_steps_per_seed"]],
                "anc_frac_delivered_first": cur["anc_frac_delivered_mean"][0],
                "anc_frac_delivered_last": cur["anc_frac_delivered_mean"][-1],
                "anc_frac_delivered_last_sd": cur["anc_frac_delivered_sd"][-1],
                "unique_frac_all_particles_last": cur["unique_frac_all_particles_mean"][-1],
                "unique_frac_all_particles_sd": cur["unique_frac_all_particles_sd"][-1],
                "unique_frac_all_particles_min": min(cur["unique_frac_all_particles_mean"]),
            }

    # the axis stays logarithmic, since the per-event geometric decay is a straight line there, but
    # the tick labels are decimal rather than scientific, and the positions come from each panel's range
    TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for j, t in enumerate(TYPES):
        ax = axes[0, j]
        ax.set_yscale("log")
        ax.set_title(t, fontsize=9, pad=4)
        ax.set_xlabel("reverse step")
        ax.set_xlim(1, 32)
        lo, hi = ax.get_ylim()
        show = [v for v in TICKS if lo <= v <= hi]
        ax.set_yticks(show)
        ax.set_yticklabels([f"{v:g}" for v in show], fontsize=7.5)
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7.5)
        ax.tick_params(labelleft=True)
    axes[0, 0].set_ylabel("surviving ancestors / N", fontsize=8.5)

    h = [plt.Line2D([0], [0], color=ARM_COLOR[a], lw=1.5) for a in ("a1", "fk_only")]
    h.append(plt.Line2D([0], [0], color="0.45", ls=(0, (1.6, 1.8)), lw=0.7))
    fig.legend(h, [ARM_LABEL["a1"], ARM_LABEL["fk_only"], "resampling fired"],
               loc="lower center", bbox_to_anchor=(0.54, 0.875), ncol=3,
               frameon=False, fontsize=8, handlelength=1.8, columnspacing=1.4)

    sidecar = {
        "figure": "fig_ancestry_collapse",
        "claim": "lineage collapse along the reverse chain, two arms, three cell types",
        "zero_sampling": True,
        "source": str(src), "source_sha256": H.sha256(src),
        "top_row": "anc_frac_delivered (the paper-table normalisation); log y axis",
        "bottom_row": "unique_frac_all_particles = distinct rows among the M LIVE particles / M (182b s3 naming); linear y axis. Population is M, NOT the N delivered cells.",
        "not_plotted": "anc_frac_all_particles stays in the json",
        "sd_band": "+-1 sample sd over the 3 sampling seeds (ddof=1)",
        "firing_marks": "vertical dashes mark the reverse steps at which ALL THREE seeds "
                        "resampled. Only Neuronal/OURS has an identical firing set across "
                        "seeds; for the other five cells the per-seed sets differ, so the "
                        "union and the per-seed sets are recorded here instead of drawn.",
        "bottom_row_y_axis": "fixed to [0.99, 1.002] with the offset notation disabled. The "
                             "series is ~1.0 everywhere, and matplotlib's autoscale otherwise "
                             "renders a '+1' offset axis that magnifies ~1e-4 noise into "
                             "apparent structure.",
        "palette_source": str(ROOT / "figures" / "style.py"),
        "palette": {ARM_LABEL[a]: ARM_COLOR[a] for a in ARM_COLOR},
        "numbers_on_the_figure": numbers,
        "numbers_source": "every value read from ancestry_trace.json; nothing transcribed",
    }
    H.save(fig, "fig_ancestry_collapse", sidecar)
    H.copy_pdf_to_paper("fig_ancestry_collapse", "ancestry_collapse.pdf")


if __name__ == "__main__":
    main()
