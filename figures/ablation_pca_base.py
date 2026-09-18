# -*- coding: utf-8 -*-
"""The ablation grid on the frozen atlas PCA basis: three arms by three requests.

Rows are the three requests top to bottom and columns the three arms left to right, in the same
order as the ablation table. Each panel draws three layers, in the same order as the baseline
grid: all real validation cells in grey at the bottom, the real cells OF THAT REQUEST above
them, and the arm's delivered cells on top.

Nothing is resampled, rescored or refitted: this reads the existing delivered arrays and the
existing scoring summaries, and every number printed on the figure also appears in the sidecar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures import panels as H                      # noqa: E402

# the three layers and their colours
# the bottom and middle layers copy the baseline grid, for one consistent look
COL_BG = "#dcdbd5"          # all real cells
COL_TARGET = "#1baf7a"      # the real cells of the request; the baseline grid uses this colour for the same meaning
# delivered cells are drawn in light purple, a colour that clashes with nothing else here
# it is not a type-identity colour, is not registered in the palette, and does not alter the style module
# the exact value was chosen by measurement, not by taste: the CIE76 distance to every colour in
# use was computed and the worst case maximised (worst 26.9; 65.9 and 88.6 to the other layers)
COL_GEN = "#c3a6e0"
COL_GEN_WORST_DE76 = {"vs": "Endothelial #6aa6ee", "dE76": 26.9}

TYPES3 = ["Endothelial", "Myeloid", "Neuronal"]
ARMS = [
    ("tilt_only", "Product tilt sampler", "purity_ablation_table10", "{t}_tilt_only"),
    ("fk_only", "FK sampler (untilted)", "purity_ablation_table10", "{t}_fk_only"),
    ("tilt_fk", "Tilted FK sampler", "purity_table1_ours", "{t}_a1"),
]
SEEDS = (20260987, 20260988, 20260989)      # the same seeds the ablation table uses

# the delivery directories. The shared registry is NOT extended: it records one request and one
# seed, and extending it would affect other figures, so paths are resolved here and asserted.
ARM_DIR = {
    "tilt_only": lambda t: H.RUNS / "ablation_table10" / f"{t}_tilt_only",
    "fk_only": lambda t: H.RUNS / "ablation_table10" / f"{t}_fk_only",
    "tilt_fk": lambda t: H.RUNS / "table1_ours" / f"{t}_a1",
}


def arm_paths(t: str, key: str) -> list[Path]:
    """The three seed arrays of one (request, arm); a missing or ambiguous one stops, never substitutes."""
    d = ARM_DIR[key](t)
    if not d.is_dir():
        raise SystemExit(f"{t}/{key}: delivery directory missing: {d}")
    out = []
    for sd in SEEDS:
        hit = sorted(q for q in d.glob("*.npy") if q.stem.endswith(f"seed{sd}"))
        if len(hit) != 1:
            raise SystemExit(f"{t}/{key}: expected exactly one array for seed {sd} in {d}, "
                             f"got {len(hit)}")
        out.append(hit[0])
    if len({q.name for q in out}) != len(SEEDS):
        raise SystemExit(f"{t}/{key}: seed resolution collapsed to fewer than {len(SEEDS)}")
    return out


def purity_row(summary_name: str, arm: str, stem: str) -> tuple[dict, dict]:
    """Take one row by exact stem, never by fuzzy arm-name matching; a miss stops the run."""
    p = H.SUMMARY / f"{summary_name}.json"
    d = H.read_json(p)
    hits = [r for r in d["rows"] if r["stem"] == stem]
    if len(hits) != 1:
        raise SystemExit(f"{p.name}: expected exactly one row with stem {stem}, got {len(hits)}")
    r = hits[0]
    if r["arm"] != arm:
        raise SystemExit(f"{p.name}: stem {stem} carries arm {r['arm']!r}, expected {arm!r}")
    return r, {"summary_file": str(p), "summary_sha256": H.sha256(p), "row_stem": stem,
               "arm": r["arm"], "purity_matched_n": r["purity_matched_n"],
               "canonical_ceiling": r["canonical_ceiling"],
               "purity_over_ceiling": r.get("purity_over_ceiling"),
               "matched_n": r["matched_n"], "n_delivered": r["n_delivered"]}


def purity_pooled(summary_name: str, arm: str, paths: list[Path]) -> dict:
    """One row per seed; report mean and sample standard deviation, and keep each row in the sidecar."""
    per = [purity_row(summary_name, arm, q.stem)[1] for q in paths]
    ceil = {r["canonical_ceiling"] for r in per}
    if len(ceil) != 1:
        raise SystemExit(f"{arm}: seeds disagree on canonical_ceiling: {sorted(ceil)}")
    v = np.array([r["purity_matched_n"] for r in per], float)
    rel = [r["purity_over_ceiling"] for r in per]
    has_rel = all(isinstance(z, (int, float)) for z in rel)
    ra = np.array(rel, float) if has_rel else None
    return {"arm": arm, "n_seeds": len(per), "seeds": list(SEEDS),
            "purity_matched_n_mean": float(v.mean()),
            "purity_matched_n_sd": float(v.std(ddof=1)),
            "canonical_ceiling": ceil.pop(),
            "purity_over_ceiling_mean": (float(ra.mean()) if has_rel else None),
            "purity_over_ceiling_sd": (float(ra.std(ddof=1)) if has_rel else None),
            "per_seed": per}


def take_panel_pooled(paths: list[Path], rng, n: int):
    """Draw n rows uniformly from the union of the three arrays, reading only the rows drawn."""
    sizes = [int(np.load(q, mmap_mode="r").shape[0]) for q in paths]
    total = sum(sizes)
    pick = np.sort(rng.choice(total, size=min(n, total), replace=False))
    edges = np.cumsum([0] + sizes)
    chunks, per_file = [], []
    for j, q in enumerate(paths):
        loc = pick[(pick >= edges[j]) & (pick < edges[j + 1])] - edges[j]
        a = np.load(q, mmap_mode="r")
        chunks.append(np.asarray(a[loc]))
        per_file.append({"path": str(q), "sha256": H.sha256(q), "n_source": sizes[j],
                         "n_genes": int(a.shape[1]), "n_plotted": int(loc.size)})
    return np.concatenate(chunks, axis=0), per_file, total


def main() -> None:
    st = H.load_style()
    st.apply_rc()
    import matplotlib.pyplot as plt

    xval, yval = H.real_cells()
    pca = H.FrozenPCA()
    gate = pca.gate_against_archive(xval)          # a failure stops the run; the basis object raises
    zr = pca.project(xval)

    # the real cells of the request: one generator consumed in row order, on the shared seed
    # a request with fewer cells than the panel size is drawn in full and not padded
    rng_real = np.random.default_rng(H.SUBSAMPLE_SEED)
    real_idx, real_rec = {}, {}
    for t in TYPES3:
        pos = np.flatnonzero(yval == t)
        k = min(pos.size, H.N_PANEL)
        sel = np.sort(rng_real.choice(pos.size, size=k, replace=False))
        real_idx[t] = pos[sel]
        real_rec[t] = {"n_total": int(pos.size), "n_plotted": int(k),
                       "subsample_seed": int(H.SUBSAMPLE_SEED),
                       "subsampled": bool(k < pos.size),
                       "rule": "sorted(rng.choice(n_of_type, min(N_PANEL, n_of_type), "
                               "replace=False)); ONE generator consumed in row order "
                               "Endothelial -> Myeloid -> Neuronal"}

    # delivered points: one generator consumed in row-major panel order, as in the earlier figures
    rng = np.random.default_rng(H.SUBSAMPLE_SEED)
    gen, arrays, numbers = {}, {}, {}
    for t in TYPES3:
        for key, _title, summary_name, arm_tpl in ARMS:
            arm = arm_tpl.format(t=t)
            paths = arm_paths(t, key)
            numbers[(t, key)] = purity_pooled(summary_name, arm, paths)
            x, per_file, total = take_panel_pooled(paths, rng, H.N_PANEL)
            gen[(t, key)] = pca.project(x)
            arrays[(t, key)] = {"pooled_over_seeds": list(SEEDS), "n_source_pooled": total,
                                "n_plotted": int(x.shape[0]),
                                "subsample_seed": int(H.SUBSAMPLE_SEED), "files": per_file}

    fig, axes = plt.subplots(3, 3, figsize=(5.5, 4.75), sharex=True, sharey=True)
    fig.subplots_adjust(left=.062, right=.995, top=.935, bottom=.085, wspace=.05, hspace=.05)

    for i, t in enumerate(TYPES3):
        for j, (key, title, _s, _a) in enumerate(ARMS):
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
                ax.set_title(title, fontsize=6.8, pad=2)
            if j == 0:
                ax.set_ylabel(t, fontsize=6.8, labelpad=2)
            r = numbers[(t, key)]
            ax.text(0.975, 0.965, f"Purity {r['purity_over_ceiling_mean']:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=st.FS_MIN - 1.5, color=st.INK, zorder=9,
                    bbox=dict(boxstyle="round,pad=0.26", fc="#ffffffe8", ec=st.GRID, lw=0.5))
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_linewidth(.5)

    h = [plt.Line2D([0], [0], ls="none", marker="o", ms=2.6, color=COL_BG),
         plt.Line2D([0], [0], ls="none", marker="o", ms=2.6, color=COL_TARGET),
         plt.Line2D([0], [0], ls="none", marker="o", ms=2.6, color=COL_GEN)]
    fig.legend(h, [f"All real cells (n={zr.shape[0]:,})",
                   "Real cells of the requested type (row)",
                   "Delivered cells"],
               loc="lower center", bbox_to_anchor=(.5, -.004), ncol=3, frameon=False,
               fontsize=6.4, handletextpad=.3, columnspacing=1.4, borderpad=0.)

    sidecar = {
        "figure": "F5 fig_ablation_pca", "ticket": 204,
        "claim": "the three ablation arms x the three requests on the frozen atlas PCA",
        "layout": {"shape": "3x3", "rows": TYPES3,
                   "cols": [t for _k, t, _s, _a in ARMS],
                   "row_col_order": "matches tab:hvg_ablation",
                   "previous_layout": "1x3, Endothelial only (renamed aside, not deleted)"},
        "zero_sampling": True, "zero_fitting": True, "zero_rescoring": True,
        "seeds": list(SEEDS),
        "seed_policy": "all three sampling seeds of tab:hvg_ablation are pooled; the panel "
                       "points are drawn from the union of the three delivered arrays and the "
                       "inset purity is their mean, so figure and table carry the same number",
        "inset_quantity": {
            "printed": "relative purity = purity_matched_n / canonical_ceiling, averaged over the three seeds",
            "why": "that is what the Purity column of tab:hvg_ablation reports, and what the figure caption already claimed the inset was: delivered purity over the real-cell classifier score for this type, the relative purity the table reports",
            "previous_1x3_printed": "purity_matched_n (ABSOLUTE), which matched neither the table nor its own caption -- see out/summary/204_A_fixes.md",
            "absolute_values_still_recorded": True},
        "frozen_basis": gate,
        "real_background": {
            "file": str(H.DATA / "val.npy"), "sha256": H.sha256(H.DATA / "val.npy"),
            "n_plotted": int(zr.shape[0]), "subsampled": False,
            "role": "grey bottom layer, every panel"},
        "real_cells_of_the_requested_type": real_rec,
        "registry_note": "figures/hvgfig.py::ABLATION registers only Endothelial and only "
                         "seed 20260987, so it cannot address a 3x3 grid. This figure "
                         "resolves paths from the delivery directories with an exact "
                         "seed-suffix match and asserts exactly one file per seed. "
                         "hvgfig.py was NOT modified (it is shared with other figures).",
        "panels": [
            {"row_request": t, "col_arm_key": key, "col_arm_title": title,
             "array": arrays[(t, key)], "purity": numbers[(t, key)],
             "real_same_type": real_rec[t],
             "purity_source_json": f"{summary_name}.json",
             "purity_row_stems": [r["row_stem"] for r in numbers[(t, key)]["per_seed"]]}
            for t in TYPES3
            for key, title, summary_name, _a in ARMS],
        "subsample": {"seed": int(H.SUBSAMPLE_SEED), "n_per_panel": int(H.N_PANEL),
                      "delivered_rule": "sorted(rng.choice(N_pooled, min(2500, N_pooled), "
                                        "replace=False)) over the concatenation of the three "
                                        "seed arrays in SEEDS order; ONE generator consumed "
                                        "in row-major panel order",
                      "real_same_type_rule": "a SECOND generator, same seed value "
                                             "SUBSAMPLE_SEED, consumed in row order; types "
                                             "below N_PANEL are drawn in full and NOT padded"},
        "palette": {"all_real_cells": COL_BG,
                    "real_cells_of_requested_type": COL_TARGET,
                    "delivered_cells": COL_GEN},
        "palette_provenance": {
            "all_real_cells": "copied from figures/fig_pca_baselines.py (the 3x4 panel), "
                              "so the two figures read as one set",
            "real_cells_of_requested_type": "copied from the same 3x4 panel, which uses this "
                                            "hex for exactly this semantic across all rows",
            "delivered_cells": "light purple, ruled by Q on 2026-09-13 ('use non-conflicting "
                               "colours, light purple for the generated cells'). It is NOT a "
                               "family-A identity colour, is NOT in TYPE_COLORS, and "
                               "style.py was NOT modified, so assert_palette is untouched.",
            "delivered_cells_selection": "chosen by measurement, not taste: CIE76 deltaE in "
                                         "Lab against every colour currently in use, largest "
                                         "worst-case minimum among light-purple candidates",
            "delivered_cells_worst_case": COL_GEN_WORST_DE76,
            "disclosure_ticket_114_ruling_4":
                "ticket 114 retired purple #9467bd because it carried TWO meanings at once "
                "(neither-type, and 'OURS generated'). This figure reintroduces a purple for "
                "the generated layer. It is a different hex (#c3a6e0, light), it carries "
                "exactly ONE meaning here, and purple appears in no other current HVG figure "
                "as a cell or arm identity -- so the ambiguity ruling 4 removed is not "
                "recreated. Flagged rather than assumed harmless."},
        "palette_source": str(H.VIZ / "style.py"),
        "palette_drift_gate": "style.assert_palette passed (unchanged registry)",
        "numbers_on_the_figure": {
            f"{t}|{key}": {
                "printed": round(numbers[(t, key)]["purity_over_ceiling_mean"], 3),
                "printed_quantity": "purity_over_ceiling_mean (RELATIVE purity)",
                "purity_over_ceiling_mean": numbers[(t, key)]["purity_over_ceiling_mean"],
                "purity_over_ceiling_sd": numbers[(t, key)]["purity_over_ceiling_sd"],
                "purity_matched_n_mean": numbers[(t, key)]["purity_matched_n_mean"],
                "purity_matched_n_sd": numbers[(t, key)]["purity_matched_n_sd"],
                "canonical_ceiling": numbers[(t, key)]["canonical_ceiling"],
                "n": int(len(gen[(t, key)])),
                "sources": [{"file": r["summary_file"], "sha256": r["summary_sha256"],
                             "row_stem": r["row_stem"],
                             "purity_matched_n": r["purity_matched_n"],
                             "purity_over_ceiling": r["purity_over_ceiling"]}
                            for r in numbers[(t, key)]["per_seed"]]}
            for t in TYPES3 for key, _t, _s, _a in ARMS},
        "why_two_summary_files": "purity_ablation_table10.json registers only the two "
                                 "ablation arms (tilt_only, fk_only) and carries all three "
                                 "requests. The tilt+FK arm is the deployed Table 1 array, "
                                 "whose authoritative purity row lives in "
                                 "purity_table1_ours.json (arms Endothelial_a1 / Myeloid_a1 / "
                                 "Neuronal_a1). Nothing is transcribed by hand: each row is "
                                 "matched by its exact stem.",
    }
    H.save(fig, "fig_ablation_pca", sidecar)
    H.copy_pdf_to_paper("fig_ablation_pca", "hvg_ablation_pca.pdf")


if __name__ == "__main__":
    main()
