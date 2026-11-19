# -*- coding: utf-8 -*-
"""Ticket 210 · block 2: PCA grid v2: **add an MDLM column** to the existing 3×4 grid.

**old script `figures/fig_pca_baselines.py` is left byte-identical; old figures stay in place.**
this file does only three things:

1. list from `[scVI, CFGen, scANVI, Ours]` becomes `[scVI, CFGen, scANVI, MDLM, Ours]`;
2. **sampling order kept**: first draw the original 12 cells in the row-column order of the original script, **then** draw the 3 MDLM cells.
   inserting in the middle would change the 2,500 rows drawn for every later cell; placed last, the original 12 cells
   indices are bit-identical (the sidecar records sha + index digest per cell, comparable with the old figure sidecar).
3. one class-A path fix: the original script `win_to_mount()` re-root the absolute Windows paths in the manifest to
   `~/mnt/crndiff/...`(at the time run on the mount point). This machine is that tree on disk, so changed to `crndiff/`
   the part after it is re-rooted at this repository root, as with `analysis/e33_deployed_arm.py::translate` the same technique.

basis, colours and points per panel (2,500) all reused, not refitted.

    python -m figures.fig_pca_baselines_v2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures import hvgfig as H                      # noqa: E402
from figures.fig_pca_baselines import (BASELINES, COL_BG, COL_GEN,   # noqa: E402
                                       COL_TARGET, N_PANEL, REGISTRY, SEED)

MDLM_SEED = 20260931                    # same sampling seed as the main conditional pools of 208/209
REPO = H.ROOT.parents[1]                # .../CRNDIFF


def translate(p: str) -> Path:
    """manifest absolute Windows paths -> local paths (this machine is that tree)."""
    s = str(p).replace("\\", "/")
    i = s.find("crndiff/")
    if i < 0:
        return Path(p)
    tail = s[i + len("crndiff/"):]
    q = REPO.parent / tail            # tail starts with CRNDIFF/
    return q


def mdlm_pool(ct: str) -> Path:
    hits = sorted((H.OUT / "cond_pools").glob(
        f"mdlm_cond_K128_{ct}_seed{MDLM_SEED}_n*.npy"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one MDLM pool for {ct}, found {hits}")
    return hits[0]


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
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))["types"]

    old_cols = BASELINES + ["Ours"]            # the four columns of the original script, order unchanged
    cols = BASELINES + ["MDLM", "Ours"]        # the five columns drawn
    rng = np.random.default_rng(SEED)
    gen, rec = {}, {}

    # ---- stage 1: the original 12 cells, in exactly the same order as the original script ----
    for ct in H.TYPES3:
        rows = reg[ct]["rows"]
        for m in old_cols:
            src = H.DELIVERED[ct] if m == "Ours" else translate(rows[m]["source"])
            if not src.is_file():
                raise SystemExit(f"missing delivery for {ct}/{m}: {src}")
            x, idx = H.take_panel(src, rng, n=N_PANEL)
            gen[(ct, m)] = pca.project(x)
            rec[f"{ct}/{m}"] = H.array_record(src, len(idx), idx, SEED)

    # ---- stage 2: the added MDLM column, drawn last ----
    for ct in H.TYPES3:
        src = mdlm_pool(ct)
        x, idx = H.take_panel(src, rng, n=N_PANEL)
        gen[(ct, "MDLM")] = pca.project(x)
        rec[f"{ct}/MDLM"] = {**H.array_record(src, len(idx), idx, SEED),
                             "sampling_seed": MDLM_SEED,
                             "drawn_last": "appended AFTER the twelve pre-existing panels "
                                           "so their row indices are unchanged"}

    fig, axes = plt.subplots(3, 5, figsize=(5.5, 3.42), sharex=True, sharey=True)
    fig.subplots_adjust(left=.048, right=.996, top=.945, bottom=.065,
                        wspace=.05, hspace=.06)
    for i, ct in enumerate(H.TYPES3):
        tgt = yval == ct
        for j, m in enumerate(cols):
            ax = axes[i, j]
            ax.scatter(zr[:, 0], zr[:, 1], s=.35, c=COL_BG, linewidths=0, zorder=1,
                       rasterized=True)
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
               loc="lower center", bbox_to_anchor=(.5, -.022), ncol=2, frameon=False,
               fontsize=6.4, handletextpad=.3, columnspacing=1.6, borderpad=0.)

    H.save(fig, "fig_pca_baselines_all_v2", {
        "figure": "PCA, five samplers x three requests (appendix, v2)",
        "ticket": 210,
        "supersedes": "fig_pca_baselines_all (v1) -- left untouched on disk",
        "types": H.TYPES3, "samplers": cols,
        "what_is_new_vs_v1": {
            "column": "MDLM", "sampling_seed": MDLM_SEED,
            "drawn": "LAST, after all twelve pre-existing panels, so their subsample "
                     "indices are bit-identical to v1 (compare rec entries with "
                     "fig_pca_baselines_all.json)",
            "geometry_note": "the grid is now 3x5 instead of 3x4, so every panel is "
                             "narrower; the DATA in the twelve original panels is "
                             "unchanged, the pixels necessarily are not"},
        "A_fixes": [{"what": "path translation for the archived baseline arrays",
                     "was": "win_to_mount(): ~/mnt/crndiff/... (the figure was first made "
                            "on a Linux mount of this tree)",
                     "now": "rejoined under this repo root",
                     "why": "same tree, same bytes; this box is the tree itself"}],
        "n_panel": N_PANEL, "subsample_seed": SEED,
        "zero_training": True, "zero_sampling": True,
        "pca_gate": gate, "arrays": rec,
        "note": "Ours uses hvgfig.DELIVERED (current runs). Every panel is subsampled to "
                "2,500 cells so panel densities are comparable across rows and columns.",
    })
    H.copy_pdf_to_paper("fig_pca_baselines_all_v2", "hvg_pca_baselines_all_v2.pdf")

    # ---- self-proof: compare the original 12 cells with the v1 sidecar cell by cell ----
    old = H.FIG / "fig_pca_baselines_all.json"
    if old.exists():
        o = H.read_json(old)["arrays"]
        same, diff = 0, []
        for k, v in rec.items():
            if k.endswith("/MDLM") or k not in o:
                continue
            a, b = o[k], v
            ok = all(a.get(f) == b.get(f) for f in ("sha256", "n_used", "idx_sha256",
                                                    "idx_first", "idx_last")
                     if f in a or f in b)
            same += ok
            if not ok:
                diff.append({"panel": k, "v1": a, "v2": b})
        print(f"[self-check] pre-existing panels identical to v1: {same}/12"
              + ("" if not diff else f"   DIFFER: {[d['panel'] for d in diff]}"))


if __name__ == "__main__":
    main()
