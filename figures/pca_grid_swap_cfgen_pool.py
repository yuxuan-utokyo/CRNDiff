# -*- coding: utf-8 -*-
"""The PCA grid with only the CFGen column replaced by its new pool.

Neither earlier version is edited; this file reuses their constants and helpers.

## Why the old draw is consumed and thrown away

The panel sampler draws rows without replacement, so the number of random values it consumes
depends on the pool size. CFGen's pool changed size, and replacing it in place would change the
indices of every column drawn after it, which would no longer be 'only one column'. So the
twelve original panels are walked in the original order, and when CFGen's turn comes the
ARCHIVED array is drawn from as before and the result discarded, before the new pool is drawn.
Every other column is then bitwise identical to the previous figure.
"""
from __future__ import annotations

import os                                                            # noqa: E402

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"

import json                                                          # noqa: E402
import sys                                                           # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures import panels as H                                      # noqa: E402
from figures.pca_grid_base import (BASELINES, COL_BG, COL_GEN,   # noqa: E402
                                       COL_TARGET, N_PANEL, REGISTRY, SEED)
from figures.pca_grid_add_mdlm import MDLM_SEED, mdlm_pool, translate  # noqa: E402

CFGEN_SEED = 20260931
N_OUT = {"Endothelial": 10057, "Myeloid": 2500, "Neuronal": 2500}


def cfgen_pool(ct: str) -> Path:
    p = H.OUT / "cond_pools" / f"cfgen_cond_{ct}_seed{CFGEN_SEED}_n{N_OUT[ct]}.npy"
    if not p.is_file():
        raise SystemExit(f"missing CFGen pool for {ct}: {p}")
    return p


def low_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return "BELOW_NORMAL_PRIORITY_CLASS"
    except Exception as e:                                           # noqa: BLE001
        return f"not set ({type(e).__name__})"


def main() -> None:
    prio = low_priority()
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

    old_cols = BASELINES + ["Ours"]             # the previous version's four columns, in the same order
    cols = BASELINES + ["MDLM", "Ours"]         # the five columns drawn, the same as before
    rng = np.random.default_rng(SEED)
    gen, rec, consumed = {}, {}, {}

    # stage one: the twelve previous panels in the same order; the CFGen draw is consumed, not used
    for ct in H.TYPES3:
        rows = reg[ct]["rows"]
        for m in old_cols:
            if m == "CFGen":
                old_src = translate(rows["CFGen"]["source"])
                _, idx = H.take_panel(old_src, rng, n=N_PANEL)      # consume this draw
                consumed[ct] = {"archived_source": str(old_src),
                                "sha256": H.sha256(old_src),
                                "n_drawn_and_discarded": int(len(idx)),
                                "why": "keeps the rng stream at exactly the position v2 "
                                       "had, so the other twelve panels are unchanged"}
                continue
            src = H.DELIVERED[ct] if m == "Ours" else translate(rows[m]["source"])
            if not src.is_file():
                raise SystemExit(f"missing delivery for {ct}/{m}: {src}")
            x, idx = H.take_panel(src, rng, n=N_PANEL)
            gen[(ct, m)] = pca.project(x)
            rec[f"{ct}/{m}"] = H.array_record(src, len(idx), idx, SEED)

    # stage two: the MDLM column, in the same position as before
    for ct in H.TYPES3:
        src = mdlm_pool(ct)
        x, idx = H.take_panel(src, rng, n=N_PANEL)
        gen[(ct, "MDLM")] = pca.project(x)
        rec[f"{ct}/MDLM"] = {**H.array_record(src, len(idx), idx, SEED),
                             "sampling_seed": MDLM_SEED}

    # stage three: the new CFGen column, drawn last
    for ct in H.TYPES3:
        src = cfgen_pool(ct)
        x, idx = H.take_panel(src, rng, n=N_PANEL)
        gen[(ct, "CFGen")] = pca.project(x)
        rec[f"{ct}/CFGen"] = {**H.array_record(src, len(idx), idx, SEED),
                              "sampling_seed": CFGEN_SEED,
                              "replaces": consumed[ct]["archived_source"],
                              "drawn_last": "so the other twelve panels keep v2's indices"}

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

    # self-check: the twelve non-CFGen panels are identical to the previous figure
    old = H.FIG / "fig_pca_baselines_all_v2.json"
    check = {"compared_against": str(old), "panels": {}, "n_same": 0, "n_diff": 0}
    if old.exists():
        o = H.read_json(old)["arrays"]
        for k, v in rec.items():
            if k.endswith("/CFGen"):
                continue
            a = o.get(k)
            same = bool(a) and all(a.get(f) == v.get(f)
                                   for f in ("sha256", "n_source", "n_plotted"))
            check["panels"][k] = {"identical_to_v2": same, "v2": a, "v3": v}
            check["n_same" if same else "n_diff"] += 1
    check["passed"] = check["n_diff"] == 0 and check["n_same"] == 12
    print(f"[self-check] non-CFGen panels identical to v2: "
          f"{check['n_same']}/12   passed={check['passed']}", flush=True)

    H.save(fig, "fig_pca_baselines_all_v3", {
        "figure": "PCA, five samplers x three requests (appendix, v3)", "ticket": 212,
        "supersedes": "fig_pca_baselines_all_v2 -- left untouched on disk",
        "types": H.TYPES3, "samplers": cols,
        "what_is_new_vs_v2": {
            "column": "CFGen", "sampling_seed": CFGEN_SEED,
            "was": {ct: consumed[ct]["archived_source"] for ct in H.TYPES3},
            "why": "section 5.2 says every table and figure reads the same delivered "
                   "cells; CFGen was still the archived single sample while every other "
                   "arm had moved to the ticket-210 / 208 deliveries",
            "rng_handling": "the archived CFGen draw is still CONSUMED at its v2 position "
                            "and discarded, and the new CFGen panels are drawn last, so "
                            "the other twelve panels keep v2's exact indices",
            "consumed_and_discarded": consumed},
        "n_panel": N_PANEL, "subsample_seed": SEED,
        "zero_training": True, "zero_sampling": True,
        "pca_gate": gate, "arrays": rec,
        "self_check_vs_v2": check,
        "cpu_threads": 2, "priority": prio,
    })
    H.copy_pdf_to_paper("fig_pca_baselines_all_v3", "hvg_pca_baselines_all_v3.pdf")
    if not check["passed"]:
        raise SystemExit("the non-CFGen panels did not reproduce v2 -- see the sidecar")


if __name__ == "__main__":
    main()
