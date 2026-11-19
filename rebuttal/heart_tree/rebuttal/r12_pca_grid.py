# -*- coding: utf-8 -*-
"""R12 CPU window item 6: PCA grid (fig:hvg-pca-grid, v3 = figures/fig_pca_baselines_v3.py) plus ONE
CountsDiff column. v3 and the paper pdf are not touched; output goes only to rebuttal/out/R12/pca_grid/.
Same handling as v3 used for its new column: every v3 draw happens first in v3's exact order (stages 1-3, code copied from v3's
main), then the CountsDiff column is drawn LAST (stage 4) from the CountsDiff final arrays, so the 15 v3 panels keep v3's indices;
the script checks them against v3's sidecar (fig_pca_baselines_all_v3.json). Frozen atlas PCA, 2,500 cells per panel, val only.
    cd Experiments/HVG && python -m rebuttal.r12_pca_grid
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = os.environ.get("R_THREADS", "2")
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from figures import hvgfig as H
from figures.fig_pca_baselines import (BASELINES, COL_BG, COL_GEN, COL_TARGET, N_PANEL, REGISTRY, SEED)
from figures.fig_pca_baselines_v2 import MDLM_SEED, mdlm_pool, translate
from figures.fig_pca_baselines_v3 import CFGEN_SEED, cfgen_pool, low_priority
OUTD = ROOT / "rebuttal" / "out" / "R12" / "pca_grid"
CD = ROOT / "out" / "runs" / "rebuttal_CD_final"
CD_FILES = {"Endothelial": CD / "Endothelial_K256_g1p0_n10057" / "CountsDiff_Endothelial_K256_g1p0_n10057_train20260625_samp20261061_20261062.npy",
            "Myeloid": CD / "Myeloid_K32_g1p0_n2500" / "CountsDiff_Myeloid_K32_g1p0_n2500_train20260625_samp20261061.npy",
            "Neuronal": CD / "Neuronal_K256_g1p0_n2500" / "CountsDiff_Neuronal_K256_g1p0_n2500_train20260625_samp20261061.npy"}


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

    old_cols = BASELINES + ["Ours"]             # v2 four-column order, unchanged
    cols = BASELINES + ["MDLM", "Ours", "CountsDiff"]   # v3 five columns + CountsDiff
    rng = np.random.default_rng(SEED)
    gen, rec, consumed = {}, {}, {}

    # ---- stage 1: the 12 old panels of v2, in exactly the same order; the CFGen cell is consumed but not used ----
    for ct in H.TYPES3:
        rows = reg[ct]["rows"]
        for m in old_cols:
            if m == "CFGen":
                old_src = translate(rows["CFGen"]["source"])
                _, idx = H.take_panel(old_src, rng, n=N_PANEL)      # consume this batch of random numbers
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

    # ---- stage 2: the MDLM column, same position as v2 ----
    for ct in H.TYPES3:
        src = mdlm_pool(ct)
        x, idx = H.take_panel(src, rng, n=N_PANEL)
        gen[(ct, "MDLM")] = pca.project(x)
        rec[f"{ct}/MDLM"] = {**H.array_record(src, len(idx), idx, SEED),
                             "sampling_seed": MDLM_SEED}

    # ---- stage 3: the new CFGen column, drawn last ----
    for ct in H.TYPES3:
        src = cfgen_pool(ct)
        x, idx = H.take_panel(src, rng, n=N_PANEL)
        gen[(ct, "CFGen")] = pca.project(x)
        rec[f"{ct}/CFGen"] = {**H.array_record(src, len(idx), idx, SEED),
                              "sampling_seed": CFGEN_SEED,
                              "replaces": consumed[ct]["archived_source"],
                              "drawn_last": "so the other twelve panels keep v2's indices"}

    # ---- stage 4 (R12): CountsDiff column, drawn last ----
    for ct in H.TYPES3:
        src = CD_FILES[ct]
        x, idx = H.take_panel(src, rng, n=N_PANEL)
        gen[(ct, "CountsDiff")] = pca.project(x)
        rec[f"{ct}/CountsDiff"] = {**H.array_record(src, len(idx), idx, SEED), "sampling_seed": 20261061, "drawn_last": True}

    def draw(cols, width):
        fig, axes = plt.subplots(3, len(cols), figsize=(width, 3.42), sharex=True, sharey=True)
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
                    ax.set_title("CRNDiff" if m == "Ours" else m, fontsize=6.8, pad=2,
                                 fontweight="bold" if m == "Ours" else "normal")
                if j == 0:
                    ax.set_ylabel(ct, fontsize=6.8, labelpad=2)

        h = [plt.Line2D([0], [0], ls="none", marker="o", ms=2.6, color=COL_TARGET),
             plt.Line2D([0], [0], ls="none", marker="o", ms=2.6, color=COL_GEN)]
        fig.legend(h, ["Real target cells", "Generated cells"],
                   loc="lower center", bbox_to_anchor=(.5, -.022), ncol=2, frameon=False,
                   fontsize=6.4, handletextpad=.3, columnspacing=1.6, borderpad=0.)
        return fig

    # v3 replica: the same five columns, the same savefig call as hvgfig.save -> png sha must equal v3's sidecar
    OUTD.mkdir(parents=True, exist_ok=True)
    f5 = draw(BASELINES + ["MDLM", "Ours"], 5.5)
    p5 = OUTD / "v3_replica.png"; f5.savefig(p5, dpi=400, bbox_inches="tight", facecolor="#ffffff"); plt.close(f5)
    v3png = H.read_json(H.FIG / "fig_pca_baselines_all_v3.json")["outputs"]["png"]["sha256"]
    replica = {"v3_png_sha256": v3png, "replica_png_sha256": H.sha256(p5), "identical": H.sha256(p5) == v3png}
    print(f"[replica] v3 png identical: {replica['identical']}", flush=True)
    fig = draw(cols, 6.6)

    # ---- self-check: the 15 v3 panels identical to v3's sidecar ----
    old = H.FIG / "fig_pca_baselines_all_v3.json"
    o = H.read_json(old)["arrays"]; check = {"compared_against": str(old), "n_same": 0, "n_diff": 0, "diff": []}
    for k, v in rec.items():
        if k.endswith("/CountsDiff"):
            continue
        a = o.get(k); same = bool(a) and all(a.get(f) == v.get(f) for f in ("sha256", "n_source", "n_plotted") if f in a or f in v)
        same = same and all(a.get(f) == v.get(f) for f in a if "idx" in f or "index" in f)
        check["n_same" if same else "n_diff"] += 1
        if not same: check["diff"].append(k)
    check["passed"] = check["n_diff"] == 0 and check["n_same"] == 15
    print(f"[self-check] v3 panels identical: {check['n_same']}/15 passed={check['passed']}", flush=True)
    OUTD.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTD / "hvg_pca_baselines_all_r12_countsdiff.pdf", dpi=400, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(OUTD / "hvg_pca_baselines_all_r12_countsdiff.png", dpi=400, bbox_inches="tight", facecolor="#ffffff")
    (OUTD / "hvg_pca_baselines_all_r12_countsdiff.json").write_text(json.dumps({
        "figure": "PCA grid v3 + CountsDiff column (rebuttal only; paper pdf untouched)", "samplers": cols,
        "n_panel": N_PANEL, "subsample_seed": SEED, "pca_gate": gate, "arrays": rec, "self_check_vs_v3": check, "v3_replica_png": replica,
        "priority": prio}, indent=1, default=str), encoding="utf-8")
    print("[out]", OUTD, flush=True)
    if not check["passed"]:
        raise SystemExit("v3 panels did not reproduce -- see the sidecar")


if __name__ == "__main__":
    main()
