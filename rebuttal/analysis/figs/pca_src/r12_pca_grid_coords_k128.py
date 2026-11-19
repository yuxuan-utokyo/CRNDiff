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
OUTD = ROOT / "rebuttal" / "out" / "R12" / "pca_grid_k128_coords"   # coords copy (vice 18:4x): no drawing, Mac redraws
CD = ROOT / "out" / "runs" / "rebuttal_CD_K128_s20260625"   # coords copy: CountsDiff column = K128 g1.0, train 20260625
CD_FILES = {"Endothelial": CD / "Endothelial_K128_g1p0_n10057" / "CountsDiff_Endothelial_K128_g1p0_n10057_train20260625_samp20261061_20261062.npy",
            "Myeloid": CD / "Myeloid_K128_g1p0_n2500" / "CountsDiff_Myeloid_K128_g1p0_n2500_train20260625_samp20261061.npy",
            "Neuronal": CD / "Neuronal_K128_g1p0_n2500" / "CountsDiff_Neuronal_K128_g1p0_n2500_train20260625_samp20261061.npy"}


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

    # ---- coords copy (vice 18:4x): the v3 self-check (unchanged code), then dump every panel's 2-D coordinates; no drawing ----
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
    if not check["passed"]:
        raise SystemExit("v3 panels did not reproduce -- not dumping")
    OUTD.mkdir(parents=True, exist_ok=True)
    arrs = {"real_xy": zr[:, :2], "real_type": np.asarray(yval).astype(str)}
    for (ct, m), g in gen.items():
        arrs[f"gen__{ct}__{m}"] = g[:, :2]
    np.savez_compressed(OUTD / "pca_grid_coords_k128.npz", **arrs)
    (OUTD / "pca_grid_coords_k128.json").write_text(json.dumps({
        "what": "PCA grid (fig:hvg-pca-grid v3 + CountsDiff column) per-panel 2-D coordinates; CountsDiff = K128 g1.0 train 20260625",
        "npz_keys": {"real_xy": "frozen atlas PCA of all val cells (background)", "real_type": "val cell type (target mask = real_type == row)",
                     "gen__<row>__<column>": "2-D coords of the 2,500 plotted generated cells"},
        "rows": H.TYPES3, "columns_in_v3_order": cols, "column_order_requested_18_50": ["scVI", "CFGen", "scANVI", "MDLM", "CountsDiff", "Ours"],
        "column_title": {"Ours": "CRNDiff (bold)"},
        "style_from_r12_pca_grid": {"COL_BG": COL_BG, "COL_TARGET": COL_TARGET, "COL_GEN": COL_GEN, "bg_s": 0.35, "target_s": 0.8, "gen_s": 0.7,
                                    "gen_alpha": 0.5, "figsize_6col": [6.6, 3.42], "title_fs": 6.8, "legend_fs_old": 6.4,
                                    "legend_labels": ["Real target cells", "Generated cells"], "rasterized": True, "dpi": 400},
        "n_panel": N_PANEL, "subsample_seed": SEED, "pca_gate": gate, "arrays": rec, "self_check_vs_v3": check, "priority": prio},
        indent=1, default=str), encoding="utf-8")
    print("[out]", OUTD, flush=True)


if __name__ == "__main__":
    main()
