# -*- coding: utf-8 -*-
"""R1 step 2: find the dense groups of generated neuronal cells in the paper's frozen UMAP.

Reads rebuttal/out/R1/r1_cells.npz (written by rebuttal.r1_fig7d); no generated cell is
re-sampled. Rule fixed before looking at the result:
  HDBSCAN(min_cluster_size=50, min_samples=25) on the pooled UMAP coordinates of the 3 OURS
  sample seeds of the reference generator (train 20260625, n=7,500).
"Main neuronal region" = the clump holding the most real val neuronal cells, where a real
cell belongs to a clump if it lies within R=0.5 UMAP units of one of the clump's cells.

    cd Experiments/HVG && python -m rebuttal.r1_clumps
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import HDBSCAN

OUT = Path(__file__).resolve().parent / "out" / "R1"
R = 0.5
SEEDS = (20260987, 20260988, 20260989)


def main() -> None:
    z = np.load(OUT / "r1_cells.npz", allow_pickle=False)
    real = z["real_xy"]
    y = z["yval"].astype(str)
    types = z["types"].astype(str).tolist()
    is_neu = y == "Neuronal"
    xy = np.vstack([z[f"xy__OURS_20260625_{s}"] for s in SEEDS])
    seed_of = np.repeat(np.arange(3), 2500)
    frac50 = np.concatenate([z[f"frac__OURS_20260625_{s}__pca50"] for s in SEEDS])
    fracfull = np.concatenate([z[f"frac__OURS_20260625_{s}__full"] for s in SEEDS])
    lab = np.concatenate([z[f"label__OURS_20260625_{s}"].astype(str) for s in SEEDS])
    lab_real = z["label__real_val"].astype(str)
    frac_real50 = z["frac__real_val__pca50"]
    real_neu_xy = real[is_neu]

    c = HDBSCAN(min_cluster_size=50, min_samples=25).fit_predict(xy)
    tree_all = cKDTree(real)
    out = []
    for k in sorted(set(c.tolist())):
        m = c == k
        t = cKDTree(xy[m])
        d, _ = t.query(real, k=1)
        near = d <= R
        comp = Counter(y[near].tolist())
        n_near = int(near.sum())
        rn = near[is_neu]
        # 15 nearest real cells in the UMAP plane around the clump's cells
        _, ii = tree_all.query(xy[m], k=15)
        umap_comp = Counter(y[ii.ravel()].tolist())
        tot = sum(umap_comp.values())
        out.append({
            "clump": int(k), "label": "unclustered" if k < 0 else f"C{k}",
            "n_cells": int(m.sum()), "share_of_generated": float(m.mean()),
            "share_per_seed": [float((c[seed_of == i] == k).mean()) for i in range(3)],
            "centroid_umap": xy[m].mean(0).tolist(),
            "extent_umap_p5_p95": [np.percentile(xy[m], 5, axis=0).tolist(),
                                   np.percentile(xy[m], 95, axis=0).tolist()],
            "pca50_neuronal_frac_mean": float(frac50[m].mean()),
            "pca50_share_ge_0p5": float((frac50[m] >= 0.5).mean()),
            "full_neuronal_frac_mean": float(fracfull[m].mean()),
            "celltypist_top": [[a, b / int(m.sum())] for a, b in Counter(lab[m].tolist()).most_common(6)],
            "real_within_R_n": n_near,
            "real_within_R_types": {a: b / max(n_near, 1) for a, b in comp.most_common(6)},
            "real_neuronal_within_R_n": int(rn.sum()),
            "real_neuronal_within_R_share": float(rn.mean()),
            "umap_15nn_types": {a: b / tot for a, b in umap_comp.most_common(6)},
        })
    out.sort(key=lambda r: -r["n_cells"])

    # the same real cells, split by their own CellTypist label and by location
    in_clump = {}
    for r in out:
        if r["clump"] < 0:
            continue
        d, _ = cKDTree(xy[c == r["clump"]]).query(real_neu_xy, k=1)
        in_clump[r["label"]] = d <= R
    real_rows = {"n": int(is_neu.sum()),
                 "celltypist": {a: b / len(lab_real) for a, b in Counter(lab_real.tolist()).most_common(8)},
                 "pca50_frac_mean": float(frac_real50.mean())}
    for name, m in in_clump.items():
        real_rows[f"in_{name}"] = {
            "n": int(m.sum()),
            "celltypist": {a: b / max(int(m.sum()), 1) for a, b in Counter(lab_real[m].tolist()).most_common(5)},
            "pca50_frac_mean": float(frac_real50[m].mean()) if m.any() else None}

    # label distribution of every arm's UMAP run, per-cell ruler call
    arm_labels = {}
    for k in z.files:
        if k.startswith("label__"):
            v = z[k].astype(str)
            arm_labels[k[7:]] = {a: b / len(v) for a, b in Counter(v.tolist()).most_common(8)}

    np.savez_compressed(OUT / "r1_clumps.npz", ours_xy=xy, clump=c, seed_of=seed_of,
                        frac50=frac50, label=lab)
    (OUT / "r1_clumps.json").write_text(json.dumps({
        "rule": "HDBSCAN(min_cluster_size=50, min_samples=25) on pooled OURS UMAP; R=0.5",
        "clumps": out, "real_val_neuronal": real_rows, "per_cell_labels": arm_labels},
        ensure_ascii=False, indent=1), encoding="utf-8")
    for r in out:
        print(f"{r['label']:12s} n={r['n_cells']:5d} share={r['share_of_generated']:.3f} "
              f"seeds={[round(v, 3) for v in r['share_per_seed']]} "
              f"ctr={[round(v, 1) for v in r['centroid_umap']]} "
              f"f50={r['pca50_neuronal_frac_mean']:.2f} ffull={r['full_neuronal_frac_mean']:.2f} "
              f"realNeu={r['real_neuronal_within_R_n']}({r['real_neuronal_within_R_share']:.2f})")
        print("   labels", [(a, round(b, 2)) for a, b in r["celltypist_top"][:4]])
        print("   real@R", {a: round(b, 2) for a, b in list(r["real_within_R_types"].items())[:4]},
              "n", r["real_within_R_n"])
    print(json.dumps(real_rows, ensure_ascii=False)[:900])
    for k, v in arm_labels.items():
        print(k, {a: round(b, 3) for a, b in list(v.items())[:5]})


if __name__ == "__main__":
    main()
