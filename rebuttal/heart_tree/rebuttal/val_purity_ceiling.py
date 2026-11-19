# -*- coding: utf-8 -*-
"""Frozen-ruler purity of the REAL val cells of each target, all val cells of the type (no subsample):
analysis.score_purity.annotate (CellTypist 1.7.1, Healthy_Adult_Heart, gate-G5 identity check) + hvg.metrics.purity_from_hist
with the v7 and v8 fixed label sets; for Neuronal also the NC1-NC3 share. Reads val only (never test).
Check: Myeloid/Neuronal use all val cells in the archived celltypist_<T>_ceiling_canonical.json row "real val cells";
their label histograms must reproduce it.
    python -m rebuttal.val_purity_ceiling
"""
import json, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from hvg import config as C
from hvg.metrics import purity_from_hist
from analysis.score_purity import annotate, ruler_check
rl = ruler_check()
D = C.LEGACY_DATA / "all"; genes = json.loads((D / "meta.json").read_text(encoding="utf-8"))["genes"]
xv = np.load(D / "val.npy", mmap_mode="r"); yv = np.load(D / "val_celltype.npy", allow_pickle=True).astype(str)
lab = {v: {t: set(x["fixed_labels"]) for t, x in json.loads((C.ASSETS / "tables" / f).read_text(encoding="utf-8"))["types"].items()}
       for v, f in (("v7", "master_purity_v7.json"), ("v8", "master_purity_v8_lineage.json"))}
out = {"ruler": rl, "split": "val (all cells of the type, no subsample); test not read", "types": {}}
for t in ("Endothelial", "Myeloid", "Neuronal"):
    idx = np.flatnonzero(yv == t)
    h = annotate(np.asarray(xv[idx], dtype=np.float64), genes); u, c = np.unique(h, return_counts=True)
    hist = {str(k): int(v) for k, v in zip(u, c)}
    p7, _ = purity_from_hist(hist, lab["v7"][t]); p8, _ = purity_from_hist(hist, lab["v8"][t])
    nc = sum(v for k, v in hist.items() if k.startswith(("NC1", "NC2", "NC3"))) / len(idx)
    arch = json.loads((C.CELLTYPIST / f"celltypist_{t}_ceiling_canonical.json").read_text(encoding="utf-8"))["label_dists"]["real val cells"]
    same = (arch == hist) if sum(arch.values()) == len(idx) else None
    out["types"][t] = {"n_val_cells": int(len(idx)), "purity_v7": p7, "purity_v8": p8, "nc1_3_share": nc, "label_dist": hist,
                       "archive_n": sum(arch.values()), "reproduces_archive_hist": same}
    print(f"{t:<12} n={len(idx):<6} v7 {p7:.4f} v8 {p8:.4f} NC1-3 {nc:.4f} | archive n {sum(arch.values())} same-hist {same}", flush=True)
p = ROOT / "out/summary/rebuttal_val_purity_ceiling.json"; p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8"); print("[out]", p)
