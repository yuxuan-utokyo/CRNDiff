# -*- coding: utf-8 -*-
"""R1 sensitivity row: the excluded training seed 20260626 (not part of the 9-run means).
    cd Experiments/HVG && python -m rebuttal.r1_seed626
"""
import json, os, sys
os.environ["NUMBA_NUM_THREADS"] = "4"; os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from figures import hvgfig as H
OUT = ROOT / "rebuttal" / "out" / "R1"; RUNS = ROOT / "out" / "runs"
c = np.load(OUT / "r1_clumps.npz"); xy, cl = c["ours_xy"], c["clump"]
t0, t1 = cKDTree(xy[cl == 0]), cKDTree(xy[cl == 1])
um = H.FrozenUMAP(); xval, yval = H.real_cells(); xval = np.asarray(xval, dtype=np.float64)
idx = NearestNeighbors(n_neighbors=15, algorithm="brute", n_jobs=4).fit(((H.cpm_log1p(xval) - um.mean) @ um.comp.T).astype(np.float32))
neu = yval == "Neuronal"
STEM = "Neuronal_residual_hvgtwist_fk_tiltevery_trigany_K32_M5000_N2500_alpha1p0_J16_int8_th0p5_tau0p2_cc256_odds_seed20260987.npy"
paths = {20260987: RUNS / "table2_seeds_s20260626" / "Neuronal_s20260626" / STEM,
         20260988: RUNS / "table2_ours_s20260626_samp20260988" / "Neuronal" / "ours_Neuronal_train20260626_samp20260988.npy",
         20260989: RUNS / "table2_ours_s20260626_samp20260989" / "Neuronal" / "ours_Neuronal_train20260626_samp20260989.npy"}
d214 = {x["sample_seed"]: x for x in json.loads((ROOT / "out/summary/214_neuronal_dual_calibre.json").read_text(encoding="utf-8"))["cells"]
        if x["arm"] == "ours" and x["train_seed"] == 20260626}
rows = []
for s, p in paths.items():
    x = np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64)
    u = um.project(x); d0, d1 = t0.query(u)[0], t1.query(u)[0]
    ii = idx.kneighbors(((H.cpm_log1p(x) - um.mean) @ um.comp.T).astype(np.float32), return_distance=False)
    fr = neu[ii].mean(1)
    r = {"train": 20260626, "samp": s, "n": int(len(x)), "main": float((d0 <= .5).mean()), "c1": float(((d1 <= .5) & (d0 > .5)).mean()),
         "elsewhere": float(((d0 > .5) & (d1 > .5)).mean()), "pca50_mean_frac": float(fr.mean()), "nn1": float(neu[ii[:, 0]].mean()),
         "v8": d214[s]["appendix_only_lineage_purity"], "v7": d214[s]["frozen_purity"],
         "label_frac_top": dict(list(d214[s]["label_frac"].items())[:4])}
    rows.append(r); print(r)
(OUT / "r1_seed626.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
