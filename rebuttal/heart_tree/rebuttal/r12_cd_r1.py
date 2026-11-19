# -*- coding: utf-8 -*-
"""R12 CPU window item 2: R1 for the CountsDiff arm only (selected cells, train 20260625, samp 20261061).
Same functions and frozen objects as rebuttal.r1_fig7d / r1_clumps / r1_addendum, nothing refitted:
  Neuronal K256_g1p0 (2,500): kNN neuronal fraction in pca2 / pca50 / full (r1_fig7d.knn_stats, k=15, index = all 45,145 val),
    frozen-UMAP projection, region shares vs the frozen OURS clumps (main / C1 / elsewhere at R = 0.25, 0.5, 1.0),
    its own HDBSCAN(50, 25), CellTypist labels (analysis.score_purity.annotate, the purity ruler's call), labels inside C1.
  Endothelial K256 (10,057) and Myeloid K32 (2,500): CellTypist fine-label composition (top 8), real val of the target alongside.
1 thread. val only; test is not read.   cd Experiments/HVG && python -m rebuttal.r12_cd_r1
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from figures import hvgfig as H
from rebuttal.r1_fig7d import knn_stats, sha, K
OUT = ROOT / "rebuttal" / "out" / "R1"
CD = ROOT / "out" / "runs" / "rebuttal_CD_final"
RUNS = {"Neuronal": CD / "Neuronal_K256_g1p0_n2500" / "CountsDiff_Neuronal_K256_g1p0_n2500_train20260625_samp20261061.npy",
        "Myeloid": CD / "Myeloid_K32_g1p0_n2500" / "CountsDiff_Myeloid_K32_g1p0_n2500_train20260625_samp20261061.npy",
        "Endothelial": CD / "Endothelial_K256_g1p0_n10057" / "CountsDiff_Endothelial_K256_g1p0_n10057_train20260625_samp20261061_20261062.npy"}
t0 = time.time(); res = {"note": __doc__, "provenance": {t: {"path": str(p), "sha256": sha(p)} for t, p in RUNS.items()}}
xval, yval = H.real_cells(); xval = np.asarray(xval, dtype=np.float64)
types = sorted(set(yval.tolist())); y_int = np.array([types.index(t) for t in yval]); neu = types.index("Neuronal")
x = np.asarray(np.load(RUNS["Neuronal"]), dtype=np.float64); assert x.shape[0] == 2500
pca2 = H.FrozenPCA(); um = H.FrozenUMAP()
spaces = {"pca2": lambda c: pca2.project(c), "pca50": lambda c: (H.cpm_log1p(c) - um.mean) @ um.comp.T, "full": lambda c: H.cpm_log1p(c)}
res["knn"] = {}
for s, f in spaces.items():
    nn = NearestNeighbors(n_neighbors=K, algorithm="brute", n_jobs=1).fit(f(xval).astype(np.float32))
    st, _ = knn_stats(nn.kneighbors(f(x).astype(np.float32), return_distance=False), y_int, len(types), neu)
    res["knn"][s] = st; print(f"[knn] {s} mean_neuronal_frac={st['mean_neuronal_frac']:.4f} ({time.time()-t0:.0f}s)", flush=True)
p = um.project(x); print(f"[umap] ({time.time()-t0:.0f}s)", flush=True)
c = np.load(OUT / "r1_clumps.npz"); xy, cl = c["ours_xy"], c["clump"]
t_main, t_c1 = cKDTree(xy[cl == 0]), cKDTree(xy[cl == 1])
d0, d1 = t_main.query(p)[0], t_c1.query(p)[0]
res["regions"] = {f"R{R}": {"main": float((d0 <= R).mean()), "c1": float(((d1 <= R) & (d0 > R)).mean()),
                            "elsewhere": float(((d0 > R) & (d1 > R)).mean())} for R in (0.25, 0.5, 1.0)}
cc = HDBSCAN(min_cluster_size=50, min_samples=25).fit_predict(p)
res["own_hdbscan"] = [{"label": int(k), "share": float((cc == k).mean()), "ctr": p[cc == k].mean(0).round(2).tolist()}
                      for k in sorted(set(cc.tolist()), key=lambda k: -(cc == k).sum())]
from analysis.score_purity import annotate
genes = json.loads((H.DATA / "meta.json").read_text(encoding="utf-8"))["genes"]
def comp(lab, top=8):
    n = len(lab); return {a: round(b / n, 4) for a, b in Counter(lab.tolist()).most_common(top)}
lab = np.asarray(annotate(x, genes)).astype(str)
in_c1 = (d1 <= 0.5) & (d0 > 0.5)
res["celltypist"] = {"Neuronal": {"generated": comp(lab), "generated_in_C1": comp(lab[in_c1]) if in_c1.any() else {}, "n_in_C1": int(in_c1.sum())}}
print(f"[celltypist] Neuronal ({time.time()-t0:.0f}s)", flush=True)
for t in ("Endothelial", "Myeloid"):
    g = np.asarray(np.load(RUNS[t]), dtype=np.float64)
    res["celltypist"][t] = {"n": int(g.shape[0]), "generated": comp(np.asarray(annotate(g, genes)).astype(str)),
                            "real_val": comp(np.asarray(annotate(xval[yval == t], genes)).astype(str))}
    print(f"[celltypist] {t} ({time.time()-t0:.0f}s)", flush=True)
res["walltime_s"] = round(time.time() - t0, 1)
(ROOT / "rebuttal" / "out" / "R1" / "r12_cd_r1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("knn", "regions", "celltypist", "walltime_s")}, ensure_ascii=False)[:3000], flush=True)
