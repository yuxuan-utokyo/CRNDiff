# -*- coding: utf-8 -*-
"""R1 on the Table 1 runs: filter the stored k-NN rows, and project each Table 1 run into the paper's
frozen UMAP to get region shares. Regions: C0/C1 from rebuttal.r1_clumps, R=0.5, unchanged.

    cd Experiments/HVG && python -m rebuttal.r1_table1
"""
import json, os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = os.environ.get("R_THREADS", "8")   # R12 CPU window: 1
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.dont_write_bytecode = True
from figures import hvgfig as H
from analysis.score_purity import annotate
from rebuttal.w2_runs import ROOT, ARMS, TABLE1, FIG7D_RUN, in_table1, run_list, load_cells
OUT = ROOT / "rebuttal/out/R1"
c = np.load(OUT / "r1_clumps.npz"); xy, cl = c["ours_xy"], c["clump"]
t0, t1 = cKDTree(xy[cl == 0]), cKDTree(xy[cl == 1])
num = json.loads((OUT / "r1_numbers.json").read_text(encoding="utf-8"))
um = H.FrozenUMAP(); genes = json.loads((H.DATA / "meta.json").read_text(encoding="utf-8"))["genes"]
res = {"runs": [], "knn": []}
for r in run_list("Neuronal"):
    t1run = in_table1(r["arm"], r["train"], r["samp"]); f7 = r["arm"] == "OURS" and (r["train"], r["samp"]) == FIG7D_RUN
    if not (t1run or f7):
        continue
    x = load_cells(r, "Neuronal"); u = um.project(x); d0, d1 = t0.query(u)[0], t1.query(u)[0]
    lab = annotate(x, genes); cnt = Counter(lab.tolist())
    q = {"arm": r["arm"], "train": r["train"], "samp": r["samp"], "in_table1": t1run, "is_fig7d": f7,
         "main": float((d0 <= .5).mean()), "c1": float(((d1 <= .5) & (d0 > .5)).mean()), "elsewhere": float(((d0 > .5) & (d1 > .5)).mean()),
         "NC1": cnt["NC1_glial"] / len(lab), "NC2": cnt["NC2_glial_NGF+"] / len(lab), "NC3": cnt["NC3_glial"] / len(lab),
         "top": [[a, b / len(lab)] for a, b in cnt.most_common(4)]}
    res["runs"].append(q); print(q, flush=True)
    for w in num["rows"]:
        if w["arm"] == r["arm"] and str(w["train"]) == str(r["train"]) and w["samp"] == r["samp"]:
            res["knn"].append({**w, "in_table1": t1run, "is_fig7d": f7})
res["refs"] = [w for w in num["rows"] if w["arm"].startswith("real")]
(OUT / "r1_table1.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
for s in ("pca50", "full", "pca2"):
    for a in ARMS:
        rr = [w for w in res["knn"] if w["arm"] == a and w["space"] == s and w["in_table1"]]
        print(s, a, len(rr), [round(w["mean_neuronal_frac"], 3) for w in rr], "nn1", [round(w["nn1_is_neuronal"], 3) for w in rr])
