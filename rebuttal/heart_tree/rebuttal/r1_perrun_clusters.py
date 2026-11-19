# -*- coding: utf-8 -*-
"""R1: per-run HDBSCAN over all 9 OURS runs, keeping the noise label apart from real clusters.
For every real cluster: its share, centre, real val cell types within 0.5 UMAP units, and the
share of real train neuronal cells located there.

    cd Experiments/HVG && python -m rebuttal.r1_perrun_clusters
"""
import json, os, sys
os.environ["NUMBA_NUM_THREADS"] = "4"; os.environ["OMP_NUM_THREADS"] = "4"
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import HDBSCAN
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from figures import hvgfig as H
from rebuttal.r1_fig7d import run_list, load_cells, tag
OUT = ROOT / "rebuttal" / "out" / "R1"
z = np.load(OUT / "r1_cells.npz")
real, y = z["real_xy"], z["yval"].astype(str)
tr = np.load(OUT / "train_neuronal_umap.npy")
um = H.FrozenUMAP()
res = []
for r in [x for x in run_list() if x["arm"] == "OURS"]:
    p = um.project(load_cells(r))
    c = HDBSCAN(min_cluster_size=50, min_samples=25).fit_predict(p)
    row = {"run": tag(r), "noise_share": float((c == -1).mean()), "clusters": []}
    for k in sorted(set(c[c >= 0].tolist()), key=lambda k: -(c == k).sum()):
        m = c == k; t = cKDTree(p[m])
        near = t.query(real)[0] <= 0.5
        cm = Counter(y[near].tolist()); nn = max(int(near.sum()), 1)
        row["clusters"].append({"share": float(m.mean()), "ctr": [round(float(v), 2) for v in p[m].mean(0)],
                                "real_types": {a: round(b / nn, 2) for a, b in cm.most_common(3)},
                                "train_neuronal_share": float((t.query(tr)[0] <= 0.5).mean())})
    res.append(row)
    print(row["run"], "noise", round(row["noise_share"], 4))
    for q in row["clusters"]:
        print("   ", round(q["share"], 4), q["ctr"], q["real_types"], "train_neu", round(q["train_neuronal_share"], 4))
(OUT / "r1_perrun_clusters.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
