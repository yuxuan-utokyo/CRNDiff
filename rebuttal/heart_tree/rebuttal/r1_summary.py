# -*- coding: utf-8 -*-
"""R1 step 3: region shares per arm, real train neuronal in the same regions, seed ranges.
Region = within R=0.5 UMAP units of a cell of OURS clump C0 (main) or C1 (fibroblast side).

    cd Experiments/HVG && python -m rebuttal.r1_summary
"""
import csv
import json
import os
from collections import Counter
from pathlib import Path

os.environ["NUMBA_NUM_THREADS"] = "8"; os.environ["OMP_NUM_THREADS"] = "8"
import numpy as np
from scipy.spatial import cKDTree

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from figures import hvgfig as H

OUT = ROOT / "rebuttal" / "out" / "R1"
R = 0.5


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


z = np.load(OUT / "r1_cells.npz")
c = np.load(OUT / "r1_clumps.npz")
xy, cl = c["ours_xy"], c["clump"]
t0, t1 = cKDTree(xy[cl == 0]), cKDTree(xy[cl == 1])
def region(p):
    d0, _ = t0.query(p, k=1); d1, _ = t1.query(p, k=1)
    return {"main": d0 <= R, "c1": (d1 <= R) & ~(d0 <= R), "elsewhere": (d0 > R) & (d1 > R)}

res = {"R": R, "arms": {}}
for k in z.files:
    if k.startswith("xy__"):
        m = region(z[k]); n = len(z[k])
        res["arms"][k[4:]] = {a: float(v.mean()) for a, v in m.items()}

y = z["yval"].astype(str)
vn = z["real_xy"][y == "Neuronal"]
m = region(vn)
res["real_val"] = {a: {"share": float(v.mean()), "n": int(v.sum()), "wilson95": wilson(int(v.sum()), len(vn))}
                   for a, v in m.items()}
don = np.load(H.DATA / "val_donor.npy", allow_pickle=True).astype(str)[y == "Neuronal"]
res["real_val_c1_donors"] = dict(Counter(don[m["c1"]].tolist()).most_common())
res["real_val_all_donors"] = dict(Counter(don.tolist()).most_common())

xtr = np.load(H.DATA / "train.npy", mmap_mode="r")
ytr = np.load(H.DATA / "train_celltype.npy", allow_pickle=True).astype(str)
ii = np.flatnonzero(ytr == "Neuronal")
um = H.FrozenUMAP()
pt = um.project(np.asarray(xtr[ii], dtype=np.float64))
m = region(pt)
res["real_train"] = {a: {"share": float(v.mean()), "n": int(v.sum()), "wilson95": wilson(int(v.sum()), len(pt))}
                     for a, v in m.items()}
np.save(OUT / "train_neuronal_umap.npy", pt)
dtr = np.load(H.DATA / "train_donor.npy", allow_pickle=True).astype(str)[ii]
res["real_train_c1_donors"] = dict(Counter(dtr[m["c1"]].tolist()).most_common())
res["real_train_all_donors"] = dict(Counter(dtr.tolist()).most_common())

num = json.loads((OUT / "r1_numbers.json").read_text(encoding="utf-8"))
tab = {}
for r in num["rows"]:
    tab.setdefault(r["space"], {}).setdefault(r["arm"], []).append(r)
res["knn"] = {}
for s, d in tab.items():
    res["knn"][s] = {}
    for a, rr in d.items():
        res["knn"][s][a] = {k: {"mean": float(np.mean([x[k] for x in rr])),
                                "min": float(np.min([x[k] for x in rr])),
                                "max": float(np.max([x[k] for x in rr])), "n_runs": len(rr)}
                            for k in ("mean_neuronal_frac", "median_neuronal_frac", "share_frac_ge_0p5",
                                      "share_frac_eq_0", "nn1_is_neuronal")}
rows = list(csv.DictReader(open(ROOT / "final results" / "all_runs_index.csv", encoding="utf-8-sig")))
t2 = [r for r in rows if r["table"] == "Table 2" and r["request"] == "Neuronal"]
res["table2_neuronal_purity"] = {}
for a in sorted(set(r["arm"] for r in t2)):
    v = [r for r in t2 if r["arm"] == a]
    res["table2_neuronal_purity"][a] = {
        "n": len(v), "abs_mean": float(np.mean([float(r["purity"]) for r in v])),
        "rel_mean": float(np.mean([float(r["purity_rel"]) for r in v]))}
(OUT / "r1_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("arms", "real_val", "real_train", "real_val_c1_donors", "real_train_c1_donors", "real_train_all_donors", "table2_neuronal_purity")}, ensure_ascii=False))
for s in res["knn"]:
    print(s, {a: {k: round(v["mean"], 3) for k, v in d.items()} for a, d in res["knn"][s].items()})
    print(s, "OURS ranges", {k: (round(v["min"], 3), round(v["max"], 3)) for k, v in res["knn"][s]["OURS"].items()})
