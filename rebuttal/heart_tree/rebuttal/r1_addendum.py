# -*- coding: utf-8 -*-
"""R1 addendum requested 11-07: radius sensitivity, all 9 OURS runs, purity lookups, donor base rates.
    cd Experiments/HVG && python -m rebuttal.r1_addendum
"""
import glob, json, os, re, sys
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
z = np.load(OUT / "r1_cells.npz"); c = np.load(OUT / "r1_clumps.npz")
xy, cl = c["ours_xy"], c["clump"]
y = z["yval"].astype(str); vn = z["real_xy"][y == "Neuronal"]; tr = np.load(OUT / "train_neuronal_umap.npy")
t0, t1 = cKDTree(xy[cl == 0]), cKDTree(xy[cl == 1])
def reg(p, R):
    d0 = t0.query(p)[0]; d1 = t1.query(p)[0]
    return {"main": float((d0 <= R).mean()), "c1": float(((d1 <= R) & (d0 > R)).mean()),
            "elsewhere": float(((d0 > R) & (d1 > R)).mean())}
res = {"radius": {str(R): {"real_val": reg(vn, R), "real_train": reg(tr, R), "ours_refgen_pooled": reg(xy, R)}
                  for R in (0.25, 0.5, 1.0)}}
um = H.FrozenUMAP()
per = []
for r in [x for x in run_list() if x["arm"] == "OURS"]:
    p = um.project(load_cells(r))
    a = {"train": r["train"], "samp": r["samp"], **{f"R{R}": reg(p, R) for R in (0.25, 0.5, 1.0)}}
    cc = HDBSCAN(min_cluster_size=50, min_samples=25).fit_predict(p)
    a["own_hdbscan"] = [{"share": float((cc == k).mean()), "ctr": p[cc == k].mean(0).round(2).tolist()}
                        for k in sorted(set(cc.tolist()), key=lambda k: -(cc == k).sum())]
    per.append(a); print(tag(r), a["R0.5"], a["own_hdbscan"], flush=True)
res["ours_9_runs"] = per
v = np.array([a["R0.5"]["c1"] for a in per]); m = np.array([a["R0.5"]["main"] for a in per])
res["ours_9_summary"] = {"c1_mean": float(v.mean()), "c1_min": float(v.min()), "c1_max": float(v.max()),
                         "main_mean": float(m.mean()), "main_min": float(m.min()), "main_max": float(m.max())}
for tn in (20260625, 20260627, 20260630):
    w = [a["R0.5"]["c1"] for a in per if a["train"] == tn]
    res["ours_9_summary"][f"c1_train{tn}"] = [round(x, 4) for x in w]
# donor base rate
ytr = np.load(H.DATA / "train_celltype.npy", allow_pickle=True).astype(str)
don = np.load(H.DATA / "train_donor.npy", allow_pickle=True).astype(str)[ytr == "Neuronal"]
HD = ["H2", "H3", "H5", "H6", "H7"]
c1 = (t1.query(tr)[0] <= .5) & (t0.query(tr)[0] > .5)
res["donors"] = {"base_share_5H": float(np.isin(don, HD).mean()), "c1_share_5H": float(np.isin(don[c1], HD).mean()),
                 "n_c1": int(c1.sum()), "n_all": int(len(don)),
                 "per_donor": {d: {"n": int((don == d).sum()), "n_c1": int(((don == d) & c1).sum()),
                                   "share_in_c1": float(((don == d) & c1).sum() / (don == d).sum())}
                               for d in sorted(set(don.tolist()))}}
# purity lookups (read only)
v8 = json.loads((ROOT / "assets/tables/master_purity_v8_lineage.json").read_text(encoding="utf-8"))
res["v8_neuronal"] = {"fixed_labels": v8["types"]["Neuronal"]["fixed_labels"], "calibration": v8.get("calibration", {}).get("Neuronal")}
hits = []
for f in glob.glob(str(ROOT / "out" / "summary" / "*.json")):
    try:
        s = open(f, encoding="utf-8").read()
    except Exception:
        continue
    if re.search(r'"?0\.97[01]\d*"?', s) and "Neuronal" in s:
        hits.append(os.path.basename(f))
res["files_with_0p970_0p971_and_Neuronal"] = sorted(hits)[:40]
(OUT / "r1_addendum.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("radius", "ours_9_summary", "donors", "v8_neuronal", "files_with_0p970_0p971_and_Neuronal")}, ensure_ascii=False))
