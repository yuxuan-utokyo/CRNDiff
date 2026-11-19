#!/usr/bin/env python
r"""R36 neighbour fraction (R29 definition, pca50 train basis 067dc34d, k = 15, brute) per rep: index = every real eval cell EXCEPT the
target cells left out of the rep's subsample (so the target's reference in the index is the 80 % subsample); per generated run (v3
manifest) the mean share of its 15 neighbours that are the target. knn_stats / cp10k_log1p / pca50 as r29_30_hlca.py.
    python r36_knn.py val|test OUT.json
"""
import hashlib, json, os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "4"
import numpy as np
from sklearn.neighbors import NearestNeighbors
SPL, OUT = sys.argv[1:3]; W = "./work"; sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
assert not os.path.exists(OUT)
def cp10k_log1p(x):
    x = np.asarray(x, dtype=np.float64); lib = x.sum(1, keepdims=True); return np.log1p(x / np.maximum(lib, 1) * 1e4)
P = np.load(f"{W}/runs/r29_30/hlca_pca50_train.npz"); assert sha(f"{W}/runs/r29_30/hlca_pca50_train.npz")[:8] == "067dc34d"
pca50 = lambda c: (cp10k_log1p(c).astype(np.float32) - P["mean"]) @ P["components"].T
ED = f"{W}/data/r16_hlca_score" if SPL == "val" else f"{W}/data/r16_hlca_test_view"
X = np.load(f"{ED}/val.npy", mmap_mode="r"); Y = np.load(f"{ED}/val_celltype.npy", allow_pickle=True).astype(str)
Z = np.concatenate([pca50(np.asarray(X[i:i + 20000])) for i in range(0, len(Y), 20000)]).astype(np.float32)
man = json.load(open(f"{W}/runs/r24_26_3seed/manifest.json")); GZ = {(e["method"], e["seed"], e["target"]): pca50(np.load(e["path"])).astype(np.float32) for e in man}
res = {"split": SPL, "script_sha256": sha(__file__), "reps": {}}
for i in range(20):
    for t in ("plasma cell", "ionocyte"):
        s = np.load(f"{W}/runs/r36/subsets_{SPL}/{SPL}_{t.replace(' ', '')}_rep{i:02d}.npz"); keep = np.ones(len(Y), bool); keep[Y == t] = False; keep[s["idx"]] = True
        nn = NearestNeighbors(n_neighbors=15, algorithm="brute", n_jobs=4).fit(Z[keep]); yk = Y[keep]
        for (m, sd, tt), g in GZ.items():
            if tt != t: continue
            res["reps"].setdefault(f"{i:02d}", {}).setdefault(t, {})[f"{m}|{sd}"] = float((yk[nn.kneighbors(g, return_distance=False)] == t).mean())
    print("rep", i, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
