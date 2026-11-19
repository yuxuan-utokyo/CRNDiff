#!/usr/bin/env python
r"""R29 in the two other heart R1 spaces.
Same code as r29_30_hlca.py (e308b3d9): knn_stats / cp10k_log1p copied verbatim, k = 15, brute, index = every real eval-split cell (all
classes), reference rows = real eval target cells (themselves excluded) and real train target cells, generated arrays = the v3 manifest.
Spaces as in the heart r1r2_metrics.py --r1-basis train: pca2 = (cp10k_log1p - mean_) @ comp2.T with the heart pca2 recipe fit on HLCA train
(runs/r28/hlca_pca2_train.npz, the R28 basis); full = cp10k_log1p on the 2,000 genes (float64, as the heart).
    python r29_spaces_hlca.py val|test OUT.json      (test: once, after val; logged in runs/r16/TEST_READ_LOG.tsv)
"""
import hashlib, json, os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "8"
import numpy as np
from sklearn.neighbors import NearestNeighbors
SPL, OUT = sys.argv[1:3]; assert SPL in ("val", "test")
W = "./work"; sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
assert not os.path.exists(OUT), f"{OUT} exists; each split is read once"
if SPL == "test": assert os.path.exists(OUT.replace("_test", "_val")), "val first"
K = 15; CL = ["plasma cell", "ionocyte"]
def knn_stats(nbr_idx, y_int, nt, neu):     # verbatim (r29_30_hlca.py / r1r2_metrics.py)
    lab = y_int[nbr_idx]
    frac = (lab == neu).mean(1)
    comp = np.stack([(lab == t).mean() for t in range(nt)])
    return {"n": int(len(frac)), "mean_neuronal_frac": float(frac.mean()), "median_neuronal_frac": float(np.median(frac)),
            "share_frac_ge_0p5": float((frac >= 0.5).mean()), "share_frac_eq_0": float((frac == 0).mean()),
            "nn1_is_neuronal": float((lab[:, 0] == neu).mean()), "neighbour_type_share": comp.tolist()}, frac
def cp10k_log1p(x):
    x = np.asarray(x, dtype=np.float64); lib = x.sum(1, keepdims=True)
    return np.log1p(x / np.maximum(lib, 1) * 1e4)
ED = f"{W}/data/r16_hlca_score" if SPL == "val" else f"{W}/data/r16_hlca_test_view"
xe_all = np.load(f"{ED}/val.npy", mmap_mode="r"); ye = np.load(f"{ED}/val_celltype.npy", allow_pickle=True).astype(str)
xt_all = np.load(f"{W}/data/r16_hlca_fit/train.npy", mmap_mode="r"); yt = np.load(f"{W}/data/r16_hlca_fit/train_celltype.npy", allow_pickle=True).astype(str)
assert len(ye) == (46796 if SPL == "val" else 116989)
real = lambda X, y, t: np.asarray(X[np.flatnonzero(y == t)], dtype=np.float64)
BF = f"{W}/runs/r28/hlca_pca2_train.npz"; P2 = np.load(BF)
SP = {"pca2": lambda c: (cp10k_log1p(c) - P2["mean_"]) @ P2["comp2"].T, "full": lambda c: cp10k_log1p(c)}
MAN = f"{W}/runs/r24_26_3seed/manifest.json"; man = json.load(open(MAN))
t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)
res = {"split": SPL, "script_sha256": sha(__file__), "pca2_basis": {"file": BF.replace(W + "/", ""), "sha256": sha(BF)}, "manifest_sha256": sha(MAN), "r29": {}, "runs": []}
types = sorted(set(ye.tolist())); y_int = np.array([types.index(t) for t in ye])
for sp, f in SP.items():
    idx = np.concatenate([f(np.asarray(xe_all[i:i + 10000])) for i in range(0, len(ye), 10000)])
    nn = NearestNeighbors(n_neighbors=K, algorithm="brute", n_jobs=8).fit(idx)
    for t in CL:
        tid = types.index(t); own = np.flatnonzero(ye == t)
        ii = nn.kneighbors(idx[own], n_neighbors=K + 1, return_distance=False)
        keep = np.array([[j for j in row if j != o][:K] for row, o in zip(ii, own)])
        st_e, _ = knn_stats(keep, y_int, len(types), tid)
        st_t, _ = knn_stats(nn.kneighbors(f(real(xt_all, yt, t)), return_distance=False), y_int, len(types), tid)
        res["r29"].setdefault(t, {})[sp] = {"real_eval_leave_one_out": st_e, "real_train": st_t}
    log(sp, "index and references done")
    for j, e in enumerate(man):
        x = np.load(e["path"]).astype(np.float64); t = e["target"]
        st, _ = knn_stats(nn.kneighbors(f(x), return_distance=False), y_int, len(types), types.index(t))
        if sp == "pca2": res["runs"].append({"method": e["method"], "seed": e["seed"], "target": t, "path": e["path"].replace(W + "/", ""), "n": len(x), "r29": {}})
        assert res["runs"][j]["path"] == e["path"].replace(W + "/", ""); res["runs"][j]["r29"][sp] = st
    log(sp, "runs done"); del idx, nn
res["types"] = types
json.dump(res, open(OUT, "w"), indent=1); log("wrote", OUT, sha(OUT)[:8])
if SPL == "test":
    with open(f"{W}/runs/r16/TEST_READ_LOG.tsv", "a") as f:
        f.write(f"{time.strftime('%m-%d_%H:%M:%S')}\tHLCA R29 kNN in pca2 + full genes test, 5 methods x 3 seeds (r29_spaces_hlca.py {sha(__file__)[:8]})\t{OUT.replace(W + '/', '')}\t{sha(OUT)[:8]}\n")
