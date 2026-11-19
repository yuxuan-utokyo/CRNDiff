#!/usr/bin/env python
r"""R29 (k-NN neighbour fraction) and R30 (sparsity) on HLCA plasma cell + ionocyte, the heart R1 / R2 code
with only data and classes changed. Functions EDGES / prof / curve / compare (r2_sparsity.py) and knn_stats (r1_fig7d.py) are copied
verbatim from scripts/r12_extra/r1r2_metrics.py (da155341), as is the R2 reference/floor block (r2_block) and the R1 reference rows.
  R30: reference = every real eval-split cell of the target; floor = train cells of the target, 5 draws of n = min(n_ref, n_train), seed
       20261107 + s; per generated run: compare(run, reference).
  R29: index = every real eval-split cell (all classes) in pca50 (fit on HLCA TRAIN, runs/r29_30/hlca_pca50_train.npz); k = 15, brute;
       per generated cell the share of its 15 neighbours that are the target class; reference rows = real eval target cells (themselves
       excluded) and real train target cells.
Generated arrays: runs/r24_26_3seed/manifest.json (the v3 table's 3 seeds per method: CRNDiff gen 25/27/28, CFGen, CountsDiff, scVI, scANVI).
    python r29_30_hlca.py val|test OUT.json      (test: once, after val; logged in runs/r16/TEST_READ_LOG.tsv)
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
# ---- copied verbatim from r1r2_metrics.py (r2_sparsity.py / r1_fig7d.py)
EDGES = np.logspace(-4, 2, 25)
def prof(x):
    gm = x.mean(0); gz = (x == 0).mean(0)
    cz = (x == 0).mean(1); det = (x > 0).sum(1); lib = x.sum(1)
    return gm, gz, cz, det, lib
def curve(gm, gz):
    idx = np.digitize(gm, EDGES)
    return np.array([gz[idx == i].mean() if (idx == i).sum() >= 5 else np.nan for i in range(len(EDGES) + 1)])
def compare(x, ref):
    gm, gz, cz, det, lib = prof(x); rgm, rgz, rcz, rdet, rlib = ref
    c, rc = curve(gm, gz), curve(rgm, rgz)
    ok = np.isfinite(c) & np.isfinite(rc)
    return {"zero_overall": float((x == 0).mean()), "gene_zero_mae_vs_real": float(np.abs(gz - rgz).mean()),
            "gene_zero_pcc_vs_real": float(np.corrcoef(gz, rgz)[0, 1]),
            "curve_rmse_vs_real": float(np.sqrt(np.nanmean((c[ok] - rc[ok]) ** 2))),
            "cell_zero_mean": float(cz.mean()), "cell_zero_sd": float(cz.std(ddof=1)),
            "detected_mean": float(det.mean()), "detected_sd": float(det.std(ddof=1)),
            "detected_cv": float(det.std(ddof=1) / det.mean()),
            "lib_mean": float(lib.mean()), "lib_median": float(np.median(lib)), "lib_sd": float(lib.std(ddof=1))}, c
def knn_stats(nbr_idx, y_int, nt, neu):
    lab = y_int[nbr_idx]
    frac = (lab == neu).mean(1)
    comp = np.stack([(lab == t).mean() for t in range(nt)])
    return {"n": int(len(frac)), "mean_neuronal_frac": float(frac.mean()), "median_neuronal_frac": float(np.median(frac)),
            "share_frac_ge_0p5": float((frac >= 0.5).mean()), "share_frac_eq_0": float((frac == 0).mean()),
            "nn1_is_neuronal": float((lab[:, 0] == neu).mean()), "neighbour_type_share": comp.tolist()}, frac
def cp10k_log1p(x):
    x = np.asarray(x, dtype=np.float64); lib = x.sum(1, keepdims=True)
    return np.log1p(x / np.maximum(lib, 1) * 1e4)
# ---- data (only change vs heart)
ED = f"{W}/data/r16_hlca_score" if SPL == "val" else f"{W}/data/r16_hlca_test_view"
xe_all = np.load(f"{ED}/val.npy", mmap_mode="r"); ye = np.load(f"{ED}/val_celltype.npy", allow_pickle=True).astype(str)
xt_all = np.load(f"{W}/data/r16_hlca_fit/train.npy", mmap_mode="r"); yt = np.load(f"{W}/data/r16_hlca_fit/train_celltype.npy", allow_pickle=True).astype(str)
assert len(ye) == (46796 if SPL == "val" else 116989)
real = lambda X, y, t: np.asarray(X[np.flatnonzero(y == t)], dtype=np.float64)
BF = f"{W}/runs/r29_30/hlca_pca50_train.npz"; P = np.load(BF)
pca50 = lambda c: (cp10k_log1p(c).astype(np.float32) - P["mean"]) @ P["components"].T
MAN = f"{W}/runs/r24_26_3seed/manifest.json"; man = json.load(open(MAN))
t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)
res = {"split": SPL, "script_sha256": sha(__file__), "basis": {"file": BF.replace(W + "/", ""), "sha256": sha(BF)}, "manifest_sha256": sha(MAN), "r30": {}, "r29": {}, "runs": []}
# R30 reference / floor
for t in CL:
    xv, xt = real(xe_all, ye, t), real(xt_all, yt, t); ref = prof(xv); fl = []
    for s in range(5):
        i = np.random.default_rng(20261107 + s).choice(len(xt), min(len(xv), len(xt)), replace=False)
        fl.append(compare(xt[i], ref)[0])
    res["r30"][t] = {"real_eval": {"n": len(xv), **compare(xv, ref)[0]},
                     "floor_train_vs_eval": {k: {"mean": float(np.mean([f[k] for f in fl])), "min": float(np.min([f[k] for f in fl])), "max": float(np.max([f[k] for f in fl]))} for k in fl[0]}}
    res["r30"][t]["_ref"] = ref
log("R30 references done")
# R29 index
types = sorted(set(ye.tolist())); y_int = np.array([types.index(t) for t in ye])
idx = np.concatenate([pca50(np.asarray(xe_all[i:i + 20000])) for i in range(0, len(ye), 20000)]).astype(np.float32)
nn = NearestNeighbors(n_neighbors=K, algorithm="brute", n_jobs=8).fit(idx)
for t in CL:
    tid = types.index(t); own = np.flatnonzero(ye == t)
    ii = nn.kneighbors(idx[own], n_neighbors=K + 1, return_distance=False)
    keep = np.array([[j for j in row if j != o][:K] for row, o in zip(ii, own)])
    st_e, _ = knn_stats(keep, y_int, len(types), tid)
    st_t, _ = knn_stats(nn.kneighbors(pca50(real(xt_all, yt, t)).astype(np.float32), return_distance=False), y_int, len(types), tid)
    res["r29"][t] = {"real_eval_leave_one_out": st_e, "real_train": st_t}
log("R29 index and references done", len(types), "classes")
for e in man:
    t = e["target"]; x = np.load(e["path"]).astype(np.float64)
    r2, _ = compare(x, res["r30"][t]["_ref"])
    st, _ = knn_stats(nn.kneighbors(pca50(x).astype(np.float32), return_distance=False), y_int, len(types), types.index(t))
    res["runs"].append({"method": e["method"], "seed": e["seed"], "target": t, "path": e["path"].replace(W + "/", ""), "n": len(x), "r30": r2, "r29": st})
log("runs done", len(res["runs"]))
for t in CL: res["r30"][t].pop("_ref")
res["types"] = types
json.dump(res, open(OUT, "w"), indent=1); log("wrote", OUT, sha(OUT)[:8])
if SPL == "test":
    with open(f"{W}/runs/r16/TEST_READ_LOG.tsv", "a") as f:
        f.write(f"{time.strftime('%m-%d_%H:%M:%S')}\tHLCA R29 kNN + R30 sparsity test, 5 methods x 3 seeds (r29_30_hlca.py {sha(__file__)[:8]})\t{OUT.replace(W + '/', '')}\t{sha(OUT)[:8]}\n")
