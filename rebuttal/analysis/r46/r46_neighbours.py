#!/usr/bin/env python
r"""R46: heart neighbour fraction for
all three targets, 50-D PCA only, test. Same recipe as Table 6's existing 50-D column = R19 test R1 (r1r2_metrics.py da155341,
--ref-split test --r1-basis train): index = all test cells (45,163) projected by the 50-D PCA fitted on train cells
(runs/r19/r1_bases_train.npz), k = 15 brute-force neighbours in float32, score per generated cell = share of its 15 test neighbours that
carry the target label. Functions knn_stats / cp10k_log1p / load_cells and the pca50 projection are copied from r1r2_metrics.py, with the
target class as a parameter instead of the hard-coded Neuronal. Runs: exactly the run entries of the R19 test jsons (CRNDiff 9 per
target, CountsDiff K128 3, MDLM / CFGen / scANVI / scVI 9 each), same thinning (N_OUT, seed 20261107).
Real cells row: the test cells of the target, each queried with K + 1 neighbours and itself removed (r1r2_metrics R1.refs, verbatim).
Gate (Neuronal only): every pca50 statistic of every run must equal the stored R19 test value exactly (max |diff| == 0).
    python -I r46_neighbours.py T OUT.json        (T in Endothelial, Myeloid, Neuronal; Neuronal is the gate and is run first)
"""
import hashlib, json, os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "4"
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors

W = Path.home() / "research/crndiff-rebuttal"; D = W / "data/hvg2k"; R19 = W / "runs/r19/test"
T, OUT = sys.argv[1], Path(sys.argv[2]); assert T in ("Endothelial", "Myeloid", "Neuronal") and not OUT.exists()
K = 15; N_OUT = {"Endothelial": 10057, "Myeloid": 2500, "Neuronal": 2500}; SUBSAMPLE_SEED = 20261107
R1K = ("mean_neuronal_frac", "median_neuronal_frac", "share_frac_ge_0p5", "share_frac_eq_0", "nn1_is_neuronal", "neighbour_type_share")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def knn_stats(nbr_idx, y_int, nt, neu):          # r1r2_metrics.py / r1_fig7d.py verbatim; "neu" = index of the target class
    lab = y_int[nbr_idx]
    frac = (lab == neu).mean(1)
    comp = np.stack([(lab == t).mean() for t in range(nt)])
    return {"n": int(len(frac)), "mean_neuronal_frac": float(frac.mean()), "median_neuronal_frac": float(np.median(frac)),
            "share_frac_ge_0p5": float((frac >= 0.5).mean()), "share_frac_eq_0": float((frac == 0).mean()),
            "nn1_is_neuronal": float((lab[:, 0] == neu).mean()), "neighbour_type_share": comp.tolist()}, frac


def cp10k_log1p(x):
    x = np.asarray(x, dtype=np.float64); lib = x.sum(1, keepdims=True)
    return np.log1p(x / np.maximum(lib, 1) * 1e4)


def load_cells(path, t, samp, train):
    x = np.load(path, mmap_mode="r"); n = N_OUT[t]
    if x.shape[0] > n:
        s = np.random.SeedSequence([SUBSAMPLE_SEED, samp, train]).generate_state(1)
        idx = np.sort(np.random.default_rng(int(s[0])).choice(x.shape[0], n, replace=False))
        return np.asarray(x[idx], dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


# run list = the R19 test json entries for this target (manifests as read in R19)
SRC = ["r1r2_crndiff_%s_test.json" % T, "r1r2_cd_k128_test.json", "r1r2_baselines_test.json"]
RUNS, STORED = [], {}
for f in SRC:
    for r in json.loads((R19 / f).read_text())["runs"]:
        if r["target"] != T: continue
        e = {k: r[k] for k in ("method", "arm", "train", "seed", "target", "path")}; RUNS.append(e)
        if r.get("r1"): STORED[(e["method"], str(e["train"]), e["seed"], e["arm"])] = r["r1"]["pca50"]
cnt = {m: sum(e["method"] == m for e in RUNS) for m in ("OURS", "CountsDiff", "MDLM", "CFGen", "scANVI", "scVI")}
assert cnt == {"OURS": 9, "CountsDiff": 3, "MDLM": 9, "CFGen": 9, "scANVI": 9, "scVI": 9}, cnt
if T == "Neuronal": assert len(STORED) == len(RUNS) == 48, len(STORED)

t0 = time.time()
xs = np.load(D / "test.npy", mmap_mode="r"); ys = np.load(D / "test_celltype.npy", allow_pickle=True).astype(str)
p50 = np.load(W / "runs/r19/r1_bases_train.npz")
proj = lambda c: (cp10k_log1p(c).astype(np.float32) - p50["mean"]) @ p50["components"].T
types = sorted(set(ys.tolist())); y_int = np.array([types.index(t) for t in ys]); tgt = types.index(T)
index = proj(np.asarray(xs, dtype=np.float64)).astype(np.float32)
nn = NearestNeighbors(n_neighbors=K, algorithm="brute", n_jobs=4).fit(index)
mask = ys == T; own = np.flatnonzero(mask)
ii = nn.kneighbors(index[mask], n_neighbors=K + 1, return_distance=False)
keep = np.array([[j for j in row if j != o][:K] for row, o in zip(ii, own)])
real = knn_stats(keep, y_int, len(types), tgt)[0]
print(f"[real] {T} n={real['n']} mean frac {real['mean_neuronal_frac']:.4f} ({time.time() - t0:.0f}s)", flush=True)
rows, gate = [], []
for e in RUNS:
    x = load_cells(e["path"], T, e["seed"], e.get("train", 0))          # as r1r2_metrics.py main loop
    st = knn_stats(nn.kneighbors(proj(x).astype(np.float32), return_distance=False), y_int, len(types), tgt)[0]
    rows.append({**e, "sha256": sha(e["path"])[:16], "pca50": st})
    key = (e["method"], str(e["train"]), e["seed"], e["arm"])
    if key in STORED:
        gate.append(max(float(np.abs(np.asarray(st[k], dtype=float) - np.asarray(STORED[key][k], dtype=float)).max()) for k in R1K))
    print(f"  {e['method']:10s} {str(e['train']):9s} {e['seed']} n {st['n']:6d} frac {st['mean_neuronal_frac']:.4f}"
          + (f"  gate {gate[-1]:.1e}" if key in STORED else ""), flush=True)
g = max(gate) if gate else None
if T == "Neuronal": assert len(gate) == 48 and g == 0.0, ("Neuronal gate vs R19 test", len(gate), g)
OUT.write_text(json.dumps({"split": "test", "space": "pca50 (train-fit, runs/r19/r1_bases_train.npz)", "k": K, "target": T, "types": types,
                           "n_test": int(len(ys)), "n_test_target": int(mask.sum()), "real_cells_loo": real, "runs": rows,
                           "gate_vs_r19_max_abs_diff": g, "gate_n_runs": len(gate), "sources": {f: sha(R19 / f)[:8] for f in SRC},
                           "bases_sha256": sha(W / "runs/r19/r1_bases_train.npz")[:8], "test_sha256": sha(D / "test.npy")[:8],
                           "test_labels_sha256": sha(D / "test_celltype.npy")[:8], "script_sha256": sha(__file__), "walltime_s": time.time() - t0}, indent=1))
print("wrote", OUT, "gate", g, f"({time.time() - t0:.0f}s)")
