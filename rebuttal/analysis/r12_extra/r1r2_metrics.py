#!/usr/bin/env python
"""R1 k-NN neighbour fraction and R2 sparsity statistics on Mac, for rows added after the Windows run.

Functions copied unchanged from Windows' rebuttal code (from_windows/rebuttal_code/rebuttal):
  R2: prof, curve, compare, EDGES (r2_sparsity.py); real reference = val cells of the target, raw counts;
      floor = train cells of the target, 5 draws of n = min(n_val, n_train), seed 20261107 + s.
  R1: knn_stats (r1_fig7d.py); k = 15; index = all real val cells (45,145) in three spaces on log1p(CP10K):
      pca2 = frozen atlas basis (frozen_atlas_pca_basis.npz), pca50 = pca50_val.npz, full = 2,000 genes.
      Neuronal only, as in R1.
Not ported (Windows only): the frozen UMAP (clump share) and the HVG2K CellTypist labels (subtype shares).
Gate (--gate): the real-cell rows of R1 and R2 and the OURS Fig 7d run (train 20260625, samp 20260987) must reproduce
results_for_check/r1_numbers.json and r2_numbers.json. Val only; test is not read.

    python -I r1r2_metrics.py --gate
    python -I r1r2_metrics.py --manifest M.json --out OUT.json     # M: [{"method","arm","seed","target","path"}]
"""
import argparse, json, os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "4"
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors

W = Path.home() / "research/crndiff-rebuttal"; D = W / "data/hvg2k"; A = W / "from_windows/rebuttal_code/assets"
REF = W / "from_windows/rebuttal_code/results_for_check"
# R20 11-10: the real reference split is a parameter (--ref-split val|test, default val); read before argparse because the reference
# arrays load at module level. The --gate path compares against val-era numbers and is only meaningful with val.
RS = os.sys.argv[os.sys.argv.index("--ref-split") + 1] if "--ref-split" in os.sys.argv else "val"; assert RS in ("val", "test"), RS
# R19 11-10: --r1-basis train uses R1 PCA spaces refit on train cells (runs/r19/r1_bases_train.npz, built by
# scripts/r12_extra/r1_bases_train.py with the original recipes); default val keeps the original val-fit spaces unchanged.
RB = os.sys.argv[os.sys.argv.index("--r1-basis") + 1] if "--r1-basis" in os.sys.argv else "val"; assert RB in ("val", "train"), RB
K = 15; N_OUT = {"Endothelial": 10057, "Myeloid": 2500, "Neuronal": 2500}; SUBSAMPLE_SEED = 20261107

# ---- copied from r2_sparsity.py
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

# ---- copied from r1_fig7d.py
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


def load_cells(path, t, samp, train):          # w2_runs.load_cells: thin only when a delivery is larger than N_OUT
    x = np.load(path, mmap_mode="r"); n = N_OUT[t]
    if x.shape[0] > n:
        s = np.random.SeedSequence([SUBSAMPLE_SEED, samp, train]).generate_state(1)
        idx = np.sort(np.random.default_rng(int(s[0])).choice(x.shape[0], n, replace=False))
        return np.asarray(x[idx], dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


xv_all = np.load(D / f"{RS}.npy", mmap_mode="r"); yv = np.load(D / f"{RS}_celltype.npy", allow_pickle=True).astype(str)
xt_all = np.load(D / "train.npy", mmap_mode="r"); yt = np.load(D / "train_celltype.npy", allow_pickle=True).astype(str)
real = lambda X, y, t: np.asarray(X[np.flatnonzero(y == t)], dtype=np.float64)


def r2_block(t):
    xv, xt = real(xv_all, yv, t), real(xt_all, yt, t); ref = prof(xv)
    fl = []
    for s in range(5):
        i = np.random.default_rng(20261107 + s).choice(len(xt), min(len(xv), len(xt)), replace=False)
        fl.append(compare(xt[i], ref)[0])
    return ref, {"real_val": {"n": len(xv), **compare(xv, ref)[0]},
                 "floor_train_vs_val": {k: {"mean": float(np.mean([f[k] for f in fl])), "min": float(np.min([f[k] for f in fl])),
                                            "max": float(np.max([f[k] for f in fl]))} for k in fl[0]}}


class R1:
    def __init__(self):
        if RB == "val": p2 = np.load(A / "frozen_atlas_pca_basis.npz"); p50 = np.load(A / "pca50_val.npz")
        else: p2 = p50 = np.load(W / "runs/r19/r1_bases_train.npz")
        self.sp = {"pca2": lambda c: (cp10k_log1p(c) - p2["mean_"]) @ p2["comp2"].T,
                   "pca50": lambda c: (cp10k_log1p(c).astype(np.float32) - p50["mean"]) @ p50["components"].T,
                   "full": lambda c: cp10k_log1p(c)}
        self.types = sorted(set(yv.tolist())); self.y_int = np.array([self.types.index(t) for t in yv]); self.neu = self.types.index("Neuronal")
        xval = np.asarray(xv_all, dtype=np.float64)
        self.index = {s: f(xval).astype(np.float32) for s, f in self.sp.items()}
        self.nn = {s: NearestNeighbors(n_neighbors=K, algorithm="brute", n_jobs=4).fit(v) for s, v in self.index.items()}

    def query(self, x):
        return {s: knn_stats(self.nn[s].kneighbors(f(x).astype(np.float32), return_distance=False), self.y_int, len(self.types), self.neu)[0]
                for s, f in self.sp.items()}

    def refs(self):
        mask = yv == "Neuronal"; own = np.flatnonzero(mask); out = {}
        for s in self.sp:
            ii = self.nn[s].kneighbors(self.index[s][mask], n_neighbors=K + 1, return_distance=False)
            keep = np.array([[j for j in row if j != o][:K] for row, o in zip(ii, own)])
            out[("real_val", s)] = knn_stats(keep, self.y_int, len(self.types), self.neu)[0]
        for s, st in self.query(real(xt_all, yt, "Neuronal")).items():
            out[("real_train", s)] = st
        return out


def maxdiff(a, b, keys):
    return max(abs(np.asarray(a[k], dtype=float) - np.asarray(b[k], dtype=float)).max() for k in keys)


ap = argparse.ArgumentParser(); ap.add_argument("--gate", action="store_true"); ap.add_argument("--manifest"); ap.add_argument("--out")
ap.add_argument("--ref-split", default="val", choices=("val", "test"))
ap.add_argument("--r1-basis", default="val", choices=("val", "train"))
a = ap.parse_args()
FIG7D = W / "from_windows/runs/table1_ours/Neuronal_a1/seed20260987.npy"
R1K = ("mean_neuronal_frac", "median_neuronal_frac", "share_frac_ge_0p5", "share_frac_eq_0", "nn1_is_neuronal", "neighbour_type_share")
if a.gate:
    r2ref = json.loads((REF / "r2_numbers.json").read_text())
    for t in ("Myeloid", "Neuronal"):
        ref, blk = r2_block(t); keys = [k for k in blk["real_val"] if k != "n"]
        d1 = maxdiff(blk["real_val"], r2ref[t]["real_val"], keys)
        d2 = max(abs(blk["floor_train_vs_val"][k]["mean"] - r2ref[t]["floor_train_vs_val"][k]["mean"]) for k in keys)
        print(f"[gate R2] {t} real_val max diff {d1:.2e}, floor mean max diff {d2:.2e}")
        if t == "Neuronal":
            w = next(r for r in r2ref[t]["runs"] if r["arm"] == "OURS" and r["train"] == 20260625 and r["samp"] == 20260987)
            st = compare(load_cells(FIG7D, t, 20260987, 20260625), ref)[0]
            print(f"[gate R2] Neuronal OURS Fig 7d run max diff {maxdiff(st, w, keys):.2e}")
    r1ref = json.loads((REF / "r1_numbers.json").read_text())["rows"]; m = R1()
    for (arm, s), st in m.refs().items():
        w = next(r for r in r1ref if r["arm"] == arm and r["space"] == s)
        print(f"[gate R1] {arm} {s} max diff {maxdiff(st, w, R1K):.2e}")
    for s, st in m.query(load_cells(FIG7D, "Neuronal", 20260987, 20260625)).items():
        w = next(r for r in r1ref if r["arm"] == "OURS" and str(r["train"]) == "20260625" and r["samp"] == 20260987 and r["space"] == s)
        print(f"[gate R1] OURS Fig 7d {s} max diff {maxdiff(st, w, R1K):.2e}")
else:
    M = json.loads(Path(a.manifest).read_text()); res = {"split": a.ref_split, "r2": {}, "r1": [], "runs": []}
    if a.r1_basis != "val": res["r1_basis"] = a.r1_basis
    m = R1() if any(e["target"] == "Neuronal" for e in M) else None
    refs = {}
    for t in sorted({e["target"] for e in M}):
        refs[t], res["r2"][t] = r2_block(t)
    for e in M:
        x = load_cells(e["path"], e["target"], e["seed"], e.get("train", 0))
        row = {**e, "n": len(x), "r2": compare(x, refs[e["target"]])[0]}
        if e["target"] == "Neuronal":
            row["r1"] = m.query(x)
        res["runs"].append(row); print(e["method"], e["arm"], e["target"], e["seed"], "zero", round(row["r2"]["zero_overall"], 4),
                                       "lib", round(row["r2"]["lib_mean"], 1), ("frac50 %.3f" % row["r1"]["pca50"]["mean_neuronal_frac"]) if "r1" in row else "", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
