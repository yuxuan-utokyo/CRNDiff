#!/usr/bin/env python
r"""R2 conditional Fano, Mac port of Windows' rebuttal/r2_addendum.py. Same definition, copied verbatim:
  fano(x) = median over genes with mean > 0 of var(ddof=1) / mean, raw counts.
  real row: all reference cells of the target; real_train_full: all train cells of the target;
  floor: 5 train subsets of size min(n_ref, n_train), rng seed 20261107 + s (draws as in r2_addendum.py).
Generated cells are loaded with r1r2_metrics.load_cells (w2_runs.load_cells: thinned only when a delivery exceeds N_OUT).
--ref-split val|test (default val). --check R2_ADDENDUM.json compares the real rows of every target and every run present in both,
exactly (bit for bit), and prints the max difference; it does not stop on a mismatch.
    python r2_fano.py --manifest MANIFEST.json --out OUT.json [--ref-split test] [--check r2_addendum.json]
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ref-split", default="val", choices=("val", "test")); ap.add_argument("--check"); ap.add_argument("--threads", default="4")
a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = a.threads
import numpy as np
W = Path.home() / "research/crndiff-rebuttal"; D = W / "data/hvg2k"
N_OUT = {"Endothelial": 10057, "Myeloid": 2500, "Neuronal": 2500}; SUBSAMPLE_SEED = 20261107   # = r1r2_metrics / w2_runs
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fano(x):                                     # verbatim from r2_addendum.py
    m = x.mean(0); v = x.var(0, ddof=1); k = m > 0
    return float(np.median(v[k] / m[k]))


def load_cells(path, t, samp, train):          # verbatim from r1r2_metrics.py (w2_runs.load_cells)
    x = np.load(path, mmap_mode="r"); n = N_OUT[t]
    if x.shape[0] > n:
        s = np.random.SeedSequence([SUBSAMPLE_SEED, samp, train]).generate_state(1)
        idx = np.sort(np.random.default_rng(int(s[0])).choice(x.shape[0], n, replace=False))
        return np.asarray(x[idx], dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


Xr = np.load(D / f"{a.ref_split}.npy", mmap_mode="r"); yr = np.load(D / f"{a.ref_split}_celltype.npy", allow_pickle=True).astype(str)
Xt = np.load(D / "train.npy", mmap_mode="r"); yt = np.load(D / "train_celltype.npy", allow_pickle=True).astype(str)
real = lambda X, y, t: np.asarray(X[np.flatnonzero(y == t)], dtype=np.float64)
M = json.loads(Path(a.manifest).read_text())
res = {"split": a.ref_split, "fano_def": "median over genes with mean > 0 of var(ddof=1)/mean, raw counts (r2_addendum.py)",
       "script_sha256": sha(__file__), "manifest_sha256": sha(a.manifest), "targets": {}}
for t in sorted({e["target"] for e in M} | (set(json.loads(Path(a.check).read_text())) if a.check else set())):
    xr, xt = real(Xr, yr, t), real(Xt, yt, t)
    d = {f"real_{a.ref_split}_fano": fano(xr), "real_train_full_fano": fano(xt), "n_ref": len(xr),
         "floor_fano": [fano(xt[np.random.default_rng(20261107 + s).choice(len(xt), min(len(xr), len(xt)), replace=False)]) for s in range(5)], "runs": []}
    for e in [e for e in M if e["target"] == t]:
        x = load_cells(e["path"], t, e["seed"], e.get("train", 0))
        d["runs"].append({k: e[k] for k in ("method", "arm", "seed") if k in e} | {"train": e.get("train"), "path": e["path"], "sha256": sha(e["path"]), "n": len(x), "fano": fano(x)})
    res["targets"][t] = d; print(t, "real", round(d[f"real_{a.ref_split}_fano"], 4), "floor", np.round(d["floor_fano"], 4).tolist(), [(r["method"], r["train"], round(r["fano"], 4)) for r in d["runs"]], flush=True)
if a.check:
    assert a.ref_split == "val", "--check compares against val numbers"
    ref = json.loads(Path(a.check).read_text()); cmp = []
    for t, d in res["targets"].items():
        w = ref[t]
        cmp += [(f"{t} real_val", d["real_val_fano"], w["real_val_fano"]), (f"{t} real_train_full", d["real_train_full_fano"], w["real_train_full_fano"])]
        cmp += [(f"{t} floor[{i}]", x, y) for i, (x, y) in enumerate(zip(d["floor_fano"], w["floor_fano"]))]
        for r in d["runs"]:
            m = [q for q in w["runs"] if q["arm"] == r["method"] and q["train"] == r["train"] and q["samp"] == r["seed"]]
            if len(m) == 1: cmp.append((f"{t} {r['method']} {r['train']}/{r['seed']}", r["fano"], m[0]["fano"]))
    res["check"] = {"against": a.check, "against_sha256": sha(a.check), "n_compared": len(cmp), "n_identical": sum(x == y for _, x, y in cmp),
                    "max_abs_diff": max(abs(x - y) for _, x, y in cmp), "rows": [{"what": w_, "mac": x, "windows": y, "identical": x == y} for w_, x, y in cmp]}
    print("check:", res["check"]["n_identical"], "of", res["check"]["n_compared"], "identical; max |diff| %.3e" % res["check"]["max_abs_diff"])
Path(a.out).write_text(json.dumps(res, indent=1)); print("[done]")
