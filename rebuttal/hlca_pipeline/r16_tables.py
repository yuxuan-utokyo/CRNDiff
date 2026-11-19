#!/usr/bin/env python
"""R16 (vice 11-09 20:4x): _data_exchange/r13_pool/tables/r13_tables.py with ONLY C changed (257 -> 4097 = NMAX 4096 + 1); run with --data
~/QIU/hlca/data_hvg2000 --pool <generator checkpoint-1 unconditional pool> --targets <the 5 HLCA targets>.
Original R13 docstring: R13: product-tilt tables on DG, with the recipe of rebuttal/build_tables.py (marg and logc_of copied unchanged).

eps=0.5 smoothing (eps/N per side), log ratio clipped at +-log(1e3), counts clipped at NMAX=256, so C=257 count bins.
Reference: all DG train cells of the target. Calibration pool: the unconditional pool of the R13 generator.

    python r13_tables.py --data DG --pool POOL.npy --out OUT_DIR [--targets GlialProg RadialGlia2 OPC Granule]
"""
import argparse, hashlib, json, os
from pathlib import Path
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "2"
import numpy as np

C = 4097


def marg(A, rows=None, chunk=5000):
    G = A.shape[1]; off = (np.arange(G, dtype=np.int64) * C)[None, :]
    H = np.zeros(G * C, dtype=np.int64)
    idx = np.arange(A.shape[0]) if rows is None else np.sort(rows)
    for k in range(0, len(idx), chunk):
        sl = idx[k:k + chunk]
        b = np.asarray(A[sl] if rows is not None else A[sl[0]:sl[-1] + 1], dtype=np.int64).clip(0, C - 1)
        H += np.bincount((b + off).ravel(), minlength=G * C)
    return H.reshape(G, C).astype(np.float64) / len(idx)


def logc_of(ht, Nref, Qg, n_gen, eps=0.5, cap=np.log(1e3)):
    return np.clip(np.log(ht + eps / Nref) - np.log(Qg + eps / n_gen), -cap, cap)


sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True); ap.add_argument("--pool", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--targets", nargs="+", default=["GlialProg", "RadialGlia2", "OPC", "Granule"])
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
P = np.load(a.pool, mmap_mode="r"); n_gen = P.shape[0]; Qg = marg(P)
X = np.load(a.data / "train.npy", mmap_mode="r"); y = np.load(a.data / "train_celltype.npy", allow_pickle=True).astype(str)
if P.shape[1] != X.shape[1]:
    raise SystemExit(f"gene count differs: pool {P.shape[1]}, data {X.shape[1]}")
log = {"C": C, "eps": 0.5, "cap": "log(1e3)", "n_pool": int(n_gen), "pool_sha256": sha(a.pool), "pool_share_above_nmax": float((np.asarray(P) > C - 1).mean()), "tables": []}
for t in a.targets:
    ii = np.flatnonzero(y == t)
    if not len(ii):
        raise SystemExit(f"no train cells for {t}")
    ht = marg(X, ii); tab = logc_of(ht, len(ii), Qg, n_gen); f = a.out / f"logc_{t}.npy"
    if f.exists():
        raise SystemExit(f"refusing to overwrite {f}")
    np.save(f, tab)
    rec = {"target": t, "n_ref": int(len(ii)), "file": f.name, "sha256": sha(f), "shape": list(tab.shape), "share_at_cap": float((np.abs(tab) >= np.log(1e3) - 1e-12).mean()),
           "mean_abs_where_target_mass": float((ht * np.abs(tab)).sum(1).mean()), "ref_share_above_nmax": float((np.asarray(X[ii]) > C - 1).mean())}
    log["tables"].append(rec); print(rec, flush=True)
(a.out / "tables_log.json").write_text(json.dumps(log, indent=1))
print("[done]", flush=True)
