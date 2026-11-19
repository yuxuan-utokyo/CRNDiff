# -*- coding: utf-8 -*-
"""R14g cross-check dump (vice 08:5x: Windows fits, Mac cross-checks). For a Windows-fitted disc in rebuttal/out/R14g/s<GEN>/disc_INTERSECT_g128:
fixed cells = the first 500 held-out positives (split_indices target_val, train.npy rows) + the first 500 held-out negatives (proposal_val,
rows of that generator's 10k pool), counts int32. Features = fit_ref_disc's own transform (norm_log1p: float32 log1p of 1e4-normalised
counts) on classifier_features, cast to float64 as the FK reader does. Writes <disc>/fixed_cells.npz with counts, train_indices,
proposal_indices, classifier_features, is_target, windows_probability (frozen hvg.sampler.PortableHGBT on the portable npz) and
windows_sklearn_probability (the fitted joblib, predict_proba[:,1]); prints max |portable - sklearn|. Mac re-reads the portable npz on
the same counts and compares with windows_probability.
    python -u rebuttal/r14g_dump.py --gen 20260627
"""
import argparse, json, os, sys
from pathlib import Path
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "4"
HVG = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HVG))
import numpy as np, joblib
from hvg.sampler import PortableHGBT
ap = argparse.ArgumentParser(); ap.add_argument("--gen", required=True, choices=("20260627", "20260630")); a = ap.parse_args()
O = HVG / "rebuttal" / "out" / "R14g" / f"s{a.gen}"; d = O / "disc_INTERSECT_g128"
assert not (d / "fixed_cells.npz").exists(), "refusing to overwrite fixed_cells.npz"
s = np.load(d / "split_indices.npz"); fe = s["classifier_features"]
X = np.load(HVG / "assets" / "legacy_root" / "_shared" / "data" / "all" / "train.npy", mmap_mode="r")
pool = O / "pool_root" / "Baseline" / "OURS" / "results" / f"linspace_T_O_K32_spilot_run214_s{a.gen}_samp20261044_batched1_chunk256_UNI_INTERSECT_Endo_g128_n10000_g0p4_noa4.npy"
q = np.load(pool, mmap_mode="r")
ti, pi = np.sort(s["target_val"])[:500], np.sort(s["proposal_val"])[:500]
counts = np.concatenate([np.asarray(X[ti], dtype=np.int32), np.asarray(q[pi], dtype=np.int32)])
z = counts.astype(np.float32); lib = z.sum(axis=1, dtype=np.float64)
xf = np.log1p(z * (1.0e4 / np.maximum(lib, 1.0)).astype(np.float32)[:, None])[:, fe]
x64 = xf.astype(np.float64)
p = PortableHGBT(d / "residual_classifier_portable.npz").predict_probability(x64)
ps = joblib.load(d / "residual_classifier.joblib").predict_proba(x64)[:, 1]
np.savez(d / "fixed_cells.npz", counts=counts, train_indices=ti, proposal_indices=pi.astype(np.int64), classifier_features=fe,
         is_target=np.r_[np.ones(len(ti), np.int8), np.zeros(len(pi), np.int8)], windows_probability=p, windows_sklearn_probability=ps)
print(json.dumps({"gen": a.gen, "n": int(len(counts)), "max_abs_portable_vs_sklearn": float(np.abs(p - ps).max()),
                  "pool_sha_prefix": __import__("hashlib").sha256(pool.read_bytes()).hexdigest()[:16]}))
