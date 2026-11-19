#!/usr/bin/env python
"""Compare discriminator predictions on one fixed batch of cells. Standard library and numpy only.

Same model on two platforms (--ref-pred): predictions should agree to float rounding. A refit (--ref-portable): no
bitwise agreement is expected, so report AUC, correlation and feature overlap.

    python disc_compare.py --portable A.npz --features A_split.npz --cells CELLS.npy --ref-pred WINDOWS_PRED.npy --out OUT.json
    python disc_compare.py --portable REFIT.npz --features REFIT_split.npz --cells CELLS.npy \
        --ref-portable ORIG.npz --ref-features ORIG_split.npz [--labels Y.npy] --out OUT.json
The prediction path copies PortableHGBT, classifier_log_reward and the odds rule from hvg/sampler.py.
"""
import argparse, json, os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "2"
import numpy as np


def predict(path, features, x):
    z = np.load(path, allow_pickle=False); nodes = z["nodes"]
    is_leaf = np.ascontiguousarray(nodes["is_leaf"]); fi = np.ascontiguousarray(nodes["feature_idx"]).astype(np.intp)
    th = np.ascontiguousarray(nodes["num_threshold"]).astype(np.float64); ml = np.ascontiguousarray(nodes["missing_go_to_left"])
    le = np.ascontiguousarray(nodes["left"]).astype(np.int64); ri = np.ascontiguousarray(nodes["right"]).astype(np.int64)
    val = np.ascontiguousarray(nodes["value"]).astype(np.float64); off = z["offsets"].astype(np.int64)
    zz = np.asarray(x, dtype=np.float32); lib = zz.sum(axis=1, dtype=np.float64)
    scale = (1.0e4 / np.maximum(lib, 1.0)).astype(np.float32)
    xf = np.log1p(zz[:, features] * scale[:, None]).astype(np.float32, copy=False)
    if xf.shape[1] != int(z["n_features"][0]):
        raise SystemExit(f"classifier expects {int(z['n_features'][0])} features, got {xf.shape[1]}")
    raw = np.full(len(xf), float(z["baseline"][0]), dtype=np.float64); rows = np.arange(len(xf))
    for s, e in zip(off[:-1], off[1:]):
        s, e = int(s), int(e); idx = np.zeros(len(xf), dtype=np.int64)
        for _ in range(e - s):
            gi = idx + s; act = is_leaf[gi] == 0
            if not act.any():
                break
            rr, ga = rows[act], gi[act]; v = xf[rr, fi[ga]]
            left = np.where(np.isnan(v), ml[ga] != 0, v <= th[ga])
            idx[act] = np.where(left, le[ga], ri[ga])
        else:
            raise RuntimeError("tree traversal did not reach leaves")
        raw += val[idx + s]
    return np.where(raw >= 0, 1.0 / (1.0 + np.exp(-raw)), np.exp(raw) / (1.0 + np.exp(raw))), raw


def rank(a):
    r = np.empty(len(a)); r[np.argsort(a, kind="stable")] = np.arange(len(a)); return r


def auc(y, s):
    r = rank(s) + 1; n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


ap = argparse.ArgumentParser()
ap.add_argument("--portable", required=True); ap.add_argument("--features", required=True); ap.add_argument("--cells", required=True)
ap.add_argument("--ref-pred"); ap.add_argument("--ref-portable"); ap.add_argument("--ref-features"); ap.add_argument("--labels"); ap.add_argument("--out", required=True)
a = ap.parse_args()
x = np.load(a.cells, mmap_mode="r"); feat = np.load(a.features)["classifier_features"]
p, raw = predict(a.portable, feat, x); res = {"n_cells": len(x), "n_features": len(feat), "numpy": np.__version__}
if a.ref_pred:
    q = np.load(a.ref_pred); d = np.abs(p - q)
    res.update(mode="same_model", max_abs_diff=float(d.max()), n_exact=int((p == q).sum()), n_diff_gt_1e12=int((d > 1e-12).sum()))
else:
    f0 = np.load(a.ref_features)["classifier_features"]; q, raw0 = predict(a.ref_portable, f0, x)
    res.update(mode="refit", pearson_logit=float(np.corrcoef(raw, raw0)[0, 1]), spearman=float(np.corrcoef(rank(p), rank(q))[0, 1]),
               max_abs_diff=float(np.abs(p - q).max()), median_abs_diff=float(np.median(np.abs(p - q))),
               feature_overlap=len(set(feat.tolist()) & set(f0.tolist())), same_feature_order=bool(np.array_equal(feat, f0)))
    if a.labels:
        y = np.load(a.labels); res.update(auc=auc(y, p), auc_ref=auc(y, q))
print(res); open(a.out, "w").write(json.dumps(res, indent=1))
