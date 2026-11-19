#!/usr/bin/env python
r"""R29: the pca50 space for the k-NN neighbour fraction, fit on HLCA TRAIN with the heart R1 train-basis recipe
(scripts/r12_extra/r1_bases_train.py): PCA(n_components=50, svd_solver="full") on log1p(CP10K) (float64) cast to float32; fit set = ALL
train cells, every type (R1 counts neighbour types over the whole atlas). Only the data changes. Chunked log1p(CP10K) to bound memory.
    python hlca_pca50_train.py OUT.npz
"""
import hashlib, json, os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "8"
import numpy as np
from sklearn.decomposition import PCA
W = "./work"; out = sys.argv[1]
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
X = np.load(f"{W}/data/r16_hlca_fit/train.npy", mmap_mode="r")
Z = np.empty(X.shape, dtype=np.float32)
for i in range(0, X.shape[0], 20000):
    x = np.asarray(X[i:i + 20000], dtype=np.float64); lib = x.sum(1, keepdims=True)
    Z[i:i + 20000] = np.log1p(x / np.maximum(lib, 1) * 1e4).astype(np.float32)
p = PCA(n_components=50, svd_solver="full").fit(Z)
np.savez(out, mean=p.mean_.astype(np.float32), components=p.components_.astype(np.float32), explained=p.explained_variance_ratio_, n_fit=len(Z))
rec = {"n_fit": int(len(Z)), "explained_sum": float(p.explained_variance_ratio_.sum()), "out_sha256": sha(out), "script_sha256": sha(__file__),
       "train_sha256": sha(f"{W}/data/r16_hlca_fit/train.npy")[:16], "recipe": "heart r1_bases_train.py pca50 (svd full, float32 log1p CP10K, all train cells)"}
json.dump(rec, open(out.replace(".npz", ".json"), "w"), indent=1); print(json.dumps(rec))
