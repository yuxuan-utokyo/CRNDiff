# -*- coding: utf-8 -*-
"""Save the R3 PCA bases (train cells of each target, log-CP10K, same settings as r3_diversity.basis) for Mac."""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "4"
sys.dont_write_bytecode = True
import numpy as np
from sklearn.decomposition import PCA
from rebuttal.w2_runs import TARGETS, real_cells, log_cp10k
out = {}
for t in TARGETS:
    p = PCA(n_components=50, svd_solver="randomized", random_state=20261107).fit(log_cp10k(real_cells("train", t)))
    out[f"{t}__mean"], out[f"{t}__components"], out[f"{t}__evr"] = p.mean_, p.components_, p.explained_variance_ratio_
    print(t, round(float(p.explained_variance_ratio_.sum()), 4), flush=True)
np.savez(sys.argv[1], **out)
