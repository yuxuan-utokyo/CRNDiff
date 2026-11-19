# -*- coding: utf-8 -*-
"""R6c: per-gene count marginal Q_g of a pool, as rebuttal/build_tables.py::marg computes it.
    python -m rebuttal.r6c_marginal <pool.npy> <out.npz>
"""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
sys.dont_write_bytecode = True
import numpy as np
from rebuttal.build_tables import marg, sha
P = np.load(sys.argv[1], mmap_mode="r"); Q = marg(P)
assert Q.shape == (2000, 513) and Q.dtype == np.float64 and abs(Q.sum(1) - 1).max() < 1e-12
np.savez_compressed(sys.argv[2], Q_g=Q, n_gen=P.shape[0], pool_sha256=sha(sys.argv[1]), pool_name=os.path.basename(sys.argv[1]))
print(Q.shape, Q.dtype, P.shape[0], float(abs(Q.sum(1) - 1).max()), sha(sys.argv[1])[:16])
