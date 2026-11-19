# -*- coding: utf-8 -*-
"""R10: fixed subsets of the Neuronal train cells. Nested (3% is the first part of 10%), so the
rarer level only removes cells. Both arms of R10 read the same file.

    cd Experiments/HVG && python -m rebuttal.r10_make_subset
"""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rebuttal" / "out" / "R10"
SEED = 20261110
y = np.load(ROOT / "assets/legacy_root/_shared/data/all/train_celltype.npy", allow_pickle=True).astype(str)
ii = np.flatnonzero(y == "Neuronal")
p = np.random.default_rng(SEED).permutation(ii)
n10, n3 = int(round(0.10 * len(ii))), int(round(0.03 * len(ii)))
f = OUT / "subset_indices.npz"
if f.exists():
    raise SystemExit(f"refusing to overwrite {f}")
np.savez(f, frac010=np.sort(p[:n10]), frac003=np.sort(p[:n3]), neuronal_train=ii, seed=SEED)
(OUT / "subset_indices.json").write_text(json.dumps({"seed": SEED, "n_neuronal_train": int(len(ii)), "n_frac010": n10, "n_frac003": n3, "nested": True}, indent=1))
print(len(ii), n10, n3)
