# -*- coding: utf-8 -*-
"""Files for Mac's cross-platform check of the portable classifier reader: fixed real cells and this machine's predictions."""
import json, os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from hvg.sampler import PortableHGBT
from rebuttal.w2_runs import DATA
out = Path(sys.argv[1]); d = ROOT / "assets/models/residual/Neuronal"; z = np.load(d / "split_indices.npz")
idx = np.sort(z["target_val"])[:1000]; X = np.asarray(np.load(DATA / "train.npy", mmap_mode="r")[idx], dtype=np.int16)
f = np.asarray(X, dtype=np.float32); xf = np.log1p(f * (1e4 / np.maximum(f.sum(1, dtype=np.float64), 1.0)).astype(np.float32)[:, None])[:, z["classifier_features"]]
p = PortableHGBT(d / "residual_classifier_portable.npz").predict_probability(xf.astype(np.float64))
np.savez_compressed(out / "fixed_cells.npz", train_indices=idx, counts=X, windows_probability=p, classifier_features=z["classifier_features"])
(out / "README.md").write_text("Portable Neuronal classifier of train 20260625 and 1,000 fixed real cells from its target_val split (train.npy row numbers in train_indices).\nInput rule: float32 counts, scale 1e4 / max(library, 1) as float32, log1p, then the classifier_features columns, then PortableHGBT.predict_probability on float64 (hvg/sampler.py).\nwindows_probability is what Windows gets. Compare with your reader. The portable npz was fitted with sklearn 1.3.0 and is read by this code without sklearn.\n", encoding="utf-8")
print(p[:3], float(p.mean()), len(p))
