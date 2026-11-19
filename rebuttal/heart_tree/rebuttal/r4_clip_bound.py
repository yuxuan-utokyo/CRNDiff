# -*- coding: utf-8 -*-
"""R4 clip: upper bound U for the odds, from the Neuronal classifier of train 20260625.
Rule (written to STATUS before any clipped run): U = median of the classifier's odds on its held-out real target
cells (target_val, from the train split), so no cell outweighs a typical real target cell. Uses no val, test or
generated cells. The 95th percentile was computed first and dropped: at 311 it sits above every candidate odds seen in R4.
    cd Experiments/HVG && python -m rebuttal.r4_clip_bound
"""
import json, os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "4"
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from hvg.sampler import PortableHGBT
from rebuttal.w2_runs import DATA
d = ROOT / "assets/models/residual/Neuronal"; z = np.load(d / "split_indices.npz"); clf = PortableHGBT(d / "residual_classifier_portable.npz")
X = np.asarray(np.load(DATA / "train.npy", mmap_mode="r")[np.sort(z["target_val"])], dtype=np.float32)
xf = np.log1p(X * (1e4 / np.maximum(X.sum(1, dtype=np.float64), 1.0)).astype(np.float32)[:, None])[:, z["classifier_features"]]
p = clf.predict_probability(xf.astype(np.float64)); o = p / np.clip(1 - p, 1e-12, None)
r = {"n": len(o), "odds_q": {q: float(np.percentile(o, q)) for q in (50, 90, 95, 99, 100)}, "U": float(np.percentile(o, 50)), "rule": "median odds on target_val"}
out = ROOT / "rebuttal/out/R4clip"; out.mkdir(parents=True, exist_ok=True); (out / "bound.json").write_text(json.dumps(r, indent=1)); print(r)
