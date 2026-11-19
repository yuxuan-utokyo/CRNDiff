#!/usr/bin/env python
"""Put a fitted discriminator in the exchange folder with a fixed batch of cells and Mac's predictions, for the cross-platform check.

Cells: the first 500 target_val rows at most (train rows, not used in the fit) and the first proposal_val rows up to 1,000 cells in total.
    python disc_publish.py --fit FIT_DIR --data DATA --pool POOL.npy --out EXCHANGE_DIR
Windows reads the portable npz, applies the input rule in hvg/sampler.py and compares with mac_probability.
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

ap = argparse.ArgumentParser(); ap.add_argument("--fit", type=Path, required=True); ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--pool", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
sys.argv = ["disc_compare", "--portable", "x", "--features", "x", "--cells", "x", "--out", "/dev/null", "--ref-pred", "x"]
import importlib.util
spec = importlib.util.spec_from_file_location("dc", Path(__file__).parent / "disc_compare.py")
src = open(spec.origin, encoding="utf-8").read().split("ap = argparse.ArgumentParser()")[0]
ns = {}; exec(compile(src, spec.origin, "exec"), ns)
s = np.load(a.fit / "split_indices.npz"); X = np.load(a.data / "train.npy", mmap_mode="r"); q = np.load(a.pool, mmap_mode="r")
tv = s["target_val"][:500]; pv = s["proposal_val"][:1000 - len(tv)]   # at most 500 target cells, so pool cells are always in
# dtype: keep the inputs' integer type (11-09 19:4x: HLCA arrays are int32 with counts up to 34,950; a cast to int16 would wrap them
# silently). For the int16 DG/HVG2K inputs this is int16 exactly as before, so the published fixed_cells.npz files are unchanged.
dt = np.result_type(X.dtype, q.dtype)
cells = np.concatenate([np.asarray(X[tv]), np.asarray(q[pv])]).astype(dt)
assert np.issubdtype(dt, np.integer) and (cells >= 0).all(), dt
p, _ = ns["predict"](a.fit / "residual_classifier_portable.npz", s["classifier_features"], cells)
a.out.mkdir(parents=True, exist_ok=True)
for f in ("residual_classifier_portable.npz", "split_indices.npz", "residual_log.json", "fit_record.json"):
    shutil.copy(a.fit / f, a.out / f)
np.savez_compressed(a.out / "fixed_cells.npz", counts=cells, train_indices=tv, proposal_indices=pv, classifier_features=s["classifier_features"], mac_probability=p,
                    is_target=np.r_[np.ones(len(tv), dtype=np.int8), np.zeros(len(pv), dtype=np.int8)])
print(json.dumps({"n": len(cells), "n_target": len(tv), "p_target_mean": float(p[:len(tv)].mean()), "p_pool_mean": float(p[len(tv):].mean())}))
