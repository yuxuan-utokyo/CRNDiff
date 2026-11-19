# -*- coding: utf-8 -*-
"""R10: tilt tables from the fixed Neuronal subsets, with the recipe in rebuttal/build_tables.py.
    cd Experiments/HVG && python -m rebuttal.r10_tables
"""
import json, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
from rebuttal.build_tables import marg, logc_of, PILOT, DATA, KZ, sha
OUT = Path(__file__).resolve().parent / "out" / "R10"
z = np.load(OUT / "subset_indices.npz")
P = np.load(PILOT, mmap_mode="r"); Qg = marg(P); n_gen = P.shape[0]
X = np.load(DATA / "train.npy", mmap_mode="r")
ii = z["neuronal_train"]; ht = marg(X, ii); full = logc_of(ht, len(ii), Qg, n_gen)
ref = np.load(KZ / "logc_Neuronal.npy"); err = float(np.abs(full - ref).max())
assert err < 1e-12, err
log = {"rebuild_max_abs_err": err, "tables": []}
for lv in ("frac010", "frac003"):
    s = z[lv]; t = logc_of(marg(X, s), len(s), Qg, n_gen)
    f = OUT / f"logc_Neuronal_{lv}.npy"; np.save(f, t)
    log["tables"].append({"level": lv, "n_ref": int(len(s)), "file": f.name, "sha256": sha(f),
                          "max_abs_diff_vs_full": float(np.abs(t - full).max()),
                          "mean_abs_diff_where_target_mass": float((ht * np.abs(t - full)).sum(1).mean())})
    print(lv, len(s), log["tables"][-1]["max_abs_diff_vs_full"], log["tables"][-1]["mean_abs_diff_where_target_mass"])
(OUT / "tables_log.json").write_text(json.dumps(log, indent=1))
print("rebuild err", err)
