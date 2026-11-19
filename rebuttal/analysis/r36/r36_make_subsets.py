#!/usr/bin/env python
r"""R36: reference-subsample index files. Reads ONLY labels (<split>_celltype.npy / _obs_names.txt), never expression.
For rep i = 0..19, seed 20261201 + i, a fresh np.random.default_rng(seed) per class: 80 % of the split's real cells of the class without
replacement (test: ionocyte 89 of 111, plasma cell 278 of 347; val: 36 of 45, 114 of 143), idx sorted, same format as make_subsets.py.
    python r36_make_subsets.py val|test OUT_DIR
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
SPL, O = sys.argv[1], Path(sys.argv[2]); O.mkdir(parents=True, exist_ok=True); D = Path.home() / "research/crndiff-rebuttal/data/r16_hlca"
y = np.load(D / f"{SPL}_celltype.npy", allow_pickle=True).astype(str); names = np.asarray((D / f"{SPL}_obs_names.txt").read_text().split("\n")); names = names[names != ""]
assert len(names) == len(y); rec = {}
for t in ("plasma cell", "ionocyte"):
    pool = np.flatnonzero(y == t); n = int(round(0.8 * len(pool)))
    for i in range(20):
        s = 20261201 + i; idx = np.sort(np.random.default_rng(s).choice(pool, n, replace=False)); assert (y[idx] == t).all()
        f = O / f"{SPL}_{t.replace(' ', '')}_rep{i:02d}.npz"
        np.savez(f, idx=idx.astype(np.int64), obs_names=names[idx], target=np.asarray(t), split=np.asarray(SPL), seed=np.asarray(s), n_class=np.asarray(len(pool)))
        rec[f.name] = {"n": n, "n_class": int(len(pool)), "seed": s, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
json.dump(rec, open(O / "subsets.json", "w"), indent=1); print({k: (v["n"], v["n_class"]) for k, v in list(rec.items())[::20]})
