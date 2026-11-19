# HVG2K B3 rule (a) (config_hvg2k.py lines 38-44): NMAX = the smallest power of two whose truncated count mass on train is <= 0.5%;
# truncated mass = sum(max(x - NMAX, 0)) / sum(x). Train only. Writes NMAX.json next to train.npy.
import json, sys, numpy as np
from pathlib import Path
D = Path(sys.argv[1]); X = np.load(D / "train.npy", mmap_mode="r"); tot = 0.0; over = {}
cands = [64, 128, 256, 512, 1024, 2048, 4096]
for i in range(0, X.shape[0], 50000):
    b = np.asarray(X[i:i + 50000], dtype=np.int64); tot += b.sum()
    for c in cands: over[c] = over.get(c, 0) + np.clip(b - c, 0, None).sum()
frac = {c: float(over[c] / tot) for c in cands}; nmax = next(c for c in cands if frac[c] <= 0.005)
ngene = {c: int((np.asarray(X).max(0) > c).sum()) for c in (nmax,)}
r = {"NMAX": nmax, "rule": "smallest power of 2 with truncated count mass <= 0.5% on train (HVG2K B3 (a))", "truncated_mass_frac": frac,
     "n_genes_with_counts_above_NMAX": ngene[nmax], "train_max": int(np.asarray(X).max()), "n_train": int(X.shape[0])}
print(r); (D / "NMAX.json").write_text(json.dumps(r, indent=1))
