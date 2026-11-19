#!/usr/bin/env python
r"""R40 views, v2 (the row-reduced v1 did not reproduce Table 3 W1: the frozen W1 code draws per type from row-index arrays through one rng
over the whole target list, so the row layout and the target list must stay as in Table 3). Here every row is kept and only the IG
columns (symbols ^IG[HKL]; JCHAIN kept) are dropped:
  runs/r40/views2/noig_train.npy (all 421,159 train cells), runs/r40/views2/noig_val/ and noig_test/ (val.npy = all cells of that split,
  celltype files symlinked to the originals, train.npy symlinked to noig_train.npy). Written in chunks; nothing else changes.
Generated arrays: runs/r40/gen/*_noig.npy (from v1, same column subset) via runs/r40/manifest_noig.json.
    python r40_make_views_fullrows.py
"""
import json, os, re
import numpy as np
W = "./work"; O = f"{W}/runs/r40/views2"; os.makedirs(O, exist_ok=True)
G = json.load(open(f"{W}/data/r16_hlca_meta/genes_cfgen_hvg2000.json")); SYM = list(G["feature_name"])
cols = np.flatnonzero(~np.array([bool(re.match(r"^IG[HKL]", s)) for s in SYM])); assert len(cols) == 1862 and SYM.index("JCHAIN") in cols
def sub(src, dst):
    X = np.load(src, mmap_mode="r"); Y = np.lib.format.open_memmap(dst, mode="w+", dtype=X.dtype, shape=(X.shape[0], len(cols)))
    for i in range(0, X.shape[0], 20000): Y[i:i + 20000] = np.asarray(X[i:i + 20000])[:, cols]
    Y.flush(); return X.shape
print("train", sub(f"{W}/data/r16_hlca_fit/train.npy", f"{O}/noig_train.npy"))
for name, src in (("noig_val", f"{W}/data/r16_hlca_score"), ("noig_test", f"{W}/data/r16_hlca_test_view")):
    d = f"{O}/{name}"; os.makedirs(d, exist_ok=True); print(name, sub(f"{src}/val.npy", f"{d}/val.npy"))
    for f, tgt in (("val_celltype.npy", os.path.realpath(f"{src}/val_celltype.npy")), ("train_celltype.npy", os.path.realpath(f"{src}/train_celltype.npy")), ("train.npy", f"{O}/noig_train.npy")):
        if not os.path.exists(f"{d}/{f}"): os.symlink(tgt, f"{d}/{f}")
json.dump({"n_genes": int(len(cols)), "cols": cols.tolist()}, open(f"{O}/cols.json", "w"))
