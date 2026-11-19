#!/usr/bin/env python
"""R16 data for Mac (vice 11-09 20:28 item 1): HLCA arrays from ~/QIU/hlca/data_hvg2000 as int32 CSR, scipy save_npz(compressed=True),
row blocks of 100,000 (each file well under 1 GB), into OUT/. Splits val, test, train (all rows), plus pca_ref.

pca_ref = Mac's table2_metrics.py rule, reproduced exactly (lines 306-313 of r16_scripts/table2_metrics.py):
  ids  = obs_names of the CFGen hlca_core_train.h5ad (467,955), in file order
  pool = positions of ids NOT in val_obs_names.txt          (--pca-ref-exclude VAL_IDS)
  pick = sort(default_rng(20261310).choice(pool, 50000, replace=False))   (--pca-fit-n 50000 --pca-fit-seed 20261310)
v2: train.npy rows are sorted by obs name (prep_r16.py line 128), so the picked cells are mapped to train.npy rows by name;
the pca_ref block keeps Mac's order (h5ad order of pick).
Test values are converted and round-trip checked only (nothing about them is printed beyond shape and nnz).
    python r16_export.py --data ~/QIU/hlca/data_hvg2000 --h5ad <hlca_core_train.h5ad> --out OUT
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "4"
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np, h5py
from scipy import sparse

BLK, SEED, NREF = 100_000, 20261310, 50_000
ap = argparse.ArgumentParser(); ap.add_argument("--data", type=Path, required=True); ap.add_argument("--h5ad", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
names = lambda s: np.array([l.rstrip("\n") for l in open(a.data / f"{s}_obs_names.txt", encoding="utf-8")])
log = {"format": "int32 CSR, scipy.sparse.save_npz(compressed=True), row blocks of 100,000", "source_dir": str(a.data), "files": {}}

def dump(tag, X, rows=None):
    n = len(rows) if rows is not None else X.shape[0]; out = []
    for lo in range(0, n, BLK):
        hi = min(lo + BLK, n); r = rows[lo:hi] if rows is not None else slice(lo, hi)
        d = np.asarray(X[r]); assert d.dtype == np.int32 and d.min() >= 0
        m = sparse.csr_matrix(d, dtype=np.int32); p = a.out / f"{tag}_rows{lo}-{hi}.npz"; tmp = p.with_name(p.name + ".part.npz")
        sparse.save_npz(tmp, m, compressed=True); os.replace(tmp, p)
        back = sparse.load_npz(p); assert back.dtype == np.int32 and back.shape == d.shape and (back != sparse.csr_matrix(d)).nnz == 0
        out.append({"file": p.name, "rows": [lo, hi], "nnz": int(m.nnz), "bytes": p.stat().st_size}); print(tag, out[-1], flush=True)
    log["files"][tag] = {"shape": [n, X.shape[1]], "blocks": out}

for s in ("val", "test", "train"):
    X = np.load(a.data / f"{s}.npy", mmap_mode="r"); dump(s, X)
    shutil.copyfile(a.data / f"{s}_celltype.npy", a.out / f"{s}_celltype.npy"); shutil.copyfile(a.data / f"{s}_obs_names.txt", a.out / f"{s}_obs_names.txt")

with h5py.File(a.h5ad, "r") as f:
    o = f["obs"]; key = o.attrs.get("_index", "_index"); ids = o[key][:]
ids = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ids]); vn = names("val"); tn = names("train")
keep = ~np.isin(ids, vn); assert keep.sum() == len(ids) - len(set(vn) & set(ids)), "exclusion count mismatch"
pool = np.flatnonzero(keep)
# v2: prep_r16.py (line 128) stores train rows sorted by obs name, not in h5ad order, so map the picked cells to train.npy rows BY NAME.
assert len(pool) == len(tn) and set(ids[pool]) == set(tn), "train.npy cells are not exactly the h5ad non-val cells"
pick = np.sort(np.random.default_rng(SEED).choice(pool, NREF, replace=False))          # Mac's draw, unchanged (h5ad positions)
row_of = {v: i for i, v in enumerate(tn)}; rows = np.array([row_of[x] for x in ids[pick]], dtype=np.int64)
assert np.array_equal(tn[rows], ids[pick]) and len(set(rows.tolist())) == NREF          # pca_ref rows are in Mac's (h5ad) order
ref_ids = ids[pick]; (a.out / "pca_ref_obs_names.txt").write_text("\n".join(ref_ids) + "\n", encoding="utf-8")
np.save(a.out / "pca_ref_train_rows.npy", rows.astype(np.int64))
Xt = np.load(a.data / "train.npy", mmap_mode="r"); dump("pca_ref", Xt, rows)
np.save(a.out / "pca_ref_celltype.npy", np.load(a.data / "train_celltype.npy", allow_pickle=True)[rows].astype(str))
log["pca_ref"] = {"rule": "pool = h5ad obs positions not in val_obs_names.txt; pick = sort(default_rng(20261310).choice(pool, 50000, replace=False)) (table2_metrics.py 306-313)",
                  "h5ad": str(a.h5ad), "n_h5ad": int(len(ids)), "n_pool": int(len(pool)), "ids_sha256": hashlib.sha256("\n".join(ref_ids).encode()).hexdigest()}
for f in ("genes.json", "meta.json", "data_card.json", "NMAX.json", "T_O.json"):
    shutil.copyfile(a.data / f, a.out / f)
(a.out / "export_log.json").write_text(json.dumps(log, indent=1))
print("[done] pca_ref ids_sha256", log["pca_ref"]["ids_sha256"], flush=True)
