"""R16: redo the pca_ref draw independently before the shared PCA is saved.
Rule (pre-registration item 9; table2_metrics.py real_train lines): ids = train h5ad obs names in file order (467,955);
pool = positions of ids not in val_obs_names.txt; pick = sort(default_rng(20261310).choice(pool, 50000, replace=False)).
Compares ids[pick] with Windows' pca_ref_obs_names.txt NAME BY NAME and IN ORDER, and checks that the pca_ref count rows equal
the train rows of the same names (train.npy is stored sorted by name). Exit 0 only if everything matches.
    check_pca_ref_draw.py R16_DATA_DIR VAL_OBS_NAMES OUT.json
"""
import hashlib, json, os, sys
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r16_data import load_split

D, VAL, OUT = sys.argv[1:4]
rd = lambda p: np.array([l.rstrip("\n") for l in open(p, encoding="utf-8") if l.strip()])
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ids = rd(f"{D}/train_h5ad_obs_names.txt"); val = rd(VAL)
assert len(ids) == 467955 and len(set(ids)) == len(ids), len(ids)
keep = ~np.isin(ids, val); assert keep.sum() == len(ids) - len(set(val) & set(ids)) == 421159, keep.sum()
pool = np.flatnonzero(keep)
pick = np.sort(np.random.default_rng(20261310).choice(pool, 50000, replace=False))
mine = ids[pick]
Xp, yp, gp, theirs = load_split(D, "pca_ref", with_names=True)
r = {"n_h5ad": int(len(ids)), "n_pool": int(len(pool)), "n_pick": int(len(mine)),
     "names_equal_in_order": bool(len(theirs) == len(mine) and np.array_equal(theirs, mine)),
     "same_set": bool(set(theirs) == set(mine)), "n_in_val": int(np.isin(mine, val).sum()),
     "ids_sha256_mine": hashlib.sha256("\n".join(mine).encode()).hexdigest(),
     "ids_sha256_theirs": hashlib.sha256("\n".join(theirs).encode()).hexdigest()}
# count rows: pca_ref rows must equal the train rows with the same names
Xt, yt, gt, tn = load_split(D, "train", with_names=True)
row_of = {v: i for i, v in enumerate(tn)}; rows = np.array([row_of[x] for x in mine])
r["counts_equal_train_rows"] = bool((Xt[rows] != Xp).nnz == 0) if r["names_equal_in_order"] else None
r["labels_equal_train_rows"] = bool(np.array_equal(yt[rows], yp)) if r["names_equal_in_order"] else None
r["genes_equal"] = bool(np.array_equal(gt, gp))
r["inputs_sha256"] = {"train_h5ad_obs_names.txt": sha(f"{D}/train_h5ad_obs_names.txt"), "val_obs_names": sha(VAL),
                      "pca_ref_obs_names.txt": sha(f"{D}/pca_ref_obs_names.txt")}
r["script_sha256"] = sha(__file__)
ok = r["names_equal_in_order"] and r["n_in_val"] == 0 and r["counts_equal_train_rows"] and r["labels_equal_train_rows"] and r["genes_equal"]
r["all_ok"] = bool(ok)
json.dump(r, open(OUT, "w"), indent=1); print(json.dumps(r, indent=1))
sys.exit(0 if ok else 1)
