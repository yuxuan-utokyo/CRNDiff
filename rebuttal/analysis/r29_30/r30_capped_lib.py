#!/usr/bin/env python
r"""R30 extra reference row: library size of the REAL target cells after capping every count at 4,096 (the generators' count
ceiling), on the same eval split and the same 2,000 HVG columns as r29_30_hlca.py's R30 reference. Reference only, no selection.
    python r30_capped_lib.py val|test OUT.json     (test: once, logged in runs/r16/TEST_READ_LOG.tsv)
"""
import hashlib, json, os, sys, time
import numpy as np
SPL, OUT = sys.argv[1:3]; assert SPL in ("val", "test"); assert not os.path.exists(OUT), f"{OUT} exists"
W = "./work"; sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
ED = f"{W}/data/r16_hlca_score" if SPL == "val" else f"{W}/data/r16_hlca_test_view"
X = np.load(f"{ED}/val.npy", mmap_mode="r"); Y = np.load(f"{ED}/val_celltype.npy", allow_pickle=True).astype(str)
CAP = 4096; res = {"split": SPL, "cap": CAP, "data_sha256": sha(f"{ED}/val.npy"), "script_sha256": sha(__file__), "types": {}}
for t in ("plasma cell", "ionocyte"):
    x = np.asarray(X[np.flatnonzero(Y == t)], dtype=np.float64); xc = np.minimum(x, CAP); lib, libc = x.sum(1), xc.sum(1)
    res["types"][t] = {"n": int(len(x)), "lib_mean": float(lib.mean()), "lib_median": float(np.median(lib)), "lib_mean_capped": float(libc.mean()),
                       "lib_median_capped": float(np.median(libc)), "share_counts_removed_by_cap": float(1 - libc.sum() / lib.sum()),
                       "share_cells_with_any_capped_entry": float((x > CAP).any(1).mean())}
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res["types"], indent=1))
if SPL == "test":
    with open(f"{W}/runs/r16/TEST_READ_LOG.tsv", "a") as f:
        f.write(f"{time.strftime('%m-%d_%H:%M:%S')}\tHLCA R30 capped-library reference row (real test cells, cap 4096; r30_capped_lib.py {sha(__file__)[:8]})\t{OUT.replace(W + '/', '')}\t{sha(OUT)[:8]}\n")
