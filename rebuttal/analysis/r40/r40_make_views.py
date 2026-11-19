#!/usr/bin/env python
r"""R40: plasma-cell metrics without immunoglobulin genes. Builds scorer data views for r13_score.py (--mode dg):
  runs/r40/views/<genes>_<split>/ with val.npy / val_celltype.npy = the split's real cells of plasma cell and ionocyte only (rows in the split's
  order), train.npy / train_celltype.npy = the train cells of those two classes, and the column set <genes> = full (all 2,000 HVGs) or noig
  (all HVGs except symbols matching ^IG[HKL]; JCHAIN kept). r13_score only uses the target's own cells (reference, floors, PCA on its train
  cells), so dropping the other classes' rows changes nothing; the full view is the gate (must reproduce Table 3). Generated arrays get the
  same column subset (runs/r40/gen/*.npy) and a manifest per gene set.
    python r40_make_views.py val|test
"""
import hashlib, json, os, re, sys
import numpy as np
SPL = sys.argv[1]; assert SPL in ("val", "test"); W = "./work"; O = f"{W}/runs/r40"
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
G = json.load(open(f"{W}/data/r16_hlca_meta/genes_cfgen_hvg2000.json")); SYM = list(G["feature_name"])
IG = np.array([bool(re.match(r"^IG[HKL]", s)) for s in SYM]); keep = {"full": np.arange(len(SYM)), "noig": np.flatnonzero(~IG)}
assert "JCHAIN" in [SYM[i] for i in keep["noig"]]
ED = f"{W}/data/r16_hlca_score" if SPL == "val" else f"{W}/data/r16_hlca_test_view"
X = np.load(f"{ED}/val.npy", mmap_mode="r"); Y = np.load(f"{ED}/val_celltype.npy", allow_pickle=True).astype(str)
XT = np.load(f"{W}/data/r16_hlca_fit/train.npy", mmap_mode="r"); YT = np.load(f"{W}/data/r16_hlca_fit/train_celltype.npy", allow_pickle=True).astype(str)
CL = ["plasma cell", "ionocyte"]; ri = np.flatnonzero(np.isin(Y, CL)); ti = np.flatnonzero(np.isin(YT, CL))
xv, xt = np.asarray(X[ri]), np.asarray(XT[ti])
rec = {"split": SPL, "n_ig_removed": int(IG.sum()), "ig_genes": [s for s, b in zip(SYM, IG) if b], "views": {}}
for g, cols in keep.items():
    d = f"{O}/views/{g}_{SPL}"; os.makedirs(d, exist_ok=True)
    np.save(f"{d}/val.npy", np.ascontiguousarray(xv[:, cols])); np.save(f"{d}/val_celltype.npy", Y[ri].astype(object), allow_pickle=True)
    np.save(f"{d}/train.npy", np.ascontiguousarray(xt[:, cols])); np.save(f"{d}/train_celltype.npy", YT[ti].astype(object), allow_pickle=True)
    rec["views"][g] = {"n_genes": int(len(cols)), "n_val": int(len(ri)), "n_train": int(len(ti)), "val_sha": sha(f"{d}/val.npy")[:8]}
if SPL == "val":   # generated arrays and manifests (split-independent)
    man = json.load(open(f"{W}/runs/r24_26_3seed/manifest.json")); os.makedirs(f"{O}/gen", exist_ok=True); M = {"full": [], "noig": []}
    for e in man:
        M["full"].append({k: e[k] for k in ("method", "seed", "target", "path")} | {"arm": "full"})
        f = f"{O}/gen/{e['method']}_{e['seed']}_{e['target'].replace(' ', '')}_noig.npy"; np.save(f, np.ascontiguousarray(np.load(e["path"])[:, keep["noig"]]))
        M["noig"].append({k: e[k] for k in ("method", "seed", "target")} | {"arm": "noig", "path": f})
    for g in M: json.dump(M[g], open(f"{O}/manifest_{g}.json", "w"), indent=1)
json.dump(rec, open(f"{O}/views_{SPL}.json", "w"), indent=1); print({g: v for g, v in rec["views"].items()}, "IG removed", rec["n_ig_removed"])
