#!/usr/bin/env python
r"""HLCA PCC real-cell reference row (the author 20:40: "real-cell reference row (val cells vs test, state this)"). For each reported target
: Pearson correlation between the per-gene mean of ALL val cells of the target and the
per-gene mean of ALL test cells of the target, both after the frozen e21 transform (CP10K + log1p, float32) -- the transform the PCC column
uses for every method (r13_score.py: pearsonr(ref.mean(0, float64), gen.mean(0, float64)) after e21.cp10k_log1p). Test is read once:
refuses if OUT exists. The method rows' PCC are already in the logged test reads (ruleA_test, cd_k128_test).
    python r16_pcc_ref_val_vs_test.py OUT.json
"""
import hashlib, importlib.util, json, os, sys, time
import numpy as np
from scipy.stats import pearsonr
W = "./work"; out = sys.argv[1]
if os.path.exists(out): sys.exit(f"{out} exists: test already read for this row")
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
s = importlib.util.spec_from_file_location("e21", f"{W}/CRNDiff/evaluation/frozen/e21_scc_mmd.py"); E = importlib.util.module_from_spec(s); s.loader.exec_module(E)
V, T = f"{W}/data/r16_hlca_score", f"{W}/data/r16_hlca_test_view"     # test view: test cells stored under the val name (R16 convention)
xv, yv = np.load(f"{V}/val.npy", mmap_mode="r"), np.load(f"{V}/val_celltype.npy", allow_pickle=True).astype(str)
xt, yt = np.load(f"{T}/val.npy", mmap_mode="r"), np.load(f"{T}/val_celltype.npy", allow_pickle=True).astype(str)
res = {"definition": "pearsonr(mean_genes(cp10k_log1p(val target cells)), mean_genes(cp10k_log1p(test target cells))), e21 transform, 2,000 genes",
       "val_npy_sha256": sha(f"{V}/val.npy")[:16], "test_view_npy_sha256": sha(f"{T}/val.npy")[:16], "read_at": time.strftime("%Y-%m-%d %H:%M %Z"),
       "script_sha256": sha(__file__), "targets": {}}
for t in ("plasma cell", "ionocyte"):
    a, _ = E.cp10k_log1p(np.asarray(xv[np.flatnonzero(yv == t)])); b, _ = E.cp10k_log1p(np.asarray(xt[np.flatnonzero(yt == t)]))
    res["targets"][t] = {"n_val": len(a), "n_test": len(b), "pcc_val_vs_test": float(pearsonr(a.mean(0, dtype=np.float64), b.mean(0, dtype=np.float64)).statistic)}
    print(t, res["targets"][t])
json.dump(res, open(out, "w"), indent=1)
