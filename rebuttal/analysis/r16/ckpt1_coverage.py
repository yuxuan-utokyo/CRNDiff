#!/usr/bin/env python
"""R16 checkpoint 1, descriptive coverage column. table2_metrics.py v3 (e690a6e4)
is imported, not changed: for each checkpoint-1 row the real test cells and the generated / reference cells are read, transformed,
projected on the loaded shared PCA and matched (5,000 per side, rng seed 20261300) by ITS functions in ITS order, so the cells are the
very ones behind the stored W2 / precision / recall. As a check, precision and recall are recomputed and must equal the stored json
exactly; then
  coverage (Naeem et al. 2020, same k = 3 as precision / recall) = share of real points whose k-NN ball (radius = distance to the k-th
  nearest other real point) contains at least one generated point.
The verdict (W2 only) is not touched. Writes OUT.json and OUT.md (coverage in the main table; recall only in the appendix table).
    python ckpt1_coverage.py OUT_PREFIX
"""
import hashlib, json, os, sys
from types import SimpleNamespace
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
W = "./work"; D = f"{W}/runs/r16/ckpt1"; out = sys.argv[1]
sys.path.insert(0, f"{W}/scripts/r16")
import table2_metrics as T2
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
assert sha(T2.__file__).startswith("e690a6e4"), "table2_metrics changed"
ROWS = [("CRNDiff seed 1", "ckpt1_uncond_s20260625_test.json"), ("CRNDiff seed 3", "ckpt1_uncond_s20260630_test.json"),
        ("CFGen", "ckpt1_uncond_cfgen_test.json"), ("real val vs test (reference)", "ref_realval_vs_test.json")]   # scLDM removed 11-10 14:24 (archived)
res = {"rule": "coverage, k = 3, on the exact checkpoint-1 cells; descriptive, verdict unchanged", "table2_metrics_sha256": sha(T2.__file__),
       "script_sha256": sha(__file__), "rows": {}}
cache = {}
for name, jf in ROWS:
    J = json.load(open(f"{D}/{jf}")); s = J["settings"]; a = SimpleNamespace(**s)
    real_dir = os.path.join(W, J["inputs"]["real"]); gen = os.path.join(W, J["inputs"]["gen"])
    assert sha(gen) == J["inputs"]["gen_sha256"], (name, "generated file changed")
    assert s["mode"] == "uncond" and s["libsize"] == "hvg" and s["pca_load"], s
    key = (real_dir, s["real_split"])
    if key not in cache: cache[key] = T2.read_real(real_dir, s["layer"], s["label_key"], s["genes"], want_libsize=True, split=s["real_split"])
    Xr, yr, gr, _, _ = cache[key]
    Xg, yg, gg, _ = T2.read_gen(gen, s["label_key"], s["gen_layer"]); assert np.array_equal(gr, gg)
    z = np.load(os.path.join(W, s["pca_load"]), allow_pickle=False); assert sha(os.path.join(W, s["pca_load"])) == J["pca"]["file_sha256"]
    Zr, Zg = T2.transform(Xr, s["transform"], None), T2.transform(Xg, s["transform"], None)
    Pr, Pg = (Zr - z["mean"]) @ z["components"].T, (Zg - z["mean"]) @ z["components"].T
    r, g = T2.matched(np.random.default_rng(s["seed"]), Pr, Pg, s["max_cells"])
    p, rc = T2.prec_rec(r, g, s["k"])
    assert p == J["uncond"]["precision"] and rc == J["uncond"]["recall"], (name, "cells differ from the stored run", p, rc, J["uncond"])
    rr = NearestNeighbors(n_neighbors=s["k"] + 1).fit(r).kneighbors(r)[0][:, s["k"]]
    cov = float((cdist(g, r) <= rr[None, :]).any(0).mean())
    res["rows"][name] = {**J["uncond"], "coverage": cov, "source": f"runs/r16/ckpt1/{jf}", "source_sha256": sha(f"{D}/{jf}")[:8], "pr_check": "identical"}
    print(f"{name}: coverage {cov:.4f} (precision/recall reproduce the stored values exactly)")
json.dump(res, open(out + ".json", "w"), indent=1)
L = ["| generator | W2 | MMD² | 1-NN | precision | coverage |", "|---|---|---|---|---|---|"]
for n, x in res["rows"].items():
    L.append(f"| {n} | {x['W2']:.3f} | {x['MMD2']:.5f} | {x['1NN']:.3f} | {x['precision']:.3f} | {x['coverage']:.3f} |")
L += ["", "appendix (recall, not in the main text):", "", "| generator | recall |", "|---|---|"] + [f"| {n} | {x['recall']:.3f} |" for n, x in res["rows"].items()]
open(out + ".md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
