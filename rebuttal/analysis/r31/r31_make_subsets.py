#!/usr/bin/env python
r"""R31: on-target generated cells only. For every array of the v3 manifest (5 methods x 3 seeds x 2 classes)
keep the cells the frozen purity LR (runs/r16/purity_lr/purity_lr_hlca.npz 219e9bce, not refit) predicts as the target, with the exact
prediction of score_with_npz.py (classes[argmax(log_cp10k(x) @ coef.T + intercept)], r13_score.log_cp10k). Generated cells only: no
reference cells are read here, so the subsets are the same for val and test.
Writes runs/r31/subsets/<method>_<seed>_<target>.npy (kept rows, original order), runs/r31/kept.json (n, fraction, sha) and
runs/r31/manifest.json: every subset with n >= 50 (n < 50 -> "too few cells", not scored), plus 2 GATE rows (full gen20260627 arrays,
must reproduce runs/r16/crn_s27/scores_val.json bit for bit).
"""
import hashlib, json, sys
import numpy as np
W = "./work"; sys.path.insert(0, f"{W}/scripts/r13_score"); import r13_score as R
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
NPZ = f"{W}/runs/r16/purity_lr/purity_lr_hlca.npz"; assert sha(NPZ)[:8] == "219e9bce"
z = np.load(NPZ); pred = lambda x: z["classes"][(R.log_cp10k(x) @ z["coef"].T + z["intercept"]).argmax(1)]
man = json.load(open(f"{W}/runs/r24_26_3seed/manifest.json")); import os; os.makedirs(f"{W}/runs/r31/subsets", exist_ok=True)
kept, M = [], []
for e in man:
    x = np.load(e["path"]); keep = pred(x) == e["target"]; n = int(keep.sum())
    f = f"{W}/runs/r31/subsets/{e['method']}_{e['seed']}_{e['target'].replace(' ', '')}.npy"
    np.save(f, x[keep]); k = {"method": e["method"], "seed": e["seed"], "target": e["target"], "source": e["path"], "source_sha256": sha(e["path"]),
                              "n_generated": int(len(x)), "n_kept": n, "frac_kept": n / len(x), "subset": f, "subset_sha256": sha(f)}
    kept.append(k)
    if n >= 50: M.append({"method": e["method"], "arm": "on_target", "seed": e["seed"], "target": e["target"], "path": f})
s27 = json.load(open(f"{W}/runs/r16/crn_s27/manifest.json"))
for e in s27:
    if e["target"] in ("plasma cell", "ionocyte"): M.append({**e, "method": "GATE"})
assert sum(m["method"] == "GATE" for m in M) == 2
json.dump({"purity_npz": NPZ, "purity_npz_sha256": sha(NPZ), "min_n": 50, "script_sha256": sha(__file__), "kept": kept}, open(f"{W}/runs/r31/kept.json", "w"), indent=1)
json.dump(M, open(f"{W}/runs/r31/manifest.json", "w"), indent=1)
print(len(M) - 2, "subsets scored;", sum(k["n_kept"] < 50 for k in kept), "below 50:", [(k["method"], k["seed"], k["target"], k["n_kept"]) for k in kept if k["n_kept"] < 50])
