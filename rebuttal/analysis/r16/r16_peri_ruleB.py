#!/usr/bin/env python
r"""Lung pericyte tau by rule B on VAL (seed 1 = generator 20260625, alpha 1, tau grid {0.2, 0.4, 0.6, 0.8, 1.0}); the rule exactly as for
plasma cell / ionocyte (code = rules() in r16_celltypist_check.py, LR purity = the main ruler, the author 00:55):
  relative scores W1 floor / W1, MMD^2 floor / MMD^2, purity / LR ceiling, coverage / held-out floor, (1 - 1-NN acc) / 0.5, each capped at 1;
  B = the tau with the smallest Euclidean distance to the all-ones point (ties -> smaller tau). Rules A and C are printed for reference.
Inputs: runs/r16/tau_val/select_lungpericyte.json and scores_lungpericyte_val.json (r16_peri_tau_score.sh). Val only.
    python r16_peri_ruleB.py OUT.json
"""
import hashlib, json, sys
import numpy as np
W = "./work"; R = f"{W}/runs/r16"; out = sys.argv[1]; t, k = "lung pericyte", "lungpericyte"
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
SF, FF = f"{R}/tau_val/select_{k}.json", f"{R}/tau_val/scores_{k}_val.json"
S = json.load(open(SF)); F = json.load(open(FF))["floors"][t]
cell = {c["tau"]: c for c in S["cells"].values() if c["alpha"] == 1.0}; taus = sorted(cell); assert taus == [0.2, 0.4, 0.6, 0.8, 1.0], taus
ref = {"W1": F["W1"]["W1"], "mmd2": F["e21"]["mmd2"], "purity": S["ceiling"], "coverage": F["two_sample_heldout_basis"]["coverage"]["mean"]}
sc = {x: {"W1": ref["W1"] / cell[x]["W1"], "mmd2": ref["mmd2"] / cell[x]["mmd2"], "purity": cell[x]["purity_logreg"] / ref["purity"],
          "coverage": cell[x]["coverage"] / ref["coverage"], "nn1_acc": (1 - cell[x]["nn1_acc"]) / 0.5} for x in taus}
cap = {x: {m: min(1.0, v) for m, v in sc[x].items()} for x in taus}
dist = {x: float(np.sqrt(sum((1 - v) ** 2 for v in cap[x].values()))) for x in taus}
A = next((x for x in taus if cell[x]["purity_logreg"] >= 0.9 * ref["purity"]), None)
B = min(taus, key=lambda x: (dist[x], x))
C = max(taus, key=lambda x: (np.exp(np.mean([np.log(max(v, 1e-12)) for v in cap[x].values()])), -x))
res = {"target": t, "B": B, "A": A, "C": C, "dist": dist, "scores": sc, "ref": ref, "raw": {x: {m: cell[x][m] for m in ("W1", "mmd2", "purity_logreg", "coverage", "nn1_acc")} for x in taus},
       "inputs": {SF.replace(W + "/", ""): sha(SF)[:8], FF.replace(W + "/", ""): sha(FF)[:8]}, "script_sha256": sha(__file__)}
json.dump(res, open(out, "w"), indent=1)
print(f"lung pericyte rule B tau = {B} (A {A}, C {C}); distance", {x: round(v, 4) for x, v in dist.items()})
for x in taus: print(" ", x, {m: round(v, 3) for m, v in sc[x].items()})
