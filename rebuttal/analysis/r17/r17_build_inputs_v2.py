#!/usr/bin/env python
"""R17 v2 input: the v1b input (runs/r17/r17_inputs.json, built by r17_build_inputs.py from delivered files only; tau 0.2 / 0.4 / 1.0)
plus the tau 0.6 / 0.8 cells scored on the Mac by scripts/r17/r17_fill_score.sh (runs/r17/fill/<T>/cells.json). Nothing typed by hand.
Per target and tau the number of samp seeds is recorded; --require3 drops a target whose 0.6 / 0.8 cells do not both have 3 seeds.
    python r17_build_inputs_v2.py OUT.json [--require3] [--targets Neuronal ...]
"""
import argparse, hashlib, json, os
W = "./work"
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--require3", action="store_true"); ap.add_argument("--targets", nargs="*")
a = ap.parse_args()
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:8]
base_f = f"{W}/runs/r17/r17_inputs.json"; J = json.load(open(base_f))
T = {}; nseeds = {}; src = {"v1b_input": f"runs/r17/r17_inputs.json {sha(base_f)}"}
for t, d in J["targets"].items():
    if a.targets and t not in a.targets: continue
    cf = f"{W}/runs/r17/fill/{t}/cells.json"
    cells = json.load(open(cf)) if os.path.exists(cf) else {}
    n = {k: len(v["seeds"]) for k, v in cells.items()}
    if a.require3 and not (n.get("0.6") == 3 and n.get("0.8") == 3): continue
    if not cells: continue
    d = json.loads(json.dumps(d))
    for k, v in cells.items():
        assert k not in d["tau"], (t, k, "already in the v1b input")
        d["tau"][k] = {m: v[m] for m in ("W1", "mmd2", "purity", "coverage", "nn1_acc")}
    T[t] = d; nseeds[t] = {**{k: 3 for k in ("0.2", "0.4", "1.0")}, **n}; src[f"cells {t}"] = f"runs/r17/fill/{t}/cells.json {sha(cf)}"
res = {"targets": T, "n_seeds": nseeds,
       "data_note": J["data_note"].replace("τ ∈ {0.2, 0.4, 1.0}(the R6 τ grid; 0.6 and 0.8 missing)", "τ ∈ {0.2, 0.4, 0.6, 0.8, 1.0}(0.6 and 0.8 are R17 cells run later on the 4090)")
                    + " τ 0.6, 0.8: W1 and MMD² scored with the same frozen ruler (bit-identical to Windows), purity from the delivered v8 json, coverage and 1-NN as above; number of seeds per cell in n_seeds.",
       "floor_note": J["floor_note"], "provenance": J["provenance"], "sources": {**J["sources"], **src}}
json.dump(res, open(a.out, "w"), indent=1)
for t in T: print(t, "deployed", T[t]["deployed_tau"], "seeds per tau", nseeds[t])
