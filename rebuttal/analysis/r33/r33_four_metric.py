#!/usr/bin/env python
r"""R33: tau selection rule B with FOUR metrics (1-NN dropped), from stored VAL scores only (no sampling, no test).
Rule (unchanged except the dropped dimension): s = (floor_W1 / W1, floor_MMD / MMD^2, purity / real purity, coverage / real coverage),
each capped at 1; pick the tau with the smallest Euclidean distance to the all-ones point; candidates tau in {0.2, 0.4, 0.6, 0.8, 1.0};
ties to the smaller tau. Unrounded numbers throughout.
Inputs:
  heart (R17): runs/r17/ruleB_final/<T>/report_numbers.json (3-seed means, refs, the 5 scores and rule B as deployed);
  HLCA (R16, plasma cell, ionocyte): runs/r16/tau_val/select_<T>.json (the val cells R16 chose tau from: generator 20260625, seed
  20260987) and the floors / real values in runs/r16/report_notes/tau_curves_val.json.
Gate: the 5-metric distance recomputed here must give the stored / deployed pick for every target (heart: equal to report_numbers.json
'B' and its dist values; HLCA: plasma cell 0.2, ionocyte 0.4).
    python r33_four_metric.py OUT.md OUT.json
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
W = Path("./work"); sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]
OM, OJ = Path(sys.argv[1]), Path(sys.argv[2]); TAUS = ["0.2", "0.4", "0.6", "0.8", "1.0"]; SRC = {}
K5 = ["W1", "mmd2", "purity", "coverage", "nn1_acc"]; K4 = K5[:4]
def dist(s, keys): return float(np.sqrt(sum((1 - min(s[k], 1.0)) ** 2 for k in keys)))
def pick(d):  # smallest distance, ties to the smaller tau
    return min(TAUS, key=lambda t: (d[t], float(t)))
res = {}
# heart
for t in ("Endothelial", "Myeloid", "Neuronal"):
    f = W / f"runs/r17/ruleB_final/{t}/report_numbers.json"; SRC[f"heart {t}"] = f"{f.relative_to(W)} {sha(f)}"; R = json.load(open(f))[t]
    ref, mean = R["ref"], R["mean"]
    sc = {tau: {"W1": ref["W1"] / mean[tau]["W1"], "mmd2": ref["mmd2"] / mean[tau]["mmd2"], "purity": mean[tau]["purity"] / ref["purity"],
                "coverage": mean[tau]["coverage"] / ref["coverage"], "nn1_acc": (1 - mean[tau]["nn1_acc"]) / 0.5} for tau in TAUS}
    for tau in TAUS:
        for k in K5: assert abs(sc[tau][k] - R["scores"][tau][k]) < 1e-12, (t, tau, k)
        assert abs(dist(sc[tau], K5) - R["dist"][tau]) < 1e-12, (t, tau)
    d5 = {tau: dist(sc[tau], K5) for tau in TAUS}; assert pick(d5) == R["B"], (t, pick(d5), R["B"])
    res[f"heart {t}"] = {"scores": sc, "deployed": str(R["deployed"]), "n_seeds": "3 mean over sampling seeds", "B5_stored": R["B"]}
# HLCA
TC = W / "runs/r16/report_notes/tau_curves_val.json"; SRC["HLCA floors"] = f"{TC.relative_to(W)} {sha(TC)}"; FL = json.load(open(TC))
for t, fn, dep in (("plasma cell", "plasmacell", "0.2"), ("ionocyte", "ionocyte", "0.4")):
    f = W / f"runs/r16/tau_val/select_{fn}.json"; SRC[f"HLCA {t}"] = f"{f.relative_to(W)} {sha(f)}"; S = json.load(open(f)); fl = FL[t]["floor"]
    assert S["split"] == "val" and S["target"] == t
    cells = {c["tau"]: c for c in S["cells"].values() if c["alpha"] == 1.0}; cells = {f"{k:.1f}": v for k, v in cells.items()}; assert sorted(cells) == TAUS, sorted(cells)
    sc = {tau: {"W1": fl["W1"] / cells[tau]["W1"], "mmd2": fl["mmd2"] / cells[tau]["mmd2"], "purity": cells[tau]["purity_logreg"] / fl["purity_logreg"],
                "coverage": cells[tau]["coverage"] / fl["coverage"], "nn1_acc": (1 - cells[tau]["nn1_acc"]) / 0.5} for tau in TAUS}
    d5 = {tau: dist(sc[tau], K5) for tau in TAUS}; assert pick(d5) == dep, (t, pick(d5), dep)
    res[f"HLCA {t}"] = {"scores": sc, "deployed": dep, "n_seeds": f"1 sampling seeds ({S['seed']})", "B5_stored": dep}
out = {}
L = ["# R33 τ selection: rule B recomputed without 1-NN (four metrics) (val)", "",
     "Uses only stored val scores, no sampling, no test reads. Score s = (W1 floor / W1, MMD² floor / MMD², purity / real purity, coverage / real coverage), each capped at 1; take the τ with the smallest Euclidean distance to the all-ones point; candidates τ ∈ {0.2, 0.4, 0.6, 0.8, 1.0}, ties go to the smaller τ."
     "five-metric distance (original rule B) plus one dimension, the 1-NN score (1 − acc) / 0.5. Normalization, candidate set and val split unchanged. All numbers use unrounded values; tables show 4 digits.", "",
     "check: the five-metric distance is recomputed here; for the 3 heart targets the scores and distances match those stored by R17 item by item (difference < 1e-12), and the selected τ is the same; for the two HLCA classes the selected τ equals the deployed value (plasma cell 0.2, ionocyte 0.4).", ""]
for name, r in res.items():
    sc = r["scores"]; d4 = {tau: dist(sc[tau], K4) for tau in TAUS}; d5 = {tau: dist(sc[tau], K5) for tau in TAUS}
    p4, p5 = pick(d4), pick(d5); o4 = sorted(TAUS, key=lambda t: (d4[t], float(t))); o5 = sorted(TAUS, key=lambda t: (d5[t], float(t)))
    out[name] = {"pick4": p4, "pick5": p5, "deployed": r["deployed"], "runner_up4": o4[1], "margin4": d4[o4[1]] - d4[p4], "runner_up5": o5[1], "margin5": d5[o5[1]] - d5[p5],
                 "dist4": d4, "dist5": d5, "scores": sc}
    L += [f"**{name}**({r['n_seeds']}; deployed τ {r['deployed']})", "",
          "| τ | W1 score | MMD² score | purity score | coverage score | 1-NN score | four-metric distance | five-metric distance |", "|---|---|---|---|---|---|---|---|"]
    for tau in TAUS:
        b4 = "**" if tau == p4 else ""; b5 = "**" if tau == p5 else ""
        L.append(f"| {tau} | " + " | ".join(f"{sc[tau][k]:.4f}" for k in K5) + f" | {b4}{d4[tau]:.4f}{b4} | {b5}{d5[tau]:.4f}{b5} |")
    L += ["", f"- four metrics select τ {p4} (runner-up {o4[1]}, gap {d4[o4[1]] - d4[p4]:.4f}); five metrics select τ {p5} (runner-up {o5[1]}, gap {d5[o5[1]] - d5[p5]:.4f})."
          + ("both rules select the same." if p4 == p5 else f"**the two rules select differently**.") + ("" if p4 == r["deployed"] else f" selection and deployment by four metrics τ {r['deployed']} different."), ""]
n_same = sum(o["pick4"] == o["pick5"] for o in out.values()); n_dep = sum(o["pick4"] == o["deployed"] for o in out.values())
L.insert(2, f"**conclusion (computed in code)**: of {len(out)} targets, four and five metrics select the same τ for {n_same}; the τ chosen by four metrics equals the deployed value for {n_dep}."
            + ";".join(f"{n} four metrics {o['pick4']} / five metrics {o['pick5']}" for n, o in out.items()) + ".")
L.insert(3, "")
L += ["Bold in the table is the τ selected by this rule. Scores above 1 are capped at 1 before computing the distance (the table shows uncapped values).", "",
      "> [!info]- reproducibility information", "> - inputs:" + ";".join(f"{k}:{v}" for k, v in SRC.items()) + ".",
      "> - Heart scores use the mean over 3 sampling seeds (input of R17 rule B); HLCA scores use the val cells used when choosing τ in R16 (generator 20260625, sampling seed 20260987, 1 seed). HLCA purity is the frozen logistic-regression purity (the same as the R16 rule); the real values of coverage and 1-NN are the train-vs-val floor.",
      f"> - this document:`scripts/r33/r33_four_metric.py`({sha(__file__)}); reads only the json above."]
OM.write_text("\n".join(L) + "\n"); json.dump({"targets": out, "sources": SRC, "script_sha256": sha(__file__)}, open(OJ, "w"), indent=1)
print("\n".join(L[:4]))
