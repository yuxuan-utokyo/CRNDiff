#!/usr/bin/env python
"""R17: build the normalized input of r17_rules.py from the delivered and existing files only (nothing typed by hand, nothing re-run).
  W1, MMD^2, v8 purity   : r6_numbers.json  grid[tau] (Windows' 3-seed means over samp 20260987/88/89, generator train 20260625)
  coverage, 1-NN         : per seed, r9_new_metrics path on the frozen R3 basis 24d2cc8f:
                           Neuronal all tau -> r3_partB.json; Endothelial tau 0.4 and Myeloid tau 0.2 (deployed) -> r3_numbers.json
                           (arm OURS, train 20260625); the other cells -> runs/r17/r9_new_metrics_<T>.json (computed 11-10 from the
                           12 delivered tau_sweep arrays)
  references             : W1 / MMD^2 floor = the frozen ruler's own floor, 201_tau_sweep_scored.json rows floor_W1 / floor_mmd2 (one
                           value per target; the same ruler as the cells; the author 11-10 12:5x, v1b; v1 used E31 block A); purity = 215_paper_numbers_v8.json real_cells purity_abs; coverage floor =
                           r3_floor_heldout.json heldout mean
  share                  : the target's fraction of data/hvg2k train labels; deployed tau from r6_numbers.json
    python r17_build_inputs.py OUT.json
"""
import hashlib, json, sys
import numpy as np
W = "./work"; L = f"{W}/from_windows/r17_hvg2k_tau"; out = sys.argv[1]
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:8]
F = {"r6": f"{L}/r6_numbers.json", "s201": f"{L}/201_tau_sweep_scored.json", "e31": f"{L}/E31_real_floors.json", "v8": f"{L}/215_paper_numbers_v8.json",
     "partB": f"{W}/from_windows/rebuttal_code/results_for_check/r3_partB.json", "r3n": f"{W}/from_windows/rebuttal_code/results_for_check/r3_numbers.json",
     "floor": f"{W}/runs/r3_floor_heldout.json", "E": f"{W}/runs/r17/r9_new_metrics_Endothelial.json", "M": f"{W}/runs/r17/r9_new_metrics_Myeloid.json"}
J = {k: json.load(open(v)) for k, v in F.items()}
y = np.load(f"{W}/data/hvg2k/train_celltype.npy", allow_pickle=True).astype(str)
SEEDS = (20260987, 20260988, 20260989); TAUS = ("0.2", "0.4", "1.0")


def per_seed(rows, key):
    by = {r["seed"]: r[key] for r in rows}; assert sorted(by) == list(SEEDS), sorted(by); return [by[s] for s in SEEDS]


def cov_rows(t, tau):
    if t == "Neuronal":
        return J["partB"]["tau0.2 (deployed)" if tau == "0.2" else f"tau{tau}"], "r3_partB.json"
    dep = J["r6"][t]["deployed"]
    if float(tau) == dep:
        rs = [dict(r, seed=r["samp"]) for r in J["r3n"][t]["runs"] if r["arm"] == "OURS" and r["train"] == 20260625]
        return rs, "r3_numbers.json (OURS, train 20260625, deployed tau)"
    arm = f"{t}_tau{tau.replace('.', 'p')}"; return J[t[0]]["runs"][arm], f"runs/r17/r9_new_metrics_{t}.json ({arm})"


T = {}; prov = {}; E31V = {}; RULV = {}
for t in ("Endothelial", "Myeloid", "Neuronal"):
    g = J["r6"][t]["grid"]; e31 = J["e31"]["types"][t]["A_train_vs_val_matched_n"]
    fr = [r for r in J["s201"]["rows"] if r["request"] == t]; fw = {r["floor_W1"] for r in fr}; fm = {r["floor_mmd2"] for r in fr}
    assert len(fr) == 9 and len(fw) == 1 and len(fm) == 1, (t, len(fr), fw, fm); E31V[t] = (e31["W1"], e31["MMD2"]); RULV[t] = (fw.pop(), fm.pop())
    d = {"share": float((y == t).mean()), "deployed_tau": J["r6"][t]["deployed"],
         "ref": {"W1": RULV[t][0], "mmd2": RULV[t][1], "purity": J["v8"]["real_cells"][t]["purity_abs"],
                 "coverage": J["floor"][t]["heldout"]["coverage"]["mean"]}, "tau": {}}
    for tau in TAUS:
        rows, src = cov_rows(t, tau); prov[f"{t} tau {tau} coverage/1-NN"] = src
        d["tau"][tau] = {"W1": [g[tau]["W1"]], "mmd2": [g[tau]["mmd2"]], "purity": [g[tau]["v8"]],
                         "coverage": per_seed(rows, "coverage"), "nn1_acc": per_seed(rows, "nn1_acc")}
    T[t] = d
res = {"targets": T,
       "data_note": "HVG2K val split (Endothelial 10,057, Myeloid 2,302, Neuronal 396 cells), generator train 20260625, α = 1, τ ∈ {0.2, 0.4, 1.0} (the R6 τ grid; 0.6 and 0.8 missing),"
                    "3 sampling seeds per cell (20260987/88/89). W1, MMD², purity (CellTypist v8) are means over 3 seeds; coverage and 1-NN are computed per seed in the frozen R3 PCA basis (24d2cc8f), then averaged."
                    "noise floor: coverage uses train vs val in the held-out basis; the purity ceiling is v8 of the real val cells.",
       "floor_note": ["W1, MMD² noise floor (from v1b) uses the frozen ruler own floor (floor_W1, floor_mmd2 in each row of 201_tau_sweep_scored.json, same ruler as the generated cells, as for HLCA):"
                      + ";".join(f"{t} W1 {RULV[t][0]:.5f},MMD² {RULV[t][1]:.6f}" for t in RULV) + ".",
                      "v1 uses block A of E31 (train vs val, matched cell counts); values:"
                      + ";".join(f"{t} W1 {E31V[t][0]:.5f},MMD² {E31V[t][1]:.6f}" for t in E31V) + "; v1 archived."],
       "provenance": prov, "sources": {k: f"{v.split('crndiff-rebuttal/')[-1]} {sha(v)}" for k, v in F.items()}}
json.dump(res, open(out, "w"), indent=1)
for t, d in T.items(): print(t, "share %.4f" % d["share"], "deployed", d["deployed_tau"], "ref", {k: round(v, 5) for k, v in d["ref"].items()})
