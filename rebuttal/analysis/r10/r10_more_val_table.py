#!/usr/bin/env python
r"""R10 conditional arm, generators s17 / s18 / s19 side by side on VAL (the author 11:2x / 11:3x: val only, no test; rows of s18 / s19 labelled
early-stopped). Reads only the score jsons written by scripts/r10/score_cond.sh (R3 coverage etc., R1 / R2, Fano) and r13_score.py
--ref-split val (frozen W1 / MMD^2 / PCC); s17's frozen-ruler row is the 11:3x re-score that reproduced the R10 report (gate).
Purity v8 / v7 is Windows' (heart CellTypist model not on the Mac): filled from a json if given, else marked pending.
    python r10_more_val_table.py OUT.md OUT.json [PURITY.json]
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
W = Path("./work"); X = W / "runs/r10_extra"; M = W / "runs/r10_more"; R = W / "from_windows/r10_more/raw"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]
ARMS = {"s17": ("Neuronal_cond_frac010", 20260987, M / "w1mmd_val_s17_gate.json"), "s18": ("Neuronal_cond_frac010_s18", 20260988, M / "w1mmd_val_s18s19.json"),
        "s19": ("Neuronal_cond_frac010_s19", 20260989, M / "w1mmd_val_s18s19.json")}
TR = {s: json.load(open(R / f"r10_cond_frac010_s2026081{s[-1]}.json")) for s in ARMS}
rows, src = {}, {}
for s, (arm, seed, wf) in ARMS.items():
    r3 = json.load(open(X / f"r3_{arm}.json")); assert r3["split"] == "val"; r3 = r3["runs"][arm]; assert len(r3) == 1 and r3[0]["file"] == f"seed{seed}.npy" and r3[0]["n_cells"] == 2500; r3 = r3[0]
    r12 = json.load(open(X / f"r1r2_{arm}.json")); assert r12["split"] == "val" if "split" in r12 else True; r12 = [x for x in r12["runs"] if x["arm"] == arm]; assert len(r12) == 1; r12 = r12[0]
    fa = json.load(open(X / f"fano_{arm}.json")); assert fa["split"] == "val"
    wm = json.load(open(wf)); assert wm["ref_split"] == "val"; w = [x for x in wm["runs"] if x["arm"] == arm]; assert len(w) == 1; w = w[0]
    t = TR[s]
    rows[s] = {"W1": w["W1"], "mmd2": w["mmd2"], "pcc": w["pcc"], "var_ratio_top10": r3["var_ratio_top10"], "recall": r3["recall"], "recall_sd": r3["recall_sd"],
               "coverage": r3["coverage"], "coverage_sd": r3["coverage_sd"], "precision": r3["precision"], "density": r3["density"], "nn1_acc": r3["nn1_acc"],
               "zero": r12["r2"]["zero_overall"], "lib_mean": r12["r2"]["lib_mean"], "lib_median": r12["r2"]["lib_median"], "detected": r12["r2"]["detected_mean"],
               "fano": fa["fano"], "r1_pca50": r12["r1"]["pca50"]["mean_neuronal_frac"], "r1_full": r12["r1"]["full"]["mean_neuronal_frac"],
               "stop_reason": t.get("stop_reason"), "best_val_step": t.get("best_val_step"), "steps_done": t.get("steps_done"), "best_val": t.get("best_val"), "sample_seed": seed}
    src[s] = {"r3": sha(X / f"r3_{arm}.json"), "r1r2": sha(X / f"r1r2_{arm}.json"), "fano": sha(X / f"fano_{arm}.json"), "w1mmd": sha(wf), "train_json": sha(R / f"r10_cond_frac010_s2026081{s[-1]}.json")}
FL = json.load(open(M / "w1mmd_val_s17_gate.json"))["floors"]["Neuronal"]
PUR = {}
if len(sys.argv) > 3:   # Windows rebuttal_R10_cond_3gen_v8.json (t215 v8 rescore of rebuttal_R10_cond_3gen_scored.json); rows keyed ..._seed<S>
    V = json.load(open(sys.argv[3]))["rows"]; src["purity_v8_json"] = sha(sys.argv[3])
    for s_, (arm, seed, _) in ARMS.items():
        k = [k for k, r in V.items() if r["arm"] == "Neuronal_cond_frac010" and k.endswith(f"_seed{seed}")]; assert len(k) == 1 and V[k[0]]["n"] == 2500, (s_, k)
        PUR[s_] = {"v8": V[k[0]]["purity_v8"], "v7": V[k[0]]["purity_v7"]}
    assert abs(PUR["s17"]["v8"] - 0.359) < 5e-4 and abs(PUR["s17"]["v7"] - 0.014) < 5e-4   # R10 report values
pur = lambda s, k: f"{PUR[s][k]:.3f}" if s in PUR and k in PUR[s] else "pending Windows"
lab = {"s17": "s17(original conditional arm, fully trained for 40,985 steps)"}
for s in ("s18", "s19"): lab[s] = f"{s}(early-stopped: best at step {rows[s]['best_val_step']} steps, number {rows[s]['steps_done']} steps stopped by the divergence guard)"
L = ["Table R10-more: three training seeds of the conditional arm on val (Neuronal, 10% labelled; 1 sampling seed per generator, 2,500 cells; reference = 396 val Neuronal cells)", "",
     "| generator | v8 ↑ | v7 ↑ | W1 ↓ | MMD² ↓ | PCC ↑ | variance ratio first 10 PCs | recall ↑ | coverage ↑ | precision | 1-NN (→ 0.5) | Neuronal neighbour fraction (PCA50 / all genes) | Fano | best val loss |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for s in ARMS:
    r = rows[s]
    L.append(f"| {lab[s]} | {pur(s, 'v8')} | {pur(s, 'v7')} | {r['W1']:.4f} | {r['mmd2']:.4f} | {r['pcc']:.3f} | {r['var_ratio_top10']:.3f} | {r['recall']:.3f}({r['recall_sd']:.3f}) | {r['coverage']:.3f}({r['coverage_sd']:.3f}) | {r['precision']:.3f} | {r['nn1_acc']:.3f} | {r['r1_pca50']:.3f} / {r['r1_full']:.3f} | {r['fano']:.3f} | {r['best_val']:.5f} |" if isinstance(r["best_val"], float) else
             f"| {lab[s]} | {pur(s, 'v8')} | {pur(s, 'v7')} | {r['W1']:.4f} | {r['mmd2']:.4f} | {r['pcc']:.3f} | {r['var_ratio_top10']:.3f} | {r['recall']:.3f}({r['recall_sd']:.3f}) | {r['coverage']:.3f}({r['coverage_sd']:.3f}) | {r['precision']:.3f} | {r['nn1_acc']:.3f} | {r['r1_pca50']:.3f} / {r['r1_full']:.3f} | {r['fano']:.3f} | {r['best_val']} |")
L.append(f"| noise floor (train vs val) | — | — | {FL['W1']['W1']:.4f} | {FL['e21']['mmd2']:.4f} | {FL['e21']['pcc']:.3f} | — | — | — | — | — | — | — | — |")
L += ["", "recall, the value in parentheses for coverage is the sd over 5 redraws; W1 / MMD² / PCC use the frozen ruler (r13_score.py --ref-split val); the s17 row is rescored and matches the R10 numbers digit for digit."]
open(sys.argv[1], "w").write("\n".join(L) + "\n")
json.dump({"rows": rows, "floor": {"W1": FL["W1"]["W1"], "mmd2": FL["e21"]["mmd2"], "pcc": FL["e21"]["pcc"]}, "purity": PUR, "sources": src, "script_sha256": sha(__file__)}, open(sys.argv[2], "w"), indent=1)
print("\n".join(L))
