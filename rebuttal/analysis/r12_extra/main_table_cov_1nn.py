#!/usr/bin/env python
r"""HVG2K main table, two added columns: coverage and 1-NN. VAL, the table's own reference cells,
PCA basis 24d2cc8f (PCA(50) on train target cells, verified), k = 5, n = min(n_val, 2,500), 5 resamples per run (r3_diversity.py procedure).
Aggregation only, nothing re-scored:
  CRNDiff, MDLM, CFGen, scANVI, scVI : Table 1's three runs per method (w2_runs.TABLE1), rows of r3_numbers.json (Windows full recompute
                                      11-08 21:48, sha 21174efd; Mac reproduced its rows bit for bit with r9_new_metrics.py, 11-07/11-09)
  CountsDiff K128                    : 3 training seeds, runs/cd_hvg2k_k128_3seed/new_metrics_<t>_s<seed>.json (Mac, same code and basis)
  real reference                     : train cells vs val, basis fitted WITHOUT those cells (heldout), 5 draws (runs/r3_floor_heldout.json)
Mean ± sd (ddof 1) over the 3 runs, 3 decimals. Coverage: higher is better; 1-NN accuracy: 0.5 is best.
    python main_table_cov_1nn.py OUT.md OUT.json
"""
import hashlib, json, sys
import numpy as np
W = "./work"; out_md, out_js = sys.argv[1:3]
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
R3 = f"{W}/from_windows/windows_full_r123/r3_numbers.json"; FL = f"{W}/runs/r3_floor_heldout.json"; KD = f"{W}/runs/cd_hvg2k_k128_3seed"
r3, fl = json.load(open(R3)), json.load(open(FL))
TABLE1 = {"OURS": [(20260625, 20260989), (20260627, 20260987), (20260630, 20260989)],          # rebuttal/w2_runs.py lines 19-23, verbatim
          "MDLM": [(20260625, 20260933), (20260934, 20260932), (20260935, 20260932)],
          "CFGen": [("run1", 20260932), ("run2", 20260932), ("run3", 20260933)],
          "scVI": [(20261050, 20261051), (20261053, 20261055), (20261054, 20261055)],
          "scANVI": [(20261060, 20261065), (20261063, 20261061), (20261064, 20261061)]}
TG = ["Endothelial", "Myeloid", "Neuronal"]; ORDER = ["scVI", "CFGen", "scANVI", "MDLM", "CountsDiff", "OURS"]
NAME = {"OURS": "**CRNDiff**", "CountsDiff": "CountsDiff (K128)"}
vals, src_runs = {}, {}
for t in TG:
    runs = r3[t]["runs"]; vals[t] = {}
    for m, pairs in TABLE1.items():
        sel = []
        for tr, sp in pairs:
            hit = [r for r in runs if r["arm"] == m and str(r["train"]) == str(tr) and int(r["samp"]) == sp]; assert len(hit) == 1, (t, m, tr, sp, len(hit))
            sel.append(hit[0])
        vals[t][m] = {"coverage": [r["coverage"] for r in sel], "nn1_acc": [r["nn1_acc"] for r in sel]}; src_runs[f"{t}/{m}"] = [f"{tr}/{sp}" for tr, sp in pairs]
    cd = [json.load(open(f"{KD}/new_metrics_{t}_s{s}.json"))["runs"][f"{t}_K128"][0] for s in (20260625, 20260626, 20260627)]
    vals[t]["CountsDiff"] = {"coverage": [r["coverage"] for r in cd], "nn1_acc": [r["nn1_acc"] for r in cd]}; src_runs[f"{t}/CountsDiff"] = ["20260625/20261061", "20260626/20261061", "20260627/20261061"]
f = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"
best_cov = {t: max(ORDER, key=lambda m: np.mean(vals[t][m]["coverage"])) for t in TG}
best_nn = {t: min(ORDER, key=lambda m: abs(np.mean(vals[t][m]["nn1_acc"]) - 0.5)) for t in TG}
title = ("HVG2K coverage and 1-NN columns added to the main table (val): highest coverage is " + ",".join(f"{t} {NAME.get(best_cov[t], best_cov[t]).strip('*')}" for t in TG)
         + "; 1-NN closest to 0.5 is " + ",".join(f"{t} {NAME.get(best_nn[t], best_nn[t]).strip('*')}" for t in TG))
L = [f"# {title}", "",
     "**Context: add coverage and 1-NN columns to the HVG2K main table, computed on the main table own val reference cells, mean ± sd over 3 runs for every method. recall does not go into the paper.", "",
     "**data.** HVG2K val, 3 targets. CRNDiff, MDLM, CFGen, scANVI, scVI use the 3 runs of Table 1 (the same batch as the other main-table columns); CountsDiff uses the 3 training seeds of K128.", "",
     "**Method.** Only aggregates existing numbers; nothing recomputed. PCA basis 24d2cc8f is PCA(50) (log-CP10K) fitted on the train cells of the target class; each run takes n = min(number of val cells, 2,500) cells, 5 redraws, k = 5."
     "coverage is the fraction of real cells with at least one generated cell in their 5-NN ball (higher is better); 1-NN is the accuracy of guessing the source by nearest neighbour after pooling real and generated cells (0.5 is best). Real reference row: another batch of train cells against val, with the PCA basis fitted without them, mean of 5.", "",
     "**results.**", "", "Table 1: coverage and 1-NN (mean ± sd, 3 runs, n − 1; bold is the best method in the column)", ""]
for t in TG:
    L += [f"**{t}**(val {r3[t]['n_val']:,} cells)", "", "| method | coverage ↑ | 1-NN (0.5 is best) |", "|---|---|---|"]
    for m in ORDER:
        c, n = f(vals[t][m]["coverage"]), f(vals[t][m]["nn1_acc"])
        L.append(f"| {NAME.get(m, m)} | {'**' + c + '**' if best_cov[t] == m else c} | {'**' + n + '**' if best_nn[t] == m else n} |")
    h = fl[t]["heldout"]; L += [f"| real reference (train vs val) | {h['coverage']['mean']:.3f} | {h['nn1_acc']['mean']:.3f} |", ""]
L += ["**what this means for the rebuttal.**",
      "- supports:" + ",".join(f"{t} highest coverage is {NAME.get(best_cov[t], best_cov[t]).strip('*')}" for t in TG) + "(by the mean over 3 runs).",
      "- does not support: any method being close to the real reference. All methods have coverage clearly below the real reference and 1-NN clearly above 0.5.",
      "- caveat: only 3 runs per method; Neuronal val has only 396 cells. The 3 CountsDiff values come from 3 training seeds, the other methods from the 3 runs of Table 1, with the same definitions (same script, same basis, same reference).", "",
      "> [!summary] summary", "> " + title + ". All methods are far from the real reference (lower coverage, higher 1-NN).", "",
      "> [!info]- reproducibility information",
      f"> - baselines and CRNDiff:`from_windows/windows_full_r123/r3_numbers.json`({sha(R3)[:8]}), runs taken as in Table 1 (the runs of each cell are written in the json); CountsDiff:`runs/cd_hvg2k_k128_3seed/new_metrics_*.json`; real reference:`runs/r3_floor_heldout.json`({sha(FL)[:8]}).",
      f"> - this document:`scripts/r12_extra/main_table_cov_1nn.py`({sha(__file__)[:8]})."]
json.dump({"values": vals, "runs": src_runs, "best_coverage": best_cov, "best_1nn": best_nn, "sources": {"r3_numbers": sha(R3)[:8], "floor": sha(FL)[:8]}, "script_sha256": sha(__file__)},
          open(out_js, "w"), indent=1)
open(out_md, "w").write("\n".join(L) + "\n"); print("\n".join(L[:40]))
