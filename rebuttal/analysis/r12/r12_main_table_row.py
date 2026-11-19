#!/usr/bin/env python
r"""R12 main-table CountsDiff row. Reads only runs/cd_hvg2k_3seed/:
  frozen_w1_mmd_pcc.json (W1, MMD², PCC per run; frozen hvg2k ruler, val) and v8_v7_3seed.json (purity v8 per run).
For each target: mean and sd (ddof = 1) over the 3 training seeds, and the range. the 9 numbers (purity, W1, MMD²) are
compared at the precision they were written in: "consistent" iff round(mean) and round(sd, ddof = 1) both equal the written values; the
script also tests whether the written ± could be a half-range or a population sd (ddof = 0).
    python r12_main_table_row.py OUT.md OUT.json
"""
import hashlib, json, sys
import numpy as np
W = "./work"; D = f"{W}/runs/cd_hvg2k_3seed"; out_md, out_js = sys.argv[1:3]
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
FZ, V8, SIDE = f"{D}/frozen_w1_mmd_pcc.json", f"{D}/v8_v7_3seed.json", f"{D}/side_table_3seed_final.json"
F, P, S = json.load(open(FZ)), json.load(open(V8)), json.load(open(SIDE))
TG = ["Endothelial", "Myeloid", "Neuronal"]; SEEDS = [20260625, 20260626, 20260627]
assert F["mode"] == "hvg2k" and S["split"] == "val" and all("n_val" in F["floors"][t] for t in TG)
# the column as written in the task notes 15:36 (value, sd, decimals)
CMD = {"purity": {"Endothelial": (0.894, 0.010), "Myeloid": (0.853, 0.004), "Neuronal": (0.946, 0.031), "dp": 3},
       "W1": {"Endothelial": (0.062, 0.011), "Myeloid": (0.173, 0.011), "Neuronal": (0.057, 0.008), "dp": 3},
       "mmd2": {"Endothelial": (0.0030, 0.0014), "Myeloid": (0.0104, 0.0031), "Neuronal": (0.0098, 0.0004), "dp": 4}}
vals = {}
for t in TG:
    rs = {r["seed"]: r for r in F["runs"] if r["target"] == t and r["method"] == "CountsDiff"}
    assert sorted(rs) == SEEDS and all(rs[s]["train"] == s for s in SEEDS), (t, sorted(rs))
    vals[t] = {"purity": [P[t][str(s)]["v8"] for s in SEEDS], "W1": [rs[s]["W1"] for s in SEEDS],
               "mmd2": [rs[s]["mmd2"] for s in SEEDS], "pcc": [rs[s]["pcc"] for s in SEEDS],
               "arm": sorted({rs[s]["arm"] for s in SEEDS}), "n_generated": sorted({rs[s]["n_generated"] for s in SEEDS})}
    for k, sk in (("W1", "W1"), ("mmd2", "mmd2"), ("pcc", "pcc"), ("purity", "v8")):   # cross-check with the 11-09 side table
        assert vals[t][k] == [S["targets"][t]["countsdiff"][str(s)][sk] for s in SEEDS], (t, k)
st = lambda x: {"mean": float(np.mean(x)), "sd_ddof1": float(np.std(x, ddof=1)), "sd_ddof0": float(np.std(x, ddof=0)),
                "min": float(min(x)), "max": float(max(x)), "half_range": float((max(x) - min(x)) / 2), "values": list(map(float, x))}
res = {"split": "val", "source": {f: sha(f) for f in (FZ, V8, SIDE)}, "script_sha256": sha(__file__), "targets": {}, "check": {}}
for t in TG:
    res["targets"][t] = {k: st(vals[t][k]) for k in ("purity", "W1", "mmd2", "pcc")} | {"arm": vals[t]["arm"], "n_generated": vals[t]["n_generated"]}
for k, c in CMD.items():
    dp = c["dp"]
    for t in TG:
        m, s = c[t]; x = res["targets"][t][k]; r = lambda v: round(v, dp)
        res["check"][f"{k}/{t}"] = {"written": [m, s], "mean_ok": r(x["mean"]) == m, "sd1_ok": r(x["sd_ddof1"]) == s,
                                    "sd0_match": r(x["sd_ddof0"]) == s, "half_range_match": r(x["half_range"]) == s,
                                    "computed": [x["mean"], x["sd_ddof1"]], "computed_rounded": [r(x["mean"]), r(x["sd_ddof1"])]}
json.dump(res, open(out_js, "w"), indent=1)
n_ok = sum(v["mean_ok"] and v["sd1_ok"] for v in res["check"].values()); n_all = len(res["check"])
bad = [k for k, v in res["check"].items() if not (v["mean_ok"] and v["sd1_ok"])]
n_sd1 = sum(v["sd1_ok"] for v in res["check"].values()); n_hr = sum(v["half_range_match"] for v in res["check"].values())
n_sd0 = sum(v["sd0_match"] for v in res["check"].values())
NM = {"purity": "purity(v8)", "W1": "W1", "mmd2": "MMD²", "pcc": "PCC"}
pcc = {t: res["targets"][t]["pcc"] for t in TG}
title = (f"HVG2K Main-table CountsDiff row: PCC is " + ",".join(f"{t} {pcc[t]['mean']:.3f} ± {pcc[t]['sd_ddof1']:.3f}" for t in TG)
         + f"; {n_ok}/{n_all} of the 9 numbers already filled in agree")
L = [f"# {title}", "",
     "**Context: the four columns of the CountsDiff row in the revised main table (HVG2K, 3 targets): purity (v8), W1, MMD², PCC, formatted as mean ± sd over 3 training seeds. This gives the PCC column and rechecks the other 9 numbers.", "",
     "**data.** val split of HVG2K (not test), 3 target classes Endothelial, Myeloid, Neuronal. CountsDiff training seeds 20260625, 20260626, 20260627,"
     "each seed uses its own setting chosen on val (" + ",".join(f"{t} {'/'.join(vals[t]['arm'])}" for t in TG) + "); the number of generated cells per class is matched to the number of val cells of the class ("
     + ",".join(f"{t} {vals[t]['n_generated'][0]:,}" for t in TG) + ").", "",
     "**method.**",
     "- W1, MMD², PCC per seed come from the frozen HVG2K scoring ruler (the same ruler as the other main-table rows); purity is the share of generated cells classified as the target class by the v8 classifier.",
     "- PCC: Pearson correlation between per-gene mean expression (log-CP10K) of generated and real val cells.",
     "- mean ± sd: arithmetic mean over 3 seeds; sd with n − 1 in the denominator:$$\\bar x=\\frac13\\sum_{i=1}^{3}x_i,\\qquad s=\\sqrt{\\frac{1}{3-1}\\sum_{i=1}^{3}(x_i-\\bar x)^2}.$$",
     "- recheck: compare the numbers written in the draft with the mean and sd of this script rounded to the same digits; consistent only if both mean and sd match. Also test whether the ± written equals the half range (max − min)/2 or the sd with n as denominator.", "",
     "**results.**", "", "Table 1: CountsDiff, HVG2K val, mean ± sd over 3 training seeds (n − 1)", "",
     "| target | purity (v8)↑ | W1 ↓ | MMD² ↓ | PCC ↑ |", "|---|---|---|---|---|"]
fmt = {"purity": 3, "W1": 3, "mmd2": 4, "pcc": 3}
for t in TG:
    L.append(f"| {t} | " + " | ".join(f"{res['targets'][t][k]['mean']:.{fmt[k]}f} ± {res['targets'][t][k]['sd_ddof1']:.{fmt[k]}f}" for k in ("purity", "W1", "mmd2", "pcc")) + " |")
L += ["", "Table 2: per-seed numbers (20260625 / 20260626 / 20260627)", "", "| target | purity (v8) | W1 | MMD² | PCC |", "|---|---|---|---|---|"]
for t in TG:
    L.append(f"| {t} | " + " | ".join(" / ".join(f"{v:.{fmt[k] + 1}f}" for v in res['targets'][t][k]['values']) for k in ("purity", "W1", "mmd2", "pcc")) + " |")
L += ["", "Table 3: recheck of 9 numbers already filled in (written value → value rounded by this script)", "", "| column | target | written in the revision | this script (n − 1) | verdict |", "|---|---|---|---|---|"]
for k in CMD:
    dp = CMD[k]["dp"]
    for t in TG:
        c = res["check"][f"{k}/{t}"]; ok = c["mean_ok"] and c["sd1_ok"]
        why = "" if ok else "(" + ",".join(x for x, f in (("mean differs", not c["mean_ok"]), ("sd different", not c["sd1_ok"])) if f) + ")"
        L.append(f"| {NM[k]} | {t} | {c['written'][0]:.{dp}f} ± {c['written'][1]:.{dp}f} | {c['computed_rounded'][0]:.{dp}f} ± {c['computed_rounded'][1]:.{dp}f} | {'consistent' if ok else 'inconsistent' + why} |")
L += ["", f"- 9 {n_ok} of the numbers agree" + ("." if not bad else "; inconsistent are " + ",".join(f"{NM[b.split('/')[0]]} {b.split('/')[1]}" for b in bad) + ", explained one by one below."),
      f"- whether the ± written is an sd or a range: of the 9, {n_sd1} equal the n − 1 sd, {n_hr} equal the half range, {n_sd0} equal the sd with n as denominator."]
for b in bad:
    k, t = b.split("/"); c = res["check"][b]; x = res["targets"][t][k]
    L.append(f"- {NM[k]} {t}: written as {c['written'][0]} ± {c['written'][1]}; the 3 seeds are " + ",".join(f"{v:.4f}" for v in x["values"])
             + f", mean {x['mean']:.5f},sd {x['sd_ddof1']:.5f}, rounded to {CMD[k]['dp']} is {c['computed_rounded'][0]} ± {c['computed_rounded'][1]}.")
L += ["", "**what this means for the rebuttal.**",
      "- supports: the PCC of the CountsDiff row of the main table can be filled directly from Table 1, in the same format as the other three columns (mean ± sd, n − 1).",
      "- needs change:" + ("none." if not bad else "numbers marked inconsistent in Table 3 are corrected per Table 1."),
      "- caveat: all these numbers are on val (the main table uses val), and the sd uses only 3 seeds.", "",
      "> [!summary] summary", f"> CountsDiff PCC on HVG2K val (3 seeds, mean ± sd, n − 1):" + ";".join(f"{t} {pcc[t]['mean']:.3f} ± {pcc[t]['sd_ddof1']:.3f}" for t in TG)
      + f". {n_ok}/{n_all} of the 9 numbers already filled in agree; the ± written is the n − 1 sd ({n_sd1}/9).", "",
      "> [!info]- reproducibility information",
      f"> - inputs:`runs/cd_hvg2k_3seed/frozen_w1_mmd_pcc.json`({sha(FZ)[:8]}),`v8_v7_3seed.json`({sha(V8)[:8]}); with `side_table_3seed_final.json`({sha(SIDE)[:8]}) checked identical seed by seed.",
      f"> - this document:`scripts/r12/r12_main_table_row.py`({sha(__file__)[:8]}); numbers in `runs/r12/R12_main_table_row.json`.",
      "> - scoring:`scripts/r13_score/r13_score.py --mode hvg2k`(the scoring run of 11-09 18:05; PCC is `pearsonr` applied to per-gene mean log-CP10K of real and generated cells); frozen module sha in frozen_sha256 of frozen_w1_mmd_pcc.json (metrics_exp08 6319272f, e21_scc_mmd e1de4039, sliced_w1 62244282)."]
open(out_md, "w").write("\n".join(L) + "\n"); print(title); print("\n".join(L[L.index("Table 3: recheck of 9 numbers already filled in (written value → value rounded by this script)"):]))
