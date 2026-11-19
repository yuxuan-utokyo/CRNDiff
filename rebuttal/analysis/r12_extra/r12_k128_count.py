#!/usr/bin/env python
r"""CRNDiff vs CountsDiff K128 cell counts on HVG2K val, from unrounded per-run values. Three cell sets, each
3 targets x 4 metrics = 12 cells, stated explicitly:
  S1 main table's own columns: purity v8, W1, MMD², PCC  
  S2 reply set: W1, MMD², purity v8, coverage
  S3 appendix set: W1, MMD², purity v8, recall
CRNDiff = Table 1's three runs (runs/cd_hvg2k_3seed/side_table_3seed_final.json ours_values, the 11-09 source); CountsDiff K128 = three
training seeds (runs/cd_hvg2k_k128_3seed/k128_rows.json: Mac frozen ruler + r9_new_metrics, Windows v8). Two counts per set:
  "mean better": CRNDiff's 3-run mean is better than CountsDiff's 3-seed mean (lower W1 / MMD², higher purity / PCC / coverage / recall);
  "ranges separate": the two [min, max] ranges do not overlap (fill sheet 11-09 section E rule).
    python r12_k128_count.py OUT.md OUT.json
"""
import hashlib, json, sys
import numpy as np
W = "./work"; out_md, out_js = sys.argv[1:3]
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
SIDE = f"{W}/runs/cd_hvg2k_3seed/side_table_3seed_final.json"; KR = f"{W}/runs/cd_hvg2k_k128_3seed/k128_rows.json"
S, K = json.load(open(SIDE)), json.load(open(KR))
CN = f"{W}/runs/r12_cov1nn/R12_main_table_cov_1nn.json"; C1 = json.load(open(CN))   # v2: 1-NN (and coverage) per run, Table 1 runs / K128 seeds
TG = ["Endothelial", "Myeloid", "Neuronal"]; LOWER = {"nn1": 2, "W1": 1, "mmd2": 1, "v8": 0, "pcc": 0, "coverage": 0, "recall": 0}
NM = {"nn1": "1-NN(0.5 is best)", "W1": "W1", "mmd2": "MMD²", "v8": "purity(v8)", "pcc": "PCC", "coverage": "coverage", "recall": "recall"}
ours = lambda t, k: C1["values"][t]["OURS"]["nn1_acc"] if k == "nn1" else S["targets"][t]["cells"][k]["ours_values"]
cd = lambda t, k: C1["values"][t]["CountsDiff"]["nn1_acc"] if k == "nn1" else K["cd_k128"][t][k]
SETS = {"S4 six main-table columns (purity, W1, MMD², PCC, coverage, 1-NN; 18 cells)": ["v8", "W1", "mmd2", "pcc", "coverage", "nn1"], "S1 four main-table columns (purity, W1, MMD², PCC)": ["v8", "W1", "mmd2", "pcc"], "S2 for the response (W1, MMD², purity, coverage)": ["W1", "mmd2", "v8", "coverage"],
        "S3 appendix (W1, MMD², purity, recall)": ["W1", "mmd2", "v8", "recall"]}
res = {"sets": {}, "sources": {"side_table": sha(SIDE)[:8], "k128_rows": sha(KR)[:8]}, "script_sha256": sha(__file__)}
L = []
for name, ks in SETS.items():
    rows = []
    for t in TG:
        for k in ks:
            o, c = ours(t, k), cd(t, k); mo, mc = float(np.mean(o)), float(np.mean(c))
            better = (abs(mo - 0.5) < abs(mc - 0.5)) if LOWER[k] == 2 else (mo < mc if LOWER[k] else mo > mc)
            sep = max(o) < min(c) or max(c) < min(o)
            rows.append({"target": t, "metric": k, "crndiff_values": o, "cd_values": c, "crndiff_mean": mo, "cd_mean": mc, "crndiff_better_mean": bool(better), "ranges_separate": bool(sep)})
    N = len(rows); nb = sum(r["crndiff_better_mean"] for r in rows); ns = sum(r["ranges_separate"] for r in rows); nsb = sum(r["ranges_separate"] and r["crndiff_better_mean"] for r in rows)
    res["sets"][name] = {"metrics": ks, "rows": rows, "crndiff_better_mean": nb, "ranges_separate": ns, "separate_and_crndiff_better": nsb}
    L += [f"### {name}: CRNDiff mean better {nb}/{N}, ranges separated {ns}/{N} (CRNDiff better in {nsb})", "",
          "| target | metric | CRNDiff 3 runs (unrounded mean) | CountsDiff K128 3 seeds | CRNDiff mean better | ranges separated |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['target']} | {NM[r['metric']]} | {r['crndiff_mean']:.6g}({min(r['crndiff_values']):.6g}–{max(r['crndiff_values']):.6g}) | "
                 f"{r['cd_mean']:.6g}({min(r['cd_values']):.6g}–{max(r['cd_values']):.6g}) | {'is' if r['crndiff_better_mean'] else '**no**'} | {'separately' if r['ranges_separate'] else 'overlap'} |")
    L.append("")
s4 = res["sets"]["S4 six main-table columns (purity, W1, MMD², PCC, coverage, 1-NN; 18 cells)"]; lost4 = [f"{r['target']} {NM[r['metric']]}" for r in s4["rows"] if not r["crndiff_better_mean"]]
s1 = res["sets"]["S1 four main-table columns (purity, W1, MMD², PCC)"]; lost = [f"{r['target']} {NM[r['metric']]}" for r in s1["rows"] if not r["crndiff_better_mean"]]
three = [r for r in s1["rows"] if r["metric"] in ("W1", "mmd2", "pcc")]
title = f"CRNDiff for CountsDiff K128 (HVG2K val): among the 18 cells of the six main-table columns, CRNDiff mean is better in {s4['crndiff_better_mean']}/18, ranges kept separate {s4['ranges_separate']}/18; the 12 cells of the original four columns are {s1['crndiff_better_mean']}/12,{s1['ranges_separate']}/12"
H = [f"# {title}", "",
     "**Context: recount on the main table own four columns (purity v8, W1, MMD², PCC) × 3 targets with unrounded numbers, giving counts for both mean is better and ranges separated, and stating which cells each count uses.", "",
     "**data.** HVG2K val. CRNDiff = the 3 runs of Table 1 (20260625/20260989, 20260627/20260987, 20260630/20260989); CountsDiff = K128, guidance 1.0, training seeds 20260625/26/27.", "",
     "**Method.** For each cell compare the 3 values on each side: which mean is better (lower W1, MMD² is better; higher for the rest); whether the two [min, max] ranges are disjoint. The three sets of cells are counted separately and listed cell by cell below.", "",
     "**results.**", "",
     f"- S4(six main-table columns, including the newly added coverage and 1-NN): cells where CountsDiff has the better mean:{','.join(lost4) if lost4 else 'none'}. For 1-NN, closer to 0.5 is better.",
     f"- S1(four main-table columns): cells where the CRNDiff mean is not better:{','.join(lost) if lost else 'none'}. Looking only at the nine W1, MMD², PCC cells: CRNDiff mean is better {sum(r['crndiff_better_mean'] for r in three)}/9.",
     "- difference from the earlier 8 of 9 cells: those nine cells were W1, MMD² and recall (12 cells with purity, i.e. S3 below), without PCC; with PCC included, CountsDiff leads in more cells.", ""] + L + [
     "**what this means for the rebuttal.**",
     f"- supports: on coverage metrics such as coverage and recall, CRNDiff is clearly better on all three targets with separated ranges (S2, S3).",
     f"- does not support: [CRNDiff is better on every column of the main table]. On four columns of the main table CountsDiff has the better mean {12 - s1['crndiff_better_mean']} cells ({','.join(lost)}); in these cells the ranges of both sides overlap.",
     "- when citing counts in the response, state which set of cells they refer to.", "",
     "> [!summary] summary", f"> four main-table columns, 12 cells: CRNDiff mean is better {s1['crndiff_better_mean']}/12, ranges kept separate {s1['ranges_separate']}/12; the 12 cells of W1, MMD², purity, coverage used in the response:{res['sets']['S2 for the response (W1, MMD², purity, coverage)']['crndiff_better_mean']}/12,{res['sets']['S2 for the response (W1, MMD², purity, coverage)']['ranges_separate']}/12. The difference comes from the PCC and coverage columns.", "",
     "> [!info]- reproducibility information", f"> - `scripts/r12_extra/r12_k128_count.py`({sha(__file__)[:8]}); input side_table_3seed_final.json ({res['sources']['side_table']}), k128_rows.json ({res['sources']['k128_rows']})."]
json.dump(res, open(out_js, "w"), indent=1); open(out_md, "w").write("\n".join(H) + "\n"); print("\n".join(H[:16]))
