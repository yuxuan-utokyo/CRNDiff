#!/usr/bin/env python
r"""CountsDiff HVG2K at K = 128, 3 training seeds, VAL. Reads runs/cd_hvg2k_k128_3seed/ (Mac: frozen ruler
W1 / MMD² / PCC, r9_new_metrics recall / coverage) and the CRNDiff Table 1 values from runs/cd_hvg2k_3seed/side_table_3seed_final.json
(ours_values, the same source as the 11-09 side table). Purity v8 for K128 is a Windows item: if V8_JSON is given ({target: {seed: v8}})
it is used, else those cells are "pending".
Outputs: main-table row (mean ± sd, ddof 1) for purity / W1 / MMD² / PCC; the 12-cell comparison (W1, MMD², purity v8, recall x 3 targets)
counted two ways, as before: (i) CRNDiff 3-run mean better; (ii) ranges [min, max] disjoint (11-09 fill sheet section E rule);
and the old (per-target K) vs K128 CountsDiff means.
    python cd_k128_rows.py OUT.md OUT.json [V8_JSON]
"""
import hashlib, json, sys
import numpy as np
W = "./work"; D = f"{W}/runs/cd_hvg2k_k128_3seed"; out_md, out_js = sys.argv[1:3]
v8 = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
TG = ["Endothelial", "Myeloid", "Neuronal"]; SEEDS = [20260625, 20260626, 20260627]
F = json.load(open(f"{D}/frozen_w1_mmd_pcc.json")); SIDE = json.load(open(f"{W}/runs/cd_hvg2k_3seed/side_table_3seed_final.json"))
OLD = json.load(open(f"{W}/runs/cd_hvg2k_3seed/frozen_w1_mmd_pcc.json"))
assert all(F["floors"][t] == OLD["floors"][t] for t in TG), "floors differ from the K-selected scoring"
cd = {}
for t in TG:
    rs = {r["seed"]: r for r in F["runs"] if r["target"] == t}; assert sorted(rs) == SEEDS
    nm = {s: json.load(open(f"{D}/new_metrics_{t}_s{s}.json"))["runs"][f"{t}_K128"][0] for s in SEEDS}
    cd[t] = {"W1": [rs[s]["W1"] for s in SEEDS], "mmd2": [rs[s]["mmd2"] for s in SEEDS], "pcc": [rs[s]["pcc"] for s in SEEDS],
             "recall": [nm[s]["recall"] for s in SEEDS], "coverage": [nm[s]["coverage"] for s in SEEDS],
             "v8": [v8[t][str(s)] for s in SEEDS] if v8 else None}
ours = {t: {k: SIDE["targets"][t]["cells"][k]["ours_values"] for k in ("W1", "mmd2", "v8", "recall", "coverage") if "ours_values" in SIDE["targets"][t]["cells"][k]} for t in TG}
oldcd = {t: {k: SIDE["targets"][t]["cells"][k]["cd_values"] for k in ("W1", "mmd2", "v8", "recall", "coverage")} for t in TG}
LOWER = {"W1": True, "mmd2": True, "v8": False, "recall": False, "coverage": False}
# v2: main count uses W1, MMD², purity, COVERAGE; the recall version is kept
# as an appendix section, unchanged.
def compare(fourth):
    cmp = []
    for t in TG:
        for k in ("W1", "mmd2", "v8", fourth):
            c, o = cd[t][k], ours[t][k]
            if c is None: cmp.append({"target": t, "metric": k, "status": "pending (v8 from Windows)"}); continue
            better = (np.mean(o) < np.mean(c)) if LOWER[k] else (np.mean(o) > np.mean(c))
            disjoint = max(o) < min(c) or max(c) < min(o)
            cmp.append({"target": t, "metric": k, "crndiff_mean": float(np.mean(o)), "crndiff_range": [min(o), max(o)], "cd_mean": float(np.mean(c)),
                        "cd_range": [min(c), max(c)], "crndiff_better_mean": bool(better), "ranges_disjoint": bool(disjoint)})
    done = [x for x in cmp if "crndiff_better_mean" in x]
    return cmp, done, sum(x["crndiff_better_mean"] for x in done), sum(x["ranges_disjoint"] for x in done), sum(x["ranges_disjoint"] and x["crndiff_better_mean"] for x in done)
cmpC, doneC, nbC, ndC, ndbC = compare("coverage"); cmp, done, n_b, n_d, n_db = compare("recall")
ms = lambda v, d: f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}"
rng = lambda v, d: f"{np.mean(v):.{d}f}({min(v):.{d}f}–{max(v):.{d}f})"
NM = {"W1": "W1", "mmd2": "MMD²", "v8": "purity(v8)", "recall": "recall", "coverage": "coverage", "pcc": "PCC"}; DP = {"W1": 4, "mmd2": 5, "v8": 4, "recall": 3, "coverage": 3, "pcc": 4}
L = [f"CountsDiff K128, HVG2K val, 3 training seeds (W1, MMD², purity, coverage × 3 targets; {len(doneC)}/12 cells available): CRNDiff mean better in {nbC}/{len(doneC)} cells, ranges separated in {ndC}/{len(doneC)} cells (CRNDiff better in {ndbC})", "",
     "main-table row (`tab:hvg_single_type_main`, mean ± sd, n − 1):", "", "| target | purity (v8) | W1 | MMD² | PCC |", "|---|---|---|---|---|"]
for t in TG:
    L.append(f"| {t} | " + (ms(cd[t]['v8'], 3) if cd[t]["v8"] else "pending Windows") + f" | {ms(cd[t]['W1'], 3)} | {ms(cd[t]['mmd2'], 4)} | {ms(cd[t]['pcc'], 3)} |")
def table(cmp_):
    out = ["| target | metric | CRNDiff | CountsDiff K128 | CRNDiff mean better | ranges separated |", "|---|---|---|---|---|---|"]
    for x in cmp_:
        if "crndiff_better_mean" not in x: out.append(f"| {x['target']} | {NM[x['metric']]} | — | pending Windows | — | — |"); continue
        d = DP[x["metric"]]; o = ours[x["target"]][x["metric"]]; c = cd[x["target"]][x["metric"]]
        out.append(f"| {x['target']} | {NM[x['metric']]} | {rng(o, d)} | {rng(c, d)} | {'is' if x['crndiff_better_mean'] else 'no'} | {'separately' if x['ranges_disjoint'] else 'overlap'} |")
    return out
L += ["", "12 cell comparison (for the response: item 4 is coverage; CRNDiff = the three runs of Table 1; mean (min–max)):", ""] + table(cmpC)
L += ["", f"appendix: old version with recall as item 4 (recall only in the appendix): CRNDiff mean better in {n_b}/{len(done)} cells, ranges separated in {n_d}/{len(done)} cells (CRNDiff better in {n_db})", ""] + table(cmp)
for x in []:
    if "crndiff_better_mean" not in x: L.append(f"| {x['target']} | {NM[x['metric']]} | — | pending Windows | — | — |"); continue
    d = DP[x["metric"]]; o = ours[x["target"]][x["metric"]]; c = cd[x["target"]][x["metric"]]
    L.append(f"| {x['target']} | {NM[x['metric']]} | {rng(o, d)} | {rng(c, d)} | {'is' if x['crndiff_better_mean'] else 'no'} | {'separately' if x['ranges_disjoint'] else 'overlap'} |")
L += ["", "old setting (K chosen per class: E K256, M K32, N K256) → CountsDiff mean at K128:", ""]
for t in TG:
    L.append(f"- {t}:" + ";".join(f"{NM[k]} {np.mean(oldcd[t][k]):.{DP[k]}f} → {np.mean(cd[t][k]):.{DP[k]}f}" for k in ("W1", "mmd2", "recall", "coverage")))
res = {"cd_k128": cd, "crndiff_table1": ours, "comparison_coverage": cmpC, "counts_coverage": {"n_cells_with_numbers": len(doneC), "crndiff_better_mean": nbC, "ranges_disjoint": ndC, "disjoint_and_crndiff_better": ndbC}, "comparison": cmp, "counts": {"n_cells_with_numbers": len(done), "crndiff_better_mean": n_b, "ranges_disjoint": n_d, "disjoint_and_crndiff_better": n_db},
       "sources": {f: sha(f)[:8] for f in (f"{D}/frozen_w1_mmd_pcc.json", f"{W}/runs/cd_hvg2k_3seed/side_table_3seed_final.json")}, "script_sha256": sha(__file__)}
json.dump(res, open(out_js, "w"), indent=1); open(out_md, "w").write("\n".join(L) + "\n"); print("\n".join(L))
