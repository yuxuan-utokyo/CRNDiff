"""HVG2K recall side by side, all methods. Reads existing numbers only; nothing is recomputed.
- OURS, CFGen, MDLM, scVI, scANVI: R3 per-run values (from_windows/rebuttal_code/results_for_check/r3_numbers.json),
  the 3 Table 1 runs per method from TABLE1 in from_windows/rebuttal_code/rebuttal/w2_runs.py (copied below).
- CountsDiff (1 seed): runs/countsdiff_hvg2k_new_metrics_<t>.json
  (r9_new_metrics.py: same function, frozen PCA basis 24d2cc8f, k=5, n=min(n_val,2500), 5 resamples).
- Noise floor: runs/r3_floor_heldout.json, held-out basis (the floor the published R3 report uses), 5 repeats.
Per method: mean and sd (ddof=1) over its runs; 1 run gives the value with no sd. Rank by mean recall; a method is tied
with OURS when the intervals mean +- sd overlap (a 1-run method's interval is its value).
Val only. Usage: recall_all_methods.py OUT.json
"""
import json, os, sys
import numpy as np

W = os.path.expanduser("~/research/crndiff-rebuttal")
TABLE1 = {"OURS": [(20260625, 20260989), (20260627, 20260987), (20260630, 20260989)],
          "MDLM": [(20260625, 20260933), (20260934, 20260932), (20260935, 20260932)],
          "CFGen": [("run1", 20260932), ("run2", 20260932), ("run3", 20260933)],
          "scVI": [(20261050, 20261051), (20261053, 20261055), (20261054, 20261055)],
          "scANVI": [(20261060, 20261065), (20261063, 20261061), (20261064, 20261061)]}
KEYS = ["recall", "coverage", "precision", "var_ratio_top10"]
TARGETS = ["Endothelial", "Myeloid", "Neuronal"]
r3 = json.load(open(f"{W}/from_windows/rebuttal_code/results_for_check/r3_numbers.json"))
fl = json.load(open(f"{W}/runs/r3_floor_heldout.json"))


def summ(rows, src):
    out = {"n_runs": len(rows), "source": src, "runs": [{k: r[k] for k in ("file", "seed", "train", "samp") if k in r} for r in rows]}
    for k in KEYS:
        v = np.array([r[k] for r in rows], float)
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "min": float(v.min()), "max": float(v.max())}
    return out


res = {}
for t in TARGETS:
    M = {}
    for a, pairs in TABLE1.items():
        rows = [r for r in r3[t]["runs"] if r["arm"] == a and (r["train"], r["samp"]) in pairs]
        assert len(rows) == 3, (t, a, len(rows))
        M[a] = summ(rows, "r3_numbers.json (Table 1 runs)")
    for name, f in (("CountsDiff", f"runs/countsdiff_hvg2k_new_metrics_{t}.json"),):   # Count-FM removed ()
        p = f"{W}/{f}"
        if not os.path.exists(p):
            M[name] = None; continue
        d = json.load(open(p)); assert d["split"] == "val" and d["basis_sha256"].startswith("24d2cc8f"), p
        (rows,) = d["runs"].values()
        M[name] = summ(rows, f)
    fh = fl[t]["heldout"]
    floor = {k: {"mean": fh[k]["mean"], "min": fh[k]["min"], "max": fh[k]["max"]} for k in KEYS}
    lo = lambda m: m["recall"]["mean"] - (m["recall"]["sd"] or 0); hi = lambda m: m["recall"]["mean"] + (m["recall"]["sd"] or 0)
    order = sorted([a for a in M if M[a]], key=lambda a: -M[a]["recall"]["mean"])
    o = M["OURS"]
    ties = [a for a in order if a != "OURS" and not (hi(M[a]) < lo(o) or lo(M[a]) > hi(o))]
    above = [a for a in order if a != "OURS" and lo(M[a]) > hi(o)]
    res[t] = {"n_compare": int(r3[t]["runs"][0]["n_compare"]), "methods": M, "floor_heldout": floor, "recall_order": order,
              "ours_rank_by_mean": order.index("OURS") + 1, "n_methods": len(order),
              "ours_tied_with": ties, "clearly_above_ours": above}
    print(t, "order", [(a, round(M[a]["recall"]["mean"], 3)) for a in order], "OURS rank", order.index("OURS") + 1,
          "tied", ties, "clearly above", above)
json.dump(res, open(sys.argv[1], "w"), indent=1)
