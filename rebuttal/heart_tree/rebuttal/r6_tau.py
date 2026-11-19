# -*- coding: utf-8 -*-
"""R6: apply the criteria in rebuttal/out/R6/criterion.md to the stored tau grid.
    cd Experiments/HVG && python -m rebuttal.r6_tau
"""
import json, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
from rebuttal.w2_runs import ROOT, RUNS, TARGETS, ours_stem
OUT = ROOT / "rebuttal" / "out" / "R6"
s = json.loads((ROOT / "out/summary/201_tau_sweep_scored.json").read_text(encoding="utf-8"))
pp = json.loads((ROOT / "out/summary/215_paper_numbers_v8.json").read_text(encoding="utf-8"))
v8 = {(r["request"], float(r["row"])): r["mean"] for r in pp["rows"] if r["table"] == "tab:hvg_tau" and r["metric"] == "purity_abs"}
TAUS = [0.2, 0.4, 1.0]; res = {}
INT = {"C1_ess_pre": "ess_pre_final_fraction", "C2_ess_min": "ess_min_fraction", "C3_lineage": "unique_lineage_fraction"}
for t in TARGETS:
    g = {}
    for tau in TAUS:
        rr = [r for r in s["rows"] if r["request"] == t and r["tau"] == tau]
        assert len(rr) == 3, (t, tau, len(rr))
        g[tau] = {"v8": v8[(t, tau)], "W1": float(np.mean([r["W1"] for r in rr])), "mmd2": float(np.mean([r["mmd2_rbf_biased"] for r in rr])),
                  "pcc": float(np.mean([r["pcc"] for r in rr])), "v7": float(np.mean([r["purity"] for r in rr]))}
        for k, f in INT.items():
            v = [r[f] for r in rr]; g[tau][k] = float(np.mean(v)); g[tau][k + "_range"] = [float(min(v)), float(max(v))]
    pick = lambda k, sign=1: max(TAUS, key=lambda x: (round(sign * g[x][k], 12), -x))
    sel = {k: pick(k) for k in INT}
    best = {"v8": pick("v8"), "W1": pick("W1", -1), "mmd2": pick("mmd2", -1), "pcc": pick("pcc")}
    res[t] = {"grid": g, "selected": sel, "best_by_metric": best, "deployed": s["deployed_tau"][t],
              "agreement": {k: {m: sel[k] == best[m] for m in best} for k in sel}}
    print(t, "selected", sel, "best", best, "deployed", s["deployed_tau"][t])
    for tau in TAUS:
        print("  ", tau, {k: round(v, 4) for k, v in g[tau].items() if not k.endswith("range")})
s626 = {}
for t in TARGETS:
    p = RUNS / "table2_seeds_s20260626" / f"{t}_s20260626" / ours_stem(t, 20260987)
    j = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    s626[t] = {k: j.get(k) for k in ("tau", "unique_lineage_fraction", "ess_min_fraction", "ess_pre_final_fraction")}
res["train20260626_deployed_tau_only"] = s626; print(s626)
(OUT / "r6_numbers.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
