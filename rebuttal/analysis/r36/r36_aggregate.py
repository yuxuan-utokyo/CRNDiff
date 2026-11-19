#!/usr/bin/env python
r"""R36 aggregate + report (test). For each class, metric and rep: a method's value = mean over its 3 seeds; then mean and sd over the 20 reps.
Rank stability against the full-reference ranking already read on test (W1 / MMD^2 / PCC / coverage: the Table 3 files r16_results_report.py
reads; neighbour fraction: runs/r29_30/r29_30_test.json): number of reps with (a) the same full 5-method order, (b) the same best method,
(c) CRNDiff at the same rank. Not independent repeats: only the reference cells are subsampled; the generated cells are the same.
    python r36_aggregate.py OUT.md OUT.json
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
W = Path("./work"); R16 = W / "runs/r16"; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]
OM, OJ = Path(sys.argv[1]), Path(sys.argv[2])
METH = ["CRNDiff", "CFGen", "CountsDiff", "scVI", "scANVI"]; CL = ["plasma cell", "ionocyte"]
MET = [("W1", "W1", -1, 4), ("mmd2", "MMD²", -1, 5), ("pcc", "PCC", 1, 4), ("coverage", "coverage", 1, 3), ("nf50", "neighbour fraction (50-dim PCA)", 1, 3)]
KN = json.load(open(W / "runs/r36/knn_test.json")); assert KN["split"] == "test"
reps = sorted(KN["reps"]); assert len(reps) == 20
val = {t: {k: {m: [] for m in METH} for k, *_ in MET} for t in CL}
for i in reps:
    S = json.load(open(W / f"runs/r36/test/rep{i}.json")); sub = {r["target"]: r for r in S["subsets"]}
    for t in CL:
        assert sub[t]["split"] == "test" and sub[t]["n"] == {"plasma cell": 278, "ionocyte": 89}[t]
        for m in METH:
            rs = [r for r in S["runs"] if r["method"] == m and r["target"] == t]; assert len(rs) == 3, (i, t, m, len(rs))
            for k in ("W1", "mmd2", "pcc", "coverage"): val[t][k][m].append(float(np.mean([r[k] for r in rs])))
            val[t]["nf50"][m].append(float(np.mean([v for kk, v in KN["reps"][i][t].items() if kk.split("|")[0] == m])))
# full-reference values (already read)
J = lambda rel: json.load(open(R16 / rel)); TAU = {"plasma cell": "tau0p2", "ionocyte": "tau0p4"}; A = J("ruleA_test/scores_test.json"); CD1 = J("cd_k128_test/scores_test.json")
def get(sc, m, t, arm=None):
    r = [x for x in sc["runs"] if x["method"] == m and x["target"] == t and (arm is None or x["arm"] == arm)]; assert len(r) == 1; return r[0]
T3 = {t: {"CRNDiff": [get(A, "CRNDiff", t, TAU[t]), get(J("crn_s27/scores_test.json"), "CRNDiff", t), get(J("crn_s28/scores_test.json"), "CRNDiff", t)],
          "CFGen": [get(A, "CFGen", t), get(J("base_cfgen_s20260988/scores_test.json"), "CFGen", t), get(J("base_cfgen_s20260989/scores_test.json"), "CFGen", t)],
          "CountsDiff": [get(CD1, "CountsDiff", t, "K128_g1p0") if t == "plasma cell" else get(A, "CountsDiff", t, "K128_g1p0"), get(J("cd_s26_k128/scores_test.json"), "CountsDiff", t, "K128_g1p0"), get(J("cd_s27_k128/scores_test.json"), "CountsDiff", t, "K128_g1p0")],
          "scVI": [get(A, "scVI", t), get(J("base_scvi_s20261053/scores_test.json"), "scVI", t), get(J("base_scvi_s20261054/scores_test.json"), "scVI", t)],
          "scANVI": [get(A, "scANVI", t), get(J("base_scanvi_s20261063/scores_test.json"), "scANVI", t), get(J("base_scanvi_s20261064/scores_test.json"), "scANVI", t)]} for t in CL}
R29 = json.load(open(W / "runs/r29_30/r29_30_test.json"))
full = {t: {**{k: {m: float(np.mean([r[k] for r in T3[t][m]])) for m in METH} for k in ("W1", "mmd2", "pcc", "coverage")},
            "nf50": {m: float(np.mean([r["r29"]["mean_neuronal_frac"] for r in R29["runs"] if r["method"] == m and r["target"] == t])) for m in METH}} for t in CL}
order = lambda d, s: sorted(METH, key=lambda m: -s * d[m])
man = json.load(open(W / "runs/r24_26_3seed/manifest.json")); vr = {}
for i in reps:
    for r in json.load(open(W / f"runs/r36/test/rep{i}.json"))["runs"]: vr.setdefault((r["method"], r["seed"], r["target"]), []).append(r["W1"])
fullrun = {}
for rel in ("ruleA_test", "cd_k128_test", "crn_s27", "crn_s28", "base_cfgen_s20260988", "base_cfgen_s20260989", "cd_s26_k128", "cd_s27_k128", "base_scvi_s20261053", "base_scvi_s20261054", "base_scanvi_s20261063", "base_scanvi_s20261064"):
    for r in J(f"{rel}/scores_test.json")["runs"]:
        if r["target"] in CL: fullrun[(r["path"].split("crndiff-rebuttal/")[-1], r["target"])] = r["W1"]
SHIFT = []
for e in man:
    k = (e["method"], e["seed"], e["target"]); pth = (e["path"].split("crndiff-rebuttal/")[-1], e["target"])
    if pth in fullrun:
        a = np.array(vr[k])
        if abs(fullrun[pth] - a.mean()) > 3 * max(a.std(ddof=1), 1e-12): SHIFT.append((k, fullrun[pth], float(a.mean()), float(a.std(ddof=1))))
out = {}
for t in CL:
    out[t] = {}
    for k, lab, s, d in MET:
        fo = order(full[t][k], s); per = [order({m: val[t][k][m][j] for m in METH}, s) for j in range(20)]
        out[t][k] = {"mean": {m: float(np.mean(val[t][k][m])) for m in METH}, "sd": {m: float(np.std(val[t][k][m], ddof=1)) for m in METH},
                     "full_order": fo, "n_same_order": sum(p == fo for p in per), "n_same_best": sum(p[0] == fo[0] for p in per),
                     "n_same_crndiff_rank": sum(p.index("CRNDiff") == fo.index("CRNDiff") for p in per), "crndiff_rank_full": fo.index("CRNDiff") + 1,
                     "crndiff_rank_counts": {str(r): sum(p.index("CRNDiff") + 1 == r for p in per) for r in range(1, 6)}}
L = ["# R36 HLCA evaluation stability: 20 draws of 80% subsamples of the reference cells (test)", "",
     "**What this is.** How sensitive the evaluation is to which real cells are used as reference: 20 subsamples without replacement of the real test target cells (80% each: 89 of 111 ionocytes, 278 of 347 plasma cells, seeds from 20261201), recomputing each method metrics with the same ruler each time. **This is not an independent replication**: generated cells stay fixed; only the reference cells change.", "",
     "**Method.** W1, MMD², PCC and coverage use the scoring script of Table 3 (with the reference replaced by the subsample; matched n and the noise floor follow the subsample); neighbour fraction uses the R29 definition (50-dim PCA, k = 15; target cells not drawn are removed from the neighbour pool). Each draw takes the mean over 3 seeds per method; the table shows mean ± sd over 20 draws."
     "note: matched n equals the subsample size (89 / 278; 111 / 347 with all references); the scorer draws n of the 2,500 generated cells per method with a fixed seed, so when n changes the drawn generated cells change too; for long-tailed generated arrays this alone moves W1 (see reading notes)."
     "target cells not drawn are removed from the neighbour pool of the neighbour fraction, so the target class in the pool is exactly this subsample."
     "Rank unchanged, using all reference cells (numbers already read for Table 3 / R29): how often the order of all 5 methods is the same, how often the top method is the same, and how often CRNDiff keeps its rank.", ""]
for t in CL:
    L += [f"**{t}**", "", "| metric | " + " | ".join(METH) + " | order with all references | order unchanged | top unchanged | CRNDiff rank unchanged (rank with all references) |", "|---|" + "---|" * (len(METH) + 4)]
    for k, lab, s, d in MET:
        o = out[t][k]
        L.append(f"| {lab}{' ↓' if s < 0 else ' ↑'} | " + " | ".join(f"{o['mean'][m]:.{d}f} ± {o['sd'][m]:.{d}f}" for m in METH) + f" | {' > '.join(o['full_order'])} | {o['n_same_order']}/20 | {o['n_same_best']}/20 | {o['n_same_crndiff_rank']}/20(number {o['crndiff_rank_full']}) |")
    L.append("")
NEAR = []
for t in CL:
    for k, lab, sg, d in MET:
        o = out[t][k]
        if o["n_same_crndiff_rank"] < 20:
            mo = sorted(METH, key=lambda m: -sg * o["mean"][m]); i = mo.index("CRNDiff")
            for j in (i - 1, i + 1):
                if 0 <= j < len(mo):
                    m2 = mo[j]; gap = abs(o["mean"]["CRNDiff"] - o["mean"][m2])
                    if gap < max(o["sd"]["CRNDiff"], o["sd"][m2]): NEAR.append(f"{t} {lab}(CRNDiff vs {m2}: difference {gap:.{d}f},sd {o['sd']['CRNDiff']:.{d}f} / {o['sd'][m2]:.{d}f})")
ni = {k: out["ionocyte"][k]["n_same_best"] for k, *_ in MET}; npl = {k: out["plasma cell"][k]["n_same_best"] for k, *_ in MET}
L += ["**how to read.**", "- times the top method stays the same out of 20:ionocyte " + ",".join(f"{lab} {ni[k]}/20" for k, lab, *_ in MET) + ";plasma cell " + ",".join(f"{lab} {npl[k]}/20" for k, lab, *_ in MET) + ".",
      "- CRNDiff rank distribution on ionocyte (20 draws):" + ";".join(f"{lab} " + ",".join(f"rank {r}: {c} times" for r, c in out['ionocyte'][k]['crndiff_rank_counts'].items() if c) for k, lab, *_ in MET) + ".", "",
      "- CRNDiff metrics where the rank differs from the all-reference rank and the mean difference to the neighbouring method is smaller than both sds:" + (";".join(NEAR) if NEAR else "none") + ". The rank differences in these cells are within subsampling fluctuation.",
      "- single runs whose W1 differs from the all-reference value by more than 3 times the sd over 20 draws:" + ";".join(f"{m} {sd} {t}: all references {fv:.3f}, subsample {mu:.3f} ± {sdv:.3f}" for (m, sd, t), fv, mu, sdv in SHIFT) + ". The generated arrays of these runs have long tails; W1 varies a lot with which generated cells are drawn. They determine how often the W1 row above keeps its order; this is not caused by subsampling the reference cells.", "",
      "> [!info]- reproducibility information", "> - subsample-only labels: scripts/r36/r36_make_subsets.py; scoring scripts/r36/r36_score.sh (score_with_npz.py ed519051 + --subset); neighbours scripts/r36/r36_knn.py (pca50 basis 067dc34d).",
      f"> - test read once (20 subsamples computed in the same read), logged in runs/r16/TEST_READ_LOG.tsv; results runs/r36/test/rep00–19.json, runs/r36/knn_test.json ({sha(W / 'runs/r36/knn_test.json')}).",
      f"> - this document:`scripts/r36/r36_aggregate.py`({sha(__file__)})."]
OM.write_text("\n".join(L) + "\n"); json.dump({"results": out, "script_sha256": sha(__file__)}, open(OJ, "w"), indent=1); print("\n".join(L[6:20]))
if "HLCA R36 evaluation stability test" not in open(W / "runs/r16/TEST_READ_LOG.tsv").read():
  with open(W / "runs/r16/TEST_READ_LOG.tsv", "a") as f:
    f.write(f"{time.strftime('%m-%d_%H:%M:%S')}\tHLCA R36 evaluation stability test (20 reference subsamples; aggregated by r36_aggregate.py {sha(__file__)})\truns/r36/test\t{sha(OJ)}\n")
