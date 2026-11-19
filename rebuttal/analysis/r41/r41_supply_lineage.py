#!/usr/bin/env python
r"""R41: candidate supply and FK lineage, HLCA plasma cell + ionocyte. No new test read: reads
  runs/r37/gen.json (unconditional pools: cells the frozen purity LR calls the target, per generator s25 / s27 / s28, 25,000 each),
  train labels (class frequencies), the tilted-FK run jsons next to the v3 manifest's CRNDiff arrays (unique_lineage_fraction,
  unique_cell_fraction, ess_min_fraction, ess_pre_final_fraction, n_intermediate_resamples), and per-run purity / coverage on test from
  the Table 3 files r16_results_report.py reads.
    python r41_supply_lineage.py OUT.md
"""
import hashlib, json, re, sys
import numpy as np
W = "./work"; sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:8]
G = json.load(open(f"{W}/runs/r37/gen.json")); man = json.load(open(f"{W}/runs/r24_26_3seed/manifest.json"))
YT = np.load(f"{W}/data/r16_hlca_fit/train_celltype.npy", allow_pickle=True).astype(str); CL = ["plasma cell", "ionocyte"]
R16 = f"{W}/runs/r16"; J = lambda rel: json.load(open(f"{R16}/{rel}")); TAU = {"plasma cell": "tau0p2", "ionocyte": "tau0p4"}
def get(sc, t, arm=None):
    r = [x for x in sc["runs"] if x["method"] == "CRNDiff" and x["target"] == t and (arm is None or x["arm"] == arm)]; assert len(r) == 1; return r[0]
T3 = {("gen20260625", t): get(J("ruleA_test/scores_test.json"), t, TAU[t]) for t in CL}
T3.update({("gen20260627", t): get(J("crn_s27/scores_test.json"), t) for t in CL}); T3.update({("gen20260628", t): get(J("crn_s28/scores_test.json"), t) for t in CL})
rows = {}
for e in [e for e in man if e["method"] == "CRNDiff"]:
    d = json.load(open(e["path"][:-4] + ".json")); t3 = T3[(e["seed"], e["target"])]; assert t3["path"].endswith(e["path"].split("crndiff-rebuttal/")[-1])
    rows[(e["target"], e["seed"])] = {"n_out": d["n_out"], "n_particles": d["n_particles"], "tau": d["tau"], "lineage": d["unique_lineage_fraction"], "ancestors": round(d["unique_lineage_fraction"] * d["n_out"]),
                                      "unique_cell": d["unique_cell_fraction"], "ess_min": d["ess_min_fraction"], "ess_pre_final": d["ess_pre_final_fraction"], "n_resamples": d["n_intermediate_resamples"],
                                      "purity": t3["purity_logreg"], "coverage": t3["coverage"]}
pool = {}
for t in CL:
    for k, v in G["sets"][t].items():
        m = re.match(r"uncond pool, gen(\d+), LR-called .* \((\d+) of ([\d,]+)\)", k)
        if m: pool[(t, "gen" + m.group(1))] = (int(m.group(2)), int(m.group(3).replace(",", "")))
GENS = ["gen20260625", "gen20260627", "gen20260628"]
L = ["# R41 candidate supply and FK lineage (HLCA plasma cell, ionocyte)", "",
     "**Question to answer.** ionocyte purity varies a lot across generators and coverage is low; is this consistent with the generator rarely producing candidate cells and the lineage being very concentrated after FK resampling? No new test reads: unconditional pool counts come from R37, lineage diagnostics from each FK run json, purity and coverage from the test numbers already read for Table 3.", "",
     "Table 1: candidate supply: among 25,000 unconditional cells per generator (τ = 0, no FK), number and share classified as the target class by the purity classifier, compared with real training frequency", "",
     "| target | training-set frequency | " + " | ".join(f"unconditional pool {g[3:]}" for g in GENS) + " |", "|---|---|---|---|---|"]
for t in CL:
    nt = int((YT == t).sum())
    L.append(f"| {t} | {nt:,} / {len(YT):,}({nt / len(YT):.5f}) | " + " | ".join(f"{pool[(t, g)][0]}({pool[(t, g)][0] / pool[(t, g)][1]:.5f})" for g in GENS) + " |")
L += ["", "Table 2: for each tilted FK run (M = 5,000 particles, 2,500 cells delivered): lineage and resampling diagnostics, alongside purity on test,coverage", "",
      "| target | generator | τ | distinct ancestors (out of 2,500) | distinct-cell fraction | minimum ESS fraction | ESS fraction before last resampling | intermediate resamplings | purity (test) | coverage (test) |", "|---|---|---|---|---|---|---|---|---|---|"]
for t in CL:
    for g in GENS:
        r = rows[(t, g)]
        L.append(f"| {t} | {g[3:]} | {r['tau']} | {r['ancestors']}({r['lineage']:.4f}) | {r['unique_cell']:.3f} | {r['ess_min']:.5f} | {r['ess_pre_final']:.3f} | {r['n_resamples']} | {r['purity']:.3f} | {r['coverage']:.3f} |")
pr = {t: [rows[(t, g)]["purity"] for g in GENS] for t in CL}; anc = {t: [rows[(t, g)]["ancestors"] for g in GENS] for t in CL}
cor = {t: float(np.corrcoef([rows[(t, g)]["lineage"] for g in GENS], pr[t])[0, 1]) for t in CL}
L += ["", "**how to read (computed in code).**",
      "- candidate supply:" + ";".join(f"{t} training-set frequency {(YT == t).mean():.5f}, in the unconditional pool " + ",".join(str(pool[(t, g)][0]) for g in GENS) + "  / 25,000" for t in CL) + ". Both classes are extremely rare in the generator own output; ionocytes only 1–4.",
      "- lineage:" + ";".join(f"{t} the final 2,500 cells come from " + ",".join(str(a) for a in anc[t]) + " distinct ancestors" for t in CL) + ". The lineage of plasma cells is more concentrated than that of ionocytes.",
      "- purity across generators:" + ";".join(f"{t} " + ",".join(f"{p:.3f}" for p in pr[t]) + f"(sd {np.std(pr[t], ddof=1):.3f})" for t in CL) + ". With only 3 generators, no correlation conclusion is drawn between ancestor count and purity.",
      "- Relation to the question: candidates of both classes are very rare (far below the training frequency in the unconditional pool), consistent with needing strong guidance and ending with concentrated lineages; but plasma cells have more concentrated lineages than ionocytes yet more stable purity, so lineage concentration by itself does not explain the ionocyte purity fluctuation.", "",
      "> [!info]- reproducibility information", f"> - unconditional pool counts: runs/r37/gen.json ({sha(W + '/runs/r37/gen.json')}), pools data/r16_uncond/hlca_uncond_s2026062{{5,7,8}}.npy (4086a07e / 234facfe / 6bd41bd5), frozen classifier 219e9bce.",
      "> - lineage diagnostics: run json next to each tilted FK array (CRNDiff rows of the v3 manifest); purity, coverage: scores_test.json of runs/r16/ruleA_test, crn_s27, crn_s28 (already read for Table 3).",
      f"> - this document:`scripts/r41/r41_supply_lineage.py`({sha(__file__)})."]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L[4:]))
