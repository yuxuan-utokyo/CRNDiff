#!/usr/bin/env python
r"""R17: HVG2K tau curves on VAL and three tau-selection rules, the same definitions as for HLCA.
Describes only. Input: a normalized json written by r17_build_inputs.py (no numbers are typed by hand):
  {"targets": {T: {"share": f, "deployed_tau": x, "ref": {"W1": floor, "mmd2": floor, "purity": real_v8, "coverage": real, },
                   "tau": {"0.2": {"W1": [s87, s88, s89], "mmd2": [...], "purity": [...], "coverage": [...], "nn1_acc": [...]}, ...}}},
   "sources": {...}}
Relative scores (1 = as good as real cells, higher = better), computed from the 3-seed MEAN of each raw metric:
  W1: floor / W1;  MMD^2: floor / MMD^2;  purity: purity / real;  coverage: coverage / real;  1-NN: (1 - acc) / 0.5
Rules (scores capped at 1 for B and C only; curves are drawn uncapped):
  A: smallest tau with purity >= 0.90 x real purity;
  B: smallest Euclidean distance of the 5 capped scores to the all-ones point;
  C: largest geometric mean of the 5 capped scores.
Ties go to the smaller tau. Figure: one row of 3 panels in the R16_tau_curves_val_row.png style (same 5 lines, colours and
legend text; y label "Relative score (1 = real cells, higher = better)"; grey bar = tau chosen by rule A; fonts 17-19).
    python r17_rules.py INPUTS.json FIG_STEM REPORT.md [PREV_NUMBERS.json]
v1b (11-10 13:0x): optional PREV_NUMBERS.json (an earlier *_numbers.json) -> the report states, per target, whether rule A/B/C moved;
the data section also prints the input's floor_note lines.
"""
import hashlib, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

inp, fig_stem, rep = sys.argv[1:4]; prev = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
J = json.load(open(inp)); sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
M = [("W1", "W1: floor / W1", "#2a78d6"), ("mmd2", "MMD²: floor / MMD²", "#7a4fc4"), ("purity", "Purity / real", "#d6455d"),
     ("coverage", "Coverage / real", "#1f9e89"), ("nn1_acc", "1-NN: (1 − acc) / 0.5", "#4a4a48")]


def scores(raw, ref):
    return {"W1": ref["W1"] / raw["W1"], "mmd2": ref["mmd2"] / raw["mmd2"], "purity": raw["purity"] / ref["purity"],
            "coverage": raw["coverage"] / ref["coverage"], "nn1_acc": (1 - raw["nn1_acc"]) / 0.5}


R = {}
for t, d in J["targets"].items():
    taus = sorted(d["tau"], key=float); mean = {x: {k: float(np.mean(v)) for k, v in d["tau"][x].items()} for x in taus}
    missing = [(x, k) for x in taus for k, _, _ in M if any(v is None for v in d["tau"][x].get(k, [None]))]
    assert not missing, (t, "missing cells", missing)
    S = {x: scores(mean[x], d["ref"]) for x in taus}; cap = {x: {k: min(1.0, v) for k, v in S[x].items()} for x in taus}
    A = [x for x in taus if mean[x]["purity"] >= 0.90 * d["ref"]["purity"]]; A = A[0] if A else None
    dist = {x: float(np.sqrt(sum((1 - v) ** 2 for v in cap[x].values()))) for x in taus}
    gm = {x: float(np.exp(np.mean([np.log(max(v, 1e-12)) for v in cap[x].values()]))) for x in taus}
    B = min(taus, key=lambda x: (dist[x], float(x))); C = max(taus, key=lambda x: (gm[x], -float(x)))
    R[t] = {"taus": taus, "mean": mean, "scores": S, "dist": dist, "geomean": gm, "A": A, "B": B, "C": C, "deployed": d["deployed_tau"], "share": d["share"], "ref": d["ref"]}
json.dump(R, open(os.path.splitext(rep)[0] + "_numbers.json", "w"), indent=1)

# ---------- figure ----------
plt.rcParams.update({"font.size": 17, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(1, len(R), figsize=(6.2 * len(R), 6.4), sharey=True)
ax = np.atleast_1d(ax)
for a, (t, r) in zip(ax, R.items()):
    xs = [float(x) for x in r["taus"]]
    if r["A"] is not None:
        a.axvspan(float(r["A"]) - 0.05, float(r["A"]) + 0.05, color="#d9d9d7", zorder=0)
    a.axhline(1.0, color="#8a8986", ls=(0, (1, 2)), lw=1.2, zorder=1)
    for k, lab, col in M:
        a.plot(xs, [r["scores"][x][k] for x in r["taus"]], "-o", color=col, lw=2.2, ms=7, zorder=3)
    a.set_title(f"{t} ({100 * r['share']:.1f}%)", fontsize=19); a.set_xlabel("τ", fontsize=18)
    a.set_xticks(xs); a.set_xticklabels([f"{x:g}" if x != 1 else "1.0" for x in xs]); a.set_xlim(min(xs) - 0.1, max(xs) + 0.1)
    a.set_ylim(0, max(1.15, max(max(r["scores"][x].values()) for x in r["taus"]) + 0.05))
    a.spines[["top", "right"]].set_visible(False); a.grid(lw=0.4, color="#e4e3de"); a.set_axisbelow(True)
ax[0].set_ylabel("Relative score\n(1 = real cells, higher = better)", fontsize=18)
H = [Line2D([], [], color=c, marker="o", lw=2.2, label=l) for _, l, c in M] + [Patch(color="#d9d9d7", label="τ chosen by rule A")]
fig.legend(handles=H, loc="lower center", ncol=3, frameon=False, fontsize=17, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(fig_stem + ".png", dpi=300); fig.savefig(fig_stem + ".pdf")

# ---------- report ----------
f3 = lambda v: f"{v:.3f}"
T = ["| target(share) | deployed τ | rule A | rule B | rule C |", "|---|---|---|---|---|"]
for t, r in R.items():
    T.append(f"| {t}({100 * r['share']:.1f}%) | {r['deployed']:g} | {r['A'] if r['A'] is not None else 'none (no level reaches 0.90 × ceiling)'} | {r['B']} | {r['C']} |")
D = ["| target | τ | W1 | MMD² | purity(v8) | coverage | 1-NN | distance to all-ones (B) | geometric mean (C) |", "|---|---|---|---|---|---|---|---|---|"]
for t, r in R.items():
    for x in r["taus"]:
        m = r["mean"][x]
        D.append(f"| {t} | {x} | {m['W1']:.4f} | {m['mmd2']:.5f} | {f3(m['purity'])} | {f3(m['coverage'])} | {f3(m['nn1_acc'])} | {r['dist'][x]:.3f} | {r['geomean'][x]:.3f} |")
    D.append(f"| {t} | real-cell reference | {r['ref']['W1']:.4f}(noise floor) | {r['ref']['mmd2']:.5f}(noise floor) | {f3(r['ref']['purity'])}(ceiling) | {f3(r['ref']['coverage'])}(noise floor) | 0.5 (ideal) | | |")
MOVE = []
if prev:
    mv = [(t, k, prev[t][k], R[t][k]) for t in R for k in ("A", "B", "C") if t in prev and prev[t][k] != R[t][k]]
    MOVE = ["compared with the previous version (different noise floor):" + ("none of the rule A, B, C selections changes for any target." if not mv else ";".join(f"{t} rule {k} changes from {a_} to {b_}" for t, k, a_, b_ in mv) + ", everything else unchanged.")]
agree = {k: sum(r[k] is not None and float(r[k]) == float(r["deployed"]) for r in R.values()) for k in ("A", "B", "C")}
L = ["# R17 HVG2K val: τ selected by the three τ rules on the 3 targets (" + ",".join(f"rule {k} equals the deployed value {v}/{len(R)}" for k, v in agree.items()) + ")", "",
     "**Context: a reviewer asked how τ is chosen in practice and whether test can be avoided. The τ curves and selection rule are placed on the main dataset HVG2K, with HLCA as cross-dataset validation. Description only; test is not read.", "",
     "**data.** " + J.get("data_note", ""), ""] + [f"- {x}" for x in J.get("floor_note", [])] + ["",
     "**Method.** Each metric is first averaged over 3 sampling seeds, then converted to a score relative to real cells (1 = as good as real cells, higher is better): W1, MMD² use noise floor ÷ value; purity, coverage use value ÷ real cells; 1-NN uses (1 − accuracy) ÷ 0.5.",
     "- rule A: the smallest τ with purity ≥ 0.90 × real ceiling.",
     "- rule B: the smallest Euclidean distance of the 5 scores (capped at 1) to the all-ones point:$$\\hat\\tau_B=\\arg\\min_\\tau\\Big(\\sum_{m=1}^{5}\\big(1-\\min(1,s_m(\\tau))\\big)^2\\Big)^{1/2}.$$",
     "- rule C: the largest geometric mean of the same 5 capped scores:$$\\hat\\tau_C=\\arg\\max_\\tau\\Big(\\prod_{m=1}^{5}\\min(1,s_m(\\tau))\\Big)^{1/5}.$$",
     "- ties go to the smaller τ. The three rules use the same definitions as for HLCA.", "",
     "**results.**", "", "Table 1: τ selected by the three rules and the currently deployed τ", ""] + T + ([""] + MOVE if prev else []) + ["",
     f"![[figs/{os.path.basename(fig_stem)}.png]]", "",
     "Figure 1: one target per panel (its share in train in parentheses). The 5 lines are scores relative to real cells (uncapped; values above 1 are drawn as is); the dotted line is 1; the grey bar is the τ chosen by rule A. Mean over 1 generator and 3 sampling seeds.", "",
     "Table 2: raw values per cell (mean over 3 sampling seeds) and the quantities of rules B and C", ""] + D + ["",
     "**what this means for the rebuttal.** Description only. The rule was set after seeing the val curves; the grid has only 3 τ levels (0.6 and 0.8 missing).", "",
     "> [!summary] summary", "> " + ";".join(f"{t}:A {r['A']},B {r['B']},C {r['C']}(deployed {r['deployed']:g})" for t, r in R.items()) + ". Uses val only.", "",
     "> [!info]- reproducibility information",
     f"> - script `scripts/r17/r17_rules.py`(sha256 {sha(__file__)[:8]}), inputs `{os.path.relpath(inp, './work')}`({sha(inp)[:8]}), command `python -I scripts/r17/r17_rules.py INPUTS FIG_STEM REPORT`.",
     "> - source:" + ";".join(f"`{k}` {v}" for k, v in J.get("sources", {}).items())]
open(rep, "w").write("\n".join(L) + "\n")
for t, r in R.items(): print(t, "A", r["A"], "B", r["B"], "C", r["C"], "deployed", r["deployed"])
