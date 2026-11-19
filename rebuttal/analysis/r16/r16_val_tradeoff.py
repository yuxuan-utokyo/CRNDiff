#!/usr/bin/env python
r"""R16 diagnostic. VAL only, existing
scores only; no test, no change to any selection. Writes the table + description (json, md), the figure (png 300 dpi + pdf, rebuttal font
spec: axis labels 16, ticks 14, legend 14, panel titles 15) and the full report. Rerun as-is when scLDM's val scores or multi-ciliated's
step-2 cells arrive: it picks them up from the files below.
Inputs (all val):
  CRNDiff  runs/r16/tau_val/select_<tok>.json (5 tau, alpha 1); multi-ciliated: select_multiciliated_step2.json if present (11 cells)
  CFGen    runs/r16/base_cfgen/scores_val.json;  scLDM runs/r16/base_scldm/scores_val.json (if present)
  CountsDiff runs/r16/cd_val/select_<tok>.json (selected K; standing rule: CountsDiff in every figure with baselines)
    python r16_val_tradeoff.py FIG_STEM REPORT.md [all]
v2 (11-10 12:4x): 4 reply classes by default, coverage instead of recall.
"""
import hashlib, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "./work"; R = f"{W}/runs/r16"; fig_stem, rep = sys.argv[1:3]
CL = [("classical monocyte", "classicalmonocyte"), ("multi-ciliated epithelial cell", "multiciliated"), ("plasma cell", "plasmacell"),
      ("ionocyte", "ionocyte"), ("type II pneumocyte", "typeIIpneumocyte")]
if not (len(sys.argv) > 3 and sys.argv[3] == "all"): CL = [c for c in CL if c[1] != "multiciliated"]
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
src = {}
def load(f):
    src[os.path.relpath(f, W)] = sha(f)[:8]; return json.load(open(f))

cf = load(f"{R}/base_cfgen/scores_val.json")
sl_f = f"{R}/base_scldm/scores_val.json"; sl = load(sl_f) if os.path.exists(sl_f) else None
res = {"split": "val", "note": "diagnostic only; selections unchanged; test not read", "classes": {}}
for t, k in CL:
    f2 = f"{R}/tau_val/select_{k}_step2.json"; step2 = os.path.exists(f2)
    s = load(f2 if step2 else f"{R}/tau_val/select_{k}.json"); cd = load(f"{R}/cd_val/select_{k}.json")
    cells = {a: {"tau": c["tau"], "alpha": c["alpha"], "W1": c["W1"], "purity": c["purity_logreg"], "coverage": c["coverage"], "candidate": c["candidate"]}
             for a, c in s["cells"].items()}
    wmin = min(cells, key=lambda a: (cells[a]["W1"], cells[a]["tau"], cells[a]["alpha"]))
    pend = bool(s["step2_triggered"]) and not step2
    b = {"CFGen": [r for r in cf["runs"] if r["target"] == t][0]}
    if sl: b["scLDM"] = [r for r in sl["runs"] if r["target"] == t][0]
    cdc = cd["cells"][cd["pick"]]
    res["classes"][t] = {"ceiling": s["ceiling"], "pick": s["pick"], "pick_provisional": pend, "step2_cells_in": step2, "W1_min_cell": wmin, "cells": cells,
                         "baselines": {n: {"W1": r["W1"], "purity": r["purity_logreg"], "coverage": r["coverage"]} for n, r in b.items()},
                         "CountsDiff": {"arm": cd["pick"], "W1": cdc["W1"], "purity": cdc["purity_logreg"], "coverage": cdc["coverage"]}}
lab = lambda a, c: f"τ {c['tau']:g}" + (f", α {c['alpha']:g}" if c["alpha"] != 1 else "")

# ---------- table + description ----------
T = ["| class | setting | W1 | purity | coverage |", "|---|---|---|---|---|"]
D = []; n_lt = 0
for t, r in res["classes"].items():
    first = True
    for a, c in sorted(r["cells"].items(), key=lambda kv: (kv[1]["alpha"], kv[1]["tau"])):
        mk = (" ★" if a == r["pick"] else "") + (" ▼" if a == r["W1_min_cell"] else "")
        T.append(f"| {t if first else ''} | CRNDiff {lab(a, c)}{mk} | {c['W1']:.4f} | {c['purity']:.3f} | {c['coverage']:.3f} |"); first = False
    T.append(f"| | CountsDiff {r['CountsDiff']['arm'].split('_')[0]}(selected) | {r['CountsDiff']['W1']:.4f} | {r['CountsDiff']['purity']:.3f} | {r['CountsDiff']['coverage']:.3f} |")
    for n, x in r["baselines"].items():
        T.append(f"| | {n} | {x['W1']:.4f} | {x['purity']:.3f} | {x['coverage']:.3f} |")
    T.append(f"| | purity ceiling of real val | | {r['ceiling']:.3f} | |")
    m = r["cells"][r["W1_min_cell"]]; g = r["baselines"]["CFGen"]; lt = m["W1"] < g["W1"]; n_lt += lt
    p = r["cells"][r["pick"]]
    D.append(f"- **{t}**: the lowest CRNDiff W1 on the grid is {lab(r['W1_min_cell'], m)} of {m['W1']:.4f}(purity {m['purity']:.3f}), CFGen is {g['W1']:.4f}(purity {g['purity']:.3f});"
             f"lowest W1 {'below' if lt else 'above'} CFGen {abs(m['W1'] - g['W1']) / g['W1'] * 100:.0f}%. Selected by the rule {lab(r['pick'], p)}{'(step 2 unfinished, provisional)' if r['pick_provisional'] else ''}:W1 {p['W1']:.4f},purity {p['purity']:.3f}."
             + ("" if r["W1_min_cell"] != r["pick"] else " the lowest-W1 cell is the rule-selected cell."))
n_cls = len(res["classes"]); res["n_classes_min_W1_below_CFGen"] = n_lt
json.dump(res, open(f"{R}/report_notes/val_tradeoff.json", "w"), indent=1)

# ---------- figure ----------
FS_AX, FS_TICK, FS_LEG, FS_PANEL = 16, 14, 14, 15
plt.rcParams.update({"font.size": FS_TICK, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
COL = {"CRNDiff": "#2a78d6", "CFGen": "#eb6834", "scLDM": "#1baf7a", "CountsDiff": "#7a4fc4"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"
fig, ax = plt.subplots(2, 3, figsize=(15, 9.4)); ax = ax.ravel()
NAMES = {"classical monocyte": "Classical monocyte", "multi-ciliated epithelial cell": "Multi-ciliated epithelial", "plasma cell": "Plasma cell",
         "ionocyte": "Ionocyte", "type II pneumocyte": "Type II pneumocyte"}
for a, (t, r) in zip(ax, res["classes"].items()):
    c1 = sorted([c for c in r["cells"].values() if c["alpha"] == 1], key=lambda c: c["tau"])
    a.plot([c["W1"] for c in c1], [c["purity"] for c in c1], "-o", color=COL["CRNDiff"], lw=1.8, ms=6, label="CRNDiff (τ grid, α = 1)", zorder=3)
    for j, c in enumerate(c1):   # alternate label sides so neighbouring tau labels do not overlap
        a.annotate(f"{c['tau']:g}", (c["W1"], c["purity"]), xytext=(4, -15) if j % 2 == 0 else (4, 7), textcoords="offset points", fontsize=12, color=COL["CRNDiff"])
    ca = [c for c in r["cells"].values() if c["alpha"] != 1]
    for al, mk in ((2, "s"), (4, "D")):
        cc = [c for c in ca if c["alpha"] == al]
        if cc: a.scatter([c["W1"] for c in cc], [c["purity"] for c in cc], marker=mk, s=40, facecolor="white", edgecolor=COL["CRNDiff"], lw=1.4, label=f"CRNDiff α = {al}", zorder=3)
    p = r["cells"][r["pick"]]
    a.scatter([p["W1"]], [p["purity"]], s=260, facecolor="none", edgecolor=INK, lw=1.4, label="Pre-registered pick", zorder=4)
    for n, x in r["baselines"].items():
        a.scatter([x["W1"]], [x["purity"]], marker="*", s=230, color=COL[n], edgecolor="white", lw=0.6, label=n, zorder=5)
    a.scatter([r["CountsDiff"]["W1"]], [r["CountsDiff"]["purity"]], marker="^", s=90, color=COL["CountsDiff"], label="CountsDiff (selected K)", zorder=5)
    a.axhline(r["ceiling"], color=MUTED, lw=1.2, ls="--", label="Purity of real val cells", zorder=1)
    a.set_xscale("log"); a.set_ylim(-0.03, 1.08)
    a.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5))); a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a.set_title(NAMES[t] + (" (step 2 pending)" if r["pick_provisional"] else ""), fontsize=FS_PANEL, fontweight="bold")
    a.spines[["top", "right"]].set_visible(False); a.grid(lw=0.4, color=GRID, which="both"); a.set_axisbelow(True)
for i in (0, 3): ax[i].set_ylabel("Purity", fontsize=FS_AX)
for i in (2, 3, 4): ax[i].set_xlabel("W1 (log scale)", fontsize=FS_AX)
h, l = [], []
for a in ax[:5]:
    for hh, ll in zip(*a.get_legend_handles_labels()):
        if ll not in l: h.append(hh); l.append(ll)
ax[5].axis("off"); ax[5].legend(h, l, loc="center", frameon=False, fontsize=FS_LEG)
fig.tight_layout(); fig.savefig(fig_stem + ".png", dpi=300); fig.savefig(fig_stem + ".pdf")

# ---------- report ----------
has_sl = sl is not None; any_pend = any(r["pick_provisional"] for r in res["classes"].values())
title = f"R16 val diagnosis: the lowest W1 CRNDiff reaches on the τ grid is below CFGen for {n_lt}/{n_cls} classes; purity of that cell in the table below"
L = [f"# {title}", "",
     "**Context: a reviewer suggested the HLCA benchmark used by CFGen and scLDM. CFGen conditional generation has lower W1 on val and test than the setting chosen by our pre-registered rule. Question: if purity is lowered a bit, can W1 catch up with CFGen? This only uses existing val readings to answer how low the grid can go; **no test reads, no change of selection, no conclusion**.", "",
     f"**data.** val split of HLCA core (50 classes), {n_cls} target classes (multi-ciliated was removed from the response; scoring still uses the original fixed order of 5 classes, numbers unchanged); all val cells of each class are used (type II pneumocyte uses a fixed subset of 2,500 for paired metrics; the ceiling still uses all 4,997)."
     "CRNDiff: 1 training seed (20260625), sampling seed 20260987, 2,500 generated per cell; τ ∈ {0.2, 0.4, 0.6, 0.8, 1.0},α = 1"
     + (("; multi-ciliated also has step 2 τ {0.4, 0.6, 0.8} × α {2, 4}" if res["classes"]["multi-ciliated epithelial cell"]["step2_cells_in"] else "") if "multi-ciliated epithelial cell" in res["classes"] else "")
     + ". CFGen: public checkpoint, sampling seed 20260987, 2,500 per class." + ("scLDM same." if has_sl else "scLDM removed from the HLCA baselines on 11-10 (reason: no scLDM on HVG2K); readings archived.")
     + "CountsDiff is the K chosen by its own pre-registered rule.", "",
     "**Method.** All three metrics are val readings; scorer, class order, subset and val readings are the same for all methods.",
     "- W1: 1D Wasserstein-1 distance between generated and real val cells per gene (raw counts, matched cell counts), averaged over 2,000 genes:$$\\mathrm{W1}=\\frac{1}{G}\\sum_{g=1}^{G}W_1\\big(\\hat P^{\\mathrm{gen}}_g,\\hat P^{\\mathrm{val}}_g\\big),\\quad G=2000.$$",
     "- purity: share of generated cells classified as the target class by a logistic classifier fitted once on train and frozen; the ceiling is the accuracy of the same classifier on all real val cells of the class.",
     "- coverage: fraction of real cells with a generated cell in their 5-NN ball (PCA space, k = 5; replaces recall).",
     "- pre-registered rule (used only on val, fixed): W1* is the smallest W1 over the cells of the class; candidates are cells with W1 ≤ 1.25 × W1*; among them take the smallest |purity − ceiling|. ▼ in the table is the lowest-W1 cell, not the rule-selected one.", "",
     "**results.**", "", "Table 1: W1, purity, coverage per class and cell on val (★ selected by the pre-registered rule, ▼ lowest W1 on the grid; 1 seed)", ""] + T + ["",
     f"![[figs/{os.path.basename(fig_stem)}.png]]", "",
     "Figure 1: one target class per panel. x axis W1 (log scale, lower is better), y axis purity (closer to the dashed line is better; the dashed line is the purity of real val cells). The blue line joins the 5 CRNDiff τ values (α = 1), labelled with τ; hollow squares and diamonds are α = 2, 4 from step 2 (when present); the black circle is the cell chosen by the pre-registered rule. Stars are CFGen" + (",scLDM" if has_sl else "") + "; triangles are the K chosen by CountsDiff. The CFGen star lies left of all CRNDiff points, meaning no cell on the grid of that class has lower W1 than CFGen (" + f"{n_cls - n_lt}/{n_cls} classes).", "",
     "per-class description (description only):", ""] + D + ["",
     f"- Total: the lowest W1 on the CRNDiff grid is below CFGen for {n_lt}/{n_cls} classes.",
     "", "**what this means for the rebuttal.** Facts only; no conclusion and no change of selection is suggested.",
     "- The lowest W1 in the table is a cell picked after looking at val, not selected by the pre-registered rule, and must not be reported as a test result.",
     "- only 1 training seed and 1 sampling seed; the W1 differences have no uncertainty estimate.",
     (("- multi-ciliated step 2 not finished; the selected cell of this class is provisional." if any_pend else "- multi-ciliated selected cell already includes step 2.") if "multi-ciliated epithelial cell" in res["classes"] else None),
     "",
     "> [!summary] summary",
     f"> Uses only existing val readings: the lowest W1 reachable on the CRNDiff τ grid is below CFGen for {n_lt}/{n_cls} classes. The purity of the lowest-W1 cell per class is listed together with the rule-selected cell in Table 1 and in the per-class description above. Test was not read; the selection is unchanged.", "",
     "> [!info]- reproducibility information",
     f"> - script:`scripts/r16/r16_val_tradeoff.py`(sha256 {sha(__file__)[:8]}), command `python -I scripts/r16/r16_val_tradeoff.py <results/figs/R16_val_tradeoff> <results/R16_val_tradeoff.md>`(crndiff-rb environment).",
     "> - scoring: score_with_npz.py ed519051, purity ruler runs/r16/purity_lr/purity_lr_hlca.npz (219e9bce), class order classical monocyte → multi-ciliated → plasma cell → ionocyte → type II, val type II subset 7816c490.",
     "> - inputs (first 8 hex of sha256):" + ";".join(f"`{k}` {v}" for k, v in src.items()) + ".",
     f"> - outputs:`runs/r16/report_notes/val_tradeoff.json`; figure `results/figs/{os.path.basename(fig_stem)}.png/.pdf`."]
open(rep, "w").write("\n".join(x for x in L if x is not None) + "\n")
print(title); print("\n".join(D)); print(f"{n_lt}/{n_cls}")
