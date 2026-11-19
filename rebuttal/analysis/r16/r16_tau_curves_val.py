#!/usr/bin/env python
r"""R16 tau-metric curves on VAL.
Seed 1, tau grid {0.2, 0.4, 0.6, 0.8, 1.0}, alpha 1; multi-ciliated step-2 alpha {2, 4} cells drawn as separate markers (not joined) when
runs/r16/tau_val/select_multiciliated_step2.json exists. Existing val reads only; no test. Describes only, proposes no rule.
Rows = classes by rarity (type II -> classical monocyte -> multi-ciliated -> plasma -> ionocyte); columns = W1, MMD^2, purity, recall,
coverage, 1-NN. Baselines (CFGen, CountsDiff selected K, scVI, scANVI; scLDM when runs/r16/base_scldm/scores_val.json exists) are
horizontal dashed lines, one colour per method across the figure; real-cell references (purity ceiling; W1 / MMD^2 / recall / coverage /
1-NN noise floor: n real train cells vs val, two_sample_heldout_basis for the PCA metrics, as the scorer's docstring says) are grey dashed.
The current rule's pick is an open circle. W1 on a log axis (ionocyte reaches 19 at tau 1.0). Rebuttal font spec 16 / 14 / 14 / 15.
    python r16_tau_curves_val.py FIG_STEM REPORT.md [all]
v3 (11-10 12:3x; v2 5ca11393 archived): 4 reply classes by default (the task notes 12:25, : multi-ciliated removed from replies;
"all" adds it back for internal use). Coverage replaces recall in the main figure and table; recall goes to an appendix table.
The scores are read unchanged from the 5-class scoring runs (fixed class order), so the 4 remaining classes are the same numbers.
"""
import hashlib, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

W = "./work"; R = f"{W}/runs/r16"; fig_stem, rep = sys.argv[1:3]
CL = [("type II pneumocyte", "typeIIpneumocyte"), ("classical monocyte", "classicalmonocyte"), ("multi-ciliated epithelial cell", "multiciliated"),
      ("plasma cell", "plasmacell"), ("ionocyte", "ionocyte")]
ALL = len(sys.argv) > 3 and sys.argv[3] == "all"
if not ALL: CL = [c for c in CL if c[1] != "multiciliated"]
MET = [("W1", "W1", "W1 ↓"), ("mmd2", "MMD²", "MMD² ↓"), ("purity_logreg", "Purity", "Purity"),
       ("coverage", "Coverage", "Coverage ↑"), ("nn1_acc", "1-NN accuracy", "1-NN (0.5 best)")]   # recall: appendix only
TAUS = (0.2, 0.4, 0.6, 0.8, 1.0)
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
src = {}
def load(f):
    src[os.path.relpath(f, W)] = sha(f)[:8]; return json.load(open(f))

base_runs = {"scVI": None, "scANVI": None}
bs = load(f"{R}/base/scores_val.json"); cf = load(f"{R}/base_cfgen/scores_val.json")
sl_f = f"{R}/base_scldm/scores_val.json"; sl = load(sl_f) if os.path.exists(sl_f) else None
D = {}
for t, k in CL:
    f2 = f"{R}/tau_val/select_{k}_step2.json"; step2 = os.path.exists(f2)
    s = load(f2 if step2 else f"{R}/tau_val/select_{k}.json")
    sc = load(f"{R}/tau_val/scores_{k}_val.json"); fl = sc["floors"][t]
    floor = {"W1": fl["W1"]["W1"], "mmd2": fl["e21"]["mmd2"], "purity_logreg": s["ceiling"], "recall": fl["two_sample_heldout_basis"]["recall"]["mean"],
             "coverage": fl["two_sample_heldout_basis"]["coverage"]["mean"], "nn1_acc": fl["two_sample_heldout_basis"]["nn1_acc"]["mean"]}
    cells = {a: c for a, c in s["cells"].items()}
    cd = load(f"{R}/cd_val/select_{k}.json"); cdc = cd["cells"][cd["pick"]]
    b = {"CFGen": [r for r in cf["runs"] if r["target"] == t][0], f"CountsDiff ({cd['pick'].split('_')[0]})": cdc}
    for r in bs["runs"]:
        if r["target"] == t: b[r["method"]] = r
    if sl: b["scLDM"] = [r for r in sl["runs"] if r["target"] == t][0]
    D[t] = {"cells": cells, "pick": s["pick"], "provisional": bool(s["step2_triggered"]) and not step2, "step2_in": step2, "floor": floor, "base": b, "ceiling": s["ceiling"]}

COL = {"CRNDiff": "#2a78d6", "CFGen": "#eb6834", "scVI": "#eda100", "scANVI": "#e87ba4", "CountsDiff": "#7a4fc4", "scLDM": "#1baf7a"}
DASH = {"CFGen": (0, (5, 2)), "scVI": (0, (5, 2)), "scANVI": (0, (5, 2)), "CountsDiff": (0, (5, 2)), "scLDM": (0, (5, 2))}
GREY, GRID = "#8a8986", "#e4e3de"
col_of = lambda n: COL[n.split(" ")[0]]
FS_AX, FS_TICK, FS_LEG, FS_PANEL = 16, 14, 14, 15
plt.rcParams.update({"font.size": FS_TICK, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(len(CL), len(MET), figsize=(22, 3.8 * len(CL) + 1.0))
NAMES = {"type II pneumocyte": "Type II pneumocyte", "classical monocyte": "Classical monocyte", "multi-ciliated epithelial cell": "Multi-ciliated",
         "plasma cell": "Plasma cell", "ionocyte": "Ionocyte"}
for i, (t, _) in enumerate(CL):
    d = D[t]; c1 = sorted([c for c in d["cells"].values() if c["alpha"] == 1], key=lambda c: c["tau"])
    for j, (key, short, title) in enumerate(MET):
        a = ax[i, j]
        a.plot([c["tau"] for c in c1], [c[key] for c in c1], "-o", color=COL["CRNDiff"], lw=2.0, ms=6, zorder=4)
        for al, mk in ((2, "s"), (4, "D")):
            cc = [c for c in d["cells"].values() if c["alpha"] == al]
            if cc: a.scatter([c["tau"] for c in cc], [c[key] for c in cc], marker=mk, s=46, facecolor="white", edgecolor=COL["CRNDiff"], lw=1.4, zorder=5)
        p = d["cells"][d["pick"]]
        a.scatter([p["tau"]], [p[key]], s=230, facecolor="none", edgecolor="#0b0b0b", lw=1.4, zorder=6)
        for n, r in d["base"].items():
            a.axhline(r[key], color=col_of(n), ls=DASH[n.split(" ")[0]], lw=1.5, zorder=2)
        a.axhline(d["floor"][key], color=GREY, ls=(0, (3, 2)), lw=1.6, zorder=1)
        if key == "W1":
            a.set_yscale("log"); a.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
            a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g")); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_xticks(TAUS); a.set_xticklabels(["0.2", "0.4", "0.6", "0.8", "1"]); a.set_xlim(0.12, 1.08)
        a.spines[["top", "right"]].set_visible(False); a.grid(lw=0.4, color=GRID, which="major"); a.set_axisbelow(True)
        if i == 0: a.set_title(title, fontsize=FS_PANEL, fontweight="bold")
        if i == len(CL) - 1: a.set_xlabel("τ", fontsize=FS_AX)
        if j == 0: a.set_ylabel(NAMES[t] + (" *" if d["provisional"] else ""), fontsize=FS_AX, fontweight="bold")
H = [Line2D([], [], color=COL["CRNDiff"], marker="o", lw=2, label="CRNDiff (α = 1)"),
     Line2D([], [], color="none", marker="o", markersize=14, markerfacecolor="none", markeredgecolor="#0b0b0b", label="Pre-registered rule's pick (archived)")]
if any(d["step2_in"] for d in D.values()):
    H += [Line2D([], [], color="none", marker="s", markerfacecolor="white", markeredgecolor=COL["CRNDiff"], label="CRNDiff α = 2"),
          Line2D([], [], color="none", marker="D", markerfacecolor="white", markeredgecolor=COL["CRNDiff"], label="CRNDiff α = 4")]
seen = []
for d in D.values():
    for n in d["base"]:
        m = n.split(" ")[0]
        if m not in seen: seen.append(m)
for m in seen:
    H.append(Line2D([], [], color=COL[m], ls=DASH[m], lw=1.5, label=m + (" (selected K)" if m == "CountsDiff" else "")))
H.append(Line2D([], [], color=GREY, ls=(0, (3, 2)), lw=1.6, label="Real cells (purity ceiling / noise floor)"))
fig.legend(handles=H, loc="upper center", ncol=len(H), frameon=False, fontsize=FS_LEG, bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.2)
fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(fig_stem + ".png", dpi=300); fig.savefig(fig_stem + ".pdf")

# ---------- table + report ----------
fmt = {"W1": "{:.4f}", "mmd2": "{:.5f}", "purity_logreg": "{:.3f}", "recall": "{:.3f}", "coverage": "{:.3f}", "nn1_acc": "{:.3f}"}
row = lambda name, x: f"| {name} | " + " | ".join(fmt[k].format(x[k]) for k, _, _ in MET) + " |"
T = []
for t, _ in CL:
    d = D[t]
    T += [f"**{t}**" + ("(step 2 unfinished, selected cell provisional)" if d["provisional"] else ""), "", "| setting | " + " | ".join(s for _, s, _ in MET) + " |", "|---|" + "---|" * len(MET)]
    for a, c in sorted(d["cells"].items(), key=lambda kv: (kv[1]["alpha"], kv[1]["tau"])):
        T.append(row(f"CRNDiff τ {c['tau']:g}" + (f", α {c['alpha']:g}" if c["alpha"] != 1 else "") + (" ○" if a == d["pick"] else ""), c))
    for n, r in d["base"].items(): T.append(row(n, r))
    T.append(row("real-cell reference", d["floor"])); T.append("")
# descriptive counts, computed
mono_p = sum(all(np.diff([c["purity_logreg"] for c in sorted([c for c in d["cells"].values() if c["alpha"] == 1], key=lambda c: c["tau"])]) >= 0) for d in D.values())
wmin = {t: min([c for c in d["cells"].values() if c["alpha"] == 1], key=lambda c: c["W1"])["tau"] for t, d in D.items()}
rmax = {t: max([c for c in d["cells"].values() if c["alpha"] == 1], key=lambda c: c["coverage"])["tau"] for t, d in D.items()}   # coverage max (name kept)
title = f"R16 val how CRNDiff metrics change with τ: classes where purity is monotone non-decreasing in τ {mono_p}/{len(D)}; at α=1 the τ with the lowest W1 is in {min(wmin.values()):g}–{max(wmin.values()):g}"
L = [f"# {title}", "",
     "**Context: a reviewer suggested the HLCA benchmark used by CFGen and scLDM. On HLCA, CFGen W1 is lower than at the τ chosen by our pre-registered rule for most classes. Before revisiting the τ selection rule, look at the full curve of every metric versus τ. This only describes the curves, proposes no new rule, and does not read test.", "",
     f"**data.** val split of HLCA core, {len(CL)} target classes, ordered by rarity (type II pneumocyte most common, ionocyte rarest)"
     + ("" if ALL else "(multi-ciliated was removed from the response; scoring still uses the original fixed order of 5 classes, and the numbers of these 4 classes are bit-identical to the 5-class case)") + "; all val cells of each class (paired metrics of type II pneumocyte use a fixed subset of 2,500). CRNDiff: training seed 20260625, sampling seed 20260987, 2,500 generated per cell, τ ∈ {0.2, 0.4, 0.6, 0.8, 1.0},α = 1"
     + (("; multi-ciliated also has τ {0.4, 0.6, 0.8} × α {2, 4}, drawn as separate points without lines" if D["multi-ciliated epithelial cell"]["step2_in"] else "") if ALL else "") + "."
     "baselines: CFGen (public checkpoint, sampling seed 20260987), CountsDiff (K chosen by its own pre-registered rule), scVI,scANVI" + (",scLDM" if sl else "(scLDM was removed from the HLCA baselines on 11-10 because there is no scLDM on HVG2K; readings archived)") + ", all with 1 seed and 2,500 per class.", "",
     "**Method.** All are existing val readings; every method uses the same scoring (scorer, class order and subset identical).",
     "- W1: 1D Wasserstein-1 distance between generated and real val cells per gene (raw counts, matched cell counts), averaged over 2,000 genes:$$\\mathrm{W1}=\\frac{1}{G}\\sum_{g=1}^{G}W_1\\big(\\hat P^{\\mathrm{gen}}_g,\\hat P^{\\mathrm{val}}_g\\big),\\quad G=2000.$$",
     "- MMD²: biased Gaussian-kernel MMD² on log-CP10K, bandwidth = median pairwise distance of real cells.",
     "- purity: share of generated cells classified as the target class by a logistic classifier fitted on train and frozen; the reference line is the accuracy of the same classifier on all real val cells of the class (ceiling).",
     "- coverage, 1-NN: computed in a PCA space fitted on the train cells of that class (d = min(50, floor(n_val/2)), k = 5). coverage is the fraction of real cells with a generated cell within their 5-NN ball; 1-NN is the accuracy of guessing the source by nearest neighbour after pooling real and generated cells (0.5 is best). recall moved to the appendix (Table A1).",
     "- noise floor (grey dashed): the same metrics for n real train cells against val; PCA metrics use a basis fitted without these cells (the version specified in the scorer docs).",
     "- pre-registered rule (archived, no longer used for selection; circles only mark the cell it selected at the time): W1* is the smallest W1 over the cells of the class; candidates are cells with W1 ≤ 1.25 × W1*; among them take the smallest |purity − ceiling|; if purity < 0.85 × ceiling add α {2, 4}. Hollow circles in the figure are its selected cells.", "",
     "**results.**", "", f"![[figs/{os.path.basename(fig_stem)}.png]]", "",
     "Figure 1: rows are target classes (rarer from top to bottom), columns are metrics. Blue lines are CRNDiff over 5 τ values (α = 1); hollow circles are the cells chosen by the pre-registered (archived) rule"
     + ("; hollow squares and diamonds are α = 2 of step 2,4" if any(d["step2_in"] for d in D.values()) else "") + ". Coloured dashed lines are the baselines (one colour per method across the figure); grey dashed lines are the real-cell reference (the upper bound for the purity column, the noise floor for the others)."
     "**W1 column uses a log y axis** (ionocyte W1 is about 19 at τ = 1.0; a linear axis would squash the other points); other columns are linear." + ("* after a row name means step 2 of that class is unfinished and the selected cell is provisional." if any(d["provisional"] for d in D.values()) else "") + "1 seed(s), no uncertainty.", "",
     "Table 1: the numbers of Figure 1, cell by cell (○ is the cell selected by the pre-registered (archived) rule)", ""] + T + [
     "per-class description (description only, computed by the script from the table above):", ""]
for t, d in D.items():
    c1 = sorted([c for c in d["cells"].values() if c["alpha"] == 1], key=lambda c: c["tau"]); p = d["cells"][d["pick"]]
    L.append(f"- **{t}**: at α=1 the lowest W1 is at τ {wmin[t]:g} ({min(c['W1'] for c in c1):.4f}), highest coverage at τ {rmax[t]:g} ({max(c['coverage'] for c in c1):.3f}); purity from τ 0.2 {c1[0]['purity_logreg']:.3f} up to τ 1.0 {c1[-1]['purity_logreg']:.3f}(ceiling {d['ceiling']:.3f}). The pre-registered (archived) rule selects τ {p['tau']:g}" + (f",α {p['alpha']:g}" if p["alpha"] != 1 else "") + ("(provisional)" if d["provisional"] else "") + ".")
# the author 11-10 11:4x: one descriptive line on ionocyte tau 1.0 (numbers from the tables, library ratio from libsize_by_tau.json)
_io = D["ionocyte"]["cells"]["tau1p0"]; _lib = load(f"{R}/report_notes/libsize_by_tau.json")["classes"]["ionocyte"]["tau"]["1p0"]
APP = ["", "### appendix: recall (not in the main text)", "",
       "recall is the fraction of real cells that fall within the 5-NN ball of some generated cell (same PCA space, k = 5).", "",
       "Table A1: recall (○ is the cell selected by the pre-registered (archived) rule)", "", "| classes | " + " | ".join(f"τ {x:g}" for x in TAUS) + " | " + " | ".join(n for n in D[CL[0][0]]["base"]) + " | real-cell reference |",
       "|---|" + "---|" * (len(TAUS) + len(D[CL[0][0]]["base"]) + 1)]
for t, _ in CL:
    d = D[t]; c1 = {c["tau"]: (a, c) for a, c in d["cells"].items() if c["alpha"] == 1}
    APP.append(f"| {t} | " + " | ".join(f"{c1[x][1]['recall']:.3f}" + (" ○" if c1[x][0] == d["pick"] else "") for x in TAUS) + " | "
               + " | ".join(f"{r['recall']:.3f}" for r in d["base"].values()) + f" | {d['floor']['recall']:.3f} |")
APP += ["", f"- **ionocyte at τ 1.0**:recall {_io['recall']:.3f} is the highest for the class, but in the same cell purity {_io['purity_logreg']:.3f},W1 {_io['W1']:.1f},"
        f"median library size of generated cells relative to real val {_lib['lib_ratio_to_real']:.0f} times (zero fraction {_lib['zero_frac']:.2f}); the recall of this cell comes from degenerate cells with inflated counts, not a better fit."]
L += ["", "**what this means for the rebuttal.** Description only; no new rule is proposed.",
      "- These curves are all numbers seen on val; any rule set from them can read test only after it is fixed, and the report must state that the rule was changed after seeing the val curves.",
      "- 1 training seed(s), 1 sampling seed; small differences between adjacent τ may be within noise.", "",
      ] + APP + ["",
      "> [!summary] summary",
      f"> HLCA val, on seed 1, CRNDiff purity is monotone non-decreasing in τ for {mono_p}/{len(D)} classes; at α = 1 the τ with the lowest W1 lies in {min(wmin.values()):g}–{max(wmin.values()):g} and the τ with the highest coverage in {min(rmax.values()):g}–{max(rmax.values()):g}. Full curves and baselines in Fig. 1 and Table 1; test was not read and the selection was not changed.", "",
      "> [!info]- reproducibility information",
      f"> - script `scripts/r16/r16_tau_curves_val.py`(sha256 {sha(__file__)[:8]}), command `python -I scripts/r16/r16_tau_curves_val.py <results/figs/R16_tau_curves_val> <results/R16_tau_curves_val.md>`(crndiff-rb environment).",
      "> - scoring: score_with_npz.py ed519051, purity ruler runs/r16/purity_lr/purity_lr_hlca.npz (219e9bce), class order classical monocyte → multi-ciliated → plasma cell → ionocyte → type II, val type II subset 7816c490.",
      "> - inputs (first 8 hex of sha256):" + ";".join(f"`{k}` {v}" for k, v in src.items()) + ".",
      f"> - output: figure `results/figs/{os.path.basename(fig_stem)}.png/.pdf`, this document."]
open(rep, "w").write("\n".join(L) + "\n")
json.dump({t: {"pick": d["pick"], "provisional": d["provisional"], "floor": d["floor"], "W1_min_tau": wmin[t], "coverage_max_tau": rmax[t]} for t, d in D.items()},
          open(f"{R}/report_notes/tau_curves_val.json", "w"), indent=1)
print(title)
