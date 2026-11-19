# -*- coding: utf-8 -*-
"""R3 figures, Table 1 runs, from rebuttal/out/R3/{r3_numbers.json,r3_partB.json}.
    cd Experiments/HVG && python -m rebuttal.r3_figures
"""
import json, os, sys
os.environ["OMP_NUM_THREADS"] = "2"
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.dont_write_bytecode = True
from rebuttal.w2_runs import ARMS, COL, TABLE1
OUT = Path(__file__).resolve().parent / "out" / "R3"
d = json.loads((OUT / "r3_numbers.json").read_text(encoding="utf-8")); b = json.loads((OUT / "r3_partB.json").read_text(encoding="utf-8"))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.5})
def t1(D, a):
    return [r for r in D["runs"] if r["arm"] == a and [r["train"], r["samp"]] in [list(x) for x in TABLE1[a]]]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.25), sharey=True)
for a, t in zip(ax, ["Endothelial", "Myeloid", "Neuronal"]):
    D = d[t]
    a.plot(range(1, 51), D["floor_var_ratio_per_pc"], color=INK, lw=1.3, ls=(0, (3, 1.5)), label="Real train vs val")
    for arm in ARMS:
        m = np.mean([r["var_ratio_per_pc"] for r in t1(D, arm)], 0)
        a.plot(range(1, 51), m, color=COL[arm], lw=1.1, label=arm)
    a.axhline(1, color=MUTED, lw=0.5); a.set_xscale("log"); a.set_xticks([1, 10, 50]); a.set_xticklabels(["1", "10", "50"]); a.set_ylim(0, 2)
    a.set_title(t, fontsize=8, pad=3); a.set_xlabel("PC, fitted on real train cells", color=MUTED, fontsize=7)
    a.spines[["top", "right"]].set_visible(False); a.grid(lw=0.3, color=GRID); a.set_axisbelow(True)
ax[0].set_ylabel("Variance ratio, generated / val", color=MUTED)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.02))
fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.17, wspace=0.08)
fig.savefig(OUT / "R3_variance_ratio.png", dpi=300); plt.close(fig)
