# -*- coding: utf-8 -*-
"""R14 figure: existing INTERSECT_Endothelial_g128 results (6 sampling seeds per arm), read from the archived summaries only.
    python -m rebuttal.r14_figure <out.png>
"""
import json, os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from pathlib import Path
import collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = Path(__file__).resolve().parents[1] / "out/summary"
d = json.loads((S / "intersect_intersect_ladder.json").read_text(encoding="utf-8")); p = json.loads((S / "purity_intersect_ladder.json").read_text(encoding="utf-8"))
pur = collections.defaultdict(list)
for r in p["rows"]: pur[r["arm"]].append(r["purity_matched_n"])
arms = [("tilt_only", "tilt only"), ("fk_only", "FK only"), ("a1", "tilted FK")]
real = d["real_uap1_rate_in_type"]["value"]
panels = [("UAP1$^+$ share of delivered cells", lambda a: d["per_arm"][a]["uap1_pos_frac_delivered"]["values"], real, "real endothelial"),
          ("Endothelial purity (CellTypist, matched n)", lambda a: pur[a], None, None),
          ("marginal W1 / noise floor", lambda a: d["per_arm"][a]["mW1_over_floor"]["values"], 1.0, "floor")]
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for k, (title, get, ref, lab) in enumerate(panels):
    for i, (a, name) in enumerate(arms):
        v = get(a); ax[k].scatter([i] * len(v), v, s=18, color="0.25", zorder=3); ax[k].hlines(sum(v) / len(v), i - 0.25, i + 0.25, color="C0", lw=2)
    if ref is not None: ax[k].axhline(ref, color="C3", ls="--", lw=1); ax[k].text(2.45, ref, lab, color="C3", fontsize=8, va="bottom", ha="right")
    ax[k].set_xticks(range(3)); ax[k].set_xticklabels([n for _, n in arms]); ax[k].set_title(title, fontsize=9); ax[k].set_xlim(-0.5, 2.5)
fig.suptitle("Endothelial ∩ UAP1$^+$ request (not a training label): 6 sampling seeds per arm, one generator; bar = mean", fontsize=9)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=160); print(sys.argv[1])
