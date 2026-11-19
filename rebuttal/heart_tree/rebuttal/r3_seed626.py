# -*- coding: utf-8 -*-
"""R3 row for the excluded train seed 20260626, same basis and procedure as r3_diversity.py."""
import json, sys
import numpy as np
sys.dont_write_bytecode = True
from rebuttal.r3_diversity import basis, score, OUT, SEED
from rebuttal.w2_runs import RUNS, TARGETS, log_cp10k, real_cells, ours_stem
res = {}
for t in TARGETS:
    p, _ = basis(t); zv = p.transform(log_cp10k(real_cells("val", t))); vv = zv.var(0, ddof=1)
    paths = [RUNS / "table2_seeds_s20260626" / f"{t}_s20260626" / ours_stem(t, 20260987)] + [RUNS / f"table2_ours_s20260626_samp{s}" / t / f"ours_{t}_train20260626_samp{s}.npy" for s in (20260988, 20260989)]
    rows = []
    for j, f in enumerate(paths):
        s = score(zv, p.transform(log_cp10k(np.asarray(np.load(f, mmap_mode="r"), dtype=np.float64))), SEED + 300 + j, vv)
        rows.append({k: v for k, v in s.items() if k != "var_ratio_per_pc"})
    res[t] = rows
    print(t, {k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("nn1_acc", "precision", "recall", "density", "coverage", "var_ratio_top10", "var_ratio_top50")}, flush=True)
(OUT / "r3_seed626.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
