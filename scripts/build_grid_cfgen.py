# -*- coding: utf-8 -*-
"""The CFGen training-by-sampling grid (nine cells).

Numbers are read only from the scored artefact written by the unmodified scoring driver. The v7
and v8 purities are recomputed from each row's stored label histogram with the frozen purity
function, on the same ceilings the other grid builders use; W1, MMD^2 and PCC are taken as they
are. Two gates guard the result: the recomputed v7 purity must equal the value stored in each
row bitwise, and one run's grid must reproduce the previously published figures.

    python scripts/build_grid_cfgen.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.metrics import purity_from_hist                             # noqa: E402
from scripts.build_grid_mdlm import rulers                              # noqa: E402

S = C.OUT / "summary"
SRC = S / "219_cfgen_cond_3runs_scored.json"
OUT = S / "219_cfgen_grid_3x3.json"
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
RUNS = [1, 2, 3]
SEEDS = [20260931, 20260932, 20260933]
PAT = re.compile(r"^cfgen_cond_(\w+?)_(?:run(\d)_)?seed(\d+)_n(\d+)$")
FIELDS = [("purity_abs_v8", 4), ("purity_rel_v8", 4), ("purity_abs_v7", 4),
          ("purity_rel_v7", 4), ("W1", 4), ("mmd2", 5), ("pcc", 4)]


def summarise(M: np.ndarray) -> dict:
    ok = ~np.isnan(M)
    out = {"n_cells": int(ok.sum()), "complete": bool(ok.all()),
           "matrix_rows_run_cols_samp": [[None if np.isnan(x) else float(x) for x in r]
                                         for r in M]}
    if ok.all():
        flat = M.ravel()
        out["C_3x3"] = {"mean": float(flat.mean()), "sd": float(flat.std(ddof=1)),
                        "n": int(flat.size)}
        b = M[0, :]
        out["B_run1_3samp"] = {"mean": float(b.mean()), "sd": float(b.std(ddof=1)),
                               "n": int(b.size)}
        out["per_run_mean"] = [float(x) for x in M.mean(axis=1)]
        within = float(np.mean([r.var(ddof=1) for r in M]))
        between = float(M.mean(axis=1).var(ddof=1))
        out["variance_decomposition"] = {
            "within_training_run_sampling_var": within,
            "between_training_run_var": between,
            "ratio_between_over_within": (between / within) if within > 0 else None}
    return out


def main() -> int:
    v7, v8, ceil = rulers()
    rows = json.loads(SRC.read_text(encoding="utf-8"))["rows"]
    cells, g1 = {}, 0.0
    for r in rows:
        m = PAT.match(str(r["stem"]))
        if not m:
            print(f"unrecognised stem {r['stem']}")
            return 2
        t, run, seed = m.group(1), int(m.group(2) or 1), int(m.group(3))
        dist = r["label_dist_whole"]
        p7, _ = purity_from_hist(dist, v7[t])
        p8, _ = purity_from_hist(dist, v8[t])
        g1 = max(g1, abs(p7 - float(r["purity_whole_set"])))
        cells[f"{t}|{run}|{seed}"] = {
            "request": t, "cfgen_run": run, "sample_seed": seed, "stem": r["stem"],
            "n_delivered": r.get("n_delivered"),
            "purity_abs_v8": p8, "purity_rel_v8": p8 / ceil[t]["v8"],
            "purity_abs_v7": p7, "purity_rel_v7": p7 / ceil[t]["v7"],
            "purity_over_ceiling_stored": r.get("purity_over_ceiling"),
            "W1": r.get("W1"), "mmd2": r.get("mmd2_rbf_biased"), "pcc": r.get("pcc"),
            "sliced_W1": r.get("sliced_W1"), "scc": r.get("scc")}
    print(f"[G1] {len(cells)}/27 cells; v7 recompute max|delta| = {g1:.3e}")
    if len(cells) != 27 or g1 != 0.0:
        print("[G1] FAIL")
        return 1

    # gate: the first run must agree bitwise with the existing rescoring list
    res = json.loads((S / "215_purity_v8_rescore.json").read_text(encoding="utf-8"))
    ref = {r["stem"]: r for r in res["rows"] if r["file"] == "210_cfgen_cond_scored.json"}
    raw = {r["stem"]: r for r in json.loads((S / "210_cfgen_cond_rows.json")
                                            .read_text(encoding="utf-8"))["rows"]}
    g2 = 0.0
    for k, c in cells.items():
        if c["cfgen_run"] != 1:
            continue
        a, b = ref[c["stem"]], raw[c["stem"]]
        for x, y in ((c["purity_abs_v8"], a["purity_v8"]), (c["purity_abs_v7"], a["purity_v7"]),
                     (c["purity_rel_v8"], a["rel_v8"]), (c["W1"], b["W1"]),
                     (c["mmd2"], b["mmd2_rbf_biased"]), (c["pcc"], b["pcc"])):
            g2 = max(g2, abs(float(x) - float(y)))
    print(f"[G2] run-1 cells vs existing rescore / 210 raw rows: max|delta| = {g2:.3e}")
    if g2 != 0.0:
        print("[G2] FAIL")
        return 1

    by = {}
    for t in TYPES:
        by[t] = {}
        for f, _ in FIELDS:
            M = np.full((len(RUNS), len(SEEDS)), np.nan)
            for i, run in enumerate(RUNS):
                for j, s in enumerate(SEEDS):
                    c = cells.get(f"{t}|{run}|{s}")
                    if c is not None and isinstance(c.get(f), (int, float)):
                        M[i, j] = c[f]
            by[t][f] = summarise(M)

    ing = json.loads((S / "219_cfgen_ingest.json").read_text(encoding="utf-8")) \
        if (S / "219_cfgen_ingest.json").exists() else {}
    blob = {"ticket": 219, "arm": "CFGen", "generator_label":
            "CFGen (Palma et al.), cell-type conditioned",
            "train_runs": RUNS, "sample_seeds": SEEDS,
            "train_runs_note": ("upstream CFGen has no seed control; runs 2/3 are independent "
                                "re-runs of the same command on machine 2, step-aligned to run 1 "
                                "(encoder 24112, flow matching 29187)"),
            "ceilings": ceil, "source": SRC.name,
            "self_check": {"n_cells": len(cells), "v7_recompute_max_abs_delta": g1,
                           "run1_vs_existing_max_abs_delta": g2,
                           "ingest_gate_R1_passed": (ing.get("gate_R1_run1_unchanged") or {})
                           .get("passed")},
            "cells": cells, "by": by,
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if OUT.exists():
        OUT.rename(OUT.with_name(f"{OUT.stem}__preexisting_{int(time.time())}.json"))
    OUT.write_text(json.dumps(blob, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'request':<13}{'metric':<15}{'run 1 (1x3)':<22}{'C 3x3':<22}"
          f"{'per-run means':<30}betw/with")
    for t in TYPES:
        for f, d in FIELDS:
            e = by[t][f]
            r = e["variance_decomposition"]["ratio_between_over_within"]
            print(f"{t:<13}{f:<15}"
                  f"{e['B_run1_3samp']['mean']:.{d}f}+-{e['B_run1_3samp']['sd']:.{d}f}     "
                  f"{e['C_3x3']['mean']:.{d}f}+-{e['C_3x3']['sd']:.{d}f}     "
                  + "/".join(f"{x:.{d}f}" for x in e["per_run_mean"]) + "   "
                  + (f"{r:.1f}x" if r else "n/a"))
        print()
    print(f"[out] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
