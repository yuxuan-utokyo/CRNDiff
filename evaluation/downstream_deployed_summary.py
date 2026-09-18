# -*- coding: utf-8 -*-
"""Aggregate the deployed-arm downstream output into one summary json.

The aggregation is identical to the earlier one (mean, sample standard deviation, per-seed, min
and max); only the input round differs. Two things are recorded that the earlier summary did not
have:

* `baseline_rng_shift`. No baseline array changed by a single byte, but the downstream blocks
  draw from ONE SHARED random stream in a fixed arm order. Our arm's row count changed, so the
  permutations the later arms receive changed with it. Our arm is drawn BEFORE every baseline,
  so our numbers are unaffected; the baselines' shift is quantified here rather than hidden.
* the per-arm delivery sizes actually used, so a reader can see which rows moved and why.

    python -m evaluation.downstream_deployed_summary
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                        # noqa: E402
from evaluation import downstream_multiseed as MS                           # noqa: E402

WORK = C.OUT / "e33_197"
SUMMARY = C.OUT / "summary"
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
ARMS_A = ["REAL (ceiling)", "OURS", "scVI", "scANVI", "CFGen"]
ARMS_B = ["OURS", "scVI", "scANVI", "CFGen"]
OURS_SEEDS = [20260987, 20260988, 20260989]


def agg(v):
    a = np.asarray(v, float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "n_seeds": int(a.size), "per_seed": [float(x) for x in a],
            "min": float(a.min()), "max": float(a.max())}


def main() -> int:
    raw = json.loads((WORK / "raw_deployed_arm.json").read_text(encoding="utf-8"))
    S = [str(s) for s in raw["seeds"]]
    A, B = raw["per_seed_block_a"], raw["per_seed_block_b"]
    K = raw["per_seed_topk"]

    block_a = {}
    for arm in ARMS_A:
        block_a[arm] = {
            "macro_F1": agg([A[s]["arms"][arm]["macro_F1"] for s in S]),
            "per_type_F1": {t: agg([A[s]["arms"][arm]["per_type_F1"][t] for s in S])
                            for t in TYPES}}
    block_b = {}
    for t in TYPES:
        block_b[t] = {"arms": {}}
        for arm in ARMS_B:
            e = {"spearman_rho_all_genes":
                 agg([B[s]["types"][t]["arms"][arm]["spearman_rho_all_genes"] for s in S])}
            for k in raw["K_registered"]:
                e[f"top{k}_overlap_fraction"] = agg(
                    [K[s][t][arm][f"top{k}_overlap_fraction"] for s in S])
            block_b[t]["arms"][arm] = e

    # baseline shift, which should come only from the random-number order
    oa = json.loads((SUMMARY / "e33_block_a_multiseed.json").read_text(encoding="utf-8"))
    ob = json.loads((SUMMARY / "e33_block_b_multiseed_topk.json").read_text(encoding="utf-8"))
    shifts = []
    for arm in [a for a in ARMS_A if a != "OURS"]:
        for met in ["macro_F1"] + TYPES:
            o = oa["arms"][arm]["macro_F1"] if met == "macro_F1" else oa["arms"][arm]["per_type_F1"][met]
            n = block_a[arm]["macro_F1"] if met == "macro_F1" else block_a[arm]["per_type_F1"][met]
            pooled = float(np.hypot(o["sd"], n["sd"]) / np.sqrt(2))
            shifts.append({"block": "a", "arm": arm, "metric": met, "was": o["mean"],
                           "now": n["mean"], "shift_in_pooled_sd": abs(n["mean"] - o["mean"]) / pooled})
    for arm in [a for a in ARMS_B if a != "OURS"]:
        for t in TYPES:
            for key in ["top100_overlap_fraction", "spearman_rho_all_genes"]:
                o, n = ob["types"][t]["arms"][arm][key], block_b[t]["arms"][arm][key]
                pooled = float(np.hypot(o["sd"], n["sd"]) / np.sqrt(2))
                shifts.append({"block": "b", "arm": arm, "metric": f"{t} {key}",
                               "was": o["mean"], "now": n["mean"],
                               "shift_in_pooled_sd": abs(n["mean"] - o["mean"]) / pooled})
    worst = max(shifts, key=lambda d: d["shift_in_pooled_sd"])

    # sensitivity to the delivered arrays
    ev = S[0]
    sens = {}
    for os_ in OURS_SEEDS:
        f = WORK / (f"raw_seed_{ev}.json" if os_ == OURS_SEEDS[0]
                    else f"raw_seed_{ev}_ours{os_}.json")
        if not f.exists():
            sens = {"status": f"not run: {f.name} missing"}
            break
        r = json.loads(f.read_text(encoding="utf-8"))
        sens[str(os_)] = {
            "macro_F1": r["per_seed_block_a"][ev]["arms"]["OURS"]["macro_F1"],
            "F1": {t: r["per_seed_block_a"][ev]["arms"]["OURS"]["per_type_F1"][t] for t in TYPES},
            "top100": {t: r["per_seed_topk"][ev][t]["OURS"]["top100_overlap_fraction"] for t in TYPES},
            "rho": {t: r["per_seed_block_b"][ev]["types"][t]["arms"]["OURS"]["spearman_rho_all_genes"]
                    for t in TYPES}}
    if "status" not in sens:
        def spread(get):
            v = [get(sens[str(o)]) for o in OURS_SEEDS]
            return {"values": v, "spread": max(v) - min(v)}
        sens = {"evaluation_seed_held_fixed": int(ev), "per_array": sens,
                "spread": {"macro_F1": spread(lambda d: d["macro_F1"]),
                           **{f"F1[{t}]": spread(lambda d, t=t: d["F1"][t]) for t in TYPES},
                           **{f"rho[{t}]": spread(lambda d, t=t: d["rho"][t]) for t in TYPES}},
                "note": "the main run uses the registered round seed 20260987; it is not the "
                        "best of the three on macro F1 or on neuronal F1 (20260988 is), so the "
                        "choice is not favourable selection"}

    out = {
        "ticket": 197,
        "table": "E33 block a (TSTR) + block b (DE), OURS arm = the DEPLOYED sampler",
        "what_changed": raw["what_changed"],
        "arm_swap": raw["arm_swap"], "deployed_seed": raw["deployed_seed"],
        "seeds_registered": raw["seeds"],
        "seed_meaning": "EVALUATION seeds: they redraw which cells are taken from each fixed "
                        "delivery pool and which real cells form the ceiling training set. The "
                        "delivered arrays are FIXED. This is not the same quantity as Table 1's "
                        "sd, which is over three SAMPLING seeds.",
        "sd": "sample standard deviation over the evaluation seeds (ddof=1)",
        "g0_environment_gate": raw["g0"], "body": raw["body"],
        "wrapper_sha256": raw["wrapper_sha256"], "multiseed_sha256": raw["multiseed_sha256"],
        "summary_sha256_of_raw": MS.sha256_file(WORK / "raw_deployed_arm.json"),
        "protocol": {"n_per_type_train": 2500, "n_genes": 2000,
                     "representation": "log(1+CP10K)", "test": "held-out real cells",
                     "n_test": raw["per_seed_block_a"][S[0]]["n_test"]},
        "block_a": block_a, "block_b": block_b,
        "K_registered": raw["K_registered"],
        "baseline_rng_shift": {
            "why": "baseline arrays are byte-identical; block_a/block_b consume ONE rng stream "
                   "in arm order (REAL, OURS, scVI, scANVI, CFGen), so changing the OURS array's "
                   "row count changes the permutations the later arms receive. OURS is drawn "
                   "before every baseline, so the OURS numbers are unaffected.",
            "n_cells_checked": len(shifts), "max_shift_in_pooled_sd": worst["shift_in_pooled_sd"],
            "n_over_2sd": sum(1 for d in shifts if d["shift_in_pooled_sd"] > 2),
            "cells": shifts},
        "delivered_array_sensitivity": sens,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    SUMMARY.mkdir(parents=True, exist_ok=True)
    p = SUMMARY / "e33_deployed_arm.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n",
                 encoding="utf-8")
    print(f"[out] {p}\n[out] sha256 {MS.sha256_file(p)}")
    print(f"[chk] baseline rng shift: max {worst['shift_in_pooled_sd']:.2f} sd "
          f"({worst['arm']} {worst['metric']}), over 2 sd: "
          f"{sum(1 for d in shifts if d['shift_in_pooled_sd'] > 2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
