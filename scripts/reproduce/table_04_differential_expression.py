# -*- coding: utf-8 -*-
"""Table 4: the differential-expression readouts.

Source: results/summary_json/212_cfgen_de.json, read only, never recomputed, never transcribed.
Four of the five rows are carried across from the earlier downstream artefacts unchanged; the
CFGen row is the one computed for this round on its new pool, by a single-arm run.

Bold marks the best mean in a column plus any arm within its own standard deviation of it.

    python scripts/reproduce/table_04_differential_expression.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "results" / "summary_json" / "212_cfgen_de.json"
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
ARMS = [("scVI", "scVI"), ("CFGen", "CFGen"), ("scANVI", "scANVI"),
        ("MDLM", "MDLM"), ("OURS", r"\textbf{Ours}")]
COLS = [("top100_overlap_fraction", 3), ("spearman_rho_all_genes", 4)]
BLOCK = {"Endothelial": r"\emph{abundant}, $80{,}463$ training cells",
         "Myeloid": r"$18{,}422$ training cells",
         "Neuronal": r"\emph{rare}, $3{,}168$ training cells"}


def main() -> int:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    if not d["g0_passed"]:
        sys.exit("the G0 environment gate did not pass -- STOP, no table")
    B = d["table3_all_arms"]

    bold = {}
    for t in TYPES:
        for key, nd in COLS:
            v = {a: B[t][a][key]["mean"] for a, _ in ARMS}
            top = max(v.values())
            bold[(t, key)] = {a for a, _ in ARMS
                              if v[a] == top or (top - v[a]) <= B[t][a][key]["sd"]}

    def cell(t, key, nd, arm):
        e = B[t][arm][key]
        s = f"{e['mean']:.{nd}f}"
        if arm in bold[(t, key)]:
            s = rf"\mathbf{{{s}}}"
        return f"${s} \\pm {e['sd']:.{nd}f}$"

    print(f"% source: {SRC.name}  sha256={hashlib.sha256(SRC.read_bytes()).hexdigest()[:16]}")
    print(f"% eval seeds: {d['eval_seeds']}  (EVALUATION seeds; the delivered arrays are fixed)")
    print("% rows scVI / scANVI / Ours are e33_deployed_arm.json VERBATIM; MDLM is "
          "209_mdlm_de_tstr.json VERBATIM; neither was recomputed.")
    print(f"% CFGen moved to the ticket-210 delivery, sampling seed {d['pool_seed']} "
          "(was the archived single sample, seed 20261061, n=20000).")
    print("% CFGen and MDLM were each measured on the single-arm route (ticket 209 "
          "amendment 1): inside block_b a second arm shifts the shared rng stream, so the "
          "published rows could not be reproduced bitwise in a joint run.")
    print(f"% G0 environment gate: {d['g0_n_compared']} numbers, "
          f"{d['g0_n_mismatch']} mismatch, passed={d['g0_passed']}")
    print("% bold: best in the column, plus any arm within its own s.d. of the best")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Sampler & Top-100 marker overlap $\uparrow$ "
          r"& Spearman, all 2,000 genes $\uparrow$ \\")
    for t in TYPES:
        print(r"\midrule")
        print(rf"\multicolumn{{3}}{{l}}{{\textbf{{{t}}} \; {BLOCK[t]}}} \\")
        for arm, lab in ARMS:
            print(f"{lab:14s} & {cell(t, 'top100_overlap_fraction', 3, arm)} "
                  f"& {cell(t, 'spearman_rho_all_genes', 4, arm)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
