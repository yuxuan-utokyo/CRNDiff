# -*- coding: utf-8 -*-
"""Table 5: training on generated cells under eleven-class replacement.

Source: results/summary_json/212_tstr11.json, read only, never recomputed, never transcribed.

The difference from the earlier three-class version is the label space. Here the classifier has
all eleven atlas classes; the three requested classes are trained on GENERATED cells only, the
other eight on real training cells, and the test set is every held-out real cell. In the
three-class version 'neuronal' effectively meant 'neither of the other two'; here it has to
compete against the real fibroblasts, pericytes and the rest for the same boundary.

    python scripts/reproduce/table_05_tstr.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "results" / "summary_json" / "212_tstr11.json"
ARMS = [("scVI", "scVI"), ("CFGen", "CFGen"), ("scANVI", "scANVI"),
        ("MDLM", "MDLM"), ("OURS", r"\textbf{Ours}")]
REF = "REAL (ceiling)"
COLS = [("macro", "Macro $F_1$ (11 classes) $\\uparrow$"),
        ("Endothelial", "Endothelial"), ("Myeloid", "Myeloid"), ("Neuronal", "Neuronal")]
ND = 3


def get(arms, k, col):
    return (arms[k]["macro_F1_11class"] if col == "macro" else arms[k][col]["F1"])


def main() -> int:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    sc = d.get("selfcheck_3class")
    if not sc or not sc.get("passed"):
        sys.exit("the 3-class self-check did not pass -- STOP, no table")
    arms = d["arms"]

    bold = {}
    for c, _ in COLS:
        v = {k: get(arms, k, c)["mean"] for k, _ in ARMS}
        top = max(v.values())
        # the best value, plus any arm within its own standard deviation of it
        bold[c] = {k for k, _ in ARMS
                   if v[k] == top or (top - v[k]) <= get(arms, k, c)["sd"]}

    def cell(k, c):
        e = get(arms, k, c)
        s = f"{e['mean']:.{ND}f}"
        if k in bold[c]:
            s = rf"\mathbf{{{s}}}"
        return f"${s} \\pm {e['sd']:.{ND}f}$"

    print(f"% source: {SRC.name}  sha256={hashlib.sha256(SRC.read_bytes()).hexdigest()[:16]}")
    print(f"% eval seeds: {d['eval_seeds']}  (EVALUATION seeds; the delivered arrays are fixed)")
    print(f"% test set: all {d['n_test']:,} held-out real cells, {d['n_classes']} classes")
    print(f"% train: the three target classes are GENERATED ONLY (n={d['n_per_type_train']} "
          f"each); the other eight are real training cells")
    print(f"% 3-class self-check vs e33_deployed_arm.json: "
          f"{sc['n_compared']} numbers, worst |delta| {sc['worst_abs_diff']:.1e}, "
          f"passed={sc['passed']}")
    print(f"% bold rule: best in the column, plus any arm within its own s.d. of the best; "
          f"the reference row is never bolded")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print("Training source & " + " & ".join(t for _, t in COLS) + r" \\")
    print(r"\midrule")
    for k, lab in ARMS:
        print(f"{lab:16s} & " + " & ".join(cell(k, c) for c, _ in COLS) + r" \\")
    print(r"\midrule")
    print("Real cells       & " + " & ".join(
        f"${get(arms, REF, c)['mean']:.{ND}f} \\pm {get(arms, REF, c)['sd']:.{ND}f}$"
        for c, _ in COLS) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
