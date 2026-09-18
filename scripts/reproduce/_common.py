# -*- coding: utf-8 -*-
"""Shared helpers for the per-table reproduction scripts.

Every number printed by these scripts is read from a file under results/ and, where a row spans
several runs, averaged over those runs. Nothing is transcribed from the paper, and a cell whose
source file is not in this repository is printed as MISSING(<file>) rather than filled in.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "results" / "summary_json"
BY_TABLE = REPO / "results" / "by_table"
SEEDS_T2 = REPO / "results" / "table02_seeds"
OUT = REPO / "out" / "tables"

TYPES = ["Endothelial", "Myeloid", "Neuronal"]
TRAIN_CELLS = {"Endothelial": "80{,}463", "Myeloid": "18{,}422", "Neuronal": "3{,}168"}
ABUNDANCE = {"Endothelial": r" \; \emph{abundant}", "Myeloid": "", "Neuronal": r" \; \emph{rare}"}


def load(name: str):
    p = SUMMARY / name
    if not p.exists():
        raise SystemExit(f"missing result file: {p}\n  -> see assets/README.md for how to rebuild it")
    return json.loads(p.read_text(encoding="utf-8"))


def mean_sd(values):
    values = [float(v) for v in values]
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def cell(m, s, digits, bold=False, with_sd=True):
    if m is None:
        return "MISSING"
    body = f"{m:.{digits}f}"
    if bold:
        body = r"\mathbf{" + body + "}"
    if with_sd and s is not None:
        return f"${body} \\pm {s:.{digits}f}$"
    return f"${body}$"


def group_header(request: str, ncols: int) -> str:
    return (f"\\multicolumn{{{ncols}}}{{l}}{{\\textbf{{{request}}}{ABUNDANCE[request]}, "
            f"${TRAIN_CELLS[request]}$ training cells}} \\\\")


def stem_index(files):
    """stem -> merged metric dict, built from every scored artefact we ship.

    A row is anything in the JSON that carries a 'stem' key; later files refine earlier ones only
    for keys they actually provide, so no file can blank out another's metric.
    """
    idx: dict[str, dict] = {}
    for name in files:
        p = SUMMARY / name
        if not p.exists():
            continue
        def walk(o):
            if isinstance(o, dict):
                s = o.get("stem")
                if isinstance(s, str):
                    d = idx.setdefault(s, {})
                    for k, v in o.items():
                        if isinstance(v, (int, float)) and k not in d:
                            d[k] = v
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(json.loads(p.read_text(encoding="utf-8")))
    return idx


SCORED_FILES = [
    "200_selftest_table1_scored.json", "201_tau_sweep_raw_scored.json",
    "201_tau_archived_raw_scored.json", "200_bestofn_FINAL_scored.json",
    "200_bestofn_e21_FINAL_scored.json", "200_bestofn_e21_3seed.json",
    "200_vgr_FINAL_scored.json", "200_aprime_v2_scored.json",
    "w1_paper_ablation_table10.json", "e21_ablation_table10.json",
    "200_tableA_sampler.json", "purity_ablation_table10.json",
]


def paper_rows():
    """The audited per-cell record: one entry per purity-bearing cell of the paper."""
    return load("215_paper_numbers_v8.json")


def over_stems(idx, stems, key):
    vals = [idx[s][key] for s in stems if s in idx and key in idx[s]]
    if not vals:
        return None, None, 0
    m, s = mean_sd(vals)
    return m, s, len(vals)


def emit(lines, out_name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (OUT / out_name).write_text(text, encoding="utf-8")
    print(text)
    print(f"[out] {OUT / out_name}")
