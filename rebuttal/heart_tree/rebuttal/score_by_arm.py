# -*- coding: utf-8 -*-
"""Frozen scoring (tools.t200_score_deliveries, rulers untouched) for experiments where several arms share one file stem.
The scorer keys its rows by stem, so such arms overwrite each other. Here the arm name is prefixed to the stem in memory.
    python -m rebuttal.score_by_arm --experiment rebuttal_R9 --out-tag rebuttal_R9_Neuronal
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools import t200_score_deliveries as T
_load = T.load_runs
def load_runs(exp):
    out = []
    for r in _load(exp):
        # staged baseline runs (CountsDiff, Count-FM) carry no "stem"; use the run file's own stem
        r["stem"] = f'{r["arm"]}__{r.get("stem") or Path(r["_path"]).stem}'; out.append(r)
    if len({r["stem"] for r in out}) != len(out):
        raise SystemExit("stems still collide")
    return out
T.load_runs = load_runs
raise SystemExit(T.main())
