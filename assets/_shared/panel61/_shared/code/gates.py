# moved from cc/experiments/EXP134_more_ckpts/run_exp134.py::save / skip (atomic writes and [SKIP] records).
# changes (item by item):
#   1. save()/skip() originally a closure over module-level RES/LOGF, here path / logger are passed explicitly; semantics unchanged
#      (still write .tmp then os.replace).
#   2. added shape_gate_gen() (shape gate of spec section 9: shape==(N,61), integer dtype, min>=0),
#      EXP134 shape_gate is meant for the time grid, not for generated outputs, so this is new rather than moved.
#   3. added note() (unified logging of markers such as [SCALE-CUT]/[CAP]/[EARLY-SUSPECT]/[P0-DEFER]).
"""shape gate, atomic writes, [SKIP]/[SCALE-CUT] records."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def save_atomic(obj, path):
    """write .tmp then os.replace: same meaning as EXP134::save."""
    path = str(path)
    t = path + ".tmp"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(t, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=float)
    os.replace(t, path)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    return json.load(open(path, encoding="utf-8-sig"))


def skip(res, cell, why, log=print, path=None):
    """EXP134::skip — log and continue; never stop the whole run."""
    log(f"[SKIP] {cell} {why}")
    res.setdefault("skipped", []).append({"cell": str(cell), "reason": str(why)[:300]})
    if path:
        save_atomic(res, path)
    return False


def note(res, tag, msg, log=print, path=None):
    """register [SCALE-CUT] / [CAP] / [EARLY-SUSPECT] / [P0-DEFER] / [FALLBACK-IMPORT] /
    [DEGRADED-EARLYSTOP] / [MIGRATION-FAIL] a class of markers."""
    line = f"[{tag}] {msg}"
    log(line)
    res.setdefault("notes", []).append({"tag": tag, "msg": str(msg)})
    if path:
        save_atomic(res, path)
    return line


def shape_gate_gen(gen, n_expect, dim_expect=61):
    """spec section 9 shape gate: shape == (n_expect, dim), integer dtype, min >= 0.
    returns (ok, problem description). On failure the caller records [SKIP]."""
    prob = []
    a = np.asarray(gen)
    if a.shape != (n_expect, dim_expect):
        prob.append(f"shape={a.shape} != ({n_expect}, {dim_expect})")
    if not np.issubdtype(a.dtype, np.integer):
        prob.append(f"dtype={a.dtype} non-integer")
    if a.size and int(a.min()) < 0:
        prob.append(f"min={int(a.min())} < 0")
    return (len(prob) == 0), "; ".join(prob)
