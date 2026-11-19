# new (thin wrapper, no numerical computation): the evaluate.py of every arm shares the same entry point,
# ensures there is only one implementation of the metrics. Change: new file; all values come from _shared/metrics.py::allmetrics.
"""general evaluation entry point: read gens npy → _shared/metrics.py::allmetrics → json."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import metrics as M
from .ctx import build_eval_ctx
from .data import load_panel
from .gates import shape_gate_gen
from .seeds import METRICS4, N_GEN

_CTX_CACHE = {}


def get_ctx():
    """ctx / REF / gamma depends only on train+val; built once, cached and reused (values bit-identical to rebuilding each time)."""
    if "v" not in _CTX_CACHE:
        p = load_panel()
        ctx, REF, gamma, diag = build_eval_ctx(p.train, p.val)
        _CTX_CACHE["v"] = (p, ctx, REF, gamma, diag)
    return _CTX_CACHE["v"]


def evaluate_array(gen):
    """gen may be integer counts or continuous floats (DDPM); integrality is checked before rounding."""
    p, ctx, REF, gamma, _ = get_ctx()
    return M.allmetrics(gen, p.val, ctx, REF, gamma)


def resolve_gen_path(path):
    """if a continuous version with the same name exists (gens_cont/), prefer it; integrality must be checked before rounding
    (EXP23::eval_gen checks integrality on raw floats, then rounds internally for count metrics).
    only DDPM writes gens_cont/; the other arms generate integers anyway, and this function is the identity for them."""
    p = Path(path)
    if p.parent.name == "gens":
        cont = p.parent.parent / "gens_cont" / p.name
        if cont.exists():
            return cont, True
    return p, False


def evaluate_file(path, n_expect=N_GEN, check_shape=True):
    src, used_cont = resolve_gen_path(path)
    gen = np.load(src)
    out = {"gen_file": str(path), "eval_source": str(src), "used_continuous": bool(used_cont),
           "shape": list(gen.shape), "dtype": str(gen.dtype)}
    if check_shape:
        probe = np.clip(np.round(gen), 0, None).astype(np.int64)
        ok, why = shape_gate_gen(probe, n_expect, probe.shape[1] if probe.ndim == 2 else -1)
        out["shape_gate_pass"] = bool(ok)
        out["shape_gate_problem"] = why
    t0 = time.time()
    out["metrics"] = evaluate_array(gen)
    out["eval_seconds"] = time.time() - t0
    out["main4"] = {k: out["metrics"].get(k) for k in METRICS4}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="EXP136 unified evaluation entry point")
    ap.add_argument("--gen", required=True, help="path of the generated npy")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--n-expect", type=int, default=N_GEN)
    ap.add_argument("--no-shape-gate", action="store_true")
    a = ap.parse_args(argv)
    out = evaluate_file(a.gen, n_expect=a.n_expect, check_shape=not a.no_shape_gate)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(out["main4"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
