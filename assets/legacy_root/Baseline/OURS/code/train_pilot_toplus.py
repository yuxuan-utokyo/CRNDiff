# -*- coding: utf-8 -*-
"""EXP179-F · **matched training** at $T_{O+}$: t is sampled from 0.9 log-uniform[F_BASE, T_O] + 0.1 point mass at t=8.0.

    python Baseline/OURS/code/train_pilot_toplus.py --seed 20260625 --steps 20000 --batch 384

## why this configuration (main line EXP179 correction section 2)

$T_{O+}$ reverse chain **queries the network only at $t=8$ and $t\\in[0,T_O]$, never in between on $(T_O, 8)$**.
while `pilot_Tm8` was trained with t drawn log-uniformly on $[0.01, 8]$, so a sizeable share of capacity went to
**a range that is never queried**. Running arm D with it would be a distracted model against a focused model,
losing would not prove anything either. This file gives $T_{O+}$ its own training configuration.

the ratio **0.9 / 0.1 is fixed in advance, not tuned**: only 1 of 257 steps is at t=8, but that step sets the starting point,
giving it 10% of the training budget is a conservative over-allocation. The fraction actually realized is given by `_STAT` counted and written to RUN_LOG.

## how this is done without changing a single byte of model_hvg2k.py

`train_ours_hvg2k` where the sampling of t is in a closure `noise` in it; cannot be overridden from outside.
so this file **does not copy those 150 lines by hand**, but instead:

    src = inspect.getsource(M.train_ours_hvg2k)      # take the source code of the original function
    src = src.replace(OLD_NOISE, NEW_NOISE)          # **one replacement that **asserts exactly one hit**
    exec(src, dict(M.__dict__) | {injected constant})        # compiled in the same globals as model_hvg2k

so this copy is **provably faithful**: apart from the asserted replacement, verbatim identical to the original function,
while `model_hvg2k.py` is only ever read. At run time the lines before and after replacement are printed for the record.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
import config_hvg2k as CFG                              # noqa: E402
import model_hvg2k as M                                 # noqa: E402

LOG = ROOT / "RUN_LOG.md"
TOPLUS_T, TOPLUS_P = 8.0, 0.1
_STAT = {"n_hi": 0, "n": 0}

OLD_NOISE = """    def noise(x0, gg):
        u = torch.rand(x0.shape[0], generator=gg).to(DEV)
        t = torch.exp(lf + u * (lt - lf))
        r = torch.exp(-t).unsqueeze(1)"""
NEW_NOISE = """    def noise(x0, gg):
        u = torch.rand(x0.shape[0], generator=gg).to(DEV)
        t = torch.exp(lf + u * (lt - lf))
        # ===== EXP179-F the only replacement: 0.9 log-uniform[F_BASE, T_O] + 0.1 point mass t=8.0 =====
        pick = torch.rand(x0.shape[0], generator=gg).to(DEV)
        hi = pick < TOPLUS_P
        t = torch.where(hi, torch.full_like(t, TOPLUS_T), t)
        _STAT["n_hi"] += int(hi.sum()); _STAT["n"] += int(x0.shape[0])
        # ==========================================================================
        r = torch.exp(-t).unsqueeze(1)"""


def rl(m):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"- `{time.strftime('%H:%M:%S')}` {m}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _build():
    src = inspect.getsource(M.train_ours_hvg2k)
    n = src.count(OLD_NOISE)
    assert n == 1, f"[ABORT] `noise` anchor matched {n} times; refusing to patch blindly"
    src2 = src.replace(OLD_NOISE, NEW_NOISE, 1)
    ns = dict(M.__dict__)
    ns.update(TOPLUS_T=TOPLUS_T, TOPLUS_P=TOPLUS_P, _STAT=_STAT)
    exec(compile(src2, "<train_ours_hvg2k::EXP179F>", "exec"), ns)
    return ns["train_ours_hvg2k"], src, src2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=CFG.SEED_PILOT)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--name", default="pilot_TOplus")
    a = ap.parse_args()

    ck = ROOT / "Baseline" / "OURS" / "models" / f"{a.name}_s{a.seed}.pt"
    for p in (ck, Path(str(ck) + ".best"), Path(str(ck) + ".last"),
              ROOT / "Baseline" / "OURS" / "models" / f"{a.name}_s{a.seed}.json"):
        if p.exists():
            raise SystemExit(f"[ABORT] name clash, refusing to overwrite:{p.name}")

    train_fn, src, src2 = _build()
    d, meta = M.load_data(("train", "val"))
    tr, va = d["train"], d["val"]
    der = M.derive(tr)
    T = M.horizon()

    rl(f"### EXP179-F training starts · {a.name} train{list(tr.shape)} val{list(va.shape)} "
       f"batch={a.batch} steps={a.steps} seed={a.seed}")
    rl(f"  **t sampling distribution = {1-TOPLUS_P:.1f} log-uniform[{CFG.F_BASE}, {T:.4f}] + "
       f"{TOPLUS_P:.1f} point mass t={TOPLUS_T}** (fixed in advance, not tuned);"
       f"e^-{TOPLUS_T}·NMAX={np.exp(-TOPLUS_T)*CFG.NMAX:.4f},"
       f"e^-T_O·NMAX={np.exp(-T)*CFG.NMAX:.4f}")
    rl(f"  copy method:`inspect.getsource(train_ours_hvg2k)` + **one asserted replacement**,"
       f"`model_hvg2k.py` read only, unchanged (original {len(src)} B -> new {len(src2)} B)")

    t0 = time.time()
    net, curve, info = train_fn(tr, va, der["V"], der["iscale"], a.seed, a.steps,
                                batch=a.batch, log_fn=lambda x: print(x, flush=True),
                                t_max=T, ckpt_path=ck)
    dt = time.time() - t0
    if ck.exists():
        raise SystemExit(f"[ABORT] name clash, refusing to overwrite:{ck.name}")
    torch.save(net.state_dict(), ck)
    for k in ("train", "val", "step"):
        np.save(ROOT / "Baseline" / "OURS" / "loss" / f"{a.name}_s{a.seed}_{k}.npy",
                np.asarray(curve[k]))
    frac = _STAT["n_hi"] / max(_STAT["n"], 1)
    (ROOT / "Baseline" / "OURS" / "models" / f"{a.name}_s{a.seed}.json").write_text(
        json.dumps({"tag": a.name, "seed": a.seed, "walltime_s": dt,
                    "dataset": CFG.DATASET, "gene_hash": CFG.GENE_HASH,
                    "t_dist": {"log_uniform_range": [CFG.F_BASE, T],
                               "point_mass_t": TOPLUS_T, "point_mass_p": TOPLUS_P,
                               "realised_fraction": frac,
                               "n_draws": _STAT["n"]},
                    "s_per_step": dt / max(a.steps, 1), **info,
                    "curve": {k: [float(x) for x in v] for k, v in curve.items()}},
                   ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    rl(f"  EXP179-F training done {dt/60:.1f} min ({dt/a.steps:.3f} s/step),"
       f"best_val={info['best_val']:.5f},n_params={info['n_params']:,};"
       f"**t=8.0 actual fraction {frac:.5f}** (configured {TOPLUS_P},"
       f"{_STAT['n_hi']:,}/{_STAT['n']:,} draws)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
