# -*- coding: utf-8 -*-
"""hvg2k · OURS pilot training + **logit adjustment (LA)**. T3 layer.

    python Baseline/OURS/code/train_pilot_la.py --tau 0 --steps 2500 --name t3base
    python Baseline/OURS/code/train_pilot_la.py --tau 1 --steps 2500 --name t3la1

## what this is

    L = -log softmax( f(n_t)_{d,:} + tau * log pi_d )_{x0_d}

`pi_d` = **the own count histogram of gene d** (counted from train,`[dim, NMAX+1]`),
no joint distribution, no resampling and no debiasing needed. 1 gene, 10 or 2000 are written exactly the same way
(spec M4: zero migration friction).`tau=0` is the status quo, used as the **same-budget control arm**.

inference uses **the same expression**:`p_hat = softmax(f + tau*log pi)`. So this is
an exact reparameterization, not an approximation; see `sample_hvg2k_la.py`.

## why it should be more useful in 2000 dimensions than in 1D (verified in T2, see exp08)

the main-line head is **shared across tokens** `Linear(d_model, NMAX+1)`, per-gene prior
only via `gene_emb` squeezed in. The priors of 2000 genes span ten orders of magnitude
(`gene_mean` from 5.5e-6 to 12.15, zerofrac 0.941); having one shared head represent it is pure waste.
LA add the prior **analytically**; the network only learns the deviation from the prior.

## how this is done without changing `model_hvg2k.py` a single byte]

following `train_pilot_toplus.py` patterns already used:

    src = inspect.getsource(M.train_ours_hvg2k)   # take the source code of the original function
    src = src.replace(OLD, NEW)                   # **two replacements that **assert exactly one hit**
    exec(src, dict(M.__dict__) | {injected constant})     # compiled in the same globals as model_hvg2k

two replacements: the CE of the training step and the CE of the val evaluation step (val also needs LA, otherwise the val curve
measures something other than the objective this arm optimizes). Apart from these two places verbatim identical to the original function,`model_hvg2k.py`
is only ever read. At run time the replaced lines are printed for the record.

## pi smoothing

`pi_d(n) = (count_d(n) + EPS) / (N + EPS*(NMAX+1))`,`EPS=0.5`.
without smoothing, unseen counts have log pi = -inf and that category can **never** be generated:
that is harder than the A4 cut (A4 only removes n > dmax_d; unseen counts below dmax remain).
EPS=0.5 the floor is 0.5/361205 = 1.4e-6, combined with the A4 cut at sampling time (`n > dmax_d` set to zero)
no spurious mass leaks out.

## red line

`model_hvg2k.py` / `sample_hvg2k.py` / `vendor_ours*.py` / `bank_dedup.py`
not a single byte changed. Output names include `--name`, does not clash with existing pilot_* names.
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
PI_EPS = 0.5

OLD_TRAIN = """        with amp:
            logits = model(n_t, t)
            loss = F.cross_entropy(logits.reshape(-1, CFG.NMAX + 1),
                                   x0.long().clamp(0, CFG.NMAX).reshape(-1))"""
NEW_TRAIN = """        with amp:
            logits = model(n_t, t) + _LA_TAU * _LA_LOGPI.unsqueeze(0)
            loss = F.cross_entropy(logits.reshape(-1, CFG.NMAX + 1),
                                   x0.long().clamp(0, CFG.NMAX).reshape(-1))"""

OLD_VAL = """                    s_ += float(F.cross_entropy(
                        model(nb, tb_).reshape(-1, CFG.NMAX + 1),
                        xb.long().clamp(0, CFG.NMAX).reshape(-1), reduction="sum"))"""
NEW_VAL = """                    s_ += float(F.cross_entropy(
                        (model(nb, tb_)
                         + _LA_TAU * _LA_LOGPI.unsqueeze(0)).reshape(-1, CFG.NMAX + 1),
                        xb.long().clamp(0, CFG.NMAX).reshape(-1), reduction="sum"))"""


def rl(m):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"- `{time.strftime('%H:%M:%S')}` {m}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_logpi(train, nmax, eps=PI_EPS):
    """pi_d(n), [dim, nmax+1] float32 (smoothed, rows normalized). **Counted from train only**."""
    X = np.clip(train, 0, nmax).astype(np.int64)
    n, dim = X.shape
    cnt = np.zeros((dim, nmax + 1), np.float64)
    for d in range(dim):
        cnt[d] = np.bincount(X[:, d], minlength=nmax + 1)
    pi = (cnt + eps) / (n + eps * (nmax + 1))
    pi /= pi.sum(1, keepdims=True)
    return np.log(pi), pi, cnt


def make_trainer(tau, logpi_t):
    """put `train_ours_hvg2k` take the source code, make two **asserted** replacements, then compile."""
    src = inspect.getsource(M.train_ours_hvg2k)
    assert src.count(OLD_TRAIN) == 1, "the training CE replacement did not match exactly once: upstream changed, stop"
    assert src.count(OLD_VAL) == 1, "val CE that replacement did not match exactly once: upstream changed, stop"
    src = src.replace(OLD_TRAIN, NEW_TRAIN).replace(OLD_VAL, NEW_VAL)
    print("  [PATCH] training CE:\n" + "\n".join("    " + l for l in NEW_TRAIN.splitlines()))
    print("  [PATCH] val  CE:\n" + "\n".join("    " + l for l in NEW_VAL.splitlines()))
    g = dict(M.__dict__)
    g["_LA_TAU"] = float(tau)
    g["_LA_LOGPI"] = logpi_t
    ns = {}
    exec(compile(src, "<train_ours_hvg2k+LA>", "exec"), g, ns)
    return ns["train_ours_hvg2k"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True,
                    help="LA strength. 0 = status-quo control arm (but through the same code path)")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=CFG.SEED_PILOT)
    ap.add_argument("--name", required=True, help="output name, e.g. t3base / t3la1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    steps = 60 if a.smoke else a.steps
    batch = a.batch or CFG.BATCH

    d, meta = M.load_data(("train", "val"))
    tr, va = d["train"], d["val"]
    der = M.derive(tr)
    T = M.horizon()
    tag = a.name

    t0 = time.time()
    lp_dir = ROOT / "Baseline" / "OURS" / "models"
    lp_dir.mkdir(parents=True, exist_ok=True)
    cache = lp_dir / f"logpi_nmax{CFG.NMAX}_eps{PI_EPS}.npy"
    if cache.exists():
        logpi = np.load(cache)
        pi = np.exp(logpi)
    else:
        logpi, pi, cnt = build_logpi(tr, CFG.NMAX)
        np.save(cache, logpi)
    logpi_t = torch.as_tensor(logpi, dtype=torch.float32, device=M.DEV)
    rl(f"### hvg2k OURS **T3/{tag}** tau={a.tau} steps={steps} batch={batch} "
       f"seed={a.seed}  ({time.time()-t0:.1f}s build log_pi)")
    rl(f"  log_pi [{logpi.shape[0]},{logpi.shape[1]}]  "
       f"min={logpi.min():.3f} max={logpi.max():.3f}  "
       f"pi(0) median={np.median(pi[:, 0]):.5f}  "
       f"median effective number of categories of per-gene pi="
       f"{np.median(np.exp(-(pi*np.log(np.maximum(pi,1e-300))).sum(1))):.3f}")
    rl(f"  train{list(tr.shape)} val{list(va.shape)}  T_MAX={T:.4f}  NMAX={CFG.NMAX}  "
       f"sum_V/lib={der['sum_V_over_lib']:.4f}")

    trainer = make_trainer(a.tau, logpi_t)
    t0 = time.time()
    try:
        net, curve, info = trainer(
            tr, va, der["V"], der["iscale"], a.seed, steps,
            batch=batch, log_fn=lambda x: print(x, flush=True), t_max=T,
            ckpt_path=lp_dir / f"{tag}_s{a.seed}.pt")
    except torch.cuda.OutOfMemoryError as e:
        rl(f"  [SCALE-CUT] batch={batch} out of memory ({repr(e)[:100]})")
        return 2
    dt = time.time() - t0
    torch.save(net.state_dict(), lp_dir / f"{tag}_s{a.seed}.pt")
    (ROOT / "Baseline" / "OURS" / "loss").mkdir(parents=True, exist_ok=True)
    for k in ("train", "val", "step"):
        np.save(ROOT / "Baseline" / "OURS" / "loss" / f"{tag}_s{a.seed}_{k}.npy",
                np.asarray(curve[k]))
    blob = {"tag": tag, "LA_tau": a.tau, "PI_EPS": PI_EPS, "seed": a.seed,
            "walltime_s": dt, "steps_requested": steps, "batch": batch,
            "dataset": CFG.DATASET, "gene_hash": CFG.GENE_HASH, "T": T,
            "logpi_cache": cache.name,
            "note": ("LA the val of the arm and the val of the tau=0 arm **cannot be compared directly**: "
                     "the two optimize different objectives (CE with tau*log pi added)."
                     "comparisons between arms always use readings on generated outputs, not val."),
            "s_per_step": dt / max(steps, 1),
            **{k: der[k] for k in ("n_distinct_V", "sum_V", "lib_mean", "sum_V_over_lib")},
            **info, "curve": {k: [float(x) for x in v] for k, v in curve.items()}}
    (lp_dir / f"{tag}_s{a.seed}.json").write_text(
        json.dumps(blob, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    rl(f"  T3/{tag} done {dt/60:.1f} min ({dt/max(steps,1):.3f} s/step),"
       f"best_val={info['best_val']:.5f},train {curve['train'][0]:.4f} -> "
       f"{curve['train'][-1]:.4f}"
       + (f", peak memory {torch.cuda.max_memory_allocated()/2**30:.1f} GB"
          if torch.cuda.is_available() else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
