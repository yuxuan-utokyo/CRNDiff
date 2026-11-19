"""5b driver: k-step inference for the published uniform-categorical D3PM. NO TRAINING.

Uses the 8 existing D3PM checkpoints (T_train = 250) unchanged, samples them at every K on
the ladder, and scores with the panel's canonical evaluator. Outputs go to NEW directories;
`D3PM/gens/`, `D3PM/results/` and `D3PM/ckpts/` are not written to at all.

Why this exists: on the NFE frontier the panel's D3PM is a SINGLE point (its T_steps is
both a training and a sampling hyper-parameter, so the published row only ever sampled at
250). The low-budget end -- K = 8/16/32, exactly where we claim an advantage -- has never
been compared against it, while the text says we lead other discrete diffusions.

K > T_train is SKIPPED, not sampled. `linspace(250, 0, K+1).round()` cannot yield more than
251 distinct levels, so K = 512 (262 repeats) and K = 1024 (774 repeats) would spend NFE on
steps whose target noise level equals their current one. Reporting those as ordinary points
would place D3PM on the frontier at a budget it never used. The skip is recorded per cell
with the degenerate-step count, not dropped silently.

Gate (already passed before this ran): at K = T_train the strided path reproduces the
published sampler bit-for-bit -- 3 checkpoints, 122,000 elements each, n_mismatch = 0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(ROOT / "D3PM"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from _shared.data import load_panel                     # noqa: E402
from _shared.seeds import N_GEN                         # noqa: E402
from _shared.evaluate_common import evaluate_array      # noqa: E402
from _shared.d3pm_kstep import sample_d3pm_kstep, plan_steps   # noqa: E402
from model import load_ckpt                             # noqa: E402

OUT = HERE / "d3pm_kstep_cells"
GENS = HERE / "d3pm_kstep_gens"
CKDIR = ROOT / "D3PM" / "ckpts"                         # READ ONLY
K_LADDER = [8, 16, 32, 64, 128, 256, 512, 1024]
SAMP_SEEDS = [20260901, 20260902]

_log: list[str] = []


def log(m: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    _log.append(line)
    (HERE / "d3pm_kstep_run.log").write_text("\n".join(_log) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default=",".join(str(k) for k in K_LADDER))
    ap.add_argument("--n-gen", type=int, default=N_GEN)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    GENS.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in a.ks.split(",")]
    p = load_panel()
    cks = sorted(CKDIR.glob("s*.pt"))
    log(f"=== 5b D3PM k-step | {len(cks)} ckpts (NOT retrained) | K={ks} | "
        f"n_gen={a.n_gen} | dim={p.dim} ===")

    n_skip = 0
    for cp in cks:
        net, abar, T, blob = load_ckpt(cp, p.dim, p.iscale)
        seed = int(cp.stem[1:])
        for K in ks:
            pl = plan_steps(T, K)
            for ss in SAMP_SEEDS:
                nm = (f"D3PM_uniform__panel61__K{K:04d}__ck{seed}__sp{ss}"
                      f"__n{a.n_gen}__Ttrain{T}__kstep")
                f = OUT / f"{nm}.json"
                if f.exists():
                    continue
                if pl["exceeds_train_chain"]:
                    f.write_text(json.dumps({
                        "experiment": "EXP140-5b", "arm": "D3PM (uniform)",
                        "panel": "61gene", "K": K, "ckpt_seed": seed, "samp_seed": ss,
                        "T_train": T, "n_gen": a.n_gen, "status": "SKIP",
                        "skip_reason": f"K={K} > T_train={T}: the subsequence has only "
                                       f"{pl['n_distinct_levels']} distinct levels, so "
                                       f"{pl['n_degenerate_steps']}/{K} steps have "
                                       f"target noise level == current one",
                        "n_distinct_levels": pl["n_distinct_levels"],
                        "n_degenerate_steps": pl["n_degenerate_steps"],
                        "nfe_nominal": K, "nfe_effective": pl["nfe_effective"],
                    }, indent=1), encoding="utf-8")
                    n_skip += 1
                    continue
                t0 = time.time()
                gen = sample_d3pm_kstep(net, abar, T, p.dim, a.n_gen, ss, K=K)
                wt = time.time() - t0
                np.save(GENS / f"{nm}.npy", gen.astype(np.int16))
                ev = evaluate_array(gen)
                rec = {"experiment": "EXP140-5b", "arm": "D3PM (uniform)",
                       "panel": "61gene", "K": K, "nfe": K, "ckpt_seed": seed,
                       "samp_seed": ss, "rep": 1, "T_train": T, "n_gen": a.n_gen,
                       "status": "OK", "retrained": False,
                       "ckpt_source": f"D3PM/ckpts/{cp.name}",
                       "grid": "linspace (uniform in t), round to integer levels",
                       "n_distinct_levels": pl["n_distinct_levels"],
                       "n_degenerate_steps": pl["n_degenerate_steps"],
                       "nfe_effective": pl["nfe_effective"],
                       "gen_max": int(gen.max()),
                       "sample_sec": wt, "gens_file": f"d3pm_kstep_gens/{nm}.npy"}
                rec.update(ev if isinstance(ev, dict) else {"metrics": ev})
                f.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
            log(f"  [ck{seed} K={K:>4d}] {'SKIP (K>T_train)' if pl['exceeds_train_chain'] else 'done'}")
    log(f"=== 5b done: {len(list(OUT.glob('*.json')))} cells, {n_skip} SKIP ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
