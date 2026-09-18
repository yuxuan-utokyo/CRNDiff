# -*- coding: utf-8 -*-
"""The composition of an unconditional pool, and the analytic ceiling it implies. Diagnostic only.

For forced-top-N selection, the achievable purity is bounded by
    ceiling(type, budget) = min(1, budget * B1(type) * pool_fraction(type) / n_out),
which is what makes the outcome at a budget of one arithmetic rather than empirical on a rare
request.

The pool fraction is the classifier's share of that type, taken through the SAME annotation
path and the same frozen label set as purity, so the two are on one ruler. It is a diagnostic:
it never enters a selection.

    python scripts/score_pool_composition.py --help
"""
from __future__ import annotations

# the thread cap must be set BEFORE numpy is imported. On a machine with many logical cores
# an uncapped BLAS saturates the machine. The thread count changes parallelism only, never a label.
import os                                                            # noqa: E402

_T = os.environ.get("T200_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse                                                      # noqa: E402
import json                                                          # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.io import sha256_file                                       # noqa: E402

TYPES = ["Endothelial", "Myeloid", "Neuronal"]
BUDGETS = [1, 2, 4, 8, 16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--generator", default="OURS")
    ap.add_argument("--chunk", type=int, default=20000)
    a = ap.parse_args()

    from evaluation.score_purity import annotate, ruler_check            # noqa: PLC0415
    from crndiff.metrics import load_fixed_label_sets                      # noqa: PLC0415

    rl = ruler_check()
    label_sets = load_fixed_label_sets(C.MASTER_PURITY_V7)
    genes = json.loads((C.LEGACY_DATA / "all" / "meta.json")
                       .read_text(encoding="utf-8"))["genes"]
    D = json.loads((C.OUT / "summary" / "200_derived_constants.json")
                   .read_text(encoding="utf-8"))

    p = Path(a.pool)
    C.require(p, "pool")
    x = np.load(p, mmap_mode="r")
    n = int(x.shape[0])
    print(f"[pool] {p.name} n={n}")

    hist: dict[str, int] = {}
    t0 = time.time()
    for lo in range(0, n, a.chunk):
        hi = min(lo + a.chunk, n)
        lab = annotate(np.asarray(x[lo:hi], dtype=np.float64), genes)
        u, c = np.unique(lab, return_counts=True)
        for k, v in zip(u, c):
            hist[str(k)] = hist.get(str(k), 0) + int(v)
        print(f"  [{hi}/{n}] {time.time()-t0:.0f}s")

    out = {"ticket": 200, "block": "M1-2 diagnostic", "generator": a.generator,
           "seed": a.seed, "pool": str(p), "pool_sha256": sha256_file(p), "n_pool": n,
           "diagnostic_not_ruler": True,
           "note": "CellTypist composition of the pool. Diagnostic only: it never enters any "
                   "selection, guidance or decision (ticket §9).",
           "celltypist": rl, "label_dist_whole": hist, "types": {}}

    for t in TYPES:
        labs = label_sets[t]
        k = sum(v for kk, v in hist.items() if kk in labs)
        pi = k / n
        N = int(D["types"][t]["N"])
        B1 = int(D["types"][t]["B1"])
        ceil_b = {}
        for b in BUDGETS:
            m = b * B1
            ceil_b[str(b)] = (None if m > D["N_pool"]
                              else float(min(1.0, m * pi / N)))
        # right-censoring rule: a type whose pool share is below the resolvable threshold reports
        out["types"][t] = {"n_labelled_as_type": int(k), "pi_hat_pool": float(pi),
                           "N": N, "B1": B1, "ceil_by_budget": ceil_b,
                           "right_censored_threshold_pi": float(N / (16 * B1)),
                           "right_censored": bool(pi < N / (16 * B1)),
                           "label_set": sorted(labs)}
        print(f"[{t}] pi_hat={pi:.6f}  N={N} B1={B1}  ceil: " +
              "  ".join(f"b{b}={('cap' if ceil_b[str(b)] is None else f'{ceil_b[str(b)]:.4f}')}"
                        for b in BUDGETS))

    dest = C.OUT / "summary" / f"200_pool_composition_{a.generator}_seed{a.seed}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest = dest.with_name(dest.stem + f"_{int(time.time())}.json")
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
