"""Is the OURS-vs-Countsdiff gap actually significant?

Raised by the self-refutation pass: the FD_frechet gap is +3.69% (1.1144 vs 1.0747), i.e.
a difference of 0.0397, while s_train is 0.026 / 0.019. That is the same order as the
error bars, so "we are behind" may not survive a test.

Both arms are evaluated at K=1024 on the SAME 8 training seeds x 2 sampling seeds, so the
comparison can be PAIRED per (ckpt_seed, samp_seed) -- which removes the seed-to-seed
variance that dominates the unpaired error bars.

Reads each arm's published results.json. NEW FILE; writes only its own output.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

SCRNA = Path(__file__).resolve().parents[1]
METRICS4 = ["resid_corr_sqrt2", "mean_abs_pairwise_corr_err", "fano_rel_err_mean",
            "FD_frechet"]
PAIR = [("OURS", "1024"), ("Countsdiff", "1024")]


def cells(arm: str, nfe: str) -> dict[tuple[int, int], dict]:
    d = json.loads((SCRNA / arm / "results" / "results.json").read_text(encoding="utf-8"))
    out = {}
    for c in d["cells"][nfe]:
        out[(int(c["ckpt_seed"]), int(c["samp_seed"]))] = c["main4"]
    return out


def perm_test(d: np.ndarray, n_perm: int = 20000, seed: int = 20260807) -> float:
    """Two-sided exact/permutation sign-flip test on the paired differences."""
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    n = len(d)
    if n <= 20:                                   # exact over all 2^n sign flips
        cnt = tot = 0
        for r in range(n + 1):
            for idx in combinations(range(n), r):
                s = np.ones(n); s[list(idx)] = -1
                cnt += abs((d * s).mean()) >= obs - 1e-15
                tot += 1
        return cnt / tot
    hits = sum(abs((d * rng.choice([-1.0, 1.0], n)).mean()) >= obs - 1e-15
               for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


def main() -> int:
    a_name, a_nfe = PAIR[0]
    b_name, b_nfe = PAIR[1]
    A, B = cells(a_name, a_nfe), cells(b_name, b_nfe)
    keys = sorted(set(A) & set(B))
    print(f"{a_name}@{a_nfe} vs {b_name}@{b_nfe}: {len(keys)} paired cells "
          f"({len({k[0] for k in keys})} ckpt seeds x {len({k[1] for k in keys})} samp seeds)")
    res = {"arms": [f"{a_name}@{a_nfe}", f"{b_name}@{b_nfe}"], "n_paired": len(keys),
           "pairing": "per (ckpt_seed, samp_seed)", "metrics": {}}
    for m in METRICS4:
        va = np.array([A[k][m] for k in keys], float)
        vb = np.array([B[k][m] for k in keys], float)
        d = va - vb                                       # >0 means OURS worse
        # cluster the paired diffs by ckpt seed: seeds are the independent unit
        by_ck: dict[int, list[float]] = {}
        for (ck, _), x in zip(keys, d):
            by_ck.setdefault(ck, []).append(x)
        dck = np.array([np.mean(v) for v in by_ck.values()], float)
        p = perm_test(dck)
        se = dck.std(ddof=1) / np.sqrt(len(dck))
        res["metrics"][m] = {
            "ours_mean": float(va.mean()), "countsdiff_mean": float(vb.mean()),
            "paired_diff_mean": float(dck.mean()), "paired_diff_se": float(se),
            "t_like": float(dck.mean() / se) if se > 0 else float("inf"),
            "n_independent_units_ckpt_seeds": int(len(dck)),
            "p_two_sided_signflip": float(p),
            "significant_at_0.05": bool(p < 0.05),
            "gap_pct": float(100 * dck.mean() / abs(vb.mean())),
        }
        r = res["metrics"][m]
        print(f"  {m:<28s} OURS {va.mean():.4f}  CD {vb.mean():.4f}  "
              f"paired Δ {dck.mean():+.4f} ± {se:.4f}  t≈{r['t_like']:+.2f}  "
              f"p={p:.4f}  {'SIGNIFICANT' if p < 0.05 else 'not significant'}")
    (SCRNA / "_shared" / "p2f_significance.json").write_text(json.dumps(res, indent=1),
                                                             encoding="utf-8")
    print("\nwrote _shared/p2f_significance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
