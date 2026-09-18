# -*- coding: utf-8 -*-
"""W1 / sliced-W1 against real cells, plus the endpoint log rho distribution.
Purpose: three readings for a delivered run.

1. Per-gene W1: the one-dimensional Wasserstein-1 gene by gene, then mean, median and quantiles.
2. Sliced W1 on the joint distribution, along fixed seeded random directions.
3. Quantiles of the endpoint log rho, which say how many orders of magnitude the weights span.

The sample size must be matched. This was learned the hard way: the residual correction is
strongly biased in n, and W1 is n-sensitive too, since a small empirical distribution is
narrower by construction. So the generated set is cut into non-overlapping blocks the size of
the real set, scored within a block, averaged, and the block spread reported alongside.

The log rho reading comes with a limitation that must be stated with it:
      a run stores only the cells AFTER delivery, that is after the terminal resampling. It does
      not store the particle cloud before resampling, nor the endpoint weights. So what is computed
      here is the log rho distribution of the SURVIVORS, which is a LOWER BOUND on the true spread,
      because resampling is exactly the operation that removes the low-weight tail.
      As anchors, the same discriminator's log rho on the real validation cells and on the untilted
      base pool is computed as well, so the delivery can be placed between them.
      Separately, ESS/M alone gives a distribution-free lower bound on the spread.

    python -m analysis.score_distribution --experiment main_single_type
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.io import load_runs                                         # noqa: E402
from crndiff.sampler import PortableHGBT, classifier_log_reward          # noqa: E402

PROJ_SEED = 20260906          # sliced-W1 directions; fixed so every arm sees the same slices
N_PROJ = 512
QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0)

# ★ W1 is n-biased: a smaller empirical sample sits closer to itself, so a W1 measured at
# n=396 and one measured at n=10057 are NOT comparable numbers. The real val counts differ a
# lot by target (Myeloid 2302 / Endothelial 10057 / Neuronal 396), so this file reports TWO
# calibres and labels which is which:
#   * "full"   n = every real val cell of that target. Best precision, and the right calibre
#              for comparing alpha WITHIN one target (n is identical on both sides).
#              Endothelial only fits ONE block of 10057 in 20000, so it gets no block spread.
#   * "common" n = COMMON_N for all three targets, so rows can be compared ACROSS targets.
# Never put a "full" number for one target next to a "full" number for another.
COMMON_N = 396                # set by Neuronal, the smallest real val population
COMMON_REPS = 20              # paired (generated block, real subsample) repetitions
COMMON_SEED = 20260906


def cp10k_log1p(x: np.ndarray) -> np.ndarray:
    """The standard scRNA scale, and the one CellTypist itself is fed."""
    z = np.asarray(x, dtype=np.float64)
    lib = z.sum(axis=1, keepdims=True)
    return np.log1p(z * (1.0e4 / np.maximum(lib, 1.0)))


def w1_equal_size(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """W1 between two equal-size empirical samples, per column: mean |sorted diff|.
    Exact for equal sample sizes -- no binning, no interpolation."""
    assert a.shape == b.shape, (a.shape, b.shape)
    return np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).mean(axis=0)


def sliced_w1(a: np.ndarray, b: np.ndarray, n_proj: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    v = rng.normal(size=(d, n_proj))
    v /= np.linalg.norm(v, axis=0, keepdims=True)
    pa, pb = a @ v, b @ v
    per = w1_equal_size(pa, pb)
    return {"mean": float(per.mean()), "median": float(np.median(per)),
            "p90": float(np.quantile(per, 0.9)), "max": float(per.max()),
            "n_proj": n_proj, "seed": seed}


def w1_at_common_n(gen: np.ndarray, real: np.ndarray, n: int, reps: int, seed: int,
                   n_proj: int) -> dict:
    """W1 at a COMMON n so different targets can be compared. Both sides are subsampled to n:
    generated takes disjoint blocks, real takes independent seeded draws."""
    rng = np.random.default_rng(seed)
    n_blocks = min(reps, gen.shape[0] // n)
    if n_blocks < 1:
        return {"error": f"generated set has {gen.shape[0]} cells, needs >= {n}"}
    pg, sw = [], []
    for b in range(n_blocks):
        g = cp10k_log1p(np.asarray(gen[b * n:(b + 1) * n], dtype=np.float64))
        idx = np.sort(rng.choice(real.shape[0], size=n, replace=False))
        r = cp10k_log1p(real[idx])
        pg.append(w1_equal_size(g, r).mean())
        sw.append(sliced_w1(g, r, n_proj, PROJ_SEED)["mean"])
    return {"n": int(n), "reps": int(n_blocks),
            "per_gene_w1_mean": float(np.mean(pg)), "per_gene_w1_spread": float(np.std(pg)),
            "sliced_w1_mean": float(np.mean(sw)), "sliced_w1_spread": float(np.std(sw)),
            "note": "COMMON n across targets -- this is the calibre to compare targets on"}


def quantiles(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    out = {f"q{q:g}": float(np.quantile(v, q)) for q in QUANTILES}
    out.update({"mean": float(v.mean()), "sd": float(v.std()), "n": int(v.size)})
    out["span_p1_p99_nats"] = out["q0.99"] - out["q0.01"]
    out["span_p1_p99_decades"] = out["span_p1_p99_nats"] / math.log(10.0)
    out["span_full_nats"] = out["q1"] - out["q0"]
    out["span_full_decades"] = out["span_full_nats"] / math.log(10.0)
    return out


def ess_to_weight_spread(ess_fraction: float, m: int) -> dict:
    """A distribution-free reading of the same question: ESS/M = 1/(M * sum w_i^2) for
    normalised w. ESS/M = f means the cloud behaves like ~f*M equally-weighted particles."""
    eff = ess_fraction * m
    return {"ess_fraction": ess_fraction, "n_particles": m,
            "effective_particles": eff,
            "note": f"at the worst step the cloud behaved like ~{eff:.1f} equally-weighted "
                    f"particles out of {m}",
            "implied_concentration_decades": math.log10(m / max(eff, 1e-12))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="main_single_type")
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-proj", type=int, default=N_PROJ)
    a = ap.parse_args(argv)

    D = C.LEGACY_DATA / "all"
    Xv = np.load(D / "val.npy", mmap_mode="r")
    yv = np.load(D / "val_celltype.npy", allow_pickle=True).astype(str)
    base = np.load(C.BASE_POOL, mmap_mode="r")

    if a.run:
        r = json.loads(Path(a.run).read_text(encoding="utf-8")); r["_path"] = a.run
        runs = [r]
    else:
        runs = load_runs(a.experiment)
    if not runs:
        raise SystemExit(f"no runs under out/runs/{a.experiment}")

    disc_cache: dict = {}
    rows = []
    for run in sorted(runs, key=lambda r: r["stem"]):
        target = run["request"]
        real = np.asarray(Xv[np.flatnonzero(yv == target)], dtype=np.float64)
        n_match = real.shape[0]
        gen = np.load(Path(run["_path"]).with_suffix(".npy"), mmap_mode="r")
        n_blocks = gen.shape[0] // n_match
        if n_blocks < 1:
            raise SystemExit(f"{run['stem']}: {gen.shape[0]} cells < matched n {n_match}")
        print(f"[dist] {run['stem'][:56]}  target={target} matched_n={n_match} "
              f"blocks={n_blocks}")

        real_l = cp10k_log1p(real)
        pg_log, pg_raw, sw = [], [], []
        for b in range(n_blocks):
            blk = np.asarray(gen[b * n_match:(b + 1) * n_match], dtype=np.float64)
            pg_log.append(w1_equal_size(cp10k_log1p(blk), real_l))
            pg_raw.append(w1_equal_size(blk, real))
            sw.append(sliced_w1(cp10k_log1p(blk), real_l, a.n_proj, PROJ_SEED))
        pg_log = np.stack(pg_log)           # [blocks, genes]
        pg_raw = np.stack(pg_raw)

        # ---- endpoint log rho, on the SURVIVING (delivered) cells ----
        rd = run.get("residual_dir") or target
        if rd not in disc_cache:
            p = C.residual_paths(rd)
            clf = PortableHGBT(p["portable"])
            feats = np.load(p["split"], allow_pickle=False)["classifier_features"]
            disc_cache[rd] = (clf, feats)
        clf, feats = disc_cache[rd]
        lr_gen = classifier_log_reward(clf, feats, np.asarray(gen, dtype=np.float64),
                                       C.DEFAULT_REWARD_FLOOR, "odds")
        lr_real = classifier_log_reward(clf, feats, real, C.DEFAULT_REWARD_FLOOR, "odds")
        lr_base = classifier_log_reward(clf, feats, np.asarray(base, dtype=np.float64),
                                        C.DEFAULT_REWARD_FLOOR, "odds")
        alpha = run.get("alpha")

        rows.append({
            "stem": run["stem"], "arm": run.get("arm"), "target": target, "alpha": alpha,
            "matched_n": int(n_match), "n_blocks": int(n_blocks),
            "n_generated": int(gen.shape[0]),
            "reference": f"real val cells of {target} (n={n_match}), all of them",
            "per_gene_w1_log1p_cp10k": {
                "mean_over_genes": float(pg_log.mean(1).mean()),
                "block_spread_of_mean": float(pg_log.mean(1).std()),
                "median_over_genes": float(np.median(pg_log.mean(0))),
                "p90_over_genes": float(np.quantile(pg_log.mean(0), 0.9)),
                "max_over_genes": float(pg_log.mean(0).max()),
                "per_block_mean": [float(v) for v in pg_log.mean(1)]},
            "per_gene_w1_raw_counts": {
                "mean_over_genes": float(pg_raw.mean(1).mean()),
                "block_spread_of_mean": float(pg_raw.mean(1).std()),
                "median_over_genes": float(np.median(pg_raw.mean(0))),
                "max_over_genes": float(pg_raw.mean(0).max())},
            "w1_common_n_cross_target_comparable": w1_at_common_n(
                gen, real, COMMON_N, COMMON_REPS, COMMON_SEED, a.n_proj),
            "calibre_warning": (
                f"the 'full' numbers below are at n={n_match} (all real val cells of {target}); "
                "they are comparable BETWEEN alphas of this target and NOT between targets. "
                f"Use w1_common_n_cross_target_comparable (n={COMMON_N}) across targets."),
            "sliced_w1_log1p_cp10k": {
                "mean_over_blocks": float(np.mean([s["mean"] for s in sw])),
                "block_spread": float(np.std([s["mean"] for s in sw])),
                "per_block": sw},
            "log_rho_endpoint": {
                "caliber": "log rho = log r - log(1-r), r clipped to [1e-8, 1-1e-8], "
                           "evaluated by the arm's own residual discriminator",
                "population": "DELIVERED cells (after the terminal resample)",
                "lower_bound_warning":
                    "this is the SURVIVING reward distribution, a LOWER BOUND on the spread the "
                    "weights actually had: the terminal resample is precisely the operation that "
                    "discards the low-weight tail. The pre-resample cloud (M=40000) and the "
                    "terminal weights were not saved, so the true span is not recoverable from "
                    "this artefact.",
                "delivered": quantiles(lr_gen),
                "alpha_times_log_rho_delivered": quantiles(alpha * lr_gen) if alpha else None,
                "real_val_cells": quantiles(lr_real),
                "untilted_base_pool": quantiles(lr_base)},
            "ess_reading": {
                "min": ess_to_weight_spread(run["ess_min_fraction"], run["n_particles"]),
                "pre_final": ess_to_weight_spread(run["ess_pre_final_fraction"],
                                                  run["n_particles"])},
            "unique_lineage_fraction": run.get("unique_lineage_fraction"),
            "unique_cell_fraction": run.get("unique_cell_fraction"),
            "n_intermediate_resamples": run.get("n_intermediate_resamples"),
        })
        r0 = rows[-1]
        print(f"    per-gene W1 (log1p cp10k) {r0['per_gene_w1_log1p_cp10k']['mean_over_genes']:.4f}"
              f" +- {r0['per_gene_w1_log1p_cp10k']['block_spread_of_mean']:.4f}"
              f" | sliced-W1 {r0['sliced_w1_log1p_cp10k']['mean_over_blocks']:.4f}")
        q = r0["log_rho_endpoint"]["delivered"]
        print(f"    log rho delivered: p1 {q['q0.01']:.2f}  med {q['q0.5']:.2f}  "
              f"p99 {q['q0.99']:.2f}  span(p1-p99) {q['span_p1_p99_decades']:.2f} decades")

    blob = {"experiment": a.experiment, "matched_n_rule": "all real val cells of the target",
            "n_proj": a.n_proj, "proj_seed": PROJ_SEED,
            "single_seed_warning": "one seed: block spreads are NOT run-to-run error bars",
            "rows": rows}
    dest = Path(a.out) if a.out else (C.OUT / "summary" / f"distribution_{a.experiment}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(blob, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[dist] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
