# -*- coding: utf-8 -*-
"""Generate the toy dataset: eight isotropic Gaussian components on one circle.

The components sit at equal angles around a circle. The four AXIAL components carry equal mass
and so do the four DIAGONAL ones.

Why the diagonal components must be equal in mass. In an earlier version one diagonal pair was
far heavier than the other, so after tilting only a small part of the proposal landed on the
request, the residual discriminator had to calibrate on a badly imbalanced problem, and its
Brier score was several times worse than the untilted arm's. With the four made equal, the
tilted proposal splits evenly between them, the discrimination problem becomes balanced, and the
ESS ceiling of the particle layer rises by an order of magnitude.

    python scripts/make_toy_dataset.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toy2d.verified.primitives import CF_MULT, kmat            # noqa: E402  frozen primitives

# conventions (frozen constants)
C = 64
S_DIM = 2
SEED = 20260905
N_TRAIN = 400_000
N_REF = 40_000
CENTER = (31.5, 31.5)
ANGLE_STEP_DEG = 45.0
AXIS = (0, 2, 4, 6)          # the axial components
CORNER = (1, 3, 5, 7)        # the diagonal components; the request and its twin are its two diagonals

COMPONENTS = [f"c{j}" for j in range(8)]
REQUESTS_BASE = {
    "diag_rare": {"components": ["c1", "c5"], "weights": [0.5, 0.5]},
    "diag_twin": {"components": ["c3", "c7"], "weights": [0.5, 0.5]},
}
MARGINAL_TWINS = [("diag_rare", "diag_twin")]

# the registered invariants
INV_TWIN_MARGINAL_MAX = 1e-14
INV_TWIN_TV_MIN = 0.999
INV_PURITY_CEILING_MIN = 0.999
INV_SEMIGROUP_MARGIN_MIN = 100.0        # 1e-9 / semigroup
INV_MEAN_RANGE = (20.0, 42.0)
SEMIGROUP_THRESHOLD = 1e-9              # the frozen self-check threshold; read, never changed


def grid():
    n1, n2 = np.meshgrid(np.arange(C, dtype=np.float64), np.arange(C, dtype=np.float64),
                         indexing="ij")
    return n1, n2


def _norm(p):
    return p / p.sum()


def cluster_center(j: int, radius: float) -> tuple[float, float]:
    th = np.deg2rad(ANGLE_STEP_DEG * j)
    return (CENTER[0] + radius * np.cos(th), CENTER[1] + radius * np.sin(th))


def cluster_pmf(j: int, radius: float, sigma: float):
    n1, n2 = grid()
    c0, c1 = cluster_center(j, radius)
    return _norm(np.exp(-((n1 - c0) ** 2 + (n2 - c1) ** 2) / (2 * sigma ** 2)))


def priors_for(mass_corner: float) -> np.ndarray:
    mass_axis = (1.0 - 4.0 * mass_corner) / 4.0
    p = np.empty(8, dtype=np.float64)
    p[list(AXIS)] = mass_axis
    p[list(CORNER)] = mass_corner
    return p


def component_tables(radius, sigma, priors):
    P = np.stack([cluster_pmf(j, radius, sigma) for j in range(8)])
    return P, np.tensordot(priors, P, axes=1)


def request_weights(name, requests):
    w = np.zeros(8, dtype=np.float64)
    for c, wc in zip(requests[name]["components"], requests[name]["weights"]):
        w[COMPONENTS.index(c)] = wc
    return w / w.sum()


def request_pmf(name, P, requests):
    return np.tensordot(request_weights(name, requests), P, axes=1)


def product_tilt(mix, q, gamma=1.0, cap=np.log(1e6)):
    lc0 = np.clip(np.log(np.maximum(q.sum(1), 1e-300) / np.maximum(mix.sum(1), 1e-300)), -cap, cap)
    lc1 = np.clip(np.log(np.maximum(q.sum(0), 1e-300) / np.maximum(mix.sum(0), 1e-300)), -cap, cap)
    return _norm(mix * np.exp(gamma * lc0)[:, None] * np.exp(gamma * lc1)[None, :])


def draw_from_table(rng, table, n):
    flat = table.ravel()
    idx = rng.choice(flat.size, size=n, p=flat / flat.sum())
    return np.stack(np.unravel_index(idx, table.shape), axis=1).astype(np.int64)


def sample_mixture(rng, P, n, priors):
    quota = np.floor(priors * n).astype(np.int64)
    quota[-1] += n - quota.sum()
    xs, ys = [], []
    for k, q in enumerate(quota):
        xs.append(draw_from_table(rng, P[k], int(q)))
        ys.append(np.full(int(q), k, dtype=np.int64))
    x = np.concatenate(xs); y = np.concatenate(ys)
    perm = rng.permutation(n)
    return x[perm], y[perm]


def stratified_roles(y, rng):
    role = np.zeros(len(y), dtype=np.int64)
    for k in np.unique(y):
        idx = np.flatnonzero(y == k)
        idx = idx[rng.permutation(len(idx))]
        role[idx[len(idx) // 2:]] = 1
    return role


def semigroup_at(V: float) -> float:
    """The frozen self_check's semigroup residual at per-dimension mean V, CF = CF_MULT * C.
    Read-only: the 1e-9 threshold and CF_MULT are never touched."""
    CF = CF_MULT * C
    return float(np.abs(kmat(0.7, V, CF) @ kmat(1.3, V, CF) - kmat(2.0, V, CF)).max())


def marginal_means(mix):
    n = np.arange(C, dtype=np.float64)
    return float((mix.sum(1) * n).sum()), float((mix.sum(0) * n).sum())


def purity_ceilings(P, mix, priors):
    resp = (priors[:, None, None] * P) / np.maximum(mix, 1e-300)[None]
    return {COMPONENTS[k]: float((P[k] * resp[k]).sum()) for k in range(8)}


def invariants(P, mix, priors, requests):
    """The five registered invariants; they must be recomputed after any change."""
    qa, qb = request_pmf("diag_rare", P, requests), request_pmf("diag_twin", P, requests)
    twin_marg = float(max(np.abs(qa.sum(1) - qb.sum(1)).max(), np.abs(qa.sum(0) - qb.sum(0)).max()))
    twin_tv = float(0.5 * np.abs(qa - qb).sum())
    ceil = purity_ceilings(P, mix, priors)
    m0, m1 = marginal_means(mix)
    semi = max(semigroup_at(m0), semigroup_at(m1))
    margin = SEMIGROUP_THRESHOLD / semi if semi > 0 else float("inf")
    rows = [
        ("twin_max_abs_marginal_diff", twin_marg, f"< {INV_TWIN_MARGINAL_MAX:g}",
         twin_marg < INV_TWIN_MARGINAL_MAX),
        ("twin_tv_joint", twin_tv, f"> {INV_TWIN_TV_MIN}", twin_tv > INV_TWIN_TV_MIN),
        ("min_purity_ceiling", min(ceil.values()), f">= {INV_PURITY_CEILING_MIN}",
         min(ceil.values()) >= INV_PURITY_CEILING_MIN),
        ("semigroup_margin", margin, f">= {INV_SEMIGROUP_MARGIN_MIN:g}x",
         margin >= INV_SEMIGROUP_MARGIN_MIN),
        ("marginal_mean_dim0", m0, f"in {list(INV_MEAN_RANGE)}",
         INV_MEAN_RANGE[0] <= m0 <= INV_MEAN_RANGE[1]),
        ("marginal_mean_dim1", m1, f"in {list(INV_MEAN_RANGE)}",
         INV_MEAN_RANGE[0] <= m1 <= INV_MEAN_RANGE[1]),
    ]
    return {"rows": [{"quantity": q, "value": v, "requirement": r, "pass": bool(p)}
                     for q, v, r, p in rows],
            "purity_ceilings": ceil, "semigroup_worst": semi,
            "semigroup_threshold": SEMIGROUP_THRESHOLD,
            "all_pass": bool(all(p for *_, p in rows))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mass-corner", type=float, default=0.005)
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--sigma", type=float, default=1.8)
    ap.add_argument("--n-train", type=int, default=N_TRAIN,
                    help="training set size (work order 194: v6 scales it with the corner "
                         "mass so each corner keeps enough training points)")
    ap.add_argument("--n-ref", type=int, default=N_REF,
                    help="reference draw size (plan 3.1 knob). X_train is drawn FIRST from the "
                         "same stream, so changing this leaves the training set -- and therefore "
                         "the checkpoint and the chain cache -- bit-identical.")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--check-only", action="store_true",
                    help="only recompute the invariants; write nothing (cheap config screen)")
    a = ap.parse_args(argv)

    priors = priors_for(a.mass_corner)
    requests = dict(REQUESTS_BASE)
    requests.update({n: {"components": [n], "weights": [1.0]} for n in COMPONENTS})
    P, mix = component_tables(a.radius, a.sigma, priors)
    inv = invariants(P, mix, priors, requests)

    print(f"[config] mass_corner={a.mass_corner} mass_axis={(1-4*a.mass_corner)/4:.6f} "
          f"radius={a.radius} sigma={a.sigma} tag={a.tag}")
    print("[invariants] (overnight plan section 3.3)")
    for r in inv["rows"]:
        print(f"  {r['quantity']:<28}{r['value']!r:<26}{r['requirement']:<20}"
              f"{'PASS' if r['pass'] else 'FAIL'}")
    for n, v in inv["purity_ceilings"].items():
        print(f"  purity_ceiling[{n}] = {v:.10f}")
    if a.check_only:
        return 0 if inv["all_pass"] else 2
    if not inv["all_pass"]:
        raise SystemExit("invariants FAILED -- configuration rejected, nothing written")

    stem = Path(a.out) / f"data2d_ring8_sym_{a.tag}"
    for suffix in (".npz", ".json", "_pmf.npz"):
        if Path(str(stem) + suffix).exists():
            raise SystemExit(f"refusing to overwrite {stem}{suffix}")

    rng = np.random.default_rng(a.seed)
    x_train, y_train = sample_mixture(rng, P, a.n_train, priors)
    x_ref, y_ref = sample_mixture(rng, P, a.n_ref, priors)
    ref_role = stratified_roles(y_ref, rng)
    mu, var = x_train.mean(0), x_train.var(0)
    print(f"[caliber] mean={mu.round(4).tolist()} var={var.round(1).tolist()}")

    resp = (priors[:, None, None] * P) / np.maximum(mix, 1e-300)[None]
    req_report = {}
    for name in requests:
        q = request_pmf(name, P, requests)
        t = product_tilt(mix, q)
        req_report[name] = {
            "mass_under_D": float(priors[request_weights(name, requests) > 0].sum()),
            "tv_product_tilt_to_request": float(0.5 * np.abs(t - q).sum()),
            "composition_of_product_tilt": {n: float(v) for n, v in
                                            zip(COMPONENTS, (resp * t[None]).sum((1, 2)))},
        }
    for name in ("diag_rare", "diag_twin"):
        r = req_report[name]
        ct = r["composition_of_product_tilt"]
        print(f"[request] {name:<10} mass_under_D={r['mass_under_D']:.5f}  "
              f"TV(tilt->req)={r['tv_product_tilt_to_request']:.6f}  "
              f"tilt purity={ct['c1']+ct['c5'] if name=='diag_rare' else ct['c3']+ct['c7']:.6f}")

    np.savez_compressed(str(stem) + ".npz",
                        X_train=x_train.astype(np.uint8), y_train=y_train.astype(np.uint8),
                        X_ref=x_ref.astype(np.uint8), y_ref=y_ref.astype(np.uint8),
                        ref_role=ref_role.astype(np.uint8))
    np.savez_compressed(str(stem) + "_pmf.npz", components=P, mixture=mix)
    cfg = {
        "version": f"toy2d_ring8_sym_{a.tag}", "C": C, "S": S_DIM, "seed": int(a.seed),
        "n_train": int(a.n_train), "n_ref": int(a.n_ref),
        "components": COMPONENTS, "priors": priors.tolist(), "center": list(CENTER),
        "knobs": {"mass_corner": a.mass_corner, "mass_axis": float((1 - 4 * a.mass_corner) / 4),
                  "radius": a.radius, "sigma": a.sigma, "n_ref": int(a.n_ref),
                  "n_train": int(a.n_train),
                  "angle_step_deg": ANGLE_STEP_DEG,
                  "counterclockwise": True},
        "clusters": {"axis": [COMPONENTS[j] for j in AXIS],
                     "corners": [COMPONENTS[j] for j in CORNER],
                     "centers": {COMPONENTS[j]: list(cluster_center(j, a.radius))
                                 for j in range(8)},
                     "expected_training_points": {COMPONENTS[j]: int(round(priors[j] * a.n_train))
                                                  for j in range(8)}},
        "requests": requests, "marginal_twins": MARGINAL_TWINS,
        "reference_role_0": "weight_classifier_training",
        "reference_role_1": "held_out_evaluation",
        "invariants": inv, "request_report": req_report,
        "marginal_mean_dim0": marginal_means(mix)[0],
        "marginal_mean_dim1": marginal_means(mix)[1],
        "train_mean": mu.tolist(), "train_var": var.tolist(),
        "derived_from": "make_data2d_ring8.py; the four square corners are made EQUAL mass so "
                        "the tilted proposal splits 25/25/25/25 over them",
    }
    Path(str(stem) + ".json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    print(f"[out] {stem}.npz / .json / _pmf.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
