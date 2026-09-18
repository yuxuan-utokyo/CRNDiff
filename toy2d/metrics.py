# -*- coding: utf-8 -*-
"""Exact readings on delivered 2-D cells, from the closed-form pmf tables.
Purpose: exact readings on a delivery. The responsibilities r_k(x) = pi_k P_k(x) / D(x) are
computed in closed form from the frozen pmf table, which gives the soft composition, the purity
(the composition mass on the requested components), the composition total variation, and the
twin gap. Grid TV is reported but not used as a criterion, alongside the noise floor of a
"""
from __future__ import annotations

import numpy as np

NOISE_FLOOR_ROOT = 20260906      # dedicated root: distinct from TWIST_ROOT (20260905) and
                                 # NOISE_ROOT (20260821); the noise-floor draw never touches run RNG


def responsibilities(pmf_components: np.ndarray, priors: np.ndarray, x: np.ndarray) -> np.ndarray:
    """r_k(x) = pi_k P_k(x) / D(x), exact, [N, 10]. D is the prior mixture of the tables."""
    xi = np.asarray(x).astype(np.int64)
    pk = pmf_components[:, xi[:, 0], xi[:, 1]]                    # [10, N]
    num = priors[:, None] * pk
    den = np.maximum(num.sum(0, keepdims=True), 1e-300)
    return (num / den).T


def request_pmf(pmf_components: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The request law on the grid: sum_k w_k P_k (w sums to 1)."""
    return np.tensordot(np.asarray(w, dtype=np.float64), pmf_components, axes=(0, 0))


def composition_tv(comp: np.ndarray, w: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(comp) - np.asarray(w)).sum())


def grid_tv(x: np.ndarray, pmf: np.ndarray) -> float:
    """TV between the empirical grid histogram of x and a pmf table (report-only; the
    N=1000 noise floor is ~0.25)."""
    C = pmf.shape[0]
    xi = np.asarray(x).astype(np.int64)
    hist = np.zeros((C, C), dtype=np.float64)
    np.add.at(hist, (xi[:, 0], xi[:, 1]), 1.0)
    hist /= len(xi)
    return float(0.5 * np.abs(hist - pmf).sum())


def grid_tv_noise_floor(pmf: np.ndarray, n: int, seed: int) -> float:
    """Same-N iid draw FROM the request pmf, then its grid TV to that pmf -- the Monte Carlo
    floor below which grid TV is indistinguishable from noise. Own stream (NOISE_FLOOR_ROOT,
    seed); never the run's RNG."""
    rng = np.random.default_rng([NOISE_FLOOR_ROOT, int(seed)])
    flat = pmf.reshape(-1)
    cdf = np.cumsum(flat)
    idx = np.searchsorted(cdf, rng.random(n) * cdf[-1], side="right").clip(0, flat.size - 1)
    C = pmf.shape[0]
    draw = np.stack([idx // C, idx % C], axis=1)
    return grid_tv(draw, pmf)


TWIN = ("s_curve", "z_curve")      # the exact marginal twins (prereg amendment 2)


def exact_readings(pmf_components: np.ndarray, cfg: dict, x: np.ndarray,
                   w: np.ndarray, seed: int, request: str | None = None) -> dict:
    """Every pre-registered metric of one delivered sample set against the request composition
    w (a vector over cfg['components']; the D priors for the 'unconditional' gate).

    purity   = the delivered responsibility mass on the request's components (the requests are
               single-component here, so purity = comp[request]).
    twin_gap = |comp[s_curve] - comp[z_curve]| (amendment 2 section 3). The twin pair is the
               whole point of the device: tilt alone cannot separate them (gap ~ 0), the
               residual correction can (gap ~ 0.9). The 'ring' request has no twin, so its
               twin_gap is null and enters no criterion."""
    comps = list(cfg["components"])
    priors = np.asarray(cfg["priors"], dtype=np.float64)
    resp = responsibilities(pmf_components, priors, x)
    comp = resp.mean(0)
    rp = request_pmf(pmf_components, w)
    if request == "ring" or not all(t in comps for t in TWIN):
        twin_gap = None
    else:
        twin_gap = float(abs(comp[comps.index(TWIN[0])] - comp[comps.index(TWIN[1])]))
    return {
        "composition": comp.tolist(),
        "components": comps,
        "purity": float(comp[np.asarray(w) > 0].sum()),
        "composition_tv": composition_tv(comp, w),
        "twin_gap": twin_gap,
        "grid_tv": grid_tv(x, rp),
        "grid_tv_noise_floor": grid_tv_noise_floor(rp, len(x), seed),
    }
