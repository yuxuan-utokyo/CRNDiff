# -*- coding: utf-8 -*-
"""The single-parameter Feynman-Kac sampler on the 2-D toy chain.
Purpose: the sampler itself. `run_fk` is line for line the 10-dimensional version, with three
changes: the imports, a purely recording trajectory hook (used by the along-chain strip figure)
and one new arm, `run_tilt_only`. The main RNG consumption order is untouched. The only model
parameter is alpha, and the weight is the discriminator odds rho = r / (1 - r). `run_tilt_only`
runs the tilted chain with no potential, no resampling and no endpoint weighting (it takes N

Target law (the only model parameter is alpha; amendment 2 fixes the weight to the ODDS):

    pi_alpha(n0)  proportional to  q(n0) * rho(n0)^alpha,    rho = r / (1 - r)

where q is the (tilted or untilted) frozen reverse chain and r the discriminator probability
(the optimal discriminator's odds equal the density ratio -- the paper's eq:rho).
`reward_mode` selects the weight: "odds" (default; every experiment) or "probability"
(log r, bit-identical to the previous sampler -- kept for port fidelity; unused in 2-D).

Potential accumulated along the reverse chain k = K-1, ..., 1 (the potential at the start of
the chain is 0 for every particle; the terminal potential closes the telescope)

    phi_k = log psi_k,   psi_k = E[ rho(n0)^alpha | n_k ]  ~=  (1/J) sum_j rho(n0_j)^alpha,  n0_j ~ p(n0 | n_k)
          = logsumexp_j( alpha * log rho(n0_j) ) - log J
    phi_0 = alpha * log rho(n0)                                 (terminal, exact: the posterior is a point)

psi_k is the optimal twist: the conditional expectation of the terminal weight given the current
state. For the EXACT twist, psi_k is a martingale along the chain, so its variance across
particles is non-decreasing between resamples and largest at the endpoint (law of total
variance) -- the design reason the sampler fires late and needs no tempering constant. The
finite-J estimate measured by the code (Var_i[log w]) is NOT covered by that theorem (log of a
martingale, Monte Carlo noise, post-fire selection); it is only plotted as an empirical check,
never used as a guarantee.

Weights telescope: log w_k = phi_k - phi_{k*} (k* = last resample step, or none), so with no
in-chain resample the delivered log-weight is exactly alpha*log rho(n0) for ANY intermediate
potential; `telescope_max_abs_error` checks that to machine precision on every run that never fired.

RNG discipline: the main stream `rng` is consumed in exactly the order of the frozen sampler --
initial state (dim by dim), then per step: systematic-resample draw only if a resample fires,
n0 draw (dim by dim), bridge draw (dim by dim). The J candidate endpoints of the twist come from
a SEPARATE stream `rng_twist` (TWIST_ROOT, seed) and the noise corruption from a per-step stream
(NOISE_ROOT, seed, step); diagnostics and the trajectory capture draw nothing.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .chain import product_x0_draw
from .config import NOISE_ROOT, TWIST_ROOT, draw_rows, feats, softmax_np


# --------------------------------------------------------------------------- primitives (frozen)
def systematic_resample(rng: np.random.Generator, weight: np.ndarray,
                        n_out: int | None = None) -> np.ndarray:
    n = len(weight) if n_out is None else int(n_out)
    positions = (rng.random() + np.arange(n, dtype=np.float64)) / n
    cdf = np.cumsum(weight)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, positions, side="right").clip(0, len(weight) - 1)


def normalized(logw: np.ndarray) -> tuple[np.ndarray, float]:
    z = np.asarray(logw, dtype=np.float64)
    z = z - float(z.max())
    w = np.exp(z)
    w /= w.sum()
    ess = 1.0 / float(np.square(w).sum())
    return w, ess


def log_reward(model, x: np.ndarray, floor: float, mode: str) -> np.ndarray:
    """log of the particle reward. The discriminator emits r in (0,1); r is clipped to
    [floor, 1-floor] in both modes (no upper clip at 1 -- odds may exceed 1).
    mode='probability': log r (the previous sampler's weight; port fidelity only).
    mode='odds': log rho = log r - log(1-r) -- the paper's weight."""
    p = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    p = np.clip(p, floor, 1.0 - floor)
    if mode == "probability":
        return np.log(p)
    if mode == "odds":
        return np.log(p) - np.log1p(-p)
    raise ValueError(f"reward_mode must be 'probability' or 'odds', got {mode!r}")


class GridRewardCache:
    """Exact memoisation of `log_reward` on the C x C count grid (ticket 163 speed-up).

        Purpose: exact memoisation of the discriminator on the lattice. Every row the sampler hands
        to log_reward is a lattice point in [0, C-1]^2 (the candidate endpoints are drawn from the
        C-column posterior, and the endpoint state equals n0 at k=1 because Kcum[0] is the identity),

    Exactness: the same row through the same tree ensemble yields the same float64 whatever the
    batch it arrives in (a row's raw score is a fixed-order serial sum over the trees), and
    neither path draws a random number, so the RNG consumption order is bit-for-bit unchanged.
    The table itself is never written to disk; its sha256 goes into the run json.
    """

    def __init__(self, model, C: int, floor: float, mode: str):
        self.C = int(C)
        self.mode = mode
        self.floor = float(floor)
        a = np.arange(self.C, dtype=np.int64)
        grid = np.stack(np.meshgrid(a, a, indexing="ij"), axis=-1).reshape(-1, 2)
        self.table = np.ascontiguousarray(
            log_reward(model, grid, floor, mode).reshape(self.C, self.C), dtype=np.float64)
        self.sha256 = hashlib.sha256(self.table.tobytes()).hexdigest()
        self.n_lookups = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xi = np.asarray(x)
        if xi.ndim != 2 or xi.shape[1] != 2:
            raise SystemExit(f"GridRewardCache expects [N, 2] cells, got shape {xi.shape}")
        if int(xi.min()) < 0 or int(xi.max()) >= self.C:
            raise SystemExit("GridRewardCache: a state left the C x C grid "
                             f"(min {int(xi.min())}, max {int(xi.max())}, C={self.C}) -- "
                             "the memoisation assumption is violated; rerun with --reward-cache off")
        self.n_lookups += len(xi)
        return self.table[xi[:, 0], xi[:, 1]]


def reward_evaluator(reward_model, C: int, floor: float, mode: str, use_cache: bool):
    """Returns (callable log-reward, cache or None). `use_cache=False` keeps the original path."""
    if use_cache:
        cache = GridRewardCache(reward_model, C, floor, mode)
        return cache, cache
    return (lambda x: log_reward(reward_model, x, floor, mode)), None


def product_posterior(net, dev, x: np.ndarray, t: float, chain, terminal: float,
                      theta: np.ndarray) -> np.ndarray:
    """p(n0 | n_k) per dim from the frozen network, multiplied by the tilt table theta (zeros = no tilt)."""
    import torch

    n = len(x)
    oh = getattr(net, "onehot", 0)
    with torch.no_grad():
        z = net(torch.tensor(feats(x, np.full(n, t), chain.CF, terminal, oh), device=dev))
        ell = torch.log_softmax(z, dim=2).cpu().numpy()
    return softmax_np(ell + theta[None, :, :], axis=2)


def logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.exp(a - m).sum(axis=axis))


# --------------------------------------------------------------------------- corruption
def corrupt_logpsi(logpsi: np.ndarray, sigma: float, sampler_seed: int, step: int) -> np.ndarray:
    """Additive Gaussian noise DIRECTLY on log psi-hat, from an independent stream seeded by
    (NOISE_ROOT, run seed, step). Under odds weights psi-hat may exceed 1, where a logit
    transform would be undefined; the corruption is therefore additive in the log domain."""
    gen = np.random.default_rng([NOISE_ROOT, int(sampler_seed), int(step)])
    return logpsi + gen.normal(0.0, sigma, size=logpsi.shape)


def middle_third_steps(K: int) -> list[int]:
    """The steps at which an intermediate potential is evaluated are k = K-1 .. 1. The corrupted
    set is the middle third of that list by count (previous corruption pre-registration)."""
    evaluated = list(range(K - 1, 0, -1))
    n = len(evaluated)
    lo, hi = n // 3, 2 * (n // 3)
    return sorted(evaluated[lo:hi])


# --------------------------------------------------------------------------- the sampler
def run_fk(net, dev, chain, terminal: float, theta: np.ndarray, reward_model, *,
           n: int, n_out: int, seed: int, alpha: float, J: int = 16,
           ess_frac: float = 0.5, resample_interval: int = 8,
           enable_intermediate_resampling: bool = True,
           reward_floor: float = 1.0e-8, reward_mode: str = "odds",
           sigma_logit: float = 0.0, degrade_steps: set[int] | None = None,
           trajectory_sink: dict | None = None, traj_steps: set[int] | None = None,
           reward_cache: bool = True,
           ) -> tuple[np.ndarray, dict]:
    """Returns (delivered cells [n_out, S] int64, diagnostics dict).

    alpha  the model parameter: target law and terminal weight rho^alpha, and the twist E[rho^alpha|n_k].
    reward_mode  "odds" (default, the paper's weight) or "probability" (port fidelity only).
    J      candidate endpoints per particle per step used to estimate the twist (a Monte Carlo
           budget, not a model parameter; J -> inf recovers the exact twist).
    theta  the tilt table used for the proposal AND for every posterior the twist is computed
           on. Pass zeros for the untilted (tilt-off) arm.
    ess_frac (theta in the paper)  resample at a scheduled step iff ESS/M < ess_frac; any value
           >= 1.0 fires UNCONDITIONALLY at every scheduled step (structural, not numerical).
    sigma_logit, degrade_steps     corrupt log psi_k (additive Gaussian on log psi) at the
           listed steps.
    trajectory_sink, traj_steps    pure recording for the chain-strip / reverse figures (work
           order sec. 2): at each k in traj_steps the current state, its log-weights and the
           endpoint sample n0 ALREADY DRAWN by that step's transition are copied into the sink.
           Draws NOTHING from any stream; None (the default) records nothing.
    reward_cache   memoise log_reward on the C x C grid (exact; see GridRewardCache). False
           keeps the per-call discriminator path.
    """
    if J < 1:
        raise ValueError("J must be >= 1")
    if not alpha > 0:
        raise ValueError("alpha must be > 0")
    if reward_mode not in ("probability", "odds"):
        raise ValueError(f"reward_mode must be 'probability' or 'odds', got {reward_mode!r}")
    degrade_steps = set(degrade_steps or ())
    traj_steps = set(traj_steps or ())

    C, CF, ts = chain.C, chain.CF, chain.ts
    S = chain.stat.shape[0]
    K = len(ts) - 1
    reward_fn, cache = reward_evaluator(reward_model, C, reward_floor, reward_mode, reward_cache)
    rng = np.random.default_rng(seed)
    rng_twist = np.random.default_rng([TWIST_ROOT, int(seed)])

    # initial state from the stationary law, dim by dim (frozen order)
    x = np.empty((n, S), dtype=np.int64)
    for g in range(S):
        cdf = np.cumsum(chain.stat[g])
        x[:, g] = np.searchsorted(cdf, rng.random(n) * cdf[-1], side="right").clip(0, CF - 1)

    lineage = np.arange(n, dtype=np.int64)
    logw = np.zeros(n, dtype=np.float64)
    prev_value = np.zeros(n, dtype=np.float64)   # potential at the start of the chain: 0 for all
    trace: list[dict] = []
    resample_steps: list[int] = []
    degraded_at: list[int] = []
    telescope_error = None
    reward_evals = 0
    if trajectory_sink is not None and K in traj_steps:      # the chain start (t = T)
        trajectory_sink[K] = {"x": x.copy(), "logw": logw.copy(), "t": float(ts[K]), "fired": False}

    for k in range(K, 0, -1):
        posterior = product_posterior(net, dev, x, float(ts[k]), chain, terminal, theta)

        if k < K:
            # ---- the twist: log E[rho(n0)^alpha | n_k] by J candidate endpoints (separate RNG) ---
            lpost = np.log(np.maximum(posterior, 1e-300))
            lr = np.empty((J, n), dtype=np.float64)
            for j in range(J):
                cand = product_x0_draw(rng_twist, lpost)
                lr[j] = reward_fn(cand)
            reward_evals += J
            logpsi = logsumexp(alpha * lr, axis=0) - math.log(J)
            if sigma_logit > 0.0 and k in degrade_steps:
                logpsi = corrupt_logpsi(logpsi, sigma_logit, seed, k)
                degraded_at.append(k)
            current_value = logpsi
            # ----------------------------------------------------------------------------------
            logw += current_value - prev_value
            weight, ess = normalized(logw)
            fired = False
            scheduled = (k % resample_interval == 0)
            rec = {"k": k, "t": float(ts[k]), "ess_fraction": ess / n,
                   "logw_var": float(np.var(logw)), "scheduled": bool(scheduled)}
            # ess_frac >= 1.0 fires unconditionally (amendment 1 section 4)
            if enable_intermediate_resampling and scheduled and (ess_frac >= 1.0 or ess < ess_frac * n):
                idx = systematic_resample(rng, weight)
                x = x[idx]
                posterior = posterior[idx]
                current_value = current_value[idx]
                lineage = lineage[idx]
                logw.fill(0.0)
                resample_steps.append(k)
                fired = True
            rec["fired"] = fired
            trace.append(rec)
            prev_value = current_value
            if trajectory_sink is not None and k in traj_steps:   # pure recording, no draws
                trajectory_sink[k] = {"x": x.copy(), "logw": logw.copy(),
                                      "t": float(ts[k]), "fired": fired,
                                      "lineage": lineage.copy()}

        # ---- transition: n0 draw then bridge, dim by dim (frozen order) -----------------------
        n0 = product_x0_draw(rng, np.log(np.maximum(posterior, 1e-300)))
        if trajectory_sink is not None and k in trajectory_sink:
            # endpoint belief at step k: the n0 THIS step's transition already drew. A copy of an
            # existing array -- no extra rng call, no change to the consumption order.
            trajectory_sink[k]["n0"] = n0.copy()
        for g in range(S):
            bridge = chain.Kcum[k - 1, g][n0[:, g], :] * chain.Kstep[k, g][:, x[:, g]].T
            x[:, g] = draw_rows(rng, bridge)

    # ---- terminal: the exact endpoint potential closes the telescope -------------------------
    final_reward = reward_fn(x)
    logw += alpha * final_reward - prev_value
    final_weight, final_ess = normalized(logw)
    trace.append({"k": 0, "t": 0.0, "ess_fraction": final_ess / n,
                  "logw_var": float(np.var(logw)), "scheduled": False, "fired": False})
    if not resample_steps:
        telescope_error = float(np.max(np.abs(logw - alpha * final_reward)))

    idx = systematic_resample(rng, final_weight, n_out=n_out)
    out = x[idx]
    lineage = lineage[idx]
    if trajectory_sink is not None and 0 in traj_steps:
        # at t = 0 the state IS the endpoint, so the belief panel equals the state panel
        trajectory_sink[0] = {"x": x.copy(), "logw": logw.copy(), "t": 0.0, "fired": True,
                              "n0": x.copy(), "lineage": lineage.copy(),
                              "final_weight": final_weight.copy(), "out_idx": idx.copy()}
    diag = {
        "n_particles": n, "n_out": n_out, "alpha": alpha, "J": J,
        "ess_threshold_fraction": ess_frac, "resample_interval": resample_interval,
        "reward_floor": reward_floor, "reward_mode": reward_mode,
        "intermediate_resampling": enable_intermediate_resampling,
        "sigma_logit": sigma_logit, "degrade_steps_requested": sorted(degrade_steps),
        "degraded_at": degraded_at,
        "n_intermediate_resamples": len(resample_steps), "resample_steps": resample_steps,
        "ess_trace": trace,
        "ess_min_fraction": float(min(v["ess_fraction"] for v in trace)),
        "ess_pre_final_fraction": float(final_ess / n),
        "unique_lineage_fraction": float(np.unique(lineage).size / n_out),
        "unique_cell_fraction": float(np.unique(np.ascontiguousarray(out), axis=0).shape[0] / n_out),
        "telescope_max_abs_error": telescope_error,
        "reward_evaluations_per_particle": reward_evals + 1,
        "NFE": K,
        # exact grid memoisation of the discriminator (ticket 163 speed-up): a fingerprint only,
        # the table itself is never written to disk
        "reward_cache": bool(reward_cache),
        "reward_table_sha256": (cache.sha256 if cache is not None else None),
        # endpoint particle index of every delivered sample (figure amendment section 1:
        # multiplicity-aware drawing). Recorded only; nothing in the sampler reads it back.
        "_delivered_idx": idx,
    }
    return out, diag


# --------------------------------------------------------------------------- pure arms
def run_tilt_only(net, dev, chain, terminal: float, theta: np.ndarray, *,
                  n: int, n_out: int, seed: int,
                  trajectory_sink: dict | None = None, traj_steps: set[int] | None = None,
                  ) -> tuple[np.ndarray, dict]:
    """The tilt-only ladder arm: the (tilted) frozen reverse chain with NO potential, NO
    in-chain resampling and NO terminal reweighting; delivery = a uniform stratified take of
    n_out (systematic resample under uniform weights). Consumes the main stream in exactly
    run_fk's order (init, then per step n0 + bridge, then one final resample draw) and
    constructs NO potential-related stream (no rng_twist, no noise stream, no discriminator)."""
    traj_steps = set(traj_steps or ())
    C, CF, ts = chain.C, chain.CF, chain.ts
    S = chain.stat.shape[0]
    K = len(ts) - 1
    rng = np.random.default_rng(seed)

    x = np.empty((n, S), dtype=np.int64)
    for g in range(S):
        cdf = np.cumsum(chain.stat[g])
        x[:, g] = np.searchsorted(cdf, rng.random(n) * cdf[-1], side="right").clip(0, CF - 1)

    lineage = np.arange(n, dtype=np.int64)
    if trajectory_sink is not None and K in traj_steps:
        trajectory_sink[K] = {"x": x.copy(), "logw": np.zeros(n), "t": float(ts[K]), "fired": False}

    for k in range(K, 0, -1):
        posterior = product_posterior(net, dev, x, float(ts[k]), chain, terminal, theta)
        if trajectory_sink is not None and k < K and k in traj_steps:
            trajectory_sink[k] = {"x": x.copy(), "logw": np.zeros(n), "t": float(ts[k]), "fired": False}
        n0 = product_x0_draw(rng, np.log(np.maximum(posterior, 1e-300)))
        if trajectory_sink is not None and k in trajectory_sink:
            trajectory_sink[k]["n0"] = n0.copy()      # see run_fk: a copy, never a new draw
        for g in range(S):
            bridge = chain.Kcum[k - 1, g][n0[:, g], :] * chain.Kstep[k, g][:, x[:, g]].T
            x[:, g] = draw_rows(rng, bridge)

    uniform = np.full(n, 1.0 / n, dtype=np.float64)
    idx = systematic_resample(rng, uniform, n_out=n_out)
    out = x[idx]
    lineage = lineage[idx]
    if trajectory_sink is not None and 0 in traj_steps:
        trajectory_sink[0] = {"x": x.copy(), "logw": np.zeros(n), "t": 0.0, "fired": False,
                              "n0": x.copy(),
                              "final_weight": uniform.copy(), "out_idx": idx.copy()}
    diag = {
        "n_particles": n, "n_out": n_out, "alpha": None, "J": None,
        "ess_threshold_fraction": None, "resample_interval": None,
        "reward_floor": None, "reward_mode": None,
        "intermediate_resampling": False,
        "sigma_logit": 0.0, "degrade_steps_requested": [], "degraded_at": [],
        "n_intermediate_resamples": 0, "resample_steps": [],
        "ess_trace": [{"k": 0, "t": 0.0, "ess_fraction": 1.0, "logw_var": 0.0,
                       "scheduled": False, "fired": False}],
        "ess_min_fraction": 1.0, "ess_pre_final_fraction": 1.0,
        "unique_lineage_fraction": float(np.unique(lineage).size / n_out),
        "unique_cell_fraction": float(np.unique(np.ascontiguousarray(out), axis=0).shape[0] / n_out),
        "telescope_max_abs_error": None,
        "reward_evaluations_per_particle": 0,
        "NFE": K,
        "reward_cache": None, "reward_table_sha256": None,   # this arm evaluates no reward
        "_delivered_idx": idx,          # same recording as run_fk; see there
    }
    return out, diag
