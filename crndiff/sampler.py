# -*- coding: utf-8 -*-
"""The single-parameter Feynman-Kac sampler on the frozen HVG generator.
Purpose: the single-parameter sampler, isomorphic to the toy package's sampler. Ported at
function level from `crndiff/frozen/fk_hvg_v2.py` (PortableHGBT, CleanLogisticReward,
classifier_log_reward, HVGFK, initial_stationary, run_fk) and from `crndiff/frozen/e38_twist.py`
(the batched draw of J candidate endpoints); sources and sha256 are in PROVENANCE.json.

The only model parameter is alpha. The target law is

    pi_alpha(n0)  ∝  q(n0) · rho(n0)^alpha,     rho = r / (1 - r)

where q is the frozen product-tilted reverse chain and r the discriminator probability, clipped
to [floor, 1 - floor] with no upper clip, since the odds may exceed one.

# ## The three changes relative to the frozen module, and only these three

1. The intermediate potential becomes a single-parameter twist (`--sampler twist`, the default):

       log psi_k = logsumexp_j( alpha · log rho(n0^(j)) ) - log J

      the n0^(j) are J candidate endpoints drawn from the tilt posterior already computed at that
      step, in one `torch.multinomial(q, J, replacement=True)` rather than J single draws, which
      would turn J=16 into roughly sixteen times the cost. It uses a dedicated torch.Generator
      (seed + TWIST_SEED_OFFSET), so the main RNG consumption order is unchanged. The frozen
   plug-in potential alpha * log r(E[n0 | n_t]) stays available as `--sampler plugin`.
   2. `intermediate_scale`, the second parameter of the two-parameter form, is removed, which is
   equivalent to pinning it to 1.0; the flag is gone and the stem no longer carries `sc...`.
   3. The weight uses odds (`--reward-mode odds`, the default): log rho = log r - log(1 - r).
   `probability` (log r) is kept as the port-fidelity setting, bitwise equal to the frozen
The endpoint potential alpha * log rho(n0) is exact, since at t = 0 the state is n0 and no twist
is needed; `final_reward` is unchanged. Everything else is untouched: the product tilt applied
at every step, the ESS threshold, systematic resampling, the telescoping identity, K, M, N, the

# ## RNG discipline

* `rng` (numpy, seed): the initial state dimension by dimension, then one draw per firing for
  systematic resampling. The same order as the frozen module.
* `torch_gen` (seed + 1): posterior n0 draws and bridge draws. Same order, same stream.
  * `twist_gen` (seed + TWIST_SEED_OFFSET): the J candidate endpoints, on an independent stream.
"""
from __future__ import annotations

import math

import numpy as np
try:
    import torch
except ModuleNotFoundError as _exc:                                      # noqa: E402
    raise SystemExit(
        "this entry point needs PyTorch, which the offline reproduction does not "
        "install. See README section 7; in short: pip install torch") from _exc

from .config import TWIST_SEED_OFFSET


# --------------------------------------------------------------------------- primitives (frozen)
def normalized(logw: np.ndarray) -> tuple[np.ndarray, float]:
    z = np.asarray(logw, dtype=np.float64)
    z = z - float(z.max())
    w = np.exp(z)
    w /= w.sum()
    return w, 1.0 / float(np.square(w).sum())


def systematic_resample(rng: np.random.Generator, weight: np.ndarray,
                        n_out: int | None = None) -> np.ndarray:
    n = len(weight) if n_out is None else int(n_out)
    cdf = np.cumsum(np.asarray(weight, dtype=np.float64))
    cdf[-1] = 1.0
    positions = (rng.random() + np.arange(n, dtype=np.float64)) / n
    return np.searchsorted(cdf, positions, side="right").clip(0, len(weight) - 1)


class PortableHGBT:
    """Inference-only binary HistGradientBoosting ensemble (frozen; fields materialised once).

        Ported verbatim from `frozen/fk_hvg_v2.py`, whose comment records why every field has to be
        materialised into a contiguous array first: reading a field view of a structured array
    """

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        nodes = z["nodes"]
        self.is_leaf = np.ascontiguousarray(nodes["is_leaf"])
        self.feature_idx = np.ascontiguousarray(nodes["feature_idx"]).astype(np.intp)
        self.num_threshold = np.ascontiguousarray(nodes["num_threshold"]).astype(np.float64)
        self.missing_left = np.ascontiguousarray(nodes["missing_go_to_left"])
        self.left = np.ascontiguousarray(nodes["left"]).astype(np.int64)
        self.right = np.ascontiguousarray(nodes["right"]).astype(np.int64)
        self.leaf_value = np.ascontiguousarray(nodes["value"]).astype(np.float64)
        self.offsets = np.ascontiguousarray(z["offsets"]).astype(np.int64)
        self.baseline = float(z["baseline"][0])
        self.n_features = int(z["n_features"][0])
        self.source_sha256 = str(z["source_sha256"][0])
        self.identity_max_abs_error = float(z["identity_max_abs_error"][0])
        del nodes, z

    def predict_probability(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != self.n_features:
            raise ValueError(f"classifier expects {self.n_features} features, got {x.shape[1]}")
        raw = np.full(len(x), self.baseline, dtype=np.float64)
        rows = np.arange(len(x))
        for start, end in zip(self.offsets[:-1], self.offsets[1:]):
            s, e = int(start), int(end)
            index = np.zeros(len(x), dtype=np.int64)
            for _ in range(e - s):
                gi = index + s
                active = self.is_leaf[gi] == 0
                if not np.any(active):
                    break
                rr = rows[active]
                ga = gi[active]
                value = x[rr, self.feature_idx[ga]]
                go_left = np.where(np.isnan(value), self.missing_left[ga] != 0,
                                   value <= self.num_threshold[ga])
                index[active] = np.where(go_left, self.left[ga], self.right[ga])
            else:
                raise RuntimeError("portable tree traversal did not reach leaves")
            raw += self.leaf_value[index + s]
        return np.where(raw >= 0, 1.0 / (1.0 + np.exp(-raw)),
                        np.exp(raw) / (1.0 + np.exp(raw)))


class CleanLogisticReward:
    """Portable smooth condition classifier fitted only on labeled real cells (frozen)."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.features = z["features"].astype(np.int64)
        self.mean = z["scaler_mean"].astype(np.float64)
        self.scale = z["scaler_scale"].astype(np.float64)
        self.coef = z["coef"].astype(np.float64)
        self.intercept = float(z["intercept"][0])
        self.n_features = len(self.features)

    def predict_probability(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != self.n_features:
            raise ValueError(f"classifier expects {self.n_features} features, got {x.shape[1]}")
        raw = ((np.asarray(x, dtype=np.float64) - self.mean) / self.scale) @ self.coef
        raw += self.intercept
        return np.where(raw >= 0, 1.0 / (1.0 + np.exp(-raw)),
                        np.exp(raw) / (1.0 + np.exp(raw)))


def classifier_log_reward(clf, features: np.ndarray, x: np.ndarray, floor: float,
                          mode: str = "odds") -> np.ndarray:
    """log of the particle reward, with the classifier's fit-time normalisation.

        mode='probability'  log r, clipped at [floor, 1]. Bitwise equal to the frozen
                                                `classifier_log_reward`; this is what gate G2 checks.
        mode='odds'         log rho = log r - log(1 - r), r clipped to [floor, 1 - floor], with no
                                                upper clip. This is the weight the paper defines.
    """
    z = np.asarray(x, dtype=np.float32)
    lib = z.sum(axis=1, dtype=np.float64)
    scale = (1.0e4 / np.maximum(lib, 1.0)).astype(np.float32)
    xf = np.log1p(z[:, features] * scale[:, None]).astype(np.float32, copy=False)
    p = np.asarray(clf.predict_probability(xf), dtype=np.float64)
    return apply_reward_mode(p, floor, mode)


def apply_reward_mode(p: np.ndarray, floor: float, mode: str) -> np.ndarray:
    """The mode rule alone, split out of `classifier_log_reward` (ticket 170).

    Split so the multi-component mixture reward (`hvg/multicluster.py`) applies the SAME
    arithmetic to its combined probability instead of writing a second copy. The body is the
    original's, moved not rewritten, so `classifier_log_reward` is unchanged bit for bit --
    and with one component the mixture path then reduces to it EXACTLY, which is what the
    k=1 degeneracy assertion checks.
    """
    p = np.asarray(p, dtype=np.float64)
    if mode == "probability":
        return np.log(np.clip(p, floor, 1.0))
    if mode == "odds":
        p = np.clip(p, floor, 1.0 - floor)
        return np.log(p) - np.log1p(-p)
    raise ValueError(f"reward_mode must be 'probability' or 'odds', got {mode!r}")


# --------------------------------------------------------------------------- the engine
class HVGEngine:
    """The frozen reverse-chain engine (posterior with the product tilt, bridge transition,
    reward pass) plus the single-parameter twist. `sampler='plugin'` reproduces the frozen
    intermediate potential; `sampler='twist'` replaces it and nothing else."""

    def __init__(self, BASE, net, bank, logpi: np.ndarray, tau: float, clf,
                 features: np.ndarray, cell_chunk: int, reward_floor: float,
                 torch_gen: torch.Generator, tilt_mode: str = "every",
                 sampler: str = "twist", J: int = 16, alpha: float = 1.0,
                 twist_gen: torch.Generator | None = None,
                 reward_mode: str = "odds", tilt_on: bool = True,
                 mixture=None):
        if tilt_mode not in ("last", "every"):
            raise ValueError("tilt_mode must be 'last' or 'every'")
        if sampler not in ("plugin", "twist", "tilt_only"):
            raise ValueError("sampler must be 'plugin', 'twist' or 'tilt_only'")
        if sampler == "twist" and J < 1:
            raise ValueError("J must be >= 1")
        if sampler != "tilt_only" and not alpha > 0:
            # ruling 164 section 2: alpha = 0 carries no information of its own --
            # rho^0 = 1 => psi_k = 1 => log psi_k = 0 => the terminal potential is 0 too
            # => the log-weights never move => ESS stays at M => it never resamples
            # => what is delivered IS the tilt proposal. That is the tilt_only arm exactly.
            raise ValueError("alpha=0 is exactly the tilt-only arm (psi == 1); "
                             "run --sampler tilt_only instead")
        self.BASE = BASE
        self.tilt_mode = tilt_mode
        # `tilt_on=False` is the residual_only ablation: NO product tilt at all. It is not the
        # same knob as tilt_mode, which only says WHEN the tilt is applied. The multiply is
        # skipped outright rather than done with a table of ones, so no pointless renormalise
        # can perturb p in the last ulp.
        self.tilt_on = bool(tilt_on)
        self.sampler = sampler
        self.J = int(J)
        self.alpha = float(alpha)
        self.reward_mode = reward_mode
        self.net = net
        self.bank = bank
        self.dev = bank.device
        self.logpi = torch.as_tensor(np.exp(float(tau) * np.asarray(logpi, dtype=np.float64)),
                                     dtype=torch.float32, device=self.dev)
        self.clf = clf
        # tilt_only evaluates no reward, so it carries no classifier and no feature index.
        # Keep them None rather than coercing: np.asarray(None, dtype=int64) raises, and a
        # silent placeholder would hide a mis-wired arm that DOES need a reward.
        self.features = None if features is None else np.asarray(features, dtype=np.int64)
        # ticket 170: a multi-component request replaces the single classifier with a
        # mixture of the components' frozen clean rewards. `last_assign` is the dominant
        # component of the CURRENT cloud, which the stratified resampler needs; it is only
        # ever written from particle-level reward calls, never from the twist candidates.
        self.mixture = mixture
        self.last_assign = None
        self.cell_chunk = int(cell_chunk)
        self.reward_floor = float(reward_floor)
        self.torch_gen = torch_gen
        self.twist_gen = twist_gen
        self.cgrid = torch.arange(bank.prior.shape[1], dtype=torch.float32, device=self.dev)
        self.discriminator_evals = 0
        self.gap_trace: list[dict] = []

    # ---- frozen: posterior with the product tilt, and the bridge transition ----
    def _posterior(self, state: torch.Tensor, step: int) -> torch.Tensor:
        b = state.shape[0]
        t = float(self.bank.tgrid[step])
        p = self.net.post(state.to(torch.float32), torch.full((b,), t, device=self.dev))
        if self.tilt_on and (self.tilt_mode == "every" or step == len(self.bank.tgrid) - 2):
            p = p * self.logpi[None]
            p = p / p.sum(-1, keepdim=True).clamp_min(1.0e-30)
        s = p.sum(-1, keepdim=True)
        return torch.where(s > 0, p / torch.clamp_min(s, 1.0e-30), p)

    def _transition(self, p: torch.Tensor, state: torch.Tensor, step: int) -> torch.Tensor:
        BASE = self.BASE
        b, dim, c = p.shape
        n0 = BASE._cat(p.reshape(b * dim, c), self.torch_gen).view(b, dim)
        del p
        knext = self.bank.Kgrid[step + 1]
        kdelt = self.bank.Kdelta_T[step]
        vi_next = BASE._slot_v_idx(knext, self.dev)[None, :].expand(b, dim)
        vi_delt = BASE._slot_v_idx(kdelt, self.dev)[None, :].expand(b, dim)
        row = knext.U[vi_next, n0]
        row.mul_(kdelt.U[vi_delt, state.clamp(0, c - 1)])
        out = BASE._cat(row.reshape(b * dim, c), self.torch_gen).view(b, dim)
        del row, n0
        return out

    def _log_reward(self, x) -> np.ndarray:
        if self.mixture is not None:
            return self.mixture.reward_and_assign(x, self.reward_floor, self.reward_mode)[0]
        if self.clf is None or self.features is None:
            raise SystemExit(
                f"sampler={self.sampler!r} asked for a reward but carries no classifier. "
                "tilt_only must never reach here; any other arm is mis-wired.")
        return classifier_log_reward(self.clf, self.features, x, self.reward_floor,
                                     self.reward_mode)

    def _log_reward_assign(self, x):
        """(log reward, dominant component or None). Called ONLY where `x` is the particle
        cloud itself -- the posterior mean in `pass_all` and the realised counts in
        `final_reward`. The twist's J candidate endpoints must NOT write `last_assign`: they
        are draws from a particle's posterior, not the particle."""
        if self.mixture is not None:
            return self.mixture.reward_and_assign(x, self.reward_floor, self.reward_mode)
        return self._log_reward(x), None

    # ---- the intermediate potential ----
    @torch.inference_mode()
    def pass_all(self, state: torch.Tensor, step: int, need_reward: bool,
                 do_transition: bool):
        """Returns (value/alpha, new_state). run_fk multiplies the first by alpha, so BOTH
        samplers return the potential DIVIDED by alpha -- see e38_twist docstring point 1."""
        n = len(state)
        value = np.empty(n, dtype=np.float64) if need_reward else None
        plugin = np.empty(n, dtype=np.float64) if (need_reward and self.sampler == "twist") \
            else None
        new_state = torch.empty_like(state) if do_transition else None
        if need_reward and self.mixture is not None:
            self.last_assign = np.empty(n, dtype=np.int64)
        J, a = self.J, self.alpha
        for lo in range(0, n, self.cell_chunk):
            hi = min(lo + self.cell_chunk, n)
            chunk = state[lo:hi]
            p = self._posterior(chunk, step)
            if need_reward:
                b, dim, c = p.shape
                xhat = torch.sum(p * self.cgrid[None, None, :], dim=-1)
                lp, asg = self._log_reward_assign(xhat.cpu().numpy())
                if asg is not None:
                    self.last_assign[lo:hi] = asg
                self.discriminator_evals += b
                if self.sampler == "plugin":
                    value[lo:hi] = lp               # run_fk restores the alpha
                else:
                    flat = p.reshape(b * dim, c)
                    # J candidate endpoints, ONE multinomial call, dedicated generator.
                    # Zero extra NETWORK cost -- p is already computed.
                    q = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                    tot = q.sum(dim=1)
                    bad = ~torch.isfinite(tot) | (tot <= 0)
                    if bad.any():                   # _cat's own fallback, verbatim
                        q = q.clone()
                        q[bad] = 0.0
                        q[bad, 0] = 1.0
                    draws = torch.multinomial(q, J, replacement=True, generator=self.twist_gen)
                    cand = (draws.transpose(0, 1).contiguous()
                            .view(J * b, dim).to(torch.int16).cpu().numpy())
                    del q, tot, bad
                    lr = self._log_reward(cand)
                    self.discriminator_evals += J * b
                    lr = lr.reshape(J, b)
                    # power first, then average, log last
                    m = (a * lr).max(axis=0)
                    log_psi = m + np.log(np.exp(a * lr - m[None, :]).mean(axis=0))
                    plugin[lo:hi] = a * lp
                    value[lo:hi] = log_psi / a      # run_fk restores the alpha
                    del draws, cand, flat, lr
                del xhat
            if do_transition:
                new_state[lo:hi] = self._transition(p, chunk, step)
            else:
                del p
        if need_reward and plugin is not None:
            gap = np.abs(plugin - a * np.asarray(value))
            self.gap_trace.append({
                "step": int(step), "t": float(self.bank.tgrid[step]),
                "abs_gap_mean": float(gap.mean()), "abs_gap_median": float(np.median(gap)),
                "abs_gap_p90": float(np.percentile(gap, 90)), "abs_gap_max": float(gap.max()),
                "signed_gap_mean": float((a * np.asarray(value) - plugin).mean())})
        return value, new_state

    def final_reward(self, x: np.ndarray) -> np.ndarray:
        """The terminal potential is EXACT: at t = 0 the state IS n0, no twist needed."""
        out = np.empty(len(x), dtype=np.float64)
        if self.mixture is not None:
            self.last_assign = np.empty(len(x), dtype=np.int64)
        for lo in range(0, len(x), self.cell_chunk):
            hi = min(lo + self.cell_chunk, len(x))
            out[lo:hi], asg = self._log_reward_assign(x[lo:hi])
            if asg is not None:
                self.last_assign[lo:hi] = asg
        self.discriminator_evals += len(x)
        return out


def initial_stationary(bank, n: int, rng: np.random.Generator) -> torch.Tensor:
    """Frozen: the initial state from the stationary law, gene by gene (the frozen order)."""
    prior = bank.prior.cpu().numpy()
    cdf = np.cumsum(prior, axis=1)
    dim = bank.dim
    x = np.empty((n, dim), dtype=np.int64)
    for d in range(dim):
        x[:, d] = np.searchsorted(cdf[d], rng.random(n) * cdf[d, -1],
                                  side="right").clip(0, prior.shape[1] - 1)
    return torch.as_tensor(x, dtype=torch.long, device=bank.device)


def make_twist_generator(dev, seed: int) -> torch.Generator:
    return torch.Generator(device=dev).manual_seed(int(seed) + TWIST_SEED_OFFSET)


# --------------------------------------------------------------------------- the sampler
def _unique_row_count(state: torch.Tensor) -> int:
    """ticket 182: how many of the M particle rows are distinct. PURE OBSERVATION.

    Reads the state, consumes nothing: no RNG of any kind is touched, no sampling function
    is called, and the caller's control flow does not branch on the result. The byte-view
    trick avoids the O(M^2) pairwise comparison.
    """
    a = state.detach().cpu().numpy()
    if a.dtype != np.int16:
        lo, hi = np.iinfo(np.int16).min, np.iinfo(np.int16).max
        if int(a.min()) < lo or int(a.max()) > hi:
            raise SystemExit("ancestry trace: a particle state does not fit int16, which the "
                             "frozen delivery cast assumes; refusing to guess")
        a = a.astype(np.int16)
    a = np.ascontiguousarray(a)
    return int(np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))).size)


def run_fk(engine: HVGEngine, n_particles: int, n_out: int, seed: int, alpha: float,
           ess_frac: float, resample_interval: int, enable_intermediate_resampling: bool,
           trigger: str = "scheduled", min_resample_gap: int = 2,
           verbose: bool = True, resample_fn=None, trace=None) -> tuple[np.ndarray, dict]:
    """Frozen `fk_hvg_v2.run_fk` with `intermediate_scale` removed (section B1).

    The weights telescope: log w_k = phi_k - phi_{k*} (k* = the last resample step, or none),
    so with no in-chain resample the delivered log-weight is EXACTLY alpha*log rho(n0) for any
    intermediate potential -- `telescope_max_abs_error` checks that on every such run (gate G1).
    """
    if trigger not in ("scheduled", "any"):
        raise ValueError("trigger must be 'scheduled' or 'any'")
    # ticket 170: the stratified-quota resampler is injected here rather than monkeypatched
    # onto the module, because run_fk resamples in TWO places -- in-chain and the final
    # N-of-M selection -- and the final one is what actually sets the delivered composition.
    # A parameter makes "both go through it" visible in the signature.
    resample = systematic_resample if resample_fn is None else resample_fn
    rng = np.random.default_rng(seed)
    state = initial_stationary(engine.bank, n_particles, rng)
    lineage = np.arange(n_particles, dtype=np.int64)
    # ticket 182: `lineage` IS the ancestry vector the frozen code already maintains -- it is
    # seeded here and re-indexed at every resample below. The trace only photographs it; it
    # adds no state the sampler reads back, so the run is bit-identical with or without it.
    if trace is not None:
        trace.update({"anc": [], "unique_per_step": [], "step": [], "t": []})
    logw = np.zeros(n_particles, dtype=np.float64)
    prev_value = np.zeros(n_particles, dtype=np.float64)
    ess_trace: list[dict] = []
    resample_steps: list[int] = []
    generator_evals = 0
    reward_evals = 0
    k_steps = len(engine.bank.tgrid) - 1

    last_resample_step = -10 ** 9
    for step in range(k_steps):
        need_reward = step > 0
        if trigger == "any":
            scheduled = need_reward and enable_intermediate_resampling
        else:
            scheduled = (need_reward and enable_intermediate_resampling
                         and (step + 1) % resample_interval == 0)
        if scheduled:
            value, _ = engine.pass_all(state, step, True, False)
            reward_evals += 1
            current_value = alpha * value          # section B1: no intermediate_scale
            logw += current_value - prev_value
            weight, ess = normalized(logw)
            ess_trace.append({"step": step, "t": float(engine.bank.tgrid[step]),
                              "ess_fraction": ess / n_particles, "scheduled": True})
            gap_ok = (step - last_resample_step) >= min_resample_gap
            if ess < ess_frac * n_particles and gap_ok:
                idx = resample(rng, weight)
                index_t = torch.as_tensor(idx, dtype=torch.long, device=engine.dev)
                state = state.index_select(0, index_t)
                current_value = current_value[idx]
                lineage = lineage[idx]
                logw.fill(0.0)
                resample_steps.append(step)
                last_resample_step = step
            prev_value = current_value
            _, new_state = engine.pass_all(state, step, False, True)
            generator_evals += 2
        else:
            value, new_state = engine.pass_all(state, step, need_reward, True)
            generator_evals += 1
            if need_reward:
                reward_evals += 1
                current_value = alpha * value
                logw += current_value - prev_value
                weight, ess = normalized(logw)
                ess_trace.append({"step": step, "t": float(engine.bank.tgrid[step]),
                                  "ess_fraction": ess / n_particles, "scheduled": False})
                prev_value = current_value
        del state
        state = new_state
        if trace is not None:
            trace["anc"].append(lineage.astype(np.int32))
            trace["unique_per_step"].append(_unique_row_count(state))
            trace["step"].append(int(step))
            trace["t"].append(float(engine.bank.tgrid[step]))
        if verbose and (step == 0 or (step + 1) % 4 == 0 or step + 1 == k_steps):
            msg = f"[FK] reverse {step + 1:02d}/{k_steps}"
            if ess_trace:
                msg += f"  ESS/N={ess_trace[-1]['ess_fraction']:.4f}"
            if resample_steps and resample_steps[-1] == step:
                msg += "  RESAMPLED"
            print(msg, flush=True)

    x0 = state.cpu().numpy().astype(np.int16, copy=False)
    del state
    final_reward = engine.final_reward(x0)
    logw += alpha * final_reward - prev_value
    final_weight, final_ess = normalized(logw)
    ess_trace.append({"step": k_steps, "t": 0.0, "ess_fraction": final_ess / n_particles,
                      "scheduled": False})
    telescope_error = None
    if not resample_steps:
        telescope_error = float(np.max(np.abs(logw - alpha * final_reward)))

    idx = resample(rng, final_weight, n_out)
    out = x0[idx]
    lineage = lineage[idx]
    if trace is not None:
        trace["delivered_idx"] = np.asarray(idx, dtype=np.int32)
    diag = {
        "method": ("sequential_Feynman_Kac_single_parameter_twist" if engine.sampler == "twist"
                   else "sequential_Feynman_Kac_difference_potential"),
        "proposal": "frozen_product_tilted_CRN_reverse_kernel",
        "reward": ("frozen_classifier_log_odds" if engine.reward_mode == "odds"
                   else "frozen_classifier_log_probability"),
        "sampler": engine.sampler,
        "reward_mode": engine.reward_mode,
        "J": (engine.J if engine.sampler == "twist" else None),
        "resample_trigger": trigger,
        "min_resample_gap": int(min_resample_gap),
        "n_particles": n_particles,
        "n_out": n_out,
        "alpha": alpha,
        "ess_threshold_fraction": ess_frac,
        "resample_interval": resample_interval,
        "intermediate_resampling": enable_intermediate_resampling,
        "n_intermediate_resamples": len(resample_steps),
        "resample_steps": resample_steps,
        "ess_trace": ess_trace,
        "ess_min_fraction": float(min(v["ess_fraction"] for v in ess_trace)),
        "ess_pre_final_fraction": float(final_ess / n_particles),
        "unique_lineage_fraction": float(np.unique(lineage).size / n_out),
        "unique_cell_fraction": float(np.unique(out, axis=0).shape[0] / n_out),
        "telescope_max_abs_error": telescope_error,
        "generator_forward_passes_per_particle": generator_evals,
        "reward_evaluations_per_particle": reward_evals + 1,
        "discriminator_evaluations_total": int(engine.discriminator_evals),
        "twist_plugin_gap_trace": (engine.gap_trace if engine.sampler == "twist" else None),
        "mixture_reward": (engine.mixture.identity() if engine.mixture is not None else None),
        "resampler": ("systematic" if resample_fn is None else "stratified_quota"),
    }
    if engine.mixture is not None:
        diag["reward"] = ("frozen_clean_mixture_log_odds" if engine.reward_mode == "odds"
                          else "frozen_clean_mixture_log_probability")
    return out, diag


def run_tilt_only(engine: HVGEngine, n_particles: int, n_out: int, seed: int,
                  verbose: bool = True) -> tuple[np.ndarray, dict]:
    """The pure tilt arm: the (tilted) frozen reverse chain with NO potential, NO in-chain
    resampling and NO terminal reweighting; delivery = a uniform stratified take of n_out.

        The tilt-only ablation arm, which doubles as the alpha = 0 arm: alpha = 0 gives rho^0 = 1,
        hence psi = 1, constant weights, no resampling ever, and the delivery is the tilt proposal.

    Consumes the main numpy stream in exactly `run_fk`'s order (initial state, then one final
    resample draw) and the torch stream in exactly its order (per step: n0 then bridge), so the
    proposal trajectory is bit-identical to an fk arm at the same seed -- the arms differ only
    in what they do with the weights. It builds NO potential-related stream: no twist generator,
    no discriminator call at all.
    """
    rng = np.random.default_rng(seed)
    state = initial_stationary(engine.bank, n_particles, rng)
    lineage = np.arange(n_particles, dtype=np.int64)
    k_steps = len(engine.bank.tgrid) - 1
    for step in range(k_steps):
        _, new_state = engine.pass_all(state, step, False, True)   # need_reward=False
        del state
        state = new_state
        if verbose and (step == 0 or (step + 1) % 4 == 0 or step + 1 == k_steps):
            print(f"[tilt_only] reverse {step + 1:02d}/{k_steps}", flush=True)
    x0 = state.cpu().numpy().astype(np.int16, copy=False)
    del state
    uniform = np.full(n_particles, 1.0 / n_particles, dtype=np.float64)
    idx = systematic_resample(rng, uniform, n_out)
    out = x0[idx]
    lineage = lineage[idx]
    diag = {
        "method": "tilt_only_proposal_no_potential",
        "proposal": "frozen_product_tilted_CRN_reverse_kernel",
        "reward": None, "sampler": "tilt_only", "reward_mode": None, "J": None,
        "resample_trigger": None, "min_resample_gap": None,
        "n_particles": n_particles, "n_out": n_out, "alpha": None,
        "ess_threshold_fraction": None, "resample_interval": None,
        "intermediate_resampling": False,
        "n_intermediate_resamples": 0, "resample_steps": [],
        "ess_trace": [{"step": k_steps, "t": 0.0, "ess_fraction": 1.0, "scheduled": False}],
        "ess_min_fraction": 1.0, "ess_pre_final_fraction": 1.0,
        "unique_lineage_fraction": float(np.unique(lineage).size / n_out),
        "unique_cell_fraction": float(np.unique(out, axis=0).shape[0] / n_out),
        "telescope_max_abs_error": None,
        "generator_forward_passes_per_particle": k_steps,
        "reward_evaluations_per_particle": 0,
        "discriminator_evaluations_total": 0,
        "twist_plugin_gap_trace": None,
        "ess_note": "no potential on this arm: the weights are uniform BY CONSTRUCTION, "
                    "ESS/M = 1 is not a measurement",
    }
    return out, diag


def middle_third_steps(K: int) -> list[int]:
    """The steps at which an intermediate potential is evaluated are 1 .. K-1; the corrupted
    set is the middle third of that list by count (the previous corruption pre-registration)."""
    evaluated = list(range(1, K))
    n = len(evaluated)
    lo, hi = n // 3, 2 * (n // 3)
    return sorted(evaluated[lo:hi])
