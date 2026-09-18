# -*- coding: utf-8 -*-
"""The multi-component request arm, ported from the retired tree rather than newly written.

Original: `old/scRNA/condition/residual_v3/fk_hvg_multicluster.py`, sha256 below; a verbatim
sha256 `c15bc02bc98e2fced29f27e4ca484dc913564940a6adc66a94e5208171e6e1e7`;
copy is kept beside the frozen modules. It is itself an extension of `fk_hvg_v2.py`: run_fk, the
kernel bank, the transition and the posterior are all reused, and only two things change, the
reward and the proposal, both of which describe the REQUEST rather than the method.

Four things were carried over with their function bodies unchanged:

| Name | What it does |
|---|---|
| `_logsumexp` | the original's numerically safe logsumexp |
| `MixtureReward` | log r_mix = logsumexp_k(log w_k + log p_k), with p_k each component's frozen clean reward; the components use different 512-gene feature sets, so they are scored separately and combined in probability space |
| `build_mixture_tilt` | mixes the tempered tilted marginals gene by gene, logc_mix = log(sum_k w_k q_k) - log prior; the engine then runs at tau = 1.0, the temperature having already entered each component |
| `make_stratified_resample` | quota resampling by dominant component (largest remainder); both the intermediate resampling and the final N-of-M selection go through it |

# ## What was changed here, and why

1. Paths: the clean rewards resolve through `config.clean_reward_paths` and the component tilt
      tables through `config.TABLES_KZ`. The original's hard-coded locations do not exist here.
2. `reward_mode`: the original returns a log probability, while every arm of this round uses
      odds. So `log_reward` hands `r_mix` to `sampler.apply_reward_mode`, the same function the
      single-classifier path uses, instead of writing the clipping and the logarithm again. That is
      also why the k = 1 degeneracy assertion holds bitwise.
      For the same reason `mixture_probability` computes sum_k w_k * clip(p_k, floor, 1) in
      probability space rather than exponentiating a logsumexp: the two are mathematically equal,
      but at k = 1 `exp(log p)` is not bitwise equal to `p` and the assertion would only be
      approximate. The original's logsumexp form is kept as `log_mix_probability` and the two are
   checked against each other at run time (`MIX_FORM_TOL`).
   3. How the stratified resampler is installed: the original monkeypatched the module global; here
      the function is passed to `run_fk(..., resample_fn=...)`. The body is unchanged and only the
      installation differs. A module-global monkeypatch is too easy to miss in this package:
   `run_fk` has two call sites, one intermediate and one final, and both must go through it, so
      passing it as an argument makes that explicit in the signature.
   4. What was not carried over: the trajectory dumping used by an earlier PCA process figure. No
   figure in this round needs it and it is unrelated to this arm's numbers; the original itself
      argued that the dump is read-only and does not touch the RNG.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C

# the two forms of the mixture reward must agree; this is a consistency check on the port,
# not a tolerance on any result
MIX_FORM_TOL = 1.0e-9


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """Verbatim from fk_hvg_multicluster.py."""
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


class MixtureReward:
    """log( sum_k w_k p_k(x) ) from frozen per-type clean logistic rewards.

    Docstring line above and the featurisation below are the original's. The class keeps the
    original's `disc_override` branch (E18: a component may be backed by a per-request
    discriminator instead of a registered clean reward) even though ticket 170 does not use it
    -- dropping a branch would be a modification, not a port.
    """

    def __init__(self, types, weights, floor: float, disc_override=None):
        from .sampler import CleanLogisticReward, PortableHGBT      # local: avoid a cycle
        self.types = list(types)
        self.weights = np.asarray(weights, dtype=np.float64)
        if abs(self.weights.sum() - 1.0) > 1e-9:
            raise SystemExit("weights must sum to 1")
        self.floor = float(floor)
        self.models = []
        self.aucs = {}
        self.reward_kind = {}
        self.sources = {}
        disc_override = dict(disc_override or {})
        for _i, t in enumerate(self.types):
            # E18: a component may be backed by a PER-REQUEST discriminator instead of a
            # registered per-type clean reward. The override is keyed by COMPONENT INDEX and
            # only replaces the reward -- the type name is still the real one, because it also
            # selects the per-type TILT table.
            if _i in disc_override:
                ddir = Path(disc_override[_i])
                port = ddir / "residual_classifier_portable.npz"
                split = ddir / "split_indices.npz"
                log = json.loads((ddir / "residual_log.json").read_text(encoding="utf-8"))
                if not log["classifier"]["gate_passed"]:
                    raise SystemExit(f"{ddir.name}: discriminator AUC gate did not "
                                     f"pass ({log['classifier']['heldout_roc_auc']:.4f} "
                                     f"< {log['classifier']['auc_gate']}) -- refusing")
                for p_ in (port, split):
                    if not p_.exists():
                        raise SystemExit(f"missing per-request artefact: {p_}")
                clf = PortableHGBT(port)
                feats = np.load(split)["classifier_features"].astype(np.int64)
                if len(feats) != clf.n_features:
                    raise SystemExit(
                        f"{ddir.name}: feature table has {len(feats)} entries but "
                        f"the portable classifier expects {clf.n_features}")
                self.models.append((clf, feats))
                self.aucs[str(t)] = log["classifier"]["heldout_roc_auc"]
                self.reward_kind[str(t)] = "per_request_discriminator: " + ddir.name
                self.sources[str(t)] = str(port.resolve())
                continue
            npz, logp = C.clean_reward_paths(t)          # <- path adapted, see the module note
            C.require(npz, f"{t} clean reward classifier")
            C.require(logp, f"{t} clean reward log")
            meta = json.loads(Path(logp).read_text(encoding="utf-8"))
            if not meta["validation"]["gate_passed"]:
                raise SystemExit(
                    f"{t} clean reward validation gate did not pass "
                    f"(roc_auc {meta['validation']['roc_auc']} < "
                    f"{meta['validation']['auc_gate']}). Ticket 170 section 4: STOP and report "
                    "the number; do not run a component on a discriminator that did not pass.")
            clf = CleanLogisticReward(npz)
            self.models.append((clf, clf.features))
            self.aucs[t] = meta["validation"]["roc_auc"]
            self.reward_kind[t] = "registered_clean_reward"
            self.sources[t] = str(Path(npz).resolve())
        self.log_w = np.log(self.weights)

    # ---- the original's featurisation, unchanged ------------------------------------
    def component_probability(self, x: np.ndarray) -> np.ndarray:
        """(n_components, n_cells) p_k(x), each on its own 512-gene feature subset.

        Body is the original `component_log_p` up to (not including) its final log; splitting it
        lets the mixture be formed in probability space -- see the module note, item 2.
        """
        z = np.asarray(x, dtype=np.float32)
        lib = z.sum(axis=1, dtype=np.float64)
        scale = (1.0e4 / np.maximum(lib, 1.0)).astype(np.float32)
        out = np.empty((len(self.models), z.shape[0]), dtype=np.float64)
        for i, (clf, feats) in enumerate(self.models):
            xf = np.log1p(z[:, feats] * scale[:, None]).astype(np.float32, copy=False)
            p = np.asarray(clf.predict_probability(xf), dtype=np.float64)
            out[i] = np.clip(p, self.floor, 1.0)
        return out

    def component_log_p(self, x: np.ndarray) -> np.ndarray:
        """(n_components, n_cells) log p_k(x) -- the original's own return value."""
        return np.log(self.component_probability(x))

    def log_mix_probability(self, x: np.ndarray) -> np.ndarray:
        """The original `log_reward`: logsumexp_k(log w_k + log p_k)."""
        return _logsumexp(self.component_log_p(x) + self.log_w[:, None], axis=0)

    # ---- what this package's engine actually calls ----------------------------------
    def reward_and_assign(self, x: np.ndarray, floor: float, mode: str):
        """(log reward under `mode`, dominant component per cell).

        The assignment is `argmax_k log p_k(x)` -- the component likelihood alone, deliberately
        weight-free (PREREG 6b/D2 item 2), so the weights cannot move the strata as well as the
        quota. Verbatim intent from the original `_reward_and_assign`.
        """
        from .sampler import apply_reward_mode
        p = self.component_probability(x)
        assign = np.argmax(p, axis=0)          # argmax of p == argmax of log p, monotone
        r_mix = self.weights @ p               # sum_k w_k p_k, exact when k == 1
        chk = float(np.max(np.abs(np.log(r_mix)
                                  - _logsumexp(np.log(p) + self.log_w[:, None], axis=0))))
        if not chk <= MIX_FORM_TOL:
            raise SystemExit(f"the two forms of the mixture reward disagree by {chk:.3e} "
                             f"(> {MIX_FORM_TOL}); the port is wrong, stopping")
        return apply_reward_mode(r_mix, floor, mode), assign

    def identity(self) -> dict:
        return {"types": self.types, "weights": self.weights.tolist(),
                "combine": "log r_mix = logsumexp_k(log w_k + log p_k)",
                "reward_kind": self.reward_kind, "heldout_auc": self.aucs,
                "sources": self.sources, "reward_floor": self.floor,
                "assignment_rule": "argmax_k log p_k (weight-free, PREREG 6b/D2 item 2)",
                "mixture_form_check_tol": MIX_FORM_TOL}


def build_mixture_tilt(prior: np.ndarray, logc_list, weights, tau: float):
    """Per-gene mixture of tempered tilted marginals; see PREREG section 4. VERBATIM.

    Done entirely in log space. The direct form `prior * exp(tau * logc)` underflows to exactly
    0 wherever the stationary prior is tiny (the Poisson tail runs to ~1e-300 at large counts),
    and the subsequent `log` then returns -inf on entries that are perfectly well defined as
    logs.
    """
    pos = prior > 0
    with np.errstate(divide="ignore"):
        logprior = np.where(pos, np.log(np.where(pos, prior, 1.0)), -np.inf)

    log_q = []
    for logc in logc_list:
        lq = logprior + tau * logc              # unnormalised log tilted marginal
        lq = np.where(pos, lq, -np.inf)
        z = _logsumexp(lq, axis=1)[:, None]     # per gene
        if not np.isfinite(z).all():
            raise SystemExit("a component tilt integrates to zero on some gene")
        log_q.append(lq - z)

    stack = np.stack([np.log(w) + lq for w, lq in zip(weights, log_q)], axis=0)
    log_qmix = _logsumexp(stack, axis=0)

    logc_mix = np.zeros_like(prior, dtype=np.float64)
    logc_mix[pos] = log_qmix[pos] - logprior[pos]
    # where the stationary prior has no mass the tilt is a no-op, not -inf
    logc_mix[~pos] = 0.0
    if not np.isfinite(logc_mix).all():
        raise SystemExit("non-finite mixture tilt table")
    # the engine exponentiates this in float64 then casts to float32
    if np.abs(logc_mix).max() > 80.0:
        raise SystemExit(f"mixture tilt out of safe range "
                         f"(max |logc_mix| = {np.abs(logc_mix).max():.1f})")
    return logc_mix, log_qmix


def make_stratified_resample(engine, targets, log: list, base):
    """D2: quota resampling by dominant component. See PREREG 6b/D2. VERBATIM body.

    Replaces `systematic_resample`, which `run_fk` uses for BOTH the intermediate resamples and
    the final N-of-M selection -- and the final selection is the one that actually sets the
    delivered composition, so both have to go through here.

    Nothing is silently topped up: a stratum with fewer particles than its quota is still filled
    (with repeats, by weight) and the shortfall is recorded, and a completely empty stratum's
    quota is drawn from the global pool and that is recorded too. A composition hit by cloning
    three particles is not a success and the diagnostics have to say so.

    (`base` is `sampler.systematic_resample`, passed in rather than reached for through a module
    global -- see the module note, item 3.)
    """
    tg = np.asarray(targets, dtype=np.float64)

    def stratified(rng, weight, n_out=None):
        n = len(weight) if n_out is None else int(n_out)
        assign = engine.last_assign
        w_all = np.asarray(weight, dtype=np.float64)
        if assign is None or len(assign) != len(w_all):
            log.append({"n": n, "fallback": "no current assignment"})
            return base(rng, weight, n_out)

        raw = tg * n                      # largest-remainder integer quota
        q = np.floor(raw).astype(np.int64)
        short = int(n - q.sum())
        if short > 0:
            q[np.argsort(-(raw - q))[:short]] += 1

        idx, rec = [], {"n": n, "quota": q.tolist(), "strata": []}
        for k in range(len(tg)):
            mem = np.flatnonzero(assign == k)
            e = {"k": k, "n_members": int(len(mem)), "quota": int(q[k])}
            if q[k] == 0:
                rec["strata"].append(e)
                continue
            if len(mem) == 0:
                idx.append(base(rng, w_all / w_all.sum(), int(q[k])))
                e["empty_filled_from_pool"] = int(q[k])
                rec["strata"].append(e)
                continue
            w = w_all[mem]
            s = w.sum()
            w = w / s if s > 0 else np.full(len(mem), 1.0 / len(mem))
            e["within_ess_fraction"] = float(1.0 / np.square(w).sum() / len(mem))
            e["shortfall"] = int(max(0, int(q[k]) - len(mem)))
            idx.append(mem[base(rng, w, int(q[k]))])
            rec["strata"].append(e)
        log.append(rec)
        return np.sort(np.concatenate(idx))

    return stratified
