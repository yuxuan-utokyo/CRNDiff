# -*- coding: utf-8 -*-
"""E38 -- the twisted engine. `fk_hvg_v2.py` is a frozen ruler and is NOT edited;
this subclasses its engine and is injected by `e38_run.py`.

What changes, and only this: the INTERMEDIATE potential.

  baseline   log G_t = alpha * log rho(nbar_0)          <- plug-in, nbar_0 = E[n0|n_t]
  E38        log G_t = log E_{Q(n0|n_t)}[ rho(n0)^alpha ]

computed in the order the prereg fixes -- power first, then average, log last --
and evaluated stably as logsumexp(alpha * log rho_j) - log J.

Two implementation points that matter and are easy to get wrong:

1. `run_fk` multiplies whatever `pass_all` returns by `alpha`:
       current_value = alpha * intermediate_scale * reward
   log_psi ALREADY contains alpha. So `pass_all` returns log_psi / alpha, and
   run_fk's own multiplication restores it exactly. Returning log_psi directly
   would silently square alpha. (alpha > 0 is asserted at construction.)

2. The J candidate draws use a SEPARATE torch Generator. If they shared the
   sampler's generator, drawing J candidates would advance the RNG stream that
   produces the transition, and the E38 arms would follow a different proposal
   trajectory than the baseline for a reason having nothing to do with the
   twist. With a separate stream the two arms propose identically and differ
   only in their weights, which is the comparison the experiment is for.

The TERMINAL potential is untouched: at t=0 the state IS n0, so alpha*log rho(x0)
is already exact and needs no twist. `final_reward` is inherited unchanged.
"""
from __future__ import annotations

import numpy as np
import torch


def make_twisted_engine(base_cls, BASE):
    """Build the E38 engine class against the frozen module's own base."""

    class TwistedHVGFK(base_cls):
        def __init__(self, *args, J: int = 4, alpha: float = 1.0,
                     twist_seed: int = 0, **kw):
            super().__init__(*args, **kw)
            if J < 1:
                raise ValueError("J must be >= 1")
            if not alpha > 0:
                raise ValueError(
                    "E38 divides log_psi by alpha to cancel run_fk's own "
                    "multiplication; alpha must be strictly positive")
            self.J = int(J)
            self.alpha = float(alpha)
            # dedicated stream -- see module docstring point 2
            self.twist_gen = torch.Generator(device=self.dev).manual_seed(twist_seed)
            self.gap_trace: list[dict] = []
            self.discriminator_evals = 0

        @torch.inference_mode()
        def pass_all(self, state, step, need_reward, do_transition):
            n = len(state)
            reward = np.empty(n, dtype=np.float64) if need_reward else None
            plugin = np.empty(n, dtype=np.float64) if need_reward else None
            new_state = torch.empty_like(state) if do_transition else None
            J, a = self.J, self.alpha

            for lo in range(0, n, self.cell_chunk):
                hi = min(lo + self.cell_chunk, n)
                chunk = state[lo:hi]
                p = self._posterior(chunk, step)
                if need_reward:
                    b, dim, c = p.shape
                    flat = p.reshape(b * dim, c)
                    # J candidate endpoints; zero extra NETWORK cost -- p is
                    # already computed. This is the count-native property the
                    # experiment is built on.
                    #
                    # Drawn in ONE torch.multinomial call with num_samples=J
                    # rather than J calls to BASE._cat. Same mathematics -- J
                    # i.i.d. draws from the same per-row categorical -- but the
                    # sanitisation and the CDF are built once instead of J
                    # times. J separate calls made J=16 roughly 16x the cost of
                    # the transition draw, which alone would have put the J=16
                    # arm at ~2.3 h per run.
                    q = torch.nan_to_num(flat, nan=0.0, posinf=0.0,
                                         neginf=0.0).clamp_min(0.0)
                    tot = q.sum(dim=1)
                    bad = ~torch.isfinite(tot) | (tot <= 0)
                    if bad.any():                      # _cat's own fallback
                        q = q.clone()
                        q[bad] = 0.0
                        q[bad, 0] = 1.0
                    draws = torch.multinomial(q, J, replacement=True,
                                              generator=self.twist_gen)
                    cand = (draws.transpose(0, 1).contiguous()
                            .view(J * b, dim).to(torch.int16).cpu().numpy())
                    del q, tot, bad
                    lr = self.classifier_log_reward_local(cand)      # (J*b,)
                    self.discriminator_evals += J * b
                    lr = lr.reshape(J, b)
                    # power first, then average, log last
                    m = (a * lr).max(axis=0)
                    log_psi = m + np.log(np.exp(a * lr - m[None, :]).mean(axis=0))
                    # the plug-in, for the gap measurement demanded by the prereg
                    xhat = torch.sum(p * self.cgrid[None, None, :], dim=-1)
                    lp = self.classifier_log_reward_local(xhat.cpu().numpy())
                    self.discriminator_evals += b
                    plugin[lo:hi] = a * lp
                    reward[lo:hi] = log_psi / a       # run_fk restores the alpha
                    del xhat, draws, cand, flat
                if do_transition:
                    new_state[lo:hi] = self._transition(p, chunk, step)
                else:
                    del p

            if need_reward:
                gap = np.abs(plugin - a * np.asarray(reward))
                self.gap_trace.append({
                    "step": int(step),
                    "t": float(self.bank.tgrid[step]),
                    "abs_gap_mean": float(gap.mean()),
                    "abs_gap_median": float(np.median(gap)),
                    "abs_gap_p90": float(np.percentile(gap, 90)),
                    "abs_gap_max": float(gap.max()),
                    "signed_gap_mean": float((a * np.asarray(reward) - plugin).mean()),
                })
            return reward, new_state

        def classifier_log_reward_local(self, x):
            from fk_hvg_v2 import classifier_log_reward
            return classifier_log_reward(self.clf, self.features, x,
                                         self.reward_floor)

    return TwistedHVGFK
