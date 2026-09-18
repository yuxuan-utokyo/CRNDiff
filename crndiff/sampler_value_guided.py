# -*- coding: utf-8 -*-
"""Value-guided candidate resampling, SVDD-style (Li et al., 2024).

Naming: this does not modify the CTMC rates (Nisonoff et al., 2024) and is not the exact
transition reweighting of Schiff et al. (2024). Each particle picks one of J candidates by a
multinomial draw on a classifier value, which is a self-normalised importance approximation to

It imports `crndiff/sampler.py` and `crndiff/frozen/` and modifies neither: the posterior, the
bridge, the initial state and `_cat` all come from those modules as they are.

At each reverse step the proposal chain is the untilted frozen reverse chain (tau = 0):

        p        = untilted clean-state posterior q(n0 | n_t)  (one generator forward, so the NFE
        n0^(j)   = J candidate endpoints drawn from p (one multinomial, dedicated generator)
        n_s^(j)  = exact bridge K_next[n0^(j)] . K_deltaT[n_t], drawn once more
    w_j      ∝ score(candidate_j)^gamma
        n_s      = one candidate chosen by a multinomial draw (no cross-particle weight, no

Two scorers:

    scorer='noised'  w_j proportional to p_noised(type | n_s^(j), s)^gamma, scored in the noised
    scorer='clean'   w_j proportional to p_S1(type | n0^(j))^gamma, scored in the clean state

For `clean` the weight does not depend on n_s, so choosing j* from n0 and then drawing the
bridge only for j* is the same law as drawing J bridges and then choosing: the bridge draws of
candidates that are not chosen cannot affect the result. This file does that, saving J-1 bridge
draws. It is an identity, not a methodological simplification, and the run json records whether
`bridge_draws_per_step` was 1 or J.
With gamma = 0 every weight is equal, so the chain is the unconditional one; gate B-V1 uses that.
"""
from __future__ import annotations

import numpy as np
try:
    import torch
except ModuleNotFoundError as _exc:                                      # noqa: E402
    raise SystemExit(
        "this entry point needs PyTorch, which the offline reproduction does not "
        "install. See README section 7; in short: pip install torch") from _exc

from . import sampler as SP


class VGRSampler:
    """Value-guided candidate resampling. Every constructor argument is explicit; no global state."""

    def __init__(self, BASE, net, bank, dmax_t, scorer, J: int = 16, gamma: float = 1.0,
                 cell_chunk: int = 256, torch_gen: torch.Generator | None = None,
                 twist_gen: torch.Generator | None = None, no_a4: bool = True):
        self.BASE = BASE                 # the frozen sample_hvg2k_noa4 module (for _cat and _slot_v_idx)
        self.net = net
        self.bank = bank
        self.dev = bank.device
        self.dmax_t = dmax_t
        self.scorer = scorer             # callable(kind, arrays...) -> np.ndarray of probability
        self.J = int(J)
        self.gamma = float(gamma)
        self.cell_chunk = int(cell_chunk)
        self.torch_gen = torch_gen
        self.twist_gen = twist_gen
        self.no_a4 = bool(no_a4)
        self.aux_calls = 0
        self.nfe = 0

    # ---------------------------------------------------------------- posterior (untilted)
    def _posterior(self, state: torch.Tensor, step: int) -> torch.Tensor:
        b = state.shape[0]
        t = float(self.bank.tgrid[step])
        p = self.net.post(state.to(torch.float32), torch.full((b,), t, device=self.dev))
        self.nfe += 1                                  # one generator forward per chunk
        if not self.no_a4:
            p = p.masked_fill(torch.arange(p.shape[2], device=self.dev)[None, None, :]
                              > self.dmax_t[None, :, None], 0.0)
        s = p.sum(-1, keepdim=True)
        return torch.where(s > 0, p / torch.clamp_min(s, 1.0e-30), p)

    def _candidates(self, p: torch.Tensor) -> torch.Tensor:
        """J candidate endpoints, one multinomial, on the dedicated generator (as in e38_twist)."""
        b, dim, c = p.shape
        q = torch.nan_to_num(p.reshape(b * dim, c), nan=0.0, posinf=0.0,
                             neginf=0.0).clamp_min(0.0)
        tot = q.sum(dim=1)
        bad = ~torch.isfinite(tot) | (tot <= 0)
        if bad.any():
            q = q.clone()
            q[bad] = 0.0
            q[bad, 0] = 1.0
        draws = torch.multinomial(q, self.J, replacement=True, generator=self.twist_gen)
        return draws.transpose(0, 1).contiguous().view(self.J, b, dim)   # [J, b, dim]

    def _bridge(self, n0: torch.Tensor, state: torch.Tensor, step: int) -> torch.Tensor:
        """Exact bridge K_next[n0] . K_deltaT[n_t], drawn once. Verbatim from sampler._transition."""
        BASE = self.BASE
        b, dim = n0.shape
        c = self.bank.prior.shape[1]
        knext = self.bank.Kgrid[step + 1]
        kdelt = self.bank.Kdelta_T[step]
        vi_next = BASE._slot_v_idx(knext, self.dev)[None, :].expand(b, dim)
        vi_delt = BASE._slot_v_idx(kdelt, self.dev)[None, :].expand(b, dim)
        row = knext.U[vi_next, n0]
        row.mul_(kdelt.U[vi_delt, state.clamp(0, c - 1)])
        out = BASE._cat(row.reshape(b * dim, c), self.torch_gen).view(b, dim)
        del row
        return out

    def _choose(self, w: np.ndarray) -> np.ndarray:
        """Choose one candidate index per particle by a multinomial draw on w ([J, b])."""
        J, b = w.shape
        ww = np.asarray(w, dtype=np.float64).T                    # [b, J]
        tot = ww.sum(axis=1, keepdims=True)
        bad = (~np.isfinite(tot[:, 0])) | (tot[:, 0] <= 0)
        if bad.any():
            ww[bad] = 0.0
            ww[bad, 0] = 1.0
            tot = ww.sum(axis=1, keepdims=True)
        t = torch.as_tensor(ww / tot, dtype=torch.float64, device=self.dev)
        pick = torch.multinomial(t, 1, generator=self.torch_gen).squeeze(1)
        return pick.detach().cpu().numpy()

    # ---------------------------------------------------------------- one whole chain
    @torch.inference_mode()
    def run(self, n_particles: int, seed: int, kind: str, verbose: bool = True):
        rng = np.random.default_rng(seed)
        state = SP.initial_stationary(self.bank, n_particles, rng)
        n_steps = len(self.bank.tgrid) - 1
        bridge_draws = 1 if kind == "clean" else self.J
        diag = {"per_step_mean_max_weight": [], "per_step_mean_ess_J": []}
        for step in range(n_steps):
            new = torch.empty_like(state)
            for lo in range(0, n_particles, self.cell_chunk):
                hi = min(lo + self.cell_chunk, n_particles)
                chunk = state[lo:hi]
                b = hi - lo
                p = self._posterior(chunk, step)
                cand = self._candidates(p)                        # [J, b, dim]
                del p
                if kind == "clean":
                    # the weight sees n0 only: choose first, then draw one bridge (an identity)
                    sc = self.scorer("clean", cand.to(torch.int16).cpu().numpy()
                                     .reshape(self.J * b, -1), step)
                    self.aux_calls += self.J * b
                    w = np.power(np.asarray(sc, dtype=np.float64).reshape(self.J, b),
                                 self.gamma)
                    pick = self._choose(w)
                    n0 = cand[pick, np.arange(b)]                 # [b, dim]
                    new[lo:hi] = self._bridge(n0, chunk, step)
                else:
                    # the weight sees n_s: draw all J bridges, then choose
                    bridged = torch.empty((self.J, b, state.shape[1]), dtype=state.dtype,
                                          device=self.dev)
                    for j in range(self.J):
                        bridged[j] = self._bridge(cand[j], chunk, step)
                    # the noised scorer lives on the GPU, so feed it tensors directly
                    sc = self.scorer("noised", bridged.reshape(self.J * b, -1), step + 1)
                    self.aux_calls += self.J * b
                    w = np.power(np.asarray(sc, dtype=np.float64).reshape(self.J, b),
                                 self.gamma)
                    pick = self._choose(w)
                    new[lo:hi] = bridged[pick, np.arange(b)]
                    del bridged
                wn = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-300)
                diag["per_step_mean_max_weight"].append(float(wn.max(axis=0).mean()))
                diag["per_step_mean_ess_J"].append(float((1.0 / np.square(wn).sum(axis=0)).mean()))
                del cand, w
            state = new
            if verbose:
                print(f"  step {step+1}/{n_steps} maxw={diag['per_step_mean_max_weight'][-1]:.3f} "
                      f"essJ={diag['per_step_mean_ess_J'][-1]:.2f}", flush=True)
        out = state.cpu().numpy().astype(np.int16)
        diag["bridge_draws_per_step_per_particle"] = bridge_draws
        diag["J"] = self.J
        diag["gamma"] = self.gamma
        diag["scorer"] = kind
        diag["generator_forward_calls"] = self.nfe
        diag["aux_calls"] = self.aux_calls
        return out, diag
