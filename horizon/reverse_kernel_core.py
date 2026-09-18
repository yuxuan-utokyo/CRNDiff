# -*- coding: utf-8 -*-
"""One-dimensional birth-death CRN diffusion: the exact Bayes solution, no sampling, no training.

The S = 1 case with k^- = 1 and k^+ = mu = E[n_0], so the mean is exactly conserved. The only
thing the horizon T controls is how bad it is to start the reverse chain from the stationary law.
"""
import numpy as np
from scipy.stats import binom, poisson
EPS = 1e-300

def q0_mixture(nmax, w, lo, hi):
    k = np.arange(nmax+1)
    p = w*poisson.pmf(k, lo) + (1-w)*poisson.pmf(k, hi)
    return p/p.sum()

def kernel(nmax, mu, t):
    """K[a, b] = P(n_t = b | n_0 = a), the one-dimensional form of the forward kernel."""
    b = float(np.exp(-t)); k = np.arange(nmax+1)
    birth = poisson.pmf(k, mu*(1.0-b))
    K = np.empty((nmax+1, nmax+1))
    for a in range(nmax+1):
        surv = np.zeros(nmax+1); surv[:a+1] = binom.pmf(np.arange(a+1), a, b)
        K[a] = np.convolve(surv, birth)[:nmax+1]
    return K

def stationary(nmax, mu):
    p = poisson.pmf(np.arange(nmax+1), mu); return p/p.sum()

def reverse_chain(q0, mu, T, K_steps, p_init):
    """Exact reverse kernel, propagated as a distribution rather than sampled. The grid is
    uniform and the spacing kernel is computed once.

    p_s = q_s * ( K_gap @ (p_t / q_t) )
    If p_init = q_T it returns q0, which is the implementation's own self-check.
    """
    nmax = q0.size-1
    Kg = kernel(nmax, mu, T/K_steps)
    marg = [q0.copy()]
    for _ in range(K_steps): marg.append(marg[-1] @ Kg)
    p = p_init.copy()
    for j in range(K_steps, 0, -1):
        qt = marg[j]
        keep = qt > qt.max()*1e-13
        r = np.zeros_like(qt); np.divide(p, qt, out=r, where=keep)
        p = marg[j-1] * (Kg @ r)
        p = np.maximum(p, 0.0); p /= p.sum()
    return p

def tv(a,b): return 0.5*float(np.abs(a-b).sum())

def tv_sampling_floor(q0, N, reps, rng):
    """Total variation between the empirical distribution of N samples and q0, 95th
    percentile: the floor below which a difference cannot be told apart."""
    k = np.arange(q0.size); v=[]
    for _ in range(reps):
        x = rng.choice(k, size=N, p=q0)
        v.append(tv(np.bincount(x, minlength=q0.size)/N, q0))
    return float(np.percentile(v,95))
