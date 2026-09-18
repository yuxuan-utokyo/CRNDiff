# 搬自 cc/experiments/EXP08_toy_countsdiff/metrics.py（整文件），改动：仅在文件顶部加入本行注释；
# 其余每一个字节逐字未动（GTReference / make_projections / swd / w1_1d / mmd_rbf /
# median_heuristic_gamma / marginal_gammas 全部原样）。EXP134 中以 `import metrics as MET` 使用。
"""Metrics for EXP08 — all computed vs the 200k ground-truth resample.

Joint:   RBF-MMD (fixed median-heuristic bandwidth per dataset) + gamma=1 MMD
         (codex convention, for continuity); SWD (sliced W1, 100 projections).
Marginal (per dim): MMD (1-D), W1; variance vs truth.
Sparsity/tail: zero_frac (overall + per dim), gen_max (per dim), P(n>k) at
         k = high GT quantile, OOD_rate (fraction of gen rows with any dim > GT dim-max).

Lower is better unless noted. Bandwidth / projection directions are fixed per dataset
(derived from ground truth) so every method is scored with the SAME kernel/projections.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Kernel / bandwidth helpers
# --------------------------------------------------------------------------- #
def median_heuristic_gamma(ref: np.ndarray, *, max_points: int = 2000, seed: int = 314159) -> float:
    """gamma = 1 / (2 * median_sq_dist) on a subsample of the reference (GT)."""
    rng = np.random.default_rng(seed)
    r = ref.astype(np.float64)
    if len(r) > max_points:
        r = r[rng.choice(len(r), size=max_points, replace=False)]
    d2 = np.sum((r[:, None, :] - r[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices(len(r), k=1)
    med = np.median(d2[iu])
    med = max(med, 1e-8)
    return float(1.0 / (2.0 * med))


def _mmd2_rbf(x: np.ndarray, y: np.ndarray, gamma: float, *, max_points: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if len(x) > max_points:
        x = x[rng.choice(len(x), size=max_points, replace=False)]
    if len(y) > max_points:
        y = y[rng.choice(len(y), size=max_points, replace=False)]
    if x.ndim == 1:
        x = x[:, None]
        y = y[:, None]
    xx = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    yy = np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    xy = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    m = np.exp(-gamma * xx).mean() + np.exp(-gamma * yy).mean() - 2.0 * np.exp(-gamma * xy).mean()
    return float(m)


def mmd_rbf(x: np.ndarray, y: np.ndarray, *, gamma: float, max_points: int = 1000, seed: int = 271828) -> float:
    return float(np.sqrt(max(_mmd2_rbf(x, y, gamma, max_points=max_points, seed=seed), 0.0)))


# --------------------------------------------------------------------------- #
# SWD (sliced Wasserstein-1, Bonneel 2015 style)
# --------------------------------------------------------------------------- #
def make_projections(dim: int, n_proj: int = 100, *, seed: int = 20260625) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n_proj, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def w1_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein-1 between empirical samples via sorted quantile coupling."""
    a = np.sort(a.astype(np.float64))
    b = np.sort(b.astype(np.float64))
    n = max(len(a), len(b))
    qa = np.interp(np.linspace(0, 1, n, endpoint=False) + 0.5 / n,
                   np.linspace(0, 1, len(a), endpoint=False) + 0.5 / len(a), a)
    qb = np.interp(np.linspace(0, 1, n, endpoint=False) + 0.5 / n,
                   np.linspace(0, 1, len(b), endpoint=False) + 0.5 / len(b), b)
    return float(np.mean(np.abs(qa - qb)))


def swd(x: np.ndarray, y: np.ndarray, projections: np.ndarray, *, max_points: int = 8000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    if len(x) > max_points:
        x = x[rng.choice(len(x), size=max_points, replace=False)]
    if len(y) > max_points:
        y = y[rng.choice(len(y), size=max_points, replace=False)]
    xp = x.astype(np.float64) @ projections.T   # [n, n_proj]
    yp = y.astype(np.float64) @ projections.T
    ds = [w1_1d(xp[:, j], yp[:, j]) for j in range(projections.shape[1])]
    return float(np.mean(ds))


# --------------------------------------------------------------------------- #
# 1-D marginal MMD (per coordinate), with per-dim bandwidth from GT
# --------------------------------------------------------------------------- #
def marginal_gammas(gt: np.ndarray, *, seed: int = 99) -> np.ndarray:
    dim = gt.shape[1]
    gammas = np.zeros(dim)
    for d in range(dim):
        col = gt[:, d].astype(np.float64)
        rng = np.random.default_rng(seed + d)
        sub = col[rng.choice(len(col), size=min(2000, len(col)), replace=False)]
        d2 = (sub[:, None] - sub[None, :]) ** 2
        iu = np.triu_indices(len(sub), k=1)
        med = max(np.median(d2[iu]), 1e-8)
        gammas[d] = 1.0 / (2.0 * med)
    return gammas


# --------------------------------------------------------------------------- #
# Full metric bundle
# --------------------------------------------------------------------------- #
class GTReference:
    """Precomputes per-dataset fixed kernel/projection state from ground truth."""

    def __init__(self, gt: np.ndarray, *, n_proj: int = 100, kquant: float = 0.99):
        self.gt = gt.astype(np.int64)
        self.dim = gt.shape[1]
        self.gamma_joint = median_heuristic_gamma(gt)
        self.gammas_marg = marginal_gammas(gt)
        self.projections = make_projections(self.dim, n_proj=n_proj)
        self.gt_max = gt.max(0)                              # per-dim max
        self.gt_var = gt.var(0)
        self.gt_zero = (gt == 0).mean(0)
        self.gt_zero_overall = float((gt == 0).mean())
        self.kquant = kquant
        self.k_thresh = np.quantile(gt, kquant, axis=0)      # per-dim high quantile

    def evaluate(self, gen: np.ndarray) -> dict[str, Any]:
        gen = gen.astype(np.int64)
        gt = self.gt
        dim = self.dim

        joint_mmd = mmd_rbf(gen, gt, gamma=self.gamma_joint, max_points=1000)
        joint_mmd_g1 = mmd_rbf(gen, gt, gamma=1.0, max_points=1000)
        swd_val = swd(gen, gt, self.projections)

        marg_mmd, marg_w1, gen_var, var_relerr = [], [], [], []
        for d in range(dim):
            marg_mmd.append(mmd_rbf(gen[:, d], gt[:, d], gamma=float(self.gammas_marg[d]), max_points=1000))
            marg_w1.append(w1_1d(gen[:, d], gt[:, d]))
            gv = float(gen[:, d].var())
            gen_var.append(gv)
            var_relerr.append(float((gv - self.gt_var[d]) / max(self.gt_var[d], 1e-9)))

        gen_zero = (gen == 0).mean(0)
        gen_max = gen.max(0)
        # --- EQUAL-N max (fix cross-N pitfall: 40k gen vs 200k GT max not comparable) ---
        # subsample GT to gen's N (fixed seed) so per-dim max is an apples-to-apples
        # order statistic at matched sample size.
        rng_en = np.random.default_rng(123457)
        n_gen = gen.shape[0]
        if len(gt) >= n_gen:
            gt_sub = gt[rng_en.choice(len(gt), size=n_gen, replace=False)]
        else:
            gt_sub = gt
        gt_max_equalN = gt_sub.max(0)
        # P(n>k): per-dim probability mass above GT high quantile, gen vs gt (probabilities
        # are N-invariant, so these survival points ARE comparable across N).
        p_gt_k = np.array([(gt[:, d] > self.k_thresh[d]).mean() for d in range(dim)])
        p_gen_k = np.array([(gen[:, d] > self.k_thresh[d]).mean() for d in range(dim)])
        # survival curve P(n>k) at a sweep of k (use heavy-tail-friendly thresholds from GT)
        ks = np.unique(np.quantile(gt, [0.90, 0.95, 0.99, 0.999], axis=0).round().astype(int).ravel())
        surv_gen = {int(k): float((gen > k).mean()) for k in ks}
        surv_gt = {int(k): float((gt > k).mean()) for k in ks}
        # OOD: any coordinate strictly exceeds the GT per-dim max (200k GT max = strict ref)
        ood_rows = np.any(gen > self.gt_max[None, :], axis=1)
        ood_rate = float(ood_rows.mean())
        ood_per_dim = (gen > self.gt_max[None, :]).mean(0)
        # OOD vs equal-N GT max (fairer tail-coverage read)
        ood_rate_equalN = float(np.any(gen > gt_max_equalN[None, :], axis=1).mean())

        return {
            "joint_mmd": joint_mmd,
            "joint_mmd_gamma1": joint_mmd_g1,
            "swd": swd_val,
            "marginal_mmd_per_dim": [float(v) for v in marg_mmd],
            "marginal_mmd_mean": float(np.mean(marg_mmd)),
            "marginal_w1_per_dim": [float(v) for v in marg_w1],
            "marginal_w1_mean": float(np.mean(marg_w1)),
            "gen_var_per_dim": gen_var,
            "gt_var_per_dim": self.gt_var.tolist(),
            "var_relerr_per_dim": var_relerr,
            "var_relerr_absmean": float(np.mean(np.abs(var_relerr))),
            "zero_frac_overall_gen": float((gen == 0).mean()),
            "zero_frac_overall_gt": self.gt_zero_overall,
            "zero_frac_per_dim_gen": gen_zero.tolist(),
            "zero_frac_per_dim_gt": self.gt_zero.tolist(),
            "gen_max_per_dim": gen_max.tolist(),
            "gt_max_per_dim": self.gt_max.tolist(),
            "gt_max_per_dim_full": self.gt_max.tolist(),
            "gt_max_per_dim_equalN": gt_max_equalN.tolist(),
            "n_gen": int(n_gen),
            "p_above_k_gen_per_dim": p_gen_k.tolist(),
            "p_above_k_gt_per_dim": p_gt_k.tolist(),
            "k_thresh_per_dim": self.k_thresh.tolist(),
            "survival_gen": surv_gen,
            "survival_gt": surv_gt,
            "ood_rate": ood_rate,  # vs full 200k GT per-dim max (back-compat)
            "ood_rate_vs_fullGTmax": ood_rate,
            "ood_rate_vs_equalNmax": ood_rate_equalN,
            "ood_per_dim": ood_per_dim.tolist(),
        }
