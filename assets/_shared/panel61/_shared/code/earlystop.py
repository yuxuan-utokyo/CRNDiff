# 新建（EXP134 / EXP23 均无对应物）—— 实现任务书 §5.1 的双判据 early stopping。改动：全新文件。
# 唯一需要说明的自定义量是 sliced-W2 探针：它**只用于 early-stop 与 ckpt 选择，从不进任何结果表**，
# 定义固定为「log1p-CPM 空间、固定 200 个固定种子随机投影、按分位耦合的 2-Wasserstein」，
# 投影种子写死 20260136，n_probe=2000，全程只用 train/val，不碰 test。
"""双判据 early stopping：native 判据 + val sliced-W2 生成判据。"""
from __future__ import annotations

import numpy as np

PROBE_PROJ_SEED = 20260136
N_PROBE = 2000
N_PROJ_PROBE = 200


def _cpm_log1p(x):
    x = np.asarray(x, float); s = x.sum(1, keepdims=True); s[s == 0] = 1
    return np.log1p(x / s * 1e4)


def make_probe_projections(dim):
    rng = np.random.default_rng(PROBE_PROJ_SEED)
    v = rng.standard_normal((N_PROJ_PROBE, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def sliced_w2(gen, ref, projections):
    """固定投影的 sliced 2-Wasserstein（分位耦合）。只作 early-stop 探针。"""
    g = _cpm_log1p(gen) @ projections.T
    r = _cpm_log1p(ref) @ projections.T
    n = min(len(g), len(r))
    q = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    tot = 0.0
    for j in range(projections.shape[0]):
        a = np.sort(g[:, j]); b = np.sort(r[:, j])
        qa = np.interp(q, np.linspace(0, 1, len(a), endpoint=False) + 0.5 / len(a), a)
        qb = np.interp(q, np.linspace(0, 1, len(b), endpoint=False) + 0.5 / len(b), b)
        tot += float(np.mean((qa - qb) ** 2))
    return float(np.sqrt(tot / projections.shape[0]))


class DualEarlyStop:
    """两个判据各自维护 best；**两个都连续 patience 次没改善**才停。任一还在改善就继续训。

    w2_enabled=False（[DEGRADED-EARLYSTOP]，如 P0-DEFER）时退化为只用 native 判据，patience 加倍。
    """

    def __init__(self, patience, w2_enabled=True, w2_every=4, min_rel=0.0):
        self.w2_enabled = bool(w2_enabled)
        self.patience = int(patience) * (1 if w2_enabled else 2)
        self.w2_every = int(w2_every)
        self.min_rel = float(min_rel)
        self.best_native = float("inf"); self.bad_native = 0; self.best_native_step = -1
        self.best_w2 = float("inf"); self.bad_w2 = 0; self.best_w2_step = -1
        self.n_native_evals = 0
        self.curve = {"step": [], "native": [], "w2": []}

    def due_for_w2(self):
        return self.w2_enabled and (self.n_native_evals % self.w2_every == 0)

    def update(self, step, native, w2=None):
        """返回 (improved_native, improved_w2, should_stop)。"""
        self.n_native_evals += 1
        imp_n = native < self.best_native * (1.0 - self.min_rel)
        if imp_n:
            self.best_native = float(native); self.bad_native = 0; self.best_native_step = int(step)
        else:
            self.bad_native += 1
        imp_w = False
        if w2 is not None:
            imp_w = w2 < self.best_w2 * (1.0 - self.min_rel)
            if imp_w:
                self.best_w2 = float(w2); self.bad_w2 = 0; self.best_w2_step = int(step)
            else:
                self.bad_w2 += 1
        self.curve["step"].append(int(step))
        self.curve["native"].append(float(native))
        self.curve["w2"].append(None if w2 is None else float(w2))
        if not self.w2_enabled:
            stop = self.bad_native >= self.patience
        else:
            # w2 评得稀疏，用「w2 的 patience 按 w2 评测次数计」：两个都耗尽才停
            stop = (self.bad_native >= self.patience) and (self.bad_w2 >= max(2, self.patience // self.w2_every))
        return imp_n, imp_w, bool(stop)

    def summary(self):
        return {"best_native": self.best_native, "best_native_step": self.best_native_step,
                "best_w2": (None if self.best_w2 == float("inf") else self.best_w2),
                "best_w2_step": self.best_w2_step, "patience": self.patience,
                "w2_enabled": self.w2_enabled, "n_native_evals": self.n_native_evals,
                "curve_tail": {k: v[-10:] for k, v in self.curve.items()}}
