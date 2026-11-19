# new (no counterpart in EXP134 / EXP23): implements the two-criterion early stopping of spec section 5.1. Change: new file.
# the only custom quantity to explain is the sliced-W2 probe: it is **used only for early stopping and ckpt selection, never in any result table**,
# definition fixed as 2-Wasserstein in log1p-CPM space with 200 fixed-seed random projections and quantile coupling,
# projection seed hard-coded to 20260136, n_probe=2000; only train/val are used, never test.
"""two-criterion early stopping: native criterion + val sliced-W2 generation criterion."""
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
    """sliced 2-Wasserstein with fixed projections (quantile coupling). Only used as an early-stop probe."""
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
    """the two criteria each keep their own best; stop only when **both fail to improve for patience consecutive times**. If either still improves, keep training.

    w2_enabled=False([DEGRADED-EARLYSTOP], e.g. P0-DEFER) degrades to the native criterion only, with doubled patience.
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
        """returns (improved_native, improved_w2, should_stop)."""
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
            # w2 evaluated sparsely; w2 patience counts w2 evaluations: stop only when both are exhausted
            stop = (self.bad_native >= self.patience) and (self.bad_w2 >= max(2, self.patience // self.w2_every))
        return imp_n, imp_w, bool(stop)

    def summary(self):
        return {"best_native": self.best_native, "best_native_step": self.best_native_step,
                "best_w2": (None if self.best_w2 == float("inf") else self.best_w2),
                "best_w2_step": self.best_w2_step, "patience": self.patience,
                "w2_enabled": self.w2_enabled, "n_native_evals": self.n_native_evals,
                "curve_tail": {k: v[-10:] for k, v in self.curve.items()}}
