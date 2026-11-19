# moved from cc/experiments/EXP20_joint_scrna/run_exp20.py::resid_corr_matrix / build_ctx /
#      two_gene_joint_w1 / offdiag_frob, and cc/experiments/EXP134_more_ckpts/run_exp134.py::main()
#      the part constructing ctx / REF / gamma (lines 161-176).
# changes (item by item):
#   1. `import metrics as MET` → `from . import metrics_exp08 as MET`(same code, see metrics_exp08.py
#      file header; the release directory must not sys.path.insert into cc/experiments/, so package-relative imports are used instead).
#   2. EXP134 the ctx/REF construction scattered in main() is collected unchanged into a single function build_eval_ctx(); statement order,
#      random seeds, slices and dtypes all unchanged (including the FIX-1 shuffle of spearmanr, gamma taken ctx["ref"].gamma_joint,
#      REF = val[permv[MHALF:2*MHALF]].astype(np.float64)).
#   3. EXP134 also computed sp_old/top_old only for printing a comparison; kept here (in the diag returned by build_eval_ctx),
#      does not affect any value.
# everything else unchanged.
"""ctx / REF / gamma construction: the single implementation for all EXP136 arms."""
from __future__ import annotations

import numpy as np

from . import metrics_exp08 as MET
from .seeds import MHALF, REF_PERM_SEED, SHUF_SEED


# --- EXP20::resid_corr_matrix(verbatim) ---
def resid_corr_matrix(X):
    X = np.asarray(X, dtype=np.float64); n, m = X.shape
    tot = X.sum(1); R = np.zeros_like(X)
    for g in range(m):
        pred = np.log1p(tot - X[:, g])
        A = np.column_stack([np.ones(n), pred])
        coef, _, _, _ = np.linalg.lstsq(A, X[:, g], rcond=None)
        R[:, g] = X[:, g] - A @ coef
    sd = R.std(0)
    if np.any(sd < 1e-9):
        ok = sd >= 1e-9; Cr = np.eye(m)
        if ok.sum() >= 2:
            sub = np.corrcoef(R[:, ok], rowvar=False); idx = np.where(ok)[0]
            for a in range(len(idx)):
                for b in range(len(idx)):
                    Cr[idx[a], idx[b]] = sub[a, b]
        return Cr
    return np.corrcoef(R, rowvar=False)


# --- EXP20::offdiag_frob(verbatim) ---
def offdiag_frob(Cm):
    m = Cm.shape[0]; iu = np.triu_indices(m, 1)
    return float(np.sqrt((Cm[iu] ** 2).sum())), float(np.mean(np.abs(Cm[iu])))


# --- EXP20::two_gene_joint_w1(verbatim) ---
def two_gene_joint_w1(gen_int, ref_int, pair):
    i, j = pair
    proj = MET.make_projections(2, n_proj=50, seed=20260625)
    return float(MET.swd(gen_int[:, [i, j]], ref_int[:, [i, j]], proj, max_points=8000, seed=42))


# --- EXP20::build_ctx(verbatim) ---
def build_ctx(ref_counts, raw_corr, resid_corr, top_pair):
    return {"ref": MET.GTReference(ref_counts, n_proj=100, kquant=0.99),
            "raw_corr": raw_corr, "resid_corr": resid_corr, "top_pair": top_pair,
            "gamma0": None}


def build_eval_ctx(train, val):
    """EXP134::main() the ctx / REF / gamma construction of lines 161-176, moved verbatim.

    returns (ctx, REF, gamma, diag). train is only used to estimate top gene pairs (FIX-1), not as an evaluation reference.
    """
    from scipy.stats import spearmanr
    dim = train.shape[1]
    raw = np.corrcoef(val.astype(float), rowvar=False); rc = resid_corr_matrix(val)
    iu = np.triu_indices(dim, 1)
    # ---- fix 1: train.npy was not shuffled; its first 20000 rows are biased (found in EXP135) ----
    sp_old = np.atleast_2d(spearmanr(train[:min(len(train), 20000)])[0])
    top_old = (int(iu[0][np.argmax(np.abs(sp_old[iu]))]), int(iu[1][np.argmax(np.abs(sp_old[iu]))]))
    perm = np.random.default_rng(SHUF_SEED).permutation(len(train))
    sp_new = np.atleast_2d(spearmanr(train[perm[:20000]])[0])
    top_new = (int(iu[0][np.argmax(np.abs(sp_new[iu]))]), int(iu[1][np.argmax(np.abs(sp_new[iu]))]))
    top = top_new
    ctx = build_ctx(val, raw, rc, top); ctx["gamma0"] = ctx["ref"].gamma_joint
    gamma = float(ctx["gamma0"])
    permv = np.random.default_rng(REF_PERM_SEED).permutation(len(val))
    REF = val[permv[MHALF:2 * MHALF]].astype(np.float64)
    diag = {"top_old": list(top_old), "top_new": list(top_new), "shuffle_seed": SHUF_SEED,
            "same": bool(top_old == top_new), "gamma": gamma,
            "rho_old": float(abs(sp_old[top_old])), "rho_new": float(abs(sp_new[top_new])),
            "REF_shape": list(REF.shape), "REF_sum": float(REF.sum())}
    return ctx, REF, gamma, diag
