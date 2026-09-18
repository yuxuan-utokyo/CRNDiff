# 搬自四处，全部逐字复制（函数体一个字符未改）：
#   1. cc/experiments/EXP18_toy_joint_battery/run_exp18.py::corr_metrics / multibw_mmd / integrality /
#      discrete_metrics / eval_gen / METRIC_KEYS
#   2. cc/experiments/EXP20_joint_scrna/run_exp20.py::eval_plus / METRIC_KEYS(E20)
#   3. cc/experiments/EXP21_strengthen_ours_joint/run_exp21.py::weakness_axes / WEAK_KEYS / eval_all
#   4. cc/experiments/EXP35_generalization/gen_eval.py::_cpm_log1p / their_metrics
#   5. cc/experiments/EXP134_more_ckpts/run_exp134.py::_kmean / _mmd / _mean_dist / energy_pair / allmetrics
# 改动（逐条）：
#   1. import 改为包内相对（`from . import metrics_exp08 as MET`、`from .ctx import ...`、
#      `from . import metrics_canonical as MC`）—— 发布目录不许 sys.path.insert 到 cc/experiments/。
#      被 import 的模块本身是逐字复制，见各自文件头。
#   2. EXP134 的 allmetrics 里 `MHALF` 原为该文件模块级常量 5000，此处改为 `from .seeds import MHALF`
#      （值仍是 5000，未变）。
#   3. eval_plus 里 `resid_corr_matrix` 原为 E20 模块内函数，此处 `from .ctx import resid_corr_matrix`
#      （同一份逐字代码）。
#   4. 新增 main4() / evaluate_gen() 两个薄封装（只做取键与形状闸门，不参与任何数值计算），
#      EXP134 里没有对应物。
# 其余逐字未动。
"""EXP136 唯一指标实现。任何 arm 都必须走 allmetrics()，禁止重写、禁止等价改写。"""
from __future__ import annotations

import numpy as np

from . import metrics_canonical as MC
from . import metrics_exp08 as MET
from .ctx import resid_corr_matrix, two_gene_joint_w1
from .seeds import METRICS4, MHALF


# =========================================================================== #
# EXP18::corr_metrics / multibw_mmd / integrality / discrete_metrics / eval_gen
# =========================================================================== #
def corr_metrics(gen, gt_corr):
    """||corr_gen - corr_data||_F (off-diag and full) + mean |pairwise corr err|."""
    g = gen.astype(np.float64)
    # guard zero-variance dims
    sd = g.std(0)
    if np.any(sd < 1e-9):
        cg = np.eye(g.shape[1])
        ok = sd >= 1e-9
        if ok.sum() >= 2:
            sub = np.corrcoef(g[:, ok], rowvar=False)
            idx = np.where(ok)[0]
            for a in range(len(idx)):
                for b in range(len(idx)):
                    cg[idx[a], idx[b]] = sub[a, b]
    else:
        cg = np.corrcoef(g, rowvar=False)
    diff = cg - gt_corr
    m = gt_corr.shape[0]; iu = np.triu_indices(m, k=1)
    return {
        "corr_frob_full": float(np.sqrt((diff ** 2).sum())),
        "corr_frob_offdiag": float(np.sqrt((diff[iu] ** 2).sum())),
        "mean_abs_pairwise_corr_err": float(np.mean(np.abs(diff[iu]))),
        "gen_mean_abs_offdiag_corr": float(np.mean(np.abs(cg[iu]))),
    }


def multibw_mmd(gen, gt, gamma0, *, factors=(0.25, 0.5, 1.0, 2.0, 4.0), max_points=1000, seed=271828):
    """Multi-bandwidth RBF MMD: summed kernel over gamma0*factors (gamma0 = median-heuristic)."""
    rng = np.random.default_rng(seed)
    x = gen.astype(np.float64); y = gt.astype(np.float64)
    if len(x) > max_points:
        x = x[rng.choice(len(x), size=max_points, replace=False)]
    if len(y) > max_points:
        y = y[rng.choice(len(y), size=max_points, replace=False)]
    xx = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    yy = np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    xy = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    kxx = np.zeros_like(xx); kyy = np.zeros_like(yy); kxy = np.zeros_like(xy)
    for f in factors:
        g = gamma0 * f
        kxx += np.exp(-g * xx); kyy += np.exp(-g * yy); kxy += np.exp(-g * xy)
    mmd2 = kxx.mean() + kyy.mean() - 2.0 * kxy.mean()
    return float(np.sqrt(max(mmd2, 0.0)))


def integrality(raw):
    """Fraction of raw generated entries that are non-negative integers needing NO rounding.
    Integer-by-construction models -> 1.0; a continuous-relaxation model (e.g. CFGen) -> <1.0."""
    r = np.asarray(raw, dtype=np.float64)
    return float(np.mean((np.abs(r - np.round(r)) < 1e-6) & (r >= -1e-6)))


def discrete_metrics(gen_int, gtref):
    """Discrete-fidelity / calibration axis: zero-prop calib, low-count (0,1,2,3,>=4) TV, Fano/dispersion match."""
    gt = gtref.gt; dim = gtref.dim
    zero_gen = float((gen_int == 0).mean()); zero_gt = float(gtref.gt_zero_overall)
    tvs, fano_abs, fano_rel = [], [], []
    for d in range(dim):
        g = gen_int[:, d]; t = gt[:, d]
        pg = np.array([(g == 0).mean(), (g == 1).mean(), (g == 2).mean(), (g == 3).mean(), (g >= 4).mean()])
        pt = np.array([(t == 0).mean(), (t == 1).mean(), (t == 2).mean(), (t == 3).mean(), (t >= 4).mean()])
        tvs.append(0.5 * float(np.abs(pg - pt).sum()))
        mg, vg = float(g.mean()), float(g.var()); mt, vt = float(t.mean()), float(t.var())
        fg = vg / max(mg, 1e-9); ft = vt / max(mt, 1e-9)
        fano_abs.append(abs(fg - ft)); fano_rel.append(abs(fg - ft) / max(ft, 1e-9))
    return {
        "zero_calib_err": abs(zero_gen - zero_gt),
        "lowcount_tv_mean": float(np.mean(tvs)),
        "fano_abs_err_mean": float(np.mean(fano_abs)),
        "fano_rel_err_mean": float(np.mean(fano_rel)),
    }


def eval_gen(raw, gtref, gt_corr, gamma0):
    """Full metric bundle for one generated sample vs GT. `raw` may be float (continuous models) -> we
    measure integrality on raw, then round-for-eval to non-neg int for all count metrics (fair to any model)."""
    integ = integrality(raw)
    gen = np.clip(np.round(np.asarray(raw, dtype=np.float64)), 0, None).astype(np.int64)
    ev = gtref.evaluate(gen)
    cm = corr_metrics(gen, gt_corr)
    dm = discrete_metrics(gen, gtref)
    return {
        "joint_mmd": ev["joint_mmd"],
        "joint_mmd_multibw": multibw_mmd(gen, gtref.gt, gamma0),
        "swd": ev["swd"],
        "marginal_w1_mean": ev["marginal_w1_mean"],
        "var_relerr_absmean": ev["var_relerr_absmean"],
        "integrality": integ,
        **cm, **dm,
    }


E18_METRIC_KEYS = ["joint_mmd", "joint_mmd_multibw", "corr_frob_offdiag", "mean_abs_pairwise_corr_err",
                   "swd", "marginal_w1_mean", "var_relerr_absmean", "gen_mean_abs_offdiag_corr", "corr_frob_full",
                   "integrality", "zero_calib_err", "lowcount_tv_mean", "fano_abs_err_mean", "fano_rel_err_mean"]


# =========================================================================== #
# EXP20::eval_plus
# =========================================================================== #
def eval_plus(raw, ctx):
    base = eval_gen(raw, ctx["ref"], ctx["raw_corr"], ctx["gamma0"])
    gen = np.clip(np.round(np.asarray(raw, dtype=np.float64)), 0, None).astype(np.int64)
    rc = resid_corr_matrix(gen)
    iu = np.triu_indices(rc.shape[0], 1)
    diff = rc - ctx["resid_corr"]
    base["resid_corr_frob_offdiag"] = float(np.sqrt((diff[iu] ** 2).sum()))
    base["resid_gen_mean_abs_offdiag"] = float(np.mean(np.abs(rc[iu])))
    base["toppair_joint_w1"] = two_gene_joint_w1(gen, ctx["ref"].gt, ctx["top_pair"])
    return base


E20_METRIC_KEYS = E18_METRIC_KEYS + ["resid_corr_frob_offdiag", "resid_gen_mean_abs_offdiag", "toppair_joint_w1"]


# =========================================================================== #
# EXP21::weakness_axes / eval_all
# =========================================================================== #
def weakness_axes(gen_int, ref_int):
    g = gen_int.astype(np.int64); r = ref_int.astype(np.int64); dim = g.shape[1]
    ng, nr = g.shape[0], r.shape[0]
    tailL1, rarehit, maxrel, skew_e, kurt_e, fano_e, histTV = [], [], [], [], [], [], []
    # equal-N subsample of ref for max-count comparison
    rng = np.random.default_rng(777)
    r_eqN = r[rng.choice(nr, size=min(ng, nr), replace=False)] if nr >= ng else r
    for d in range(dim):
        gd = g[:, d]; rd = r[:, d]
        ks = np.unique(np.quantile(rd, [0.90, 0.95, 0.99, 0.999]).round().astype(int))
        sg = np.array([(gd > k).mean() for k in ks]); sr = np.array([(rd > k).mean() for k in ks])
        tailL1.append(float(np.abs(sg - sr).sum()))
        k99 = max(1, int(np.quantile(rd, 0.99)))
        rarehit.append(abs(float((gd > k99).mean()) - float((rd > k99).mean())))
        maxrel.append(abs(int(gd.max()) - int(r_eqN[:, d].max())) / max(int(r_eqN[:, d].max()), 1))
        # high-order moments
        def mom(x):
            x = x.astype(np.float64); m = x.mean(); s = x.std()
            if s < 1e-9:
                return 0.0, 0.0
            z = (x - m) / s
            return float((z ** 3).mean()), float((z ** 4).mean() - 3.0)
        sg3, kg = mom(gd); sr3, kr = mom(rd)
        skew_e.append(abs(sg3 - sr3)); kurt_e.append(abs(kg - kr))
        mg, vg = gd.mean(), gd.var(); mr_, vr = rd.mean(), rd.var()
        fano_e.append(abs((vg / max(mg, 1e-9)) - (vr / max(mr_, 1e-9))))
        mx = int(max(gd.max(), rd.max()))
        hg = np.bincount(gd, minlength=mx + 1)[:mx + 1].astype(float); hg /= max(hg.sum(), 1)
        hr = np.bincount(rd, minlength=mx + 1)[:mx + 1].astype(float); hr /= max(hr.sum(), 1)
        histTV.append(0.5 * float(np.abs(hg - hr).sum()))
    # Fano error focused on the most-overdispersed genes (top decile by ref Fano)
    ref_fano = np.array([r[:, d].var() / max(r[:, d].mean(), 1e-9) for d in range(dim)])
    top = np.argsort(-ref_fano)[: max(1, dim // 10)]
    return {
        "tail_survivalL1_mean": float(np.mean(tailL1)),
        "rare_hit_abserr_mean": float(np.mean(rarehit)),
        "maxcount_relerr_mean": float(np.mean(maxrel)),
        "skew_abserr_mean": float(np.mean(skew_e)),
        "kurtosis_abserr_mean": float(np.mean(kurt_e)),
        "fano_abserr_mean": float(np.mean(fano_e)),
        "fano_abserr_top10pctOverdisp": float(np.mean(np.array(fano_e)[top])),
        "pergene_histTV_mean": float(np.mean(histTV)),
    }


def eval_all(raw, ctx):
    """EXP20 eval_plus + Part B weakness axes (round-for-eval to int once)."""
    base = eval_plus(raw, ctx)
    gen = np.clip(np.round(np.asarray(raw, dtype=np.float64)), 0, None).astype(np.int64)
    base.update(weakness_axes(gen, ctx["ref"].gt))
    return base


# =========================================================================== #
# EXP35/gen_eval::their_metrics
# =========================================================================== #
def _cpm_log1p(x):
    x = np.asarray(x, float); s = x.sum(1, keepdims=True); s[s == 0] = 1
    return np.log1p(x / s * 1e4)


def their_metrics(gen, ref, rng, scale=10.0):
    g = _cpm_log1p(gen); r = _cpm_log1p(ref)
    m = min(len(g), len(r)); g = g[:m]; r = r[rng.choice(len(r), m, replace=False)]
    P = rng.standard_normal((g.shape[1], g.shape[1])); P /= np.linalg.norm(P, axis=0)
    gp = np.sort(g @ P / scale, 0); rp = np.sort(r @ P / scale, 0)
    w2 = float(np.sqrt(np.mean((gp - rp) ** 2)))
    from scipy.spatial.distance import cdist
    sub = min(m, 1000); gi = g[:sub] / scale / 10; ri = r[:sub] / scale / 10
    XX = cdist(gi, gi, "sqeuclidean"); YY = cdist(ri, ri, "sqeuclidean"); XY = cdist(gi, ri, "sqeuclidean")
    mmd = float(max(0.0, np.exp(-XX).mean() + np.exp(-YY).mean() - 2 * np.exp(-XY).mean()) ** 0.5)
    mg, mr = g.mean(0), r.mean(0); cg, cr = np.cov(g.T), np.cov(r.T)
    ev = np.linalg.eigvals(np.atleast_2d(cg) @ np.atleast_2d(cr)).real
    fd = float(((mg - mr) ** 2).sum() + np.trace(np.atleast_2d(cg)) + np.trace(np.atleast_2d(cr)) - 2 * np.sqrt(np.clip(ev, 0, None)).sum())
    return {"W2_sliced": w2, "MMD_rbf": mmd, "FD_frechet": fd}


# =========================================================================== #
# EXP134::_kmean / _mmd / _mean_dist / energy_pair / allmetrics
# =========================================================================== #
def _kmean(A, B, g, block=2000):
    aa = (A * A).sum(1); bb = (B * B).sum(1); tot = 0.0; cnt = 0
    for i in range(0, len(A), block):
        d2 = aa[i:i + block, None] + bb[None, :] - 2.0 * (A[i:i + block] @ B.T); np.maximum(d2, 0, out=d2)
        tot += float(np.exp(-g * d2).sum()); cnt += int(d2.size)
    return tot / cnt


def _mmd(A, B, g):
    return float(np.sqrt(max(_kmean(A, A, g) + _kmean(B, B, g) - 2.0 * _kmean(A, B, g), 0.0)))


def _mean_dist(A, B, block=2000):
    aa = (A * A).sum(1); bb = (B * B).sum(1); tot = 0.0; cnt = 0
    for i in range(0, len(A), block):
        d2 = aa[i:i + block, None] + bb[None, :] - 2.0 * (A[i:i + block] @ B.T); np.maximum(d2, 0, out=d2)
        tot += float(np.sqrt(d2).sum()); cnt += int(d2.size)
    return tot / cnt


def energy_pair(A, B):
    A = A.astype(np.float64); B = B.astype(np.float64)
    return float(2 * _mean_dist(A, B) - _mean_dist(A, A) - _mean_dist(B, B))


def allmetrics(gen, val, ctx, REF, gamma):
    ev = eval_all(gen, ctx); their = their_metrics(gen, val, np.random.default_rng(0))
    d = {k: float(v) for k, v in ev.items() if isinstance(v, (int, float, np.floating, np.integer))}
    d.update({k: float(v) for k, v in their.items()})
    d["resid_corr_sqrt2"] = float(MC.resid_corr_sqrt2(gen, val))
    g5 = gen[np.random.default_rng(7).choice(len(gen), min(MHALF, len(gen)), replace=False)].astype(np.float64)
    d["joint_mmd_dh"] = _mmd(g5, REF, gamma); d["energy_dh"] = energy_pair(g5, REF)
    return d


# =========================================================================== #
# 薄封装（不参与数值计算）
# =========================================================================== #
def main4(d):
    """从 allmetrics 的输出里取 4 个主指标。"""
    return {k: d[k] for k in METRICS4 if k in d}
