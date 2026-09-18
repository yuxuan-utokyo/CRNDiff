# 搬自 _shared_discussion/metrics_canonical.py（整文件），改动：仅在文件顶部加入本行注释；
# 其余逐字未动。EXP134 中以 `import metrics_canonical as MC` 使用，提供 resid_corr_sqrt2。
"""CANONICAL METRICS — claude 与 cc 必须用【这同一段代码】算，杜绝口径漂移。
resid_corr 对 VAL 算、报 /sqrt2(上三角约定)。所有 arm 一律用这个。
用法:
    import metrics_canonical as M
    stats = M.all_metrics(gen_npy, train_npy, val_npy)   # dict
产物: fano_relerr, mean_rel_signed_median, resid_corr_sqrt2, raw_corr_sqrt2,
      FanoL, EL, pmf_bins(21_50, 51p 等)
"""
import numpy as np

SQRT2 = np.sqrt(2.0)

def _resid_corr_matrix(X):
    """逐基因回归掉 log(total - gene) 后的残差相关矩阵。VERBATIM EXP20::resid_corr_matrix。"""
    X = np.asarray(X, dtype=np.float64); n, m = X.shape
    tot = X.sum(1); R = np.zeros_like(X)
    for g in range(m):
        pred = np.log1p(tot - X[:, g])
        A = np.column_stack([np.ones(n), pred])
        coef, _, _, _ = np.linalg.lstsq(A, X[:, g], rcond=None)
        R[:, g] = X[:, g] - A @ coef
    return np.corrcoef(R, rowvar=False)

def resid_corr_sqrt2(gen, val):
    """||C_resid(gen) - C_resid(val)||_F / sqrt2 . 【对 VAL 算参照。】"""
    Cv = _resid_corr_matrix(val)
    return float(np.linalg.norm(_resid_corr_matrix(gen) - Cv) / SQRT2)

def raw_corr_sqrt2(gen, val):
    Cv = np.corrcoef(np.asarray(val, float), rowvar=False)
    return float(np.linalg.norm(np.corrcoef(np.asarray(gen, float), rowvar=False) - Cv) / SQRT2)

def fano_relerr(gen, train):
    g = np.asarray(gen, float); t = np.asarray(train, float)
    fv = t.var(0) / np.maximum(t.mean(0), 1e-9)
    fg = g.var(0) / np.maximum(g.mean(0), 1e-9)
    return float(np.median(np.abs(fg / fv - 1)))

def mean_rel_signed_median(gen, train):
    mu = np.asarray(train, float).mean(0)
    return float(np.median(np.asarray(gen, float).mean(0) / mu - 1))

def pmf_bins(gen, train, nmax=128):
    ns = np.arange(nmax + 1); G = train.shape[1]
    def pool(X):
        return np.stack([np.bincount(np.clip(X[:, g], 0, nmax), minlength=nmax + 1) / len(X)
                         for g in range(G)]).mean(0)
    Pt = pool(train); Pg = pool(gen)
    out = {}
    for name, (a, b) in [("0", (0, 0)), ("1_5", (1, 5)), ("6_20", (6, 20)),
                          ("21_50", (21, 50)), ("51p", (51, nmax))]:
        m = (ns >= a) & (ns <= b); d = Pt[m].sum()
        out[name] = float(Pg[m].sum() / d) if d > 0 else float("nan")
    return out

def null_stats(train, val, n_gen=20000, reps=25, seed=9):
    """完美模型 null: 从 val bootstrap n_gen, 对 val 算 metric。"""
    rng = np.random.default_rng(seed); N = len(val)
    rc, fa, me = [], [], []
    for _ in range(reps):
        b = val[rng.choice(N, n_gen, replace=True)]
        rc.append(resid_corr_sqrt2(b, val)); fa.append(fano_relerr(b, train)); me.append(mean_rel_signed_median(b, train))
    f = lambda a: (float(np.mean(a)), float(np.std(a)))
    return {"resid_corr_sqrt2": f(rc), "fano_relerr": f(fa), "mean_rel": f(me)}

def all_metrics(gen, train, val):
    L = np.asarray(gen, float).sum(1)
    return {
        "fano_relerr": fano_relerr(gen, train),
        "mean_rel_signed_median": mean_rel_signed_median(gen, train),
        "resid_corr_sqrt2": resid_corr_sqrt2(gen, val),
        "raw_corr_sqrt2": raw_corr_sqrt2(gen, val),
        "FanoL": float(L.var() / L.mean()),
        "EL": float(L.mean()),
        "pmf_bins": pmf_bins(gen, train),
    }

if __name__ == "__main__":
    import sys, glob, json
    # 自检: 对 A1 的既有 gens 报 resid_corr_sqrt2, 应 ≈ 1.306 (claude 存档值)
    ROOT = "cc/experiments/EXP35_generalization/data"
    tr = np.load(f"{ROOT}/train.npy").astype(np.int64)
    va = np.load(f"{ROOT}/val.npy").astype(np.int64)
    fs = sorted(glob.glob("cc/experiments/EXP50_conserved_catalysts/gens/A1_*n20000.npy"))
    if fs:
        vals = [resid_corr_sqrt2(np.load(f), va) for f in fs]
        print("A1 resid_corr_sqrt2 (canonical, 对val):", [round(v, 4) for v in vals],
              " 均值 %.4f  (claude 存档 1.306, cc 上轮误报 0.977)" % np.mean(vals))
