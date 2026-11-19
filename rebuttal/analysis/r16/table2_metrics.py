#!/usr/bin/env python
"""R16: our implementation of the scLDM Table 2 metrics (arXiv 2511.02986v2), for (generated, real test) on HLCA.
Written from the papers only (no scLDM evaluation code exists; see results/R16_HLCA_evaluation_definition.md). Every setting the papers leave open is
an explicit option with a default fixed BEFORE any HLCA result is seen; the defaults and their sources:

  genes        the file's var["highly_variable"] (2,000 of 27,997; CFGen's released HLCA files; Table 2 caption "highly variable
               genes"; that Table 2 used exactly this set is not stated anywhere)                          --genes highly_variable
  counts       real: layers["X_counts"] (raw counts); generated: counts in the same gene order (checked by gene ID)
  transform    per-cell normalise to 1e4 total counts, then log1p, same for real and generated (CFGen App. F.2)    --transform cp10k_log1p
  library size the total used for that normalisation: over the selected genes (default; works for any generator) or over all
               27,997 genes (standard scanpy order: normalise, log1p, then subset). "all" needs the generated cells' full-gene totals
               (npz array library_size); a generator trained on the 2,000 genes alone cannot supply them. Not stated in either paper.
                                                                                                           --libsize hvg
  PRE-REGISTERED:
               PCA fit ONCE on 50,000 real TRAIN cells (train minus val, sampled with seed 20261310), shared by every method and class:
               --pca-fit real_train --pca-ref hlca_core_train.h5ad --pca-ref-exclude VAL_IDS.txt --pca-fit-n 50000 --pca-fit-seed 20261310
               --pca-save PCA.npz (first run), then --pca-load PCA.npz for every other method;
               library size over the 2,000 HVGs (--libsize hvg; baselines cut to the 2,000 genes before normalising);
               classes: the 42 with >= 100 real test cells (--min-real 100). The paper-literal test-fit PCA (--pca-fit all_real) is kept
               only for a protocol note (a CFGen row compared with Table 2's 12.43), not for the main table.
  PCA          30 components (scLDM App. I.8), fitted on the real data (I.8). all_real = all real cells passed in --real (after the class
               filter); real_train = real train cells; generated cells projected with those loadings
               Caveat found in the self-test (11-09): fitting PCA on the same real cells that are then compared makes the real cloud
               wider than an independent sample in the noise PCs, so precision > recall even for identical distributions (synthetic
               3 x 1,200 cells, 200 genes: P 0.92 / R 0.49; PCA on pooled cells 0.79 / 0.76). --pca-fit real_train --pca-ref TRAIN.h5ad
               fits PCA on real TRAIN cells instead (out of sample for both sides); which reading Table 2 used is not stated.
  cells        cond: per class, real = all real test cells of that class, generated = the generated cells of that class; both
               shuffled with --seed and truncated to the smaller n, then capped at --max-cells (CFGen distribution_distances.py
               truncates to min n). uncond: all real vs all generated, same rule.                        --max-cells 5000
  MMD^2        RBF, sum of kernels exp(-||x-y||^2 / (2 s^2)) over s in {0.01, 0.1, 1, 10, 100} (CFGen App. F.1 / mix_rbf_mmd2),
               unbiased U-statistic (scLDM App. I.1, Eq. 25)                                              --mmd-sigmas, --mmd-est unbiased
  W2           exact optimal transport between the two equal-size empirical sets, uniform weights, squared Euclidean cost, sqrt
               of the optimum (scLDM App. I.2; CFGen optimal_transport.py ot.emd2)                         --w2 exact (POT), fallback Hungarian
  1-NN         leave-one-out 1-NN two-sample accuracy on the pooled equal-size sets, Euclidean (scLDM App. I.4)
  P / R        Kynkäänniemi et al. 2019, k = 3 (scLDM App. I.5): radius of a point = distance to its k-th nearest neighbour in its own
               set; precision = share of generated points inside some real ball; recall = share of real points inside some
               generated ball                                                                              --k 3
  aggregate    cond: per class, then unweighted mean and sd (ddof=0 and ddof=1 both written) over classes; the cell-weighted mean is
               written too. Classes with fewer than --min-cells cells on either side after truncation are skipped and listed.
               (Neither paper states the aggregation; CFGen App. F.2 says metrics are computed per cell type.)   --min-cells 10
Nothing here reads HLCA test labels to choose a setting; the script only computes.
v3 (11-09 20:4x; v2 d39541de archived in _archive_r16/): input reading only, no computation changed. --real / --pca-ref may be the
r16_data/ export folder (with --real-split / --pca-ref-split); --pca-only fits and saves the shared PCA without --real or --gen
(the real_train PCA lines moved unchanged into fit_shared_pca()).

Input formats
  --real  H5AD (e.g. hlca_core_test.h5ad): obs[--label-key], var["highly_variable"], layers[--layer] read with h5py (CSR or dense).
  --gen   NPZ with arrays counts (n x G), labels (n,), genes (G,) [gene IDs, must equal the selected real genes in order]
          or H5AD with X or layers[--gen-layer] counts, obs[--label-key], var_names.
Usage
  python table2_metrics.py --real hlca_core_test.h5ad --gen cfgen_cond.npz --mode cond --out OUT.json [--classes A B ...]
  python table2_metrics.py --real hlca_core_test.h5ad --gen cfgen_uncond.npz --mode uncond --out OUT.json
  python table2_metrics.py --selftest
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

SIGMAS_DEFAULT = "0.01,0.1,1,10,100"


# ---------------------------------------------------------------- reading
def _read_matrix(h5, path):
    g = h5[path]
    if hasattr(g, "keys") and "indptr" in g:  # CSR / CSC group
        enc = g.attrs.get("encoding-type", "csr_matrix")
        enc = enc.decode() if isinstance(enc, bytes) else enc
        shape = tuple(g.attrs["shape"])
        m = (sp.csr_matrix if "csr" in enc else sp.csc_matrix)((g["data"][...], g["indices"][...], g["indptr"][...]), shape=shape)
        return m.tocsr()
    return sp.csr_matrix(g[...])


def read_real(path, layer, label_key, genes_mode, want_libsize=False, split=None):
    """Returns X (selected genes), labels, gene IDs, full-gene totals (or None), obs names.
    v3 (11-09 20:4x): path may also be the r16_data/ export folder (Windows' r16_export.py: int32 CSR blocks of the 2,000 HVGs);
    then split names the part (test, val, train, pca_ref). That folder holds only the 2,000 genes, so full-gene totals are None."""
    if os.path.isdir(path):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from r16_data import load_split
        assert genes_mode == "highly_variable" and split, "the r16_data folder holds the 2,000 HVGs only; give --real-split / --pca-ref-split"
        X, y, g, names = load_split(path, split, with_names=True)
        return X, y, g, None, names
    import anndata as ad, h5py
    a = ad.read_h5ad(path, backed="r")
    labels = np.asarray(a.obs[label_key].astype(str)); obs_names = np.asarray(a.obs_names.astype(str))
    var_names = np.asarray(a.var_names.astype(str))
    if genes_mode == "highly_variable":
        mask = np.asarray(a.var["highly_variable"]).astype(bool)
    elif genes_mode == "all":
        mask = np.ones(len(var_names), bool)
    else:
        raise ValueError(genes_mode)
    a.file.close()
    with h5py.File(path, "r") as h5:
        X = _read_matrix(h5, f"layers/{layer}" if layer != "X" else "X")
    lib_all = np.asarray(X.sum(1), dtype=np.float64).ravel() if want_libsize else None
    X = X[:, np.flatnonzero(mask)]
    return X, labels, var_names[mask], lib_all, obs_names


def read_gen(path, label_key, gen_layer):
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        lib = z["library_size"].astype(np.float64) if "library_size" in z.files else None
        return sp.csr_matrix(z["counts"]), z["labels"].astype(str), z["genes"].astype(str), lib
    import anndata as ad
    a = ad.read_h5ad(path)
    X = a.layers[gen_layer] if gen_layer in a.layers else a.X
    lib = np.asarray(a.obs["library_size"], dtype=np.float64) if "library_size" in a.obs else None
    return sp.csr_matrix(X), np.asarray(a.obs[label_key].astype(str)), np.asarray(a.var_names.astype(str)), lib


# ---------------------------------------------------------------- transform
def transform(X, how, lib=None):
    """lib: per-cell totals to normalise by (None = total over the columns of X, i.e. the selected genes)."""
    X = sp.csr_matrix(X, dtype=np.float64)
    if how == "raw":
        return X.toarray()
    if how == "log1p":
        return np.log1p(X.toarray())
    if how == "cp10k_log1p":
        lib = (np.asarray(X.sum(1)).ravel() if lib is None else np.asarray(lib, np.float64).copy()); lib[lib == 0] = 1.0
        return np.log1p(sp.diags(1e4 / lib) @ X).toarray()
    raise ValueError(how)


# ---------------------------------------------------------------- metrics (inputs: equal-size arrays in PCA space)
def mmd2(x, y, sigmas, est):
    def ksum(a, b):
        d2 = cdist(a, b, "sqeuclidean")
        return sum(np.exp(-d2 / (2.0 * s * s)) for s in sigmas)
    kxx, kyy, kxy = ksum(x, x), ksum(y, y), ksum(x, y)
    n, m = len(x), len(y)
    if est == "unbiased":
        return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def w2(x, y, how):
    M = cdist(x, y, "sqeuclidean")
    if how == "exact":
        try:
            import ot
            a = np.full(len(x), 1.0 / len(x)); b = np.full(len(y), 1.0 / len(y))
            return float(np.sqrt(ot.emd2(a, b, M, numItermax=int(1e7))))
        except ImportError:
            how = "hungarian"
    if how == "hungarian":
        from scipy.optimize import linear_sum_assignment
        assert len(x) == len(y)
        r, c = linear_sum_assignment(M)
        return float(np.sqrt(M[r, c].mean()))
    raise ValueError(how)


def one_nn(x, y):
    z = np.vstack([x, y]); lab = np.r_[np.zeros(len(x)), np.ones(len(y))]
    nn = NearestNeighbors(n_neighbors=2).fit(z)
    _, idx = nn.kneighbors(z)
    return float((lab[idx[:, 1]] == lab).mean())  # leave-one-out: neighbour 0 is the point itself


def prec_rec(real, fake, k):
    def radii(a):
        d, _ = NearestNeighbors(n_neighbors=k + 1).fit(a).kneighbors(a)
        return d[:, k]  # k-th neighbour excluding self
    rr, rf = radii(real), radii(fake)
    d_fr = cdist(fake, real)  # fake x real
    precision = float((d_fr <= rr[None, :]).any(1).mean())
    recall = float((d_fr.T <= rf[None, :]).any(1).mean())
    return precision, recall


def all_metrics(real, fake, a):
    p, r = prec_rec(real, fake, a.k)
    return {"W2": w2(real, fake, a.w2), "MMD2": mmd2(real, fake, a.sigmas, a.mmd_est), "1NN": one_nn(real, fake), "precision": p, "recall": r,
            "n": len(real)}


def matched(rng, xr, xg, cap):
    n = min(len(xr), len(xg), cap)
    return xr[rng.permutation(len(xr))[:n]], xg[rng.permutation(len(xg))[:n]]


# ---------------------------------------------------------------- driver
def fit_shared_pca(a, pca):
    """The real_train PCA (lines unchanged from v2, moved here in v3 so --pca-only can run them without a generated file)."""
    Zt = transform(a.pca_ref_matrix, a.transform, getattr(a, "pca_ref_lib", None))
    pca.fit(Zt); mean, comp = pca.mean_, pca.components_; evr = float(pca.explained_variance_ratio_.sum())
    pca_info = {"fit": "real_train", "n_fit": int(len(Zt)), "ref_ids_sha256": getattr(a, "pca_ref_ids_sha", None)}
    if getattr(a, "pca_save", None):
        np.savez(a.pca_save, mean=mean, components=comp, explained_variance_ratio=pca.explained_variance_ratio_, n_fit=len(Zt),
                 ref_ids_sha256=str(a.pca_ref_ids_sha), transform=a.transform, libsize=a.libsize, n_genes=Zt.shape[1])
        pca_info["saved_to"] = a.pca_save
    return mean, comp, evr, pca_info


def evaluate(Xr, yr, Xg, yg, a, lr=None, lg=None):
    t0 = time.time()
    if a.classes:
        keep_r, keep_g = np.isin(yr, a.classes), np.isin(yg, a.classes)
        Xr, yr, Xg, yg = Xr[keep_r], yr[keep_r], Xg[keep_g], yg[keep_g]
        lr = None if lr is None else lr[keep_r]; lg = None if lg is None else lg[keep_g]
    Zr, Zg = transform(Xr, a.transform, lr), transform(Xg, a.transform, lg)
    pca = PCA(n_components=a.n_pcs, random_state=a.seed, svd_solver="full" if Zr.shape[1] <= 4000 else "auto")
    loaded = getattr(a, "pca_loaded", None)
    if loaded is not None:  # the one shared PCA (fit once on real train cells, saved with --pca-save)
        mean, comp = loaded["mean"], loaded["components"]; assert comp.shape == (a.n_pcs, Zr.shape[1]), comp.shape
        pca_info = {"fit": "loaded", **{k: loaded[k].item() if loaded[k].ndim == 0 else None for k in ("n_fit",)}, "file": a.pca_load,
                    "file_sha256": loaded["_sha"]}
        evr = float(loaded["explained_variance_ratio"].sum())
    elif a.pca_fit == "all_real":
        pca.fit(Zr); mean, comp = pca.mean_, pca.components_; pca_info = {"fit": "all_real", "n_fit": int(len(Zr))}; evr = float(pca.explained_variance_ratio_.sum())
    elif a.pca_fit == "real_train":
        mean, comp, evr, pca_info = fit_shared_pca(a, pca)
    else:
        raise ValueError(a.pca_fit)
    Pr, Pg = (Zr - mean) @ comp.T, (Zg - mean) @ comp.T
    rng = np.random.default_rng(a.seed)
    res = {"settings": {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in vars(a).items()
                        if k not in ("real", "gen", "out", "pca_ref_matrix", "pca_ref_lib", "pca_loaded")},
           "pca": {**pca_info, "explained_variance_ratio_sum": evr}}
    if a.mode == "uncond":
        r, g = matched(rng, Pr, Pg, a.max_cells)
        res["uncond"] = all_metrics(r, g, a)
    else:
        per, skipped = {}, {}
        for c in sorted(set(yr) | set(yg)):
            xr, xg = Pr[yr == c], Pg[yg == c]
            n = min(len(xr), len(xg), a.max_cells)
            if len(xr) < getattr(a, "min_real", 0) or n < max(a.min_cells, a.k + 2):
                skipped[c] = {"n_real": int(len(xr)), "n_gen": int(len(xg)),
                              "reason": "real test cells < --min-real" if len(xr) < getattr(a, "min_real", 0) else "cells < --min-cells"}; continue
            r, g = matched(rng, xr, xg, a.max_cells)
            per[c] = {**all_metrics(r, g, a), "n_real_total": int(len(xr)), "n_gen_total": int(len(xg))}
        keys = ["W2", "MMD2", "1NN", "precision", "recall"]
        w = np.array([per[c]["n_real_total"] for c in per], float)
        agg = {}
        for k in keys:
            v = np.array([per[c][k] for c in per], float)
            agg[k] = {"mean": float(v.mean()), "sd_ddof0": float(v.std()), "sd_ddof1": float(v.std(ddof=1)) if len(v) > 1 else None,
                      "weighted_mean": float((v * w).sum() / w.sum())}
        res["cond"] = {"n_classes": len(per), "aggregate": agg, "per_class": per, "skipped": skipped}
    res["walltime_s"] = round(time.time() - t0, 1)
    return res


def selftest():
    """Synthetic checks only (no HLCA data).
    1. Metric level, Gaussian in 30 dims: same distribution -> 1-NN ~ 0.5, P ~ R, MMD^2 ~ 0; shifted -> all worse. W2 exact = Hungarian.
    2. Pipeline on counts (3 classes x 1,200 cells, 200 genes): a non-uniform change (gene means swapped for 30% of genes; a uniform
       scale would be removed by the CP10K normalisation) is worse than an independent draw from the same distribution, under both
       PCA readings; the P/R asymmetry of --pca-fit all_real on identical distributions is printed, not asserted."""
    rng = np.random.default_rng(0)
    x, y, z = rng.normal(size=(1200, 30)), rng.normal(size=(1200, 30)), rng.normal(0.5, 1, size=(1200, 30))
    sig = [float(s) for s in SIGMAS_DEFAULT.split(",")]
    p, r = prec_rec(x, y, 3); m0, m1 = mmd2(x, y, sig, "unbiased"), mmd2(x, z, sig, "unbiased")
    print("gaussian same : 1NN %.3f P %.3f R %.3f MMD2 %.5f W2 %.3f" % (one_nn(x, y), p, r, m0, w2(x, y, "exact")))
    print("gaussian shift: 1NN %.3f P %.3f R %.3f MMD2 %.5f W2 %.3f" % (one_nn(x, z), *prec_rec(x, z, 3), m1, w2(x, z, "exact")))
    assert abs(one_nn(x, y) - 0.5) < 0.04 and abs(p - r) < 0.05 and abs(m0) < 0.01
    assert one_nn(x, z) > 0.6 and m1 > m0 + 0.01 and w2(x, z, "exact") > w2(x, y, "exact")
    u, v = rng.normal(size=(300, 5)), rng.normal(0.3, 1, size=(300, 5))
    e, h = w2(u, v, "exact"), w2(u, v, "hungarian"); print("W2 exact %.6f hungarian %.6f" % (e, h)); assert abs(e - h) < 1e-6
    G, n = 200, 1200
    mu = rng.gamma(0.6, 2.0, size=(3, G)); mu_bad = mu.copy(); sw = rng.permutation(G)[: int(0.3 * G)]; mu_bad[:, sw] = mu[:, rng.permutation(sw)]
    lab = np.repeat(np.array(["a", "b", "c"]), n)
    draw = lambda M: sp.csr_matrix(rng.poisson(rng.gamma(2.0, M[np.repeat(np.arange(3), n)] / 2.0)))
    R, Gs, Gb, Tr = draw(mu), draw(mu), draw(mu_bad), draw(mu)
    class A: pass
    a = A(); a.transform = "cp10k_log1p"; a.n_pcs = 30; a.seed = 0; a.max_cells = 5000; a.min_cells = 10; a.k = 3
    a.sigmas = sig; a.mmd_est = "unbiased"; a.w2 = "exact"; a.classes = None; a.mode = "cond"; a.pca_ref_matrix = Tr
    for fit in ("all_real", "real_train"):
        a.pca_fit = fit
        good, bad = evaluate(R, lab, Gs, lab, a)["cond"]["aggregate"], evaluate(R, lab, Gb, lab, a)["cond"]["aggregate"]
        print(f"[{fit}] same   :", {k: round(v["mean"], 4) for k, v in good.items()})
        print(f"[{fit}] changed:", {k: round(v["mean"], 4) for k, v in bad.items()})
        assert bad["1NN"]["mean"] > good["1NN"]["mean"] + 0.05 and bad["W2"]["mean"] > good["W2"]["mean"] and bad["MMD2"]["mean"] > good["MMD2"]["mean"]
    a.mode = "uncond"; a.pca_fit = "real_train"; uu = evaluate(R, lab, Gs, lab, a)["uncond"]; print("uncond same [real_train]:", {k: round(v, 4) for k, v in uu.items()})
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real"); ap.add_argument("--gen"); ap.add_argument("--out"); ap.add_argument("--mode", choices=("cond", "uncond"))
    ap.add_argument("--classes", nargs="*"); ap.add_argument("--label-key", default="cell_type"); ap.add_argument("--layer", default="X_counts")
    ap.add_argument("--gen-layer", default="X_counts"); ap.add_argument("--genes", default="highly_variable", choices=("highly_variable", "all"))
    ap.add_argument("--transform", default="cp10k_log1p", choices=("cp10k_log1p", "log1p", "raw")); ap.add_argument("--n-pcs", type=int, default=30)
    ap.add_argument("--libsize", default="hvg", choices=("hvg", "all"), help="hvg = total over the selected genes")
    ap.add_argument("--pca-fit", default="real_train", choices=("all_real", "real_train")); ap.add_argument("--max-cells", type=int, default=5000)
    ap.add_argument("--pca-ref", help="real TRAIN h5ad for --pca-fit real_train")
    ap.add_argument("--real-split", help="v3: part of the r16_data folder given as --real (test or val)")
    ap.add_argument("--pca-ref-split", default="pca_ref", help="v3: part of the r16_data folder given as --pca-ref")
    ap.add_argument("--pca-only", action="store_true", help="v3: fit the real_train PCA, write --pca-save (and --out json), stop")
    ap.add_argument("--pca-fit-n", "--pca-ref-n", dest="pca_fit_n", type=int, default=50000)
    ap.add_argument("--pca-fit-seed", type=int, default=20261310)
    ap.add_argument("--pca-ref-exclude", help="text file, one obs name per line, removed from --pca-ref before sampling (the val cells)")
    ap.add_argument("--pca-save", help="write the fitted PCA (mean, components) here, to share across methods")
    ap.add_argument("--pca-load", help="use a PCA written by --pca-save instead of fitting")
    ap.add_argument("--min-cells", type=int, default=10); ap.add_argument("--min-real", type=int, default=100,
                    help="cond: skip classes with fewer real test cells")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mmd-sigmas", default=SIGMAS_DEFAULT); ap.add_argument("--mmd-est", default="unbiased", choices=("unbiased", "biased"))
    ap.add_argument("--w2", default="exact", choices=("exact", "hungarian")); ap.add_argument("--seed", type=int, default=20261300)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    a.sigmas = [float(s) for s in a.mmd_sigmas.split(",")]
    full = a.libsize == "all"
    if a.pca_only:  # v3: fit and save the shared real_train PCA only; reads --pca-ref, never --real or --gen
        assert a.pca_fit == "real_train" and a.pca_save and not a.pca_load and not full
        Xt, _, gt, _, ids = read_real(a.pca_ref, a.layer, a.label_key, a.genes, split=a.pca_ref_split)
        keep = np.ones(len(ids), bool)
        if a.pca_ref_exclude:  # same lines as below
            ex = set(l.strip() for l in open(a.pca_ref_exclude) if l.strip()); keep = ~np.isin(ids, list(ex))
            assert keep.sum() == len(ids) - len(ex & set(ids)), "exclusion count mismatch"
        pool = np.flatnonzero(keep)
        pick = pool if len(pool) <= a.pca_fit_n else np.sort(np.random.default_rng(a.pca_fit_seed).choice(pool, a.pca_fit_n, replace=False))
        a.pca_ref_ids_sha = hashlib.sha256("\n".join(ids[pick]).encode()).hexdigest(); a.pca_ref_matrix = Xt[pick]; a.pca_ref_lib = None
        pca = PCA(n_components=a.n_pcs, random_state=a.seed, svd_solver="full" if Xt.shape[1] <= 4000 else "auto")
        mean, comp, evr, info = fit_shared_pca(a, pca)
        info.update({"explained_variance_ratio_sum": evr, "n_genes": int(Xt.shape[1]), "pca_file_sha256": hashlib.sha256(open(a.pca_save, "rb").read()).hexdigest(),
                     "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "pca_ref": a.pca_ref, "pca_ref_split": a.pca_ref_split})
        if a.out:
            json.dump(info, open(a.out, "w"), indent=1)
        print(json.dumps(info, indent=1)); return 0
    Xr, yr, gr, lr, _ = read_real(a.real, a.layer, a.label_key, a.genes, want_libsize=True, split=a.real_split)
    Xg, yg, gg, lg = read_gen(a.gen, a.label_key, a.gen_layer)
    assert len(gr) == len(gg) and np.array_equal(gr, gg), "generated genes must equal the selected real genes, in the same order"
    if full:
        assert lg is not None and len(lg) == Xg.shape[0], "--libsize all needs the generated cells' full-gene totals (npz library_size)"
    else:
        lr, lg = None, None
    a.pca_ref_matrix = None; a.pca_ref_lib = None; a.pca_loaded = None; a.pca_ref_ids_sha = None
    if a.pca_load:
        z = np.load(a.pca_load, allow_pickle=False); a.pca_loaded = {k: z[k] for k in z.files}
        a.pca_loaded["_sha"] = hashlib.sha256(open(a.pca_load, "rb").read()).hexdigest()
        assert str(z["transform"]) == a.transform and str(z["libsize"]) == a.libsize and int(z["n_genes"]) == len(gr), "saved PCA was fit with other settings"
    elif a.pca_fit == "real_train":
        Xt, _, gt, lt, ids = read_real(a.pca_ref, a.layer, a.label_key, a.genes, want_libsize=True, split=a.pca_ref_split); assert np.array_equal(gt, gr), "train genes differ from test genes"
        keep = np.ones(len(ids), bool)
        if a.pca_ref_exclude:  # e.g. the val cells: PCA is fit on train minus val
            ex = set(l.strip() for l in open(a.pca_ref_exclude) if l.strip()); keep = ~np.isin(ids, list(ex))
            assert keep.sum() == len(ids) - len(ex & set(ids)), "exclusion count mismatch"
        pool = np.flatnonzero(keep)
        pick = pool if len(pool) <= a.pca_fit_n else np.sort(np.random.default_rng(a.pca_fit_seed).choice(pool, a.pca_fit_n, replace=False))
        a.pca_ref_ids_sha = hashlib.sha256("\n".join(ids[pick]).encode()).hexdigest()
        a.pca_ref_matrix = Xt[pick]; a.pca_ref_lib = lt[pick] if full else None
    res = evaluate(Xr, yr, Xg, yg, a, lr, lg)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    res["inputs"] = {"real": a.real, "real_bytes": os.path.getsize(a.real), "gen": a.gen, "gen_sha256": sha(a.gen), "n_genes": int(len(gr)),
                     "n_real": int(Xr.shape[0]), "n_gen": int(Xg.shape[0]), "script_sha256": sha(__file__)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res.get("uncond") or res["cond"]["aggregate"], indent=1))


if __name__ == "__main__":
    sys.exit(main())
