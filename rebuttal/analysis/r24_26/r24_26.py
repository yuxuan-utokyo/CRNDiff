#!/usr/bin/env python
r"""R24-R26 on HLCA plasma cell and ionocyte, seed 1 of every method (rules: results/R24-26_prereg.md incl. the 11-11 00:35 additions).
  python r24_26.py val|test OUTDIR
Expression is log-CP10K with the library size taken over the 2,000 HVGs (real and generated alike).
  R24  marker genes (present in the panel; missing ones listed, not replaced): fraction expressing (count > 0) and mean log-CP10K for each
       method's generated cells, real eval target cells, all real eval non-target cells; per method mean |frac diff|, |mean diff| vs real
       eval target cells, over all / IG* / non-IG* markers.
  R25  "others" = 50,000 train non-target cells (seed 20261111, one draw per class, shared by all rows). Wilcoxon (scanpy rank_genes_groups,
       defaults except method='wilcoxon' and n_genes=2000 so every gene gets a logFC) of group vs others. Reference = real eval target cells
       vs others. Per row: top-50 (by score) overlap with the reference, Spearman of the 2,000 logFCs with the reference. Test mode adds the
       row real val target cells vs others.
  R26  negatives = 50,000 train non-target cells (seed 20261112); PCA(50) fit on those + all train target cells (log-CP10K, centred, full
       SVD); binary logistic regression, class_weight='balanced', max_iter 2000. C from {0.01, 0.1, 1} chosen ONCE on val in the
       no-augmentation, all-real setting by val F1 at threshold 0.5 (ties -> smaller C); test mode reads that C from the val output.
       n_aug = real val target count (143 / 45); generated subset seed 20261114 (same rule for every method).
       (a) all train targets + n_aug; (b) k = 20 train targets (5 draws, seed 20261113) + n_aug. Rows: no augmentation, + each method,
       + the n_aug real val target cells (test mode only; in val mode it would train on the eval cells).
       Eval: every cell of the eval split, natural proportions; F1, recall, precision at 0.5, average precision.
In val mode the eval split is val (pipeline check; numbers not for the paper); in test mode it is test (the CFGen held-out file).
"""
import hashlib, json, os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "8"
import numpy as np
SPL, OUT = sys.argv[1], sys.argv[2]; assert SPL in ("val", "test")
W = "./work"; R = f"{W}/runs/r16"
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
os.makedirs(OUT, exist_ok=True); res_f = f"{OUT}/r24_26_{SPL}.json"
assert not os.path.exists(res_f), f"{res_f} exists; each split is run once"
VALJ = f"{os.path.dirname(OUT.rstrip('/'))}/val/r24_26_val.json"
if SPL == "test": assert os.path.exists(VALJ), "val first"
t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

G = json.load(open(f"{W}/data/r16_hlca_meta/genes_cfgen_hvg2000.json"))
assert G["genes"] == json.load(open(f"{W}/data/r16_hlca_test_view/meta.json"))["genes"] == json.load(open(f"{W}/data/r16_hlca/genes.json"))["genes"]
SYM = list(G["feature_name"])
MARK = {"plasma cell": "JCHAIN MZB1 XBP1 PRDM1 SDC1 TNFRSF17 IGKC IGHG1 IGHA1 IGLC2".split(),
        "ionocyte": "FOXI1 CFTR ASCL3 BSND ATP6V1B1 ATP6V1G3 CLNK FOXI2".split()}
CL = list(MARK); NVAL = {"plasma cell": 143, "ionocyte": 45}

def logcp(x):
    x = np.asarray(x, dtype=np.float64); lib = x.sum(1, keepdims=True); lib[lib == 0] = 1
    return np.log1p(x / lib * 1e4).astype(np.float32)
def logcp_chunked(x, rows=None, ch=20000):
    rows = np.arange(x.shape[0]) if rows is None else rows
    return np.concatenate([logcp(x[rows[i:i + ch]]) for i in range(0, len(rows), ch)]) if len(rows) else np.zeros((0, x.shape[1]), np.float32)

# data
TRX = np.load(f"{W}/data/r16_hlca_fit/train.npy", mmap_mode="r"); TRC = np.load(f"{W}/data/r16_hlca_fit/train_celltype.npy", allow_pickle=True)
VD = f"{W}/data/r16_hlca_score"; VX = np.load(f"{VD}/val.npy", mmap_mode="r"); VC = np.load(f"{VD}/val_celltype.npy", allow_pickle=True)
ED = VD if SPL == "val" else f"{W}/data/r16_hlca_test_view"
EX = np.load(f"{ED}/val.npy", mmap_mode="r"); EC = np.load(f"{ED}/val_celltype.npy", allow_pickle=True)
assert EX.shape[0] == (46796 if SPL == "val" else 116989)
for t in CL: assert int((VC == t).sum()) == NVAL[t]
log("eval", SPL, EX.shape, {t: int((EC == t).sum()) for t in CL})

# generated cells, seed 1 of every method (the R16 test manifest + CountsDiff K128 plasma cell)
man = json.load(open(f"{R}/ruleA_test/manifest_test.json")) + json.load(open(f"{R}/cd_k128_test/manifest_test.json"))
TAU = {"plasma cell": "tau0p2", "ionocyte": "tau0p4"}
def gen_path(m, t):
    arm = {"CRNDiff": TAU[t], "CountsDiff": "K128_g1p0"}.get(m)
    c = [e for e in man if e["method"] == m and e["target"] == t and (arm is None or e["arm"] == arm)]
    assert len({e["path"] for e in c}) == 1, (m, t, c); return c[0]
METHODS = ["CRNDiff", "CFGen", "CountsDiff", "scVI", "scANVI"]
GEN = {}; GEN_REC = {}
for t in CL:
    for m in METHODS:
        e = gen_path(m, t); x = np.load(e["path"]); assert x.shape == (2500, 2000), (m, t, x.shape)
        GEN[(m, t)] = x; GEN_REC[f"{m}|{t}"] = {"path": e["path"].replace(W + "/", ""), "sha256": sha(e["path"]), "arm": e["arm"], "seed": e["seed"]}
log("generated loaded", len(GEN))

def draw(seed, pool, n):
    return np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))

out = {"split": SPL, "script_sha256": sha(__file__), "generated": GEN_REC, "R24": {}, "R25": {}, "R26": {}, "draws": {}}
for t in CL:
    e_t = np.flatnonzero(EC == t); e_o = np.flatnonzero(EC != t)
    tr_t = np.flatnonzero(TRC == t); tr_o = np.flatnonzero(TRC != t)
    v_t = np.flatnonzero(VC == t)
    real_t = logcp_chunked(EX, e_t); cnt_t = np.asarray(EX[e_t]); valt_x = logcp_chunked(VX, v_t)
    gl = {m: logcp(GEN[(m, t)]) for m in METHODS}

    # ---------- R24
    present = [g for g in MARK[t] if g in SYM]; missing = [g for g in MARK[t] if g not in SYM]; gi = [SYM.index(g) for g in present]
    ig = [g.startswith("IG") for g in present]
    # non-target eval cells: library size over the full 2,000 genes, chunked
    fr_o = np.zeros(len(gi)); mu_o = np.zeros(len(gi))
    for i in range(0, len(e_o), 20000):
        xx = np.asarray(EX[e_o[i:i + 20000]], dtype=np.float64); lib = xx.sum(1, keepdims=True); lib[lib == 0] = 1
        fr_o += (xx[:, gi] > 0).sum(0); mu_o += np.log1p(xx[:, gi] / lib * 1e4).sum(0)
    rows = {"real eval target": {"frac": (cnt_t[:, gi] > 0).mean(0), "mean": real_t[:, gi].mean(0)},
            "real eval non-target": {"frac": fr_o / len(e_o), "mean": mu_o / len(e_o)}}
    for m in METHODS: rows[m] = {"frac": (GEN[(m, t)][:, gi] > 0).mean(0), "mean": gl[m][:, gi].mean(0)}
    summ = {}
    for m in METHODS + ["real eval non-target"]:
        df = np.abs(rows[m]["frac"] - rows["real eval target"]["frac"]); dm = np.abs(rows[m]["mean"] - rows["real eval target"]["mean"])
        s = {}
        for nm, msk in (("all", np.ones(len(gi), bool)), ("IG", np.array(ig)), ("nonIG", ~np.array(ig))):
            s[nm] = {"abs_frac_diff": float(df[msk].mean()) if msk.any() else None, "abs_mean_diff": float(dm[msk].mean()) if msk.any() else None, "n_genes": int(msk.sum())}
        summ[m] = s
    out["R24"][t] = {"present": present, "missing": missing, "IG": [g for g, b in zip(present, ig) if b],
                     "rows": {k: {"frac": [float(a) for a in v["frac"]], "mean": [float(a) for a in v["mean"]]} for k, v in rows.items()},
                     "summary": summ, "n_eval_target": len(e_t), "n_eval_nontarget": len(e_o)}
    log(t, "R24 done", present, "missing", missing)

    # ---------- R25
    import anndata as ad, scanpy as sc
    from scipy.stats import spearmanr
    oth = draw(20261111, tr_o, 50000); oth_x = logcp_chunked(TRX, oth)
    out["draws"][f"R25_others|{t}"] = {"seed": 20261111, "n": 50000, "idx_sha256": hashlib.sha256(oth.astype(np.int64).tobytes()).hexdigest()}
    def de(grp):
        X = np.vstack([grp, oth_x]); a = ad.AnnData(X); a.obs["grp"] = np.array(["g"] * len(grp) + ["o"] * len(oth_x)); a.obs["grp"] = a.obs["grp"].astype("category")
        a.uns["log1p"] = {"base": None}; a.var_names = list(G["genes"])
        sc.tl.rank_genes_groups(a, "grp", groups=["g"], method="wilcoxon", n_genes=2000)
        r = a.uns["rank_genes_groups"]; names = np.array([n[0] for n in r["names"]]); sc_ = np.array([s[0] for s in r["scores"]], float); lf = np.array([l[0] for l in r["logfoldchanges"]], float)
        order = np.argsort(-sc_, kind="stable"); top = set(names[order[:50]].tolist())
        pos = {g: i for i, g in enumerate(names)}; lfc = np.array([lf[pos[g]] for g in G["genes"]])
        return top, lfc
    ref_top, ref_lfc = de(real_t)
    r25 = {"reference": f"real {SPL} target cells ({len(e_t)}) vs others", "n_nonfinite_ref_lfc": int((~np.isfinite(ref_lfc)).sum())}
    groups = {m: gl[m] for m in METHODS}
    if SPL == "test": groups["real val target"] = valt_x
    for m, x in groups.items():
        top, lfc = de(x); ok = np.isfinite(lfc) & np.isfinite(ref_lfc)
        r25[m] = {"top50_overlap": len(top & ref_top), "spearman_logfc": float(spearmanr(lfc[ok], ref_lfc[ok]).statistic), "n_genes_spearman": int(ok.sum()), "n_cells": len(x)}
        log(t, "R25", m, r25[m])
    out["R25"][t] = r25

    # ---------- R26
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score
    neg = draw(20261112, tr_o, 50000); neg_x = logcp_chunked(TRX, neg); trt_x = logcp_chunked(TRX, tr_t)
    out["draws"][f"R26_negatives|{t}"] = {"seed": 20261112, "n": 50000, "idx_sha256": hashlib.sha256(neg.astype(np.int64).tobytes()).hexdigest()}
    pca = PCA(50, svd_solver="full").fit(np.vstack([neg_x, trt_x]))
    Zneg, Ztr = pca.transform(neg_x), pca.transform(trt_x)
    # eval: all cells, chunked projection
    ev_rows = np.arange(EX.shape[0]); Zev = np.concatenate([pca.transform(logcp(EX[i:i + 20000])) for i in range(0, EX.shape[0], 20000)])
    yev = (EC == t).astype(int)
    nA = NVAL[t]; gsub = np.sort(np.random.default_rng(20261114).choice(2500, nA, replace=False))
    AUG = {m: pca.transform(gl[m][gsub]) for m in METHODS}
    if SPL == "test": AUG["real val target"] = pca.transform(valt_x)
    def fit_eval(pos, C):
        X = np.vstack([Zneg, pos]); y = np.r_[np.zeros(len(Zneg)), np.ones(len(pos))]
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X, y)
        p = clf.predict_proba(Zev)[:, 1]; yh = (p >= 0.5).astype(int)
        return {"F1": float(f1_score(yev, yh)), "recall": float(recall_score(yev, yh)), "precision": float(precision_score(yev, yh, zero_division=0)),
                "AP": float(average_precision_score(yev, p)), "n_iter": int(clf.n_iter_[0])}
    if SPL == "val":
        csel = {C: fit_eval(Ztr, C) for C in (0.01, 0.1, 1.0)}
        Cbest = sorted(csel, key=lambda c: (-csel[c]["F1"], c))[0]
        r26 = {"C_selection_val": {str(c): v for c, v in csel.items()}, "C": Cbest}
    else:
        Cbest = json.load(open(VALJ))["R26"][t]["C"]; r26 = {"C": Cbest, "C_from": VALJ.replace(W + "/", "")}
    kdraw = np.random.default_rng(20261113); ksets = [np.sort(kdraw.choice(len(tr_t), 20, replace=False)) for _ in range(5)]
    r26.update({"n_aug": nA, "n_train_target": len(tr_t), "n_eval": len(yev), "n_eval_target": int(yev.sum()), "gen_subset_seed": 20261114, "k": 20, "k_seed": 20261113})
    a = {"no augmentation": fit_eval(Ztr, Cbest)}
    for m, z in AUG.items(): a[f"+ {m}"] = fit_eval(np.vstack([Ztr, z]), Cbest)
    b = {"no augmentation": [fit_eval(Ztr[s], Cbest) for s in ksets]}
    for m, z in AUG.items(): b[f"+ {m}"] = [fit_eval(np.vstack([Ztr[s], z]), Cbest) for s in ksets]
    r26["a_all_real"] = a; r26["b_k20"] = b
    r26["pca_explained_var_sum"] = float(pca.explained_variance_ratio_.sum())
    out["R26"][t] = r26
    log(t, "R26 C", Cbest, "a:", {k: round(v["F1"], 3) for k, v in a.items()}, "b F1 mean:", {k: round(np.mean([r["F1"] for r in v]), 3) for k, v in b.items()})

json.dump(out, open(res_f, "w"), indent=1); log("wrote", res_f, sha(res_f)[:8])
