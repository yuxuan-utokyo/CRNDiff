#!/usr/bin/env python
r"""R24-R26 on HLCA plasma cell and ionocyte with 3 seeds per method. Same rules and code path as
r24_26.py (prereg results/R24-26_prereg.md incl. the 11-11 00:35 additions); the only change is that each method has a list of seeds, read
from a manifest, and every per-method quantity is computed per seed (mean ± sd over seeds in the report).
  python r24_26_3seed.py val|test OUTDIR MANIFEST.json
MANIFEST: [{"method", "seed", "target", "path"}, ...], exactly 3 entries per (method, target) for CRNDiff, CFGen, CountsDiff, scVI, scANVI.
Seed-independent rows (real target / non-target, real val target, no augmentation) are computed once.
R26 C: chosen once on val by F1 at 0.5 in the no-augmentation, all-real setting (ties -> smaller C; posted on the run log 11-11 before any
test read). This depends on no generated cells; val mode recomputes it and asserts it equals C_POSTED; test mode reads it from val output.
(b) per seed: 5 draws of k = 20 train targets (seed 20261113, the same 5 draws for every seed and method); the generated subset of n_aug
cells is drawn with seed 20261114 from each seed's 2,500 cells.
"""
import hashlib, json, os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "8"
import numpy as np
SPL, OUT, MAN = sys.argv[1], sys.argv[2], sys.argv[3]; assert SPL in ("val", "test")
W = "./work"
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
os.makedirs(OUT, exist_ok=True); res_f = f"{OUT}/r24_26_{SPL}.json"
assert not os.path.exists(res_f), f"{res_f} exists; each split is run once"
VALJ = f"{os.path.dirname(OUT.rstrip('/'))}/val/r24_26_val.json"
if SPL == "test": assert os.path.exists(VALJ), "val first"
C_POSTED = {"plasma cell": 0.01, "ionocyte": 0.1}   # F1 rule; replaced only if the author chooses AP before 01:30
t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

G = json.load(open(f"{W}/data/r16_hlca_meta/genes_cfgen_hvg2000.json"))
assert G["genes"] == json.load(open(f"{W}/data/r16_hlca_test_view/meta.json"))["genes"] == json.load(open(f"{W}/data/r16_hlca/genes.json"))["genes"]
SYM = list(G["feature_name"])
MARK = {"plasma cell": "JCHAIN MZB1 XBP1 PRDM1 SDC1 TNFRSF17 IGKC IGHG1 IGHA1 IGLC2".split(),
        "ionocyte": "FOXI1 CFTR ASCL3 BSND ATP6V1B1 ATP6V1G3 CLNK FOXI2".split()}
CL = list(MARK); NVAL = {"plasma cell": 143, "ionocyte": 45}
METHODS = ["CRNDiff", "CFGen", "CountsDiff", "scVI", "scANVI"]

def logcp(x):
    x = np.asarray(x, dtype=np.float64); lib = x.sum(1, keepdims=True); lib[lib == 0] = 1
    return np.log1p(x / lib * 1e4).astype(np.float32)
def logcp_chunked(x, rows, ch=20000):
    return np.concatenate([logcp(x[rows[i:i + ch]]) for i in range(0, len(rows), ch)])

TRX = np.load(f"{W}/data/r16_hlca_fit/train.npy", mmap_mode="r"); TRC = np.load(f"{W}/data/r16_hlca_fit/train_celltype.npy", allow_pickle=True)
VD = f"{W}/data/r16_hlca_score"; VX = np.load(f"{VD}/val.npy", mmap_mode="r"); VC = np.load(f"{VD}/val_celltype.npy", allow_pickle=True)
ED = VD if SPL == "val" else f"{W}/data/r16_hlca_test_view"
EX = np.load(f"{ED}/val.npy", mmap_mode="r"); EC = np.load(f"{ED}/val_celltype.npy", allow_pickle=True)
assert EX.shape[0] == (46796 if SPL == "val" else 116989)
for t in CL: assert int((VC == t).sum()) == NVAL[t]

man = json.load(open(MAN)); GEN = {}; REC = {}
for t in CL:
    for m in METHODS:
        es = sorted([e for e in man if e["method"] == m and e["target"] == t], key=lambda e: str(e["seed"]))
        assert len(es) == 3 and len({e["path"] for e in es}) == 3 and len({str(e["seed"]) for e in es}) == 3, (m, t, es)
        for e in es:
            x = np.load(e["path"]); assert x.shape == (2500, 2000), (m, t, x.shape)
            GEN[(m, t, str(e["seed"]))] = x; REC[f"{m}|{t}|{e['seed']}"] = {"path": e["path"].replace(W + "/", ""), "sha256": sha(e["path"])}
SEEDS = {(m, t): sorted({s for (mm, tt, s) in GEN if mm == m and tt == t}) for m in METHODS for t in CL}
log("eval", SPL, EX.shape, "generated arrays", len(GEN))

def draw(seed, pool, n): return np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))
out = {"split": SPL, "script_sha256": sha(__file__), "manifest_sha256": sha(MAN), "generated": REC, "seeds": {f"{m}|{t}": s for (m, t), s in SEEDS.items()},
       "R24": {}, "R25": {}, "R26": {}, "draws": {}}
import anndata as ad, scanpy as sc
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score
for t in CL:
    e_t = np.flatnonzero(EC == t); e_o = np.flatnonzero(EC != t); tr_t = np.flatnonzero(TRC == t); tr_o = np.flatnonzero(TRC != t); v_t = np.flatnonzero(VC == t)
    real_t = logcp_chunked(EX, e_t); cnt_t = np.asarray(EX[e_t]); valt_x = logcp_chunked(VX, v_t)
    gl = {k: logcp(v) for k, v in GEN.items() if k[1] == t}

    # R24
    present = [g for g in MARK[t] if g in SYM]; missing = [g for g in MARK[t] if g not in SYM]; gi = [SYM.index(g) for g in present]; ig = np.array([g.startswith("IG") for g in present])
    fr_o = np.zeros(len(gi)); mu_o = np.zeros(len(gi))
    for i in range(0, len(e_o), 20000):
        xx = np.asarray(EX[e_o[i:i + 20000]], dtype=np.float64); lib = xx.sum(1, keepdims=True); lib[lib == 0] = 1
        fr_o += (xx[:, gi] > 0).sum(0); mu_o += np.log1p(xx[:, gi] / lib * 1e4).sum(0)
    ref = {"frac": (cnt_t[:, gi] > 0).mean(0), "mean": real_t[:, gi].mean(0)}
    rows = {"real eval target": ref, "real eval non-target": {"frac": fr_o / len(e_o), "mean": mu_o / len(e_o)}}
    for (m, tt, s) in [k for k in GEN if k[1] == t]: rows[f"{m}|{s}"] = {"frac": (GEN[(m, t, s)][:, gi] > 0).mean(0), "mean": gl[(m, t, s)][:, gi].mean(0)}
    def summ(r):
        df = np.abs(r["frac"] - ref["frac"]); dm = np.abs(r["mean"] - ref["mean"]); o = {}
        for nm, msk in (("all", np.ones(len(gi), bool)), ("IG", ig), ("nonIG", ~ig)):
            o[nm] = {"abs_frac_diff": float(df[msk].mean()) if msk.any() else None, "abs_mean_diff": float(dm[msk].mean()) if msk.any() else None, "n_genes": int(msk.sum())}
        return o
    out["R24"][t] = {"present": present, "missing": missing, "IG": [g for g, b in zip(present, ig) if b], "n_eval_target": len(e_t), "n_eval_nontarget": len(e_o),
                     "rows": {k: {"frac": [float(a) for a in v["frac"]], "mean": [float(a) for a in v["mean"]]} for k, v in rows.items()},
                     "summary": {k: summ(v) for k, v in rows.items() if k != "real eval target"}}
    log(t, "R24 done; missing", missing)

    # R25
    oth = draw(20261111, tr_o, 50000); oth_x = logcp_chunked(TRX, oth)
    out["draws"][f"R25_others|{t}"] = {"seed": 20261111, "n": 50000, "idx_sha256": hashlib.sha256(oth.astype(np.int64).tobytes()).hexdigest()}
    def de(grp):
        a = ad.AnnData(np.vstack([grp, oth_x])); a.obs["grp"] = np.array(["g"] * len(grp) + ["o"] * len(oth_x)); a.obs["grp"] = a.obs["grp"].astype("category")
        a.uns["log1p"] = {"base": None}; a.var_names = list(G["genes"])
        sc.tl.rank_genes_groups(a, "grp", groups=["g"], method="wilcoxon", n_genes=2000)
        r = a.uns["rank_genes_groups"]; names = np.array([n[0] for n in r["names"]]); s_ = np.array([s[0] for s in r["scores"]], float); lf = np.array([l[0] for l in r["logfoldchanges"]], float)
        top = set(names[np.argsort(-s_, kind="stable")[:50]].tolist()); pos = {g: i for i, g in enumerate(names)}
        return top, np.array([lf[pos[g]] for g in G["genes"]])
    ref_top, ref_lfc = de(real_t); r25 = {"reference": f"real {SPL} target cells ({len(e_t)}) vs others"}
    groups = {f"{m}|{s}": gl[(m, t, s)] for (m, tt, s) in GEN if tt == t}
    if SPL == "test": groups["real val target"] = valt_x
    for k, x in groups.items():
        top, lfc = de(x); ok = np.isfinite(lfc) & np.isfinite(ref_lfc)
        r25[k] = {"top50_overlap": len(top & ref_top), "spearman_logfc": float(spearmanr(lfc[ok], ref_lfc[ok]).statistic), "n_genes_spearman": int(ok.sum())}
    out["R25"][t] = r25; log(t, "R25 done")

    # R26
    neg = draw(20261112, tr_o, 50000); neg_x = logcp_chunked(TRX, neg); trt_x = logcp_chunked(TRX, tr_t)
    out["draws"][f"R26_negatives|{t}"] = {"seed": 20261112, "n": 50000, "idx_sha256": hashlib.sha256(neg.astype(np.int64).tobytes()).hexdigest()}
    pca = PCA(50, svd_solver="full").fit(np.vstack([neg_x, trt_x])); Zneg, Ztr = pca.transform(neg_x), pca.transform(trt_x)
    Zev = np.concatenate([pca.transform(logcp(EX[i:i + 20000])) for i in range(0, EX.shape[0], 20000)]); yev = (EC == t).astype(int)
    nA = NVAL[t]; gsub = np.sort(np.random.default_rng(20261114).choice(2500, nA, replace=False))
    def fit_eval(pos, C):
        X = np.vstack([Zneg, pos]); y = np.r_[np.zeros(len(Zneg)), np.ones(len(pos))]
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X, y); p = clf.predict_proba(Zev)[:, 1]; yh = (p >= 0.5).astype(int)
        return {"F1": float(f1_score(yev, yh)), "recall": float(recall_score(yev, yh)), "precision": float(precision_score(yev, yh, zero_division=0)), "AP": float(average_precision_score(yev, p)), "n_iter": int(clf.n_iter_[0])}
    if SPL == "val":
        csel = {C: fit_eval(Ztr, C) for C in (0.01, 0.1, 1.0)}; Cb = sorted(csel, key=lambda c: (-csel[c]["F1"], c))[0]
        assert Cb == C_POSTED[t], (t, Cb, C_POSTED[t]); r26 = {"C_selection_val": {str(c): v for c, v in csel.items()}, "C": Cb}
    else:
        Cb = json.load(open(VALJ))["R26"][t]["C"]; assert Cb == C_POSTED[t]; r26 = {"C": Cb, "C_from": VALJ.replace(W + "/", "")}
    kd = np.random.default_rng(20261113); ksets = [np.sort(kd.choice(len(tr_t), 20, replace=False)) for _ in range(5)]
    r26.update({"n_aug": nA, "n_train_target": len(tr_t), "n_eval": len(yev), "n_eval_target": int(yev.sum())})
    AUG = {f"{m}|{s}": pca.transform(gl[(m, t, s)][gsub]) for (m, tt, s) in GEN if tt == t}
    if SPL == "test": AUG["real val target"] = pca.transform(valt_x)
    a = {"no augmentation": fit_eval(Ztr, Cb)}; b = {"no augmentation": [fit_eval(Ztr[s], Cb) for s in ksets]}
    for k, z in AUG.items():
        a[k] = fit_eval(np.vstack([Ztr, z]), Cb); b[k] = [fit_eval(np.vstack([Ztr[s], z]), Cb) for s in ksets]
    r26["a_all_real"] = a; r26["b_k20"] = b; out["R26"][t] = r26
    log(t, "R26 C", Cb, "a F1", {k: round(v["F1"], 3) for k, v in a.items()})

json.dump(out, open(res_f, "w"), indent=1); log("wrote", res_f, sha(res_f)[:8])
