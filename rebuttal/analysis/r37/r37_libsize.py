#!/usr/bin/env python
r"""R37: where plasma-cell (and ionocyte) library size comes from. Statistics per cell set, on the 2,000 HVG counts:
library mean / median, IG count share (sum of counts in IG genes / all counts; IG = HVG symbols matching ^IG[HKL]), and the non-IG library
(mean / median of the count sum over the non-IG genes).
  real  SPLIT OUT.json : real train target cells, real eval target cells, real eval target cells capped at 4,096 per count (the generators'
                         ceiling). SPLIT = val | test (test: once, logged in runs/r16/TEST_READ_LOG.tsv).
  gen   OUT.json [POOL_DIR] : generated sets, no reference cells read: tilted FK = the v3 manifest's CRNDiff arrays (3 generators);
                         unconditional pools (r16_make_pool --tau 0, 25,000 cells; files _R16_uncond_s2026062{5,7,8}*.npy in POOL_DIR) ->
                         the cells the frozen purity LR (219e9bce, prediction as score_with_npz.py) calls the target, n reported;
                         tilt-only arm of R35 if present (runs/r35/arrays/<target>_tilt_only/*.npy).
"""
import glob, hashlib, json, os, re, sys, time
import numpy as np
MODE = sys.argv[1]; OUT = sys.argv[-1]   # real: real SPLIT OUT; gen: gen [POOL_DIR] OUT
W = "./work"; sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
G = json.load(open(f"{W}/data/r16_hlca_meta/genes_cfgen_hvg2000.json")); SYM = list(G["feature_name"])
IG = np.array([bool(re.match(r"^IG[HKL]", s)) for s in SYM]); CL = ["plasma cell", "ionocyte"]; CAP = 4096
def stats(x):
    x = np.asarray(x, dtype=np.float64); lib = x.sum(1); ig = x[:, IG].sum(1); non = lib - ig
    return {"n": int(len(x)), "lib_mean": float(lib.mean()), "lib_median": float(np.median(lib)), "ig_share": float(ig.sum() / max(lib.sum(), 1)),
            "nonig_lib_mean": float(non.mean()), "nonig_lib_median": float(np.median(non))}
res = {"mode": MODE, "ig_genes": [s for s, b in zip(SYM, IG) if b], "script_sha256": sha(__file__), "sets": {}}
if MODE == "real":
    SPL = sys.argv[2]; assert SPL in ("val", "test") and not os.path.exists(OUT)
    ED = f"{W}/data/r16_hlca_score" if SPL == "val" else f"{W}/data/r16_hlca_test_view"
    X = np.load(f"{ED}/val.npy", mmap_mode="r"); Y = np.load(f"{ED}/val_celltype.npy", allow_pickle=True).astype(str)
    XT = np.load(f"{W}/data/r16_hlca_fit/train.npy", mmap_mode="r"); YT = np.load(f"{W}/data/r16_hlca_fit/train_celltype.npy", allow_pickle=True).astype(str)
    res["split"] = SPL
    for t in CL:
        xe = np.asarray(X[np.flatnonzero(Y == t)], dtype=np.float64)
        res["sets"][t] = {"real train": stats(XT[np.flatnonzero(YT == t)]), f"real {SPL}": stats(xe), f"real {SPL}, capped at 4,096": stats(np.minimum(xe, CAP))}
    json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res["sets"], indent=0)[:1200])
    if SPL == "test":
        with open(f"{W}/runs/r16/TEST_READ_LOG.tsv", "a") as f:
            f.write(f"{time.strftime('%m-%d_%H:%M:%S')}\tHLCA R37 library-size sources, real test target cells (r37_libsize.py {sha(__file__)[:8]})\t{OUT.replace(W + '/', '')}\t{sha(OUT)[:8]}\n")
else:
    PD = sys.argv[2] if len(sys.argv) == 4 else None   # gen POOL_DIR OUT
    sys.path.insert(0, f"{W}/scripts/r13_score"); import r13_score as R
    NPZ = f"{W}/runs/r16/purity_lr/purity_lr_hlca.npz"; assert sha(NPZ)[:8] == "219e9bce"; z = np.load(NPZ)
    pred = lambda x: z["classes"][(R.log_cp10k(x) @ z["coef"].T + z["intercept"]).argmax(1)]
    man = json.load(open(f"{W}/runs/r24_26_3seed/manifest.json")); res["inputs"] = {}
    for t in CL:
        S = res["sets"].setdefault(t, {})
        for e in [e for e in man if e["method"] == "CRNDiff" and e["target"] == t]:
            S[f"tilted FK, {e['seed']}"] = stats(np.load(e["path"])); res["inputs"][e["path"].replace(W + "/", "")] = sha(e["path"])[:8]
        for f in sorted(glob.glob(f"{W}/runs/r35/arrays/{t.replace(' ', '')}_tilt_only/*.npy")):
            S[f"tilt only (R35), {os.path.basename(f)}"] = stats(np.load(f)); res["inputs"][f.replace(W + "/", "")] = sha(f)[:8]
    if PD:
        for g in ("20260625", "20260627", "20260628"):
            fs = sorted(glob.glob(f"{PD}/hlca_uncond_s{g}.npy")); assert len(fs) == 1, (g, fs)
            x = np.load(fs[0]); p = np.concatenate([pred(x[i:i + 5000]) for i in range(0, len(x), 5000)]); res["inputs"][os.path.basename(fs[0])] = sha(fs[0])[:8]
            for t in CL:
                k = p == t; res["sets"][t][f"uncond pool, gen{g}, LR-called {t} ({int(k.sum())} of {len(x):,})"] = stats(x[k]) if k.sum() else {"n": 0}
    json.dump(res, open(OUT, "w"), indent=1); print(json.dumps({t: {k: (v["n"], round(v.get("lib_mean", 0), 1)) for k, v in s.items()} for t, s in res["sets"].items()}, indent=0))
