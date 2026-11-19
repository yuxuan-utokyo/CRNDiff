#!/usr/bin/env python
"""R12: recompute R1 (k-NN neighbour fraction), R2 (sparsity) and R3 (coverage) for every Table 1 run of every method
in one pass on Mac, then add the new arms. Val only; test is not read.

Inputs: Windows' load_cells output per run, `CELLS/<target>/<arm>_<train>_<samp>.npz` (key x; .npy also accepted).
Functions: R2 prof/curve/compare and R1 knn_stats from r1r2_metrics.py (copied unchanged from Windows, gate passed);
R3 score = r3_diversity.score, basis = target_pca_bases.npz (the frozen PCA(50) of r3_diversity.basis).
R3 resample seed = 20261107 + j, j = the position of the run in w2_runs.run_list(t) (45 runs: OURS 9, then per sampling
seed CFGen run1-3 and MDLM 625/934/935, then scVI 9, scANVI 9). New arms get j = 0, as in r9_new_metrics.py.
Gate (--gate): every Table 1 row must reproduce r1_numbers.json, r2_numbers.json and r3_numbers.json (|diff| <= 1e-9).

    python -I r12_all_rows.py --cells CELLS --gate
    python -I r12_all_rows.py --cells CELLS --extra EXTRA.json --out OUT.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

W = Path.home() / "research/crndiff-rebuttal"; REF = W / "from_windows/rebuttal_code/results_for_check"
ap = argparse.ArgumentParser(); ap.add_argument("--cells", required=True); ap.add_argument("--gate", action="store_true")
ap.add_argument("--extra"); ap.add_argument("--out"); ap.add_argument("--targets", nargs="+", default=["Endothelial", "Myeloid", "Neuronal"])
a = ap.parse_args()
src = (W / "scripts/r12_extra/r1r2_metrics.py").read_text()
assert src.count("\nap = argparse.ArgumentParser()") == 1
G = {}; exec(compile(src.split("\nap = argparse.ArgumentParser()")[0], "r1r2_metrics.py", "exec"), G)   # its functions and data, not its CLI
sys.path.insert(0, str(W / "scripts/r13_score")); from r13_score import log_cp10k
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

K, NMAX, DRAWS, SEED = 5, 2500, 5, 20261107
TABLE1 = {"OURS": [(20260625, 20260989), (20260627, 20260987), (20260630, 20260989)],
          "MDLM": [(20260625, 20260933), (20260934, 20260932), (20260935, 20260932)],
          "CFGen": [("run1", 20260932), ("run2", 20260932), ("run3", 20260933)],
          "scVI": [(20261050, 20261051), (20261053, 20261055), (20261054, 20261055)],
          "scANVI": [(20261060, 20261065), (20261063, 20261061), (20261064, 20261061)]}


def run_order():          # w2_runs.run_list order, without paths
    r = [("OURS", 20260625, s) for s in (20260987, 20260988, 20260989)] + [("OURS", 20260627, s) for s in (20260987, 20260988, 20260989)]
    r += [("OURS", 20260630, s) for s in (20260987, 20260988, 20260989)]
    for s in (20260931, 20260932, 20260933):
        r += [("CFGen", "run1", s), ("CFGen", "run2", s), ("CFGen", "run3", s), ("MDLM", 20260625, s), ("MDLM", 20260934, s), ("MDLM", 20260935, s)]
    for arm, trs, sps in (("scVI", (20261050, 20261053, 20261054), (20261051, 20261055, 20261056)),
                          ("scANVI", (20261060, 20261063, 20261064), (20261061, 20261065, 20261066))):
        r += [(arm, tr, s) for tr in trs for s in sps]
    return r


# ---- copied from r3_diversity.py (n_jobs as there)
def nn1_acc(x, y):
    z = np.vstack([x, y]); lab = np.r_[np.zeros(len(x)), np.ones(len(y))]
    i = NearestNeighbors(n_neighbors=2, n_jobs=4).fit(z).kneighbors(z, return_distance=False)
    j = np.where(i[:, 0] == np.arange(len(z)), i[:, 1], i[:, 0])
    return float((lab[j] == lab).mean())
def prdc(real, fake):
    rr = pairwise_distances(real, n_jobs=4); np.fill_diagonal(rr, np.inf)
    ff = pairwise_distances(fake, n_jobs=4); np.fill_diagonal(ff, np.inf)
    rad_r = np.partition(rr, K - 1, axis=1)[:, K - 1]; rad_f = np.partition(ff, K - 1, axis=1)[:, K - 1]
    d = pairwise_distances(real, fake, n_jobs=4)
    inr = d <= rad_r[:, None]
    return {"precision": float(inr.any(0).mean()), "recall": float((d <= rad_f[None, :]).any(1).mean()),
            "density": float(inr.sum() / (K * fake.shape[0])), "coverage": float(inr.any(1).mean())}
def score(zreal, zfake, seed, var_val):
    n = min(len(zreal), NMAX)
    rng = np.random.default_rng(seed); out = []
    for s in range(DRAWS):
        ri = rng.choice(len(zreal), n, replace=False) if len(zreal) > n else np.arange(len(zreal))
        fi = rng.choice(len(zfake), n, replace=len(zfake) < n)
        out.append({"nn1_acc": nn1_acc(zreal[ri], zfake[fi]), **prdc(zreal[ri], zfake[fi])})
    v = zfake.var(0, ddof=1)
    m = {k: float(np.mean([o[k] for o in out])) for k in out[0]}
    m.update({"n_compare": n, "var_ratio_top10": float(v[:10].sum() / var_val[:10].sum()), "var_ratio_top50": float(v.sum() / var_val.sum())})
    return m


def load(t, arm, train, samp):
    for ext in (".npz", ".npy"):
        f = Path(a.cells) / t / f"{arm}_{train}_{samp}{ext}"
        if f.exists():
            z = np.load(f); x = z["x"] if ext == ".npz" else z
            return np.asarray(x, dtype=np.float64), f
    raise SystemExit(f"missing {t}/{arm}_{train}_{samp}")


B = np.load(W / "from_windows/rebuttal_code/assets/target_pca_bases.npz")
R3K = ("nn1_acc", "precision", "recall", "density", "coverage", "var_ratio_top10", "var_ratio_top50")
R2K = ("zero_overall", "gene_zero_mae_vs_real", "gene_zero_pcc_vs_real", "curve_rmse_vs_real", "cell_zero_mean", "cell_zero_sd",
       "detected_mean", "detected_sd", "detected_cv", "lib_mean", "lib_median", "lib_sd")
R1K = ("mean_neuronal_frac", "median_neuronal_frac", "share_frac_ge_0p5", "share_frac_eq_0", "nn1_is_neuronal")
r1ref = json.loads((REF / "r1_numbers.json").read_text())["rows"]; r2ref = json.loads((REF / "r2_numbers.json").read_text())
r3ref = json.loads((REF / "r3_numbers.json").read_text())
m1 = G["R1"]() if "Neuronal" in a.targets else None
res = {"split": "val", "rows": []}; worst = {"R1": 0.0, "R2": 0.0, "R3": 0.0}
for t in a.targets:
    proj = lambda x: (log_cp10k(x) - B[f"{t}__mean"]) @ B[f"{t}__components"].T
    zv = proj(G["real"](G["xv_all"], G["yv"], t)); var_val = zv.var(0, ddof=1)
    ref2, _ = G["r2_block"](t)
    todo = [(j, arm, tr, s) for j, (arm, tr, s) in enumerate(run_order()) if (tr, s) in TABLE1[arm]]
    extra = [] if not a.extra else [(0, e["arm_label"], e["train"], e["samp"]) for e in json.loads(Path(a.extra).read_text()) if e["target"] == t]
    for j, arm, tr, s in todo + extra:
        x, f = load(t, arm, tr, s)
        row = {"target": t, "arm": arm, "train": tr, "samp": s, "n": len(x), "file": f.name, "r3_seed": SEED + j,
               "r3": score(zv, proj(x), SEED + j, var_val), "r2": G["compare"](x, ref2)[0]}
        if t == "Neuronal":
            row["r1"] = m1.query(x)
        if a.gate and arm in TABLE1:
            w3 = next(q for q in r3ref[t]["runs"] if q["arm"] == arm and str(q["train"]) == str(tr) and q["samp"] == s)
            w2 = next(q for q in r2ref[t]["runs"] if q["arm"] == arm and str(q["train"]) == str(tr) and q["samp"] == s)
            d3 = max(abs(row["r3"][k] - w3[k]) for k in R3K); d2 = max(abs(row["r2"][k] - w2[k]) for k in R2K)
            worst["R3"] = max(worst["R3"], d3); worst["R2"] = max(worst["R2"], d2); msg = f"R3 {d3:.1e} R2 {d2:.1e}"
            if t == "Neuronal":
                d1 = max(abs(row["r1"][sp][k] - next(q for q in r1ref if q["arm"] == arm and str(q["train"]) == str(tr) and q["samp"] == s and q["space"] == sp)[k])
                         for sp in ("pca2", "pca50", "full") for k in R1K)
                worst["R1"] = max(worst["R1"], d1); msg += f" R1 {d1:.1e}"
            print(f"[gate] {t} {arm} {tr}/{s} j={j}: {msg}", flush=True)
        else:
            print(f"[row] {t} {arm} {tr}/{s} recall {row['r3']['recall']:.3f} zero {row['r2']['zero_overall']:.4f}", flush=True)
        res["rows"].append(row)
if a.gate:
    print("[gate] worst |diff|", {k: f"{v:.1e}" for k, v in worst.items()}, "PASS" if max(worst.values()) <= 1e-9 else "FAIL")
if a.out:
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
