# -*- coding: utf-8 -*-
"""R3 (z7bw: rare target over-concentrated; PCA on real target, per-PC and total variance; 1-NN, P/R).

Basis: PCA(50) fitted on TRAIN cells of the target, log-CP10K (basis only). Variance ratio denominator
and all two-sample comparisons use VAL cells of the target. n = min(n_val, 2500) on both sides; generated
cells resampled 5 times (val also resampled when n_val > n). Noise floor: real train n cells vs real val,
same procedure. Test split not read.

1-NN accuracy: pool real and fake, predict each point's origin from its nearest other point. 0.5 = no
separation, 1.0 = fully separated. P/R/D/C: Naeem et al. 2020, k=5, Euclidean in the PCA space.

Part B (Neuronal only, 11-07 01:40 criteria): where the concentration comes from.

    cd Experiments/HVG && python -m rebuttal.r3_diversity [--only-b]
"""
import glob, json, os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = os.environ.get("R_THREADS", "8")   # R12 CPU window: 1
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
sys.dont_write_bytecode = True
from rebuttal.w2_runs import ROOT, RUNS, DATA, TARGETS, ARMS, run_list, load_cells, real_cells, log_cp10k
OUT = ROOT / "rebuttal" / "out" / "R3"; OUT.mkdir(parents=True, exist_ok=True)
K, NMAX, DRAWS, SEED = 5, 2500, 5, 20261107
NJ = int(os.environ.get("R_THREADS", "8"))


def nn1_acc(a, b):
    z = np.vstack([a, b]); y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    i = NearestNeighbors(n_neighbors=2, n_jobs=NJ).fit(z).kneighbors(z, return_distance=False)
    j = np.where(i[:, 0] == np.arange(len(z)), i[:, 1], i[:, 0])
    return float((y[j] == y).mean())


def prdc(real, fake):
    rr = pairwise_distances(real, n_jobs=NJ); np.fill_diagonal(rr, np.inf)
    ff = pairwise_distances(fake, n_jobs=NJ); np.fill_diagonal(ff, np.inf)
    rad_r = np.partition(rr, K - 1, axis=1)[:, K - 1]; rad_f = np.partition(ff, K - 1, axis=1)[:, K - 1]
    d = pairwise_distances(real, fake, n_jobs=NJ)
    inr = d <= rad_r[:, None]
    return {"precision": float(inr.any(0).mean()), "recall": float((d <= rad_f[None, :]).any(1).mean()),
            "density": float(inr.sum() / (K * fake.shape[0])), "coverage": float(inr.any(1).mean())}


def score(zreal, zfake, seed, var_val):
    n = min(len(zreal), NMAX)
    rng = np.random.default_rng(seed); out = []
    for s in range(DRAWS):
        ri = rng.choice(len(zreal), n, replace=False) if len(zreal) > n else np.arange(len(zreal))
        fi = rng.choice(len(zfake), n, replace=len(zfake) < n)
        a, b = zreal[ri], zfake[fi]
        out.append({"nn1_acc": nn1_acc(a, b), **prdc(a, b)})
    v = zfake.var(0, ddof=1)
    m = {k: float(np.mean([o[k] for o in out])) for k in out[0]}
    m.update({f"{k}_sd": float(np.std([o[k] for o in out], ddof=1)) for k in out[0]})
    m.update({"n_compare": n, "var_ratio_top10": float(v[:10].sum() / var_val[:10].sum()),
              "var_ratio_top50": float(v.sum() / var_val.sum()), "var_ratio_per_pc": (v / var_val).tolist()})
    return m


def basis(t):
    xt = real_cells("train", t)
    return PCA(n_components=50, svd_solver="randomized", random_state=SEED).fit(log_cp10k(xt)), xt


def part_a():
    res = {}
    for t in TARGETS:
        p, xt = basis(t)
        zv = p.transform(log_cp10k(real_cells("val", t))); var_val = zv.var(0, ddof=1)
        rows, fl = [], []
        for s in range(5):      # noise floor: fresh train subset each time, n = n_val or 2500
            i = np.random.default_rng(SEED + 100 + s).choice(len(xt), min(len(zv), NMAX, len(xt)), replace=False)
            fl.append(score(zv, p.transform(log_cp10k(xt[i])), SEED + 200 + s, var_val))
        keys = ("nn1_acc", "precision", "recall", "density", "coverage", "var_ratio_top10", "var_ratio_top50")
        res[t] = {"n_val": len(zv), "explained_top50": float(p.explained_variance_ratio_.sum()),
                  "floor": {k: {"mean": float(np.mean([f[k] for f in fl])), "min": float(np.min([f[k] for f in fl])), "max": float(np.max([f[k] for f in fl]))} for k in keys},
                  "floor_var_ratio_per_pc": np.mean([f["var_ratio_per_pc"] for f in fl], 0).tolist()}
        print(t, "floor", {k: round(res[t]["floor"][k]["mean"], 3) for k in keys}, flush=True)
        for j, r in enumerate(run_list(t)):
            z = p.transform(log_cp10k(load_cells(r, t)))
            rows.append({"arm": r["arm"], "train": r["train"], "samp": r["samp"], "r3_seed": SEED + r.get("r3_j", j), **score(zv, z, SEED + r.get("r3_j", j), var_val)})
        res[t]["runs"] = rows
        for a in ARMS:
            rr = [q for q in rows if q["arm"] == a]
            print(" ", a, {k: round(float(np.mean([q[k] for q in rr])), 3) for k in keys}, flush=True)
        (OUT / "r3_numbers.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


def part_b():
    from figures import hvgfig as H
    from analysis.score_purity import annotate
    t = "Neuronal"
    genes = json.loads((H.DATA / "meta.json").read_text(encoding="utf-8"))["genes"]
    p, xt = basis(t); xv_raw = real_cells("val", t); zv = p.transform(log_cp10k(xv_raw)); var_val = zv.var(0, ddof=1)
    um = H.FrozenUMAP(); xval, yval = H.real_cells(); xval = np.asarray(xval, dtype=np.float64); neu = yval == t
    idx = NearestNeighbors(n_neighbors=15, algorithm="brute", n_jobs=NJ).fit(((H.cpm_log1p(xval) - um.mean) @ um.comp.T).astype(np.float32))
    c = np.load(ROOT / "rebuttal/out/R1/r1_clumps.npz"); t0 = cKDTree(c["ours_xy"][c["clump"] == 0])
    def spread(u):
        m = t0.query(u)[0] <= 0.5
        return {"main_share": float(m.mean()), "main_y_sd": float(u[m, 1].std(ddof=1)), "main_share_y_gt_2p9": float((u[m, 1] > 2.9).mean())}
    arms = {}
    stem = "Neuronal_residual_hvgtwist_fk_tiltevery_trigany_K32_M5000_N2500_alpha1p0_J16_int8_th0p5_tau{}_cc256_odds_seed{}.npy"
    S = (20260987, 20260988, 20260989)
    arms["tau0.2 (deployed)"] = [RUNS / "table1_ours/Neuronal_a1" / stem.format("0p2", s) for s in S]
    arms["tau0.4"] = [RUNS / "tau_sweep/Neuronal_tau0p4" / stem.format("0p4", s) for s in S]
    arms["tau1.0"] = [RUNS / "tau_sweep/Neuronal_tau1p0" / stem.format("1p0", s) for s in S]
    arms["top0 (R9)"] = [Path(f) for f in sorted(glob.glob(str(RUNS / "rebuttal_R9/Neuronal_top0/*.npy")))]
    arms["FK only (ablation_table10)"] = [Path(f) for f in sorted(glob.glob(str(RUNS / "ablation_table10/Neuronal_fk_only/*.npy")))]
    arms["tilt only (ablation_table10)"] = [Path(f) for f in sorted(glob.glob(str(RUNS / "ablation_table10/Neuronal_tilt_only/*.npy")))]
    v8, v7 = {"NC1_glial", "NC2_glial_NGF+", "NC3_glial"}, {"NC1_glial"}
    out = {}
    real_um = um.real_xy[neu]
    out["real_val_spread"] = spread(real_um)
    out["real_train_spread"] = spread(np.load(ROOT / "rebuttal/out/R1/train_neuronal_umap.npy"))
    for a, fs in arms.items():
        rows = []
        for j, f in enumerate(fs):
            x = np.asarray(np.load(f, mmap_mode="r"), dtype=np.float64)
            js = json.loads(f.with_suffix(".json").read_text(encoding="utf-8")) if f.with_suffix(".json").exists() else {}
            lab = annotate(x, genes)
            ii = idx.kneighbors(((H.cpm_log1p(x) - um.mean) @ um.comp.T).astype(np.float32), return_distance=False)
            sc = score(zv, p.transform(log_cp10k(x)), SEED + j, var_val)
            rows.append({"file": f.name, "seed": js.get("seed"), "checkpoint_sha256": str(js.get("checkpoint_sha256"))[:12],
                         "unique_lineage_fraction": js.get("unique_lineage_fraction"), "v8": float(np.isin(lab, list(v8)).mean()),
                         "v7": float(np.isin(lab, list(v7)).mean()), "pca50_frac": float(neu[ii].mean()), **spread(um.project(x)),
                         **{k: v for k, v in sc.items() if k != "var_ratio_per_pc"}})
        out[a] = rows
        if rows:
            print(a, {k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("var_ratio_top10", "var_ratio_top50", "recall", "coverage", "precision", "v8", "v7", "pca50_frac", "main_share", "main_y_sd", "main_share_y_gt_2p9")}, [r["unique_lineage_fraction"] for r in rows], {r["checkpoint_sha256"] for r in rows}, flush=True)
    (OUT / "r3_partB.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    if "--only-b" not in sys.argv:
        part_a()
    if "--only-a" not in sys.argv:   # R12: part B has no CountsDiff arm
        part_b()
    print("[done]", flush=True)
