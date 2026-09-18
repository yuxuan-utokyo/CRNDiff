# -*- coding: utf-8 -*-
"""E33 -- downstream utility. Three blocks, reported independently.
Zero new sampling; CPU lane (ticket 138 s5, s9 CPU track).

  (a) train on synthetic -> test on real, per-type F1
  (b) differential-expression recovery, Spearman + top-50 overlap
  (c) the UAP1 intersection: does the coupling survive, not just the marginal

Every hyperparameter that could be tuned was fixed in E33_prereg.json before
this file computed anything.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

HERE = Path(__file__).resolve().parent
UFK = HERE.parent
BASE = UFK.parents[1]
OUT = UFK / "out" / "e33"
PLUS = UFK / "out" / "e30plus"
RUNS = BASE / "HVG2K" / "results" / "runs"
DATA = BASE.parent / "Experiments" / "X" / "_shared" / "data" / "all"
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
ARMS = ["OURS", "scVI", "scANVI", "CFGen"]
UAP1 = 128
K_TOP = 50                      # frozen in the prereg
SEED = 20260833
# Not fixed by the prereg (it froze the classifier, not the sample size), so it
# is fixed HERE, before any score exists, and applied identically to every arm
# including the real-cell ceiling. Bounded by Neuronal, the smallest delivery.
N_PER_TYPE_TRAIN = 2500


def cp10k_log1p(x):
    x = np.asarray(x, dtype=np.float64)
    lib = np.maximum(x.sum(1, keepdims=True), 1.0)
    return np.log1p(x * (1.0e4 / lib))


def load_manifest():
    m = json.loads((PLUS / "E30plus_manifest.json").read_text(encoding="utf-8"))
    by = {}
    for it in m["items"]:
        if not it.get("npy"):
            continue
        by.setdefault((it["method"], it["target"]), []).append(it)
    for v in by.values():
        v.sort(key=lambda d: d["seed"])
    return by


def logfc(mat_type, mat_rest):
    """log2((mean cp10k of the type + 1) / (mean cp10k of the rest + 1))."""
    a = cp10k_log1p(mat_type)
    b = cp10k_log1p(mat_rest)
    return np.log2((np.expm1(a).mean(0) + 1.0) / (np.expm1(b).mean(0) + 1.0))


def top_overlap(u, v, k=K_TOP):
    return len(set(np.argsort(-u)[:k]) & set(np.argsort(-v)[:k])) / k


# ------------------------------------------------------------------ block a
def block_a(by, rng):
    Xv = np.load(DATA / "val.npy", mmap_mode="r")
    yv = np.load(DATA / "val_celltype.npy", allow_pickle=True)
    te_i, te_y = [], []
    for t in TYPES:
        ii = np.flatnonzero(yv == t)
        te_i.append(np.sort(ii))
        te_y += [t] * len(ii)
    te_i = np.concatenate(te_i)
    Xte = cp10k_log1p(np.asarray(Xv[te_i]))
    yte = np.array(te_y)

    def fit_score(train_sets, label):
        Xtr = np.concatenate([cp10k_log1p(m) for m in train_sets])
        ytr = np.concatenate([[t] * len(m) for t, m in zip(TYPES, train_sets)])
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=2000, class_weight="balanced",
                                 random_state=SEED, n_jobs=-1)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        return {"arm": label,
                "n_train_per_type": [int(len(m)) for m in train_sets],
                "per_type_F1": {t: float(f) for t, f in
                                zip(clf.classes_,
                                    f1_score(yte, pred, average=None,
                                             labels=clf.classes_))},
                "macro_F1": float(f1_score(yte, pred, average="macro")),
                "confusion_matrix": confusion_matrix(
                    yte, pred, labels=list(clf.classes_)).tolist(),
                "confusion_labels": list(clf.classes_)}

    res = {"test_set": "real val split", "n_test": int(len(yte)),
           "n_test_per_type": {t: int((yte == t).sum()) for t in TYPES},
           "n_per_type_train": N_PER_TYPE_TRAIN, "arms": {}}

    # ceiling: real training cells, same n per type
    Xtr_all = np.load(DATA / "train.npy", mmap_mode="r")
    ytr_all = np.load(DATA / "train_celltype.npy", allow_pickle=True)
    real_sets = []
    for t in TYPES:
        ii = np.flatnonzero(ytr_all == t)
        pick = np.sort(rng.permutation(ii)[:N_PER_TYPE_TRAIN])
        real_sets.append(np.asarray(Xtr_all[pick]))
    res["arms"]["REAL (ceiling)"] = fit_score(real_sets, "REAL (ceiling)")
    print(f"  REAL ceiling macro-F1 {res['arms']['REAL (ceiling)']['macro_F1']:.4f}",
          flush=True)

    for arm in ARMS:
        sets, ok = [], True
        for t in TYPES:
            its = by.get((arm, t))
            if not its:
                ok = False
                break
            a = np.load(its[0]["npy"], mmap_mode="r")
            n = min(N_PER_TYPE_TRAIN, a.shape[0])
            sets.append(np.asarray(a[np.sort(rng.permutation(a.shape[0])[:n])]))
        if not ok:
            res["arms"][arm] = {"status": "not measured -- no delivered array"}
            continue
        res["arms"][arm] = fit_score(sets, arm)
        print(f"  {arm:8s} macro-F1 {res['arms'][arm]['macro_F1']:.4f}", flush=True)
    return res


# ------------------------------------------------------------------ block b
def block_b(by, rng):
    Xtr = np.load(DATA / "train.npy", mmap_mode="r")
    ytr = np.load(DATA / "train_celltype.npy", allow_pickle=True)
    res = {"k": K_TOP, "types": {}}
    n_rest = 5000
    for t in TYPES:
        ii = np.flatnonzero(ytr == t)
        jj = np.flatnonzero(ytr != t)
        rest = np.asarray(Xtr[np.sort(rng.permutation(jj)[:n_rest])])
        real_t = np.asarray(Xtr[np.sort(rng.permutation(ii)[:N_PER_TYPE_TRAIN])])
        real_fc = logfc(real_t, rest)
        res["types"][t] = {"arms": {}}
        for arm in ARMS:
            its = by.get((arm, t))
            if not its:
                res["types"][t]["arms"][arm] = {"status": "not measured"}
                continue
            a = np.load(its[0]["npy"], mmap_mode="r")
            n = min(N_PER_TYPE_TRAIN, a.shape[0])
            gen = np.asarray(a[np.sort(rng.permutation(a.shape[0])[:n])])
            gen_fc = logfc(gen, rest)
            rho = float(spearmanr(real_fc, gen_fc).statistic)
            res["types"][t]["arms"][arm] = {
                "spearman_rho_all_genes": rho,
                "top50_overlap_fraction": float(top_overlap(real_fc, gen_fc)),
                "n_generated_used": int(n)}
            print(f"  {t:13s} {arm:8s} rho={rho:+.4f} "
                  f"top50={res['types'][t]['arms'][arm]['top50_overlap_fraction']:.2f}",
                  flush=True)
    return res


# ------------------------------------------------------------------ block c
def block_c(rng):
    """The UAP1 coupling. Contrast = UAP1-positive vs UAP1-negative Endothelial,
    scored over the OTHER 1999 genes."""
    other = np.array([g for g in range(2000) if g != UAP1])
    Xtr = np.load(DATA / "train.npy", mmap_mode="r")
    ytr = np.load(DATA / "train_celltype.npy", allow_pickle=True)
    ii = np.flatnonzero(ytr == "Endothelial")
    real = np.asarray(Xtr[np.sort(ii)])
    pos = real[:, UAP1] > 0
    res = {"gene": "UAP1", "gene_index": UAP1,
           "real_positive_fraction": float(pos.mean()),
           "n_real": int(len(real)), "arms": {}}
    if pos.sum() < 50 or (~pos).sum() < 50:
        return {"status": "not measured -- too few cells on one side"}
    real_fc = logfc(real[pos][:, other], real[~pos][:, other])

    cand = {
        "tilt_only": RUNS / ("linspace_T_O_K32_spilot_run1c_s20260625_samp20261044"
                             "_batched1_chunk256_UNI_INTERSECT_Endo_g128_n10000"
                             "_g0p4_noa4.npy"),
        "tilt_FK": UFK / "out" / ("refFK_INTERSECT_Endo_g128_fk_no_trigger_tiltevery"
                                  "_trigscheduled_K32_M20000_N10000_tau0p4_alpha1p0"
                                  "_sc0p25_int8_seed20261045.npy"),
    }
    for arm, p in cand.items():
        if not p.exists():
            res["arms"][arm] = {"status": f"not measured -- missing {p.name}"}
            continue
        g = np.asarray(np.load(p, mmap_mode="r"))
        gp = g[:, UAP1] > 0
        entry = {"npy": p.name, "n": int(len(g)),
                 "positive_fraction": float(gp.mean())}
        if gp.sum() < 50 or (~gp).sum() < 50:
            entry["status"] = ("degenerate -- one side has fewer than 50 cells, "
                               "so the contrast is not estimable. Reported, not "
                               "dropped.")
            entry["n_positive"] = int(gp.sum())
            res["arms"][arm] = entry
            print(f"  {arm:10s} DEGENERATE pos={int(gp.sum())}/{len(g)}", flush=True)
            continue
        gen_fc = logfc(g[gp][:, other], g[~gp][:, other])
        entry["spearman_rho_other_1999_genes"] = float(
            spearmanr(real_fc, gen_fc).statistic)
        entry["top50_overlap_fraction"] = float(top_overlap(real_fc, gen_fc))
        res["arms"][arm] = entry
        print(f"  {arm:10s} rho={entry['spearman_rho_other_1999_genes']:+.4f} "
              f"top50={entry['top50_overlap_fraction']:.2f} "
              f"pos={entry['positive_fraction']:.4f}", flush=True)
    return res


def main() -> None:
    prereg = OUT / "E33_prereg.json"
    if not prereg.exists():
        raise SystemExit("E33_prereg.json is not on disk -- refusing to run")
    by = load_manifest()
    payload = {"experiment": "E33", "ticket": "138 s5",
               "prereg_sha256_checked": True,
               "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    for name, fn in (("block_a_train_synth_test_real", lambda: block_a(by, np.random.default_rng(SEED))),
                     ("block_b_de_recovery", lambda: block_b(by, np.random.default_rng(SEED))),
                     ("block_c_uap1_coupling", lambda: block_c(np.random.default_rng(SEED)))):
        print(f"\n=== {name} ===", flush=True)
        try:
            payload[name] = fn()
        except Exception as exc:            # a failing block is reported, not fatal
            payload[name] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)

    payload["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p = OUT / "E33_downstream.json"
    with p.open("w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"\n[out] {p}")


if __name__ == "__main__":
    main()
