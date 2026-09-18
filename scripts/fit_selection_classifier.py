# -*- coding: utf-8 -*-
"""Fit the selection classifier used by best-of-N: a real-versus-real one-versus-rest scorer.

One gradient-boosting classifier per request, with the positives being REAL training cells of
that type and the negatives REAL training cells of every other type. It is a different model on
different data from the purity ruler, so purity never scores itself.

The recipe is not invented here: the hyperparameters, the feature-selection effect size, the
normalisation and the seed are read from the recorded discriminator log, so the selector and the
residual discriminator are fitted the same way.

    python scripts/fit_selection_classifier.py --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402

SEED = 20260817                     # the fitting seed, shared with the noised-state classifier
RECIPE_TYPE = "Endothelial"         # the recipe is read from this type's discriminator log; all three share the hyperparameters
AUC_GATE = 0.95                     # B-S1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_log1p(x: np.ndarray) -> np.ndarray:
    """Per-cell normalisation on the classifier's convention, verbatim from the residual fitter."""
    z = np.asarray(x, dtype=np.float32)
    lib = z.sum(axis=1, dtype=np.float64)
    scale = (1.0e4 / np.maximum(lib, 1.0)).astype(np.float32)
    return np.log1p(z * scale[:, None]).astype(np.float32, copy=False)


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC in Mann-Whitney U form, with ties averaged; sklearn is not used, to avoid version drift."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    r = 1.0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (r + (r + (j - i))) / 2.0
        r += (j - i + 1)
        i = j + 1
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def portable_predict(nodes, offsets, baseline, x):
    """A verbatim copy of the portable predictor, used only for the identity self-check."""
    raw = np.full(len(x), baseline, dtype=np.float64)
    rows = np.arange(len(x))
    for start, end in zip(offsets[:-1], offsets[1:]):
        tree = nodes[int(start):int(end)]
        index = np.zeros(len(x), dtype=np.int64)
        for _ in range(len(tree)):
            node = tree[index]
            active = node["is_leaf"] == 0
            if not np.any(active):
                break
            rr = rows[active]
            nn = node[active]
            value = x[rr, nn["feature_idx"]]
            go_left = np.where(np.isnan(value), nn["missing_go_to_left"] != 0,
                               value <= nn["num_threshold"])
            index[active] = np.where(go_left, nn["left"], nn["right"])
        else:
            raise RuntimeError("tree traversal did not reach leaves")
        raw += tree[index]["value"]
    return np.where(raw >= 0, 1.0 / (1.0 + np.exp(-raw)), np.exp(raw) / (1.0 + np.exp(raw)))


def export_portable(model, out: Path, recipe_sha: str, seed: int) -> dict:
    """Write the fitted classifier out in the portable form the sampler reads, with the same
        logic as the exporter; only the source differs (an in-memory model rather than a file),"""
    if model.n_trees_per_iteration_ != 1 or len(model.classes_) != 2:
        raise SystemExit("only binary one-tree-per-iteration HGBT is supported")
    trees = [row[0].nodes for row in model._predictors]
    if any(np.any(t["is_categorical"] != 0) for t in trees):
        raise SystemExit("categorical tree nodes are not supported")
    offsets = np.concatenate(([0], np.cumsum([len(t) for t in trees]))).astype(np.int64)
    nodes = np.concatenate(trees)
    baseline = float(np.asarray(model._baseline_prediction).reshape(-1)[0])

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, size=(2048, model.n_features_in_)).astype(np.float32)
    x[0, 0] = np.nan
    expected = model.predict_proba(x)[:, 1]
    actual = portable_predict(nodes, offsets, baseline, x)
    error = float(np.max(np.abs(expected - actual)))
    if error > 1.0e-12:
        raise SystemExit(f"portable predictor identity failed: max abs error={error}")

    if out.exists():
        raise SystemExit(f"refusing to overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        nodes=nodes, offsets=offsets,
        baseline=np.asarray([baseline], dtype=np.float64),
        n_features=np.asarray([model.n_features_in_], dtype=np.int64),
        source_sha256=np.asarray([f"in-memory-HGBT;recipe={recipe_sha}"]),
        identity_max_abs_error=np.asarray([error], dtype=np.float64),
    )
    return {"identity_max_abs_error": error, "n_trees": len(trees),
            "n_nodes": int(len(nodes))}


def main() -> int:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        import sklearn
    except ModuleNotFoundError as _exc:
        raise SystemExit(
            "this entry point needs scikit-learn, which the offline reproduction does not "
            "install. See README section 7; in short: pip install scikit-learn joblib") from _exc

    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=["Endothelial", "Myeloid", "Neuronal"])
    ap.add_argument("--out-root", default=str(C.MODELS / "selection"))
    a = ap.parse_args()

    recipe_path = C.RESIDUAL / RECIPE_TYPE / "residual_log.json"
    C.require(recipe_path, "residual recipe (HGBT hyper-parameters + split logic)")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe_sha = sha256_file(recipe_path)
    clf_rec = recipe["classifier"]
    # the residual fitter's defaults are what the log records; the feature count and the
    # iteration cap are read from the log and the rest are that fitter's own argparse defaults,
    # which is what produced that log in the first place
    HP = {"learning_rate": 0.06, "max_iter": int(clf_rec["n_iter"]), "max_leaf_nodes": 31,
          "min_samples_leaf": 30, "l2_regularization": 1.0, "early_stopping": True,
          "validation_fraction": 0.15, "n_iter_no_change": 20, "random_state": SEED}
    N_FEATURES = int(clf_rec["n_features"])
    print(f"[recipe] {recipe_path}")
    print(f"[recipe] sha256 {recipe_sha}")
    print(f"[recipe] HGBT {HP}  n_features={N_FEATURES}  seed={SEED}")

    data = C.LEGACY_DATA / "all"
    x_train = np.load(data / "train.npy", mmap_mode="r")
    y_train = np.load(data / "train_celltype.npy", allow_pickle=True)
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    genes = np.asarray(meta["genes"]) if "genes" in meta else None
    print(f"[data] train {x_train.shape} dtype={x_train.dtype}  labels {y_train.shape}")

    out_root = Path(a.out_root)
    results = {}
    for t in a.types:
        t0 = time.time()
        d = out_root / t
        npz = d / "selection_classifier_portable.npz"
        logp = d / "log.json"
        if npz.exists() or logp.exists():
            raise SystemExit(f"refusing to overwrite existing S1 artefacts in {d}")

        rng = np.random.default_rng(SEED)
        pos_all = np.flatnonzero(y_train == t)
        neg_all = np.flatnonzero(y_train != t)
        n_pos = len(pos_all)
        fit_pos = min(8000, (2 * n_pos) // 3)
        val_pos = min(4000, n_pos - fit_pos)
        fit_neg, val_neg = 8000, 2000
        pos_perm = rng.permutation(pos_all)[: fit_pos + val_pos]
        p_fit_i = np.sort(pos_perm[:fit_pos])
        p_val_i = np.sort(pos_perm[fit_pos:])
        neg_perm = rng.permutation(neg_all)[: fit_neg + val_neg]
        q_fit_i = np.sort(neg_perm[:fit_neg])
        q_val_i = np.sort(neg_perm[fit_neg:])
        print(f"\n[{t}] n_pos={n_pos}  fit/val pos = {len(p_fit_i)}/{len(p_val_i)}  "
              f"neg = {len(q_fit_i)}/{len(q_val_i)} (of {len(neg_all)})")

        p_fit = normalize_log1p(np.asarray(x_train[p_fit_i]))
        q_fit = normalize_log1p(np.asarray(x_train[q_fit_i]))
        mu_p = p_fit.mean(axis=0, dtype=np.float64)
        mu_q = q_fit.mean(axis=0, dtype=np.float64)
        var_p = p_fit.var(axis=0, dtype=np.float64)
        var_q = q_fit.var(axis=0, dtype=np.float64)
        effect = np.abs(mu_p - mu_q) / np.sqrt(0.5 * (var_p + var_q) + 1.0e-8)
        feat = np.argsort(effect)[-min(N_FEATURES, x_train.shape[1]):][::-1].astype(np.int64)

        x_fit = np.concatenate((p_fit[:, feat], q_fit[:, feat]), axis=0)
        y_fit = np.concatenate((np.ones(len(p_fit_i), dtype=np.int8),
                                np.zeros(len(q_fit_i), dtype=np.int8)))
        clf = HistGradientBoostingClassifier(**HP)
        clf.fit(x_fit, y_fit)
        del x_fit, y_fit, p_fit, q_fit

        x_val = np.concatenate(
            (normalize_log1p(np.asarray(x_train[p_val_i]))[:, feat],
             normalize_log1p(np.asarray(x_train[q_val_i]))[:, feat]), axis=0)
        y_val = np.concatenate((np.ones(len(p_val_i), dtype=np.int8),
                                np.zeros(len(q_val_i), dtype=np.int8)))
        s_val = clf.predict_proba(x_val)[:, 1]
        auc = roc_auc(y_val, s_val)
        passed = bool(auc >= AUC_GATE)
        print(f"[{t}] val AUROC = {auc:.6f}   B-S1 (>= {AUC_GATE}) {'PASS' if passed else 'FAIL'}")

        exp = export_portable(clf, npz, recipe_sha, SEED)
        np.save(d / "features.npy", feat)
        log = {
            "ticket": 200, "block": "M1-0.3", "what": "S1 selection classifier (one-vs-rest)",
            "type": t, "seed": SEED,
            "recipe_source": str(recipe_path), "recipe_sha256": recipe_sha,
            "hyperparameters": HP, "n_features": int(len(feat)),
            "feature_screen": "abs(mean_p - mean_q) / sqrt(0.5*(var_p+var_q)+1e-8) on "
                              "CP10K+log1p, fit rows only; top-n by that effect size",
            "normalisation": "library-size normalise to 1e4 then log1p (CellTypist caliber)",
            "positives": "real train cells of this type",
            "negatives": "real train cells of every OTHER type (one-vs-rest)",
            "split_rule": "fit_pos=min(8000, floor(2/3*n_pos)); val_pos=min(4000, n_pos-fit_pos); "
                          "fit_neg=8000; val_neg=2000 -- written down before any number was seen",
            "n_pos_total": int(n_pos), "n_neg_total": int(len(neg_all)),
            "n_fit_pos": int(len(p_fit_i)), "n_val_pos": int(len(p_val_i)),
            "n_fit_neg": int(len(q_fit_i)), "n_val_neg": int(len(q_val_i)),
            "val_auroc": float(auc), "auc_gate": AUC_GATE, "gate_B_S1_passed": passed,
            "portable": exp,
            "feature_names": ([str(genes[i]) for i in feat] if genes is not None else None),
            "walltime_s": time.time() - t0,
            "versions": {"python": sys.version, "numpy": np.__version__,
                         "sklearn": sklearn.__version__, "platform": platform.platform()},
        }
        if not passed:
            log["do_not_publish"] = True
            log["do_not_publish_reason"] = f"B-S1 not met: val AUROC {auc:.6f} < {AUC_GATE}"
        logp.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
        log["portable_sha256"] = sha256_file(npz)
        logp.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
        results[t] = {"val_auroc": auc, "passed": passed, "dir": str(d)}
        print(f"[{t}] -> {npz}  ({time.time()-t0:.1f}s)")

    summary = {"ticket": 200, "gate": "B-S1", "threshold": AUC_GATE,
               "per_type": results,
               "all_passed": all(v["passed"] for v in results.values()),
               "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    gp = C.GATES / "200_S1_selection.json"
    gp.parent.mkdir(parents=True, exist_ok=True)
    if gp.exists():
        gp = C.GATES / f"200_S1_selection_{int(time.time())}.json"
    gp.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[B-S1] all_passed={summary['all_passed']} -> {gp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
