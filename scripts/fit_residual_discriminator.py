"""Fit and apply a clean-endpoint residual correction for HVG generation.

This wrapper is intentionally downstream of the frozen CRN sampler: it reads
an existing product-tilted proposal, learns target-vs-proposal discrepancy on
disjoint clean endpoints, and resamples a candidate-only split.  It does not
modify the generator or any baseline numerical path.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

try:
    import joblib
except ModuleNotFoundError as _exc:                                      # noqa: E402
    raise SystemExit(
        "this entry point needs joblib, which the offline reproduction does not "
        "install. See README section 7; in short: pip install joblib scikit-learn") from _exc
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
COND = HERE.parent
DEFAULT_X_ROOT = Path(__file__).resolve().parents[1] / "assets" / "legacy_root"
DEFAULT_PROPOSAL = (
    DEFAULT_X_ROOT
    / "Baseline"
    / "OURS"
    / "results"
    / (
        "linspace_T_O_K32_spilot_run1c_s20260625_samp20260902_"
        "batched1_chunk256_CONDFULL_Endothelial_tables_kz_"
        "n20000_g0p4_noa4.npy"
    )
)


def normalize_log1p(x: np.ndarray) -> np.ndarray:
    """CellTypist-compatible per-cell normalization, returned as float32."""
    z = np.asarray(x, dtype=np.float32)
    lib = z.sum(axis=1, dtype=np.float64)
    scale = (1.0e4 / np.maximum(lib, 1.0)).astype(np.float32)
    return np.log1p(z * scale[:, None]).astype(np.float32, copy=False)


def systematic_resample(weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or len(w) == 0 or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("invalid resampling weights")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("all resampling weights are zero")
    cdf = np.cumsum(w / s)
    cdf[-1] = 1.0
    u = (rng.random() + np.arange(n, dtype=np.float64)) / n
    return np.searchsorted(cdf, u, side="right").astype(np.int64)


def weight_stats(w: np.ndarray, chosen: np.ndarray) -> dict[str, float]:
    w = np.asarray(w, dtype=np.float64)
    ess = float(w.sum() ** 2 / (len(w) * np.square(w).sum()))
    return {
        "ess_fraction": ess,
        "unique_source_fraction": float(np.unique(chosen).size / len(chosen)),
        "weight_min": float(w.min()),
        "weight_median": float(np.median(w)),
        "weight_max": float(w.max()),
    }


def marginal_stats(x: np.ndarray, target_mean: np.ndarray) -> dict[str, float]:
    z = np.asarray(x, dtype=np.float64)
    mean = z.mean(axis=0)
    pos = target_mean > 0
    rel = np.abs(mean[pos] - target_mean[pos]) / target_mean[pos]
    lib = z.sum(axis=1)
    return {
        "mean_relative_median": float(np.median(rel)),
        "mean_relative_p90": float(np.quantile(rel, 0.9)),
        "mean_absolute_median": float(np.median(np.abs(mean - target_mean))),
        "zero_fraction": float(np.mean(z == 0)),
        "library_median": float(np.median(lib)),
        "library_mean": float(np.mean(lib)),
    }


def corr_matrix(x: np.ndarray, genes: np.ndarray) -> np.ndarray:
    z = normalize_log1p(np.asarray(x)[:, genes]).astype(np.float64)
    z -= z.mean(axis=0, keepdims=True)
    sd = np.sqrt(np.square(z).sum(axis=0))
    z /= np.maximum(sd, 1.0e-12)
    c = z.T @ z
    np.fill_diagonal(c, 1.0)
    return c


def corr_error(x: np.ndarray, target_corr: np.ndarray, genes: np.ndarray) -> float:
    c = corr_matrix(x, genes)
    mask = ~np.eye(len(genes), dtype=bool)
    denom = np.linalg.norm(target_corr[mask])
    return float(np.linalg.norm((c - target_corr)[mask]) / max(denom, 1.0e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-root", default=str(DEFAULT_X_ROOT))
    ap.add_argument("--proposal", default=str(DEFAULT_PROPOSAL))
    ap.add_argument("--target", default="Endothelial")
    ap.add_argument("--fit-neg", type=int, default=8000)
    ap.add_argument("--val-neg", type=int, default=2000)
    ap.add_argument("--candidate", type=int, default=10000)
    ap.add_argument("--fit-pos", type=int, default=8000)
    ap.add_argument("--val-pos", type=int, default=4000)
    ap.add_argument("--n-features", type=int, default=256)
    ap.add_argument("--corr-features", type=int, default=64)
    ap.add_argument("--n-out", type=int, default=2500)
    ap.add_argument("--alphas", default="0,0.5,1,2")
    ap.add_argument("--auc-gate", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--max-iter", type=int, default=180)
    ap.add_argument("--learning-rate", type=float, default=0.06)
    ap.add_argument("--max-leaf-nodes", type=int, default=31)
    ap.add_argument("--min-samples-leaf", type=int, default=30)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("-o", "--out",
                    default=str(Path(__file__).resolve().parents[1]
                                / "out" / "residual_Endothelial"))
    a = ap.parse_args()

    xroot = Path(a.x_root)
    data = xroot / "_shared" / "data" / "all"
    proposal_path = Path(a.proposal)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in (proposal_path, data / "train.npy", data / "train_celltype.npy", data / "meta.json"):
        if not p.exists():
            raise SystemExit(f"missing required input: {p}")

    rng = np.random.default_rng(a.seed)
    q = np.load(proposal_path, mmap_mode="r")
    if q.ndim != 2:
        raise SystemExit(f"proposal must be a matrix, got {q.shape}")
    need_q = a.fit_neg + a.val_neg + a.candidate
    if need_q > q.shape[0]:
        raise SystemExit(f"proposal needs {need_q} rows, only has {q.shape[0]}")
    q_perm = rng.permutation(q.shape[0])[:need_q]
    q_fit_i = np.sort(q_perm[: a.fit_neg])
    q_val_i = np.sort(q_perm[a.fit_neg : a.fit_neg + a.val_neg])
    q_cand_i = np.sort(q_perm[a.fit_neg + a.val_neg :])

    x_train = np.load(data / "train.npy", mmap_mode="r")
    y_train = np.load(data / "train_celltype.npy", allow_pickle=True)
    pos_all = np.flatnonzero(y_train == a.target)
    need_pos = a.fit_pos + a.val_pos
    if need_pos > len(pos_all):
        raise SystemExit(f"target {a.target} needs {need_pos} references, only has {len(pos_all)}")
    pos_perm = rng.permutation(pos_all)[:need_pos]
    p_fit_i = np.sort(pos_perm[: a.fit_pos])
    p_val_i = np.sort(pos_perm[a.fit_pos :])

    print(f"[split] proposal fit/val/candidate = {len(q_fit_i)}/{len(q_val_i)}/{len(q_cand_i)}")
    print(f"[split] target fit/val = {len(p_fit_i)}/{len(p_val_i)} of {len(pos_all)}")

    p_fit_full = np.asarray(x_train[p_fit_i])
    q_fit_full = np.asarray(q[q_fit_i])
    p_fit_norm = normalize_log1p(p_fit_full)
    q_fit_norm = normalize_log1p(q_fit_full)

    # Fit-only screen.  It is deterministic and never sees validation or candidate rows.
    mu_p = p_fit_norm.mean(axis=0, dtype=np.float64)
    mu_q = q_fit_norm.mean(axis=0, dtype=np.float64)
    var_p = p_fit_norm.var(axis=0, dtype=np.float64)
    var_q = q_fit_norm.var(axis=0, dtype=np.float64)
    effect = np.abs(mu_p - mu_q) / np.sqrt(0.5 * (var_p + var_q) + 1.0e-8)
    feat = np.argsort(effect)[-min(a.n_features, q.shape[1]) :][::-1].astype(np.int64)

    x_fit = np.concatenate((p_fit_norm[:, feat], q_fit_norm[:, feat]), axis=0)
    y_fit = np.concatenate((np.ones(len(p_fit_i), dtype=np.int8), np.zeros(len(q_fit_i), dtype=np.int8)))
    clf = HistGradientBoostingClassifier(
        learning_rate=a.learning_rate,
        max_iter=a.max_iter,
        max_leaf_nodes=a.max_leaf_nodes,
        min_samples_leaf=a.min_samples_leaf,
        l2_regularization=a.l2,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=a.seed,
    )
    clf.fit(x_fit, y_fit)
    del x_fit, y_fit, p_fit_norm, q_fit_norm

    p_val = np.asarray(x_train[p_val_i])
    q_val = np.asarray(q[q_val_i])
    x_val = np.concatenate(
        (normalize_log1p(p_val)[:, feat], normalize_log1p(q_val)[:, feat]), axis=0
    )
    y_val = np.concatenate((np.ones(len(p_val_i), dtype=np.int8), np.zeros(len(q_val_i), dtype=np.int8)))
    val_score = clf.predict_proba(x_val)[:, 1]
    auc = float(roc_auc_score(y_val, val_score))
    passed = bool(auc >= a.auc_gate)
    print(f"[gate] held-out ROC AUC = {auc:.6f}; threshold {a.auc_gate:.3f}; passed={passed}")

    candidate = np.asarray(q[q_cand_i])
    score = clf.predict_proba(normalize_log1p(candidate)[:, feat])[:, 1]
    score = np.clip(score.astype(np.float64), 1.0e-8, 1.0)
    if not passed:
        score[:] = 1.0

    genes = json.loads((data / "meta.json").read_text(encoding="utf-8"))["genes"]
    target_mean = np.asarray(p_fit_full, dtype=np.float64).mean(axis=0)
    corr_rank = np.argsort(var_p)[-min(a.corr_features, q.shape[1]) :][::-1].astype(np.int64)
    target_corr = corr_matrix(p_val, corr_rank)

    split_file = out / "split_indices.npz"
    np.savez_compressed(
        split_file,
        proposal_fit=q_fit_i,
        proposal_val=q_val_i,
        proposal_candidate=q_cand_i,
        target_fit=p_fit_i,
        target_val=p_val_i,
        classifier_features=feat,
        correlation_features=corr_rank,
    )
    joblib.dump(clf, out / "residual_classifier.joblib")

    alphas = [float(v.strip()) for v in a.alphas.split(",") if v.strip()]
    records = []
    for j, alpha in enumerate(alphas):
        if alpha == 0.0 or not passed:
            w = np.ones_like(score)
        else:
            w = np.power(score, alpha)
        arm_rng = np.random.default_rng(a.seed + 1000 + j)
        chosen = systematic_resample(w, a.n_out, arm_rng)
        sample = candidate[chosen]
        tag = str(alpha).replace(".", "p")
        safe_target = "".join(c if (c.isalnum() or c in "-_") else "_" for c in a.target)
        sample_path = out / f"{safe_target}_residual_alpha{tag}_n{a.n_out}.npy"
        np.save(sample_path, sample)
        rec = {
            "alpha": alpha,
            "primary": bool(math.isclose(alpha, 1.0)),
            "sample": sample_path.name,
            **weight_stats(w, chosen),
            **marginal_stats(sample, target_mean),
            "correlation_relative_frobenius": corr_error(sample, target_corr, corr_rank),
        }
        records.append(rec)
        print(
            f"[arm alpha={alpha:g}] ESS={rec['ess_fraction']:.4f} "
            f"unique={rec['unique_source_fraction']:.4f} "
            f"mean-rel-med={rec['mean_relative_median']:.4f} "
            f"corr={rec['correlation_relative_frobenius']:.4f}"
        )

    candidate_stats = {
        **marginal_stats(candidate, target_mean),
        "correlation_relative_frobenius": corr_error(candidate, target_corr, corr_rank),
    }
    def marginal_ok(rec: dict[str, float]) -> bool:
        return bool(
            rec["mean_relative_median"]
            <= max(0.35, 1.5 * candidate_stats["mean_relative_median"])
        )

    primary = next((r for r in records if r["primary"]), None)
    marginal_gate = bool(primary is not None and marginal_ok(primary))
    production = None
    if passed:
        for alpha in np.round(np.arange(1.0, 0.0, -0.05), 2):
            w = np.power(score, float(alpha))
            chosen = systematic_resample(
                w, a.n_out, np.random.default_rng(a.seed + 7077)
            )
            sample = candidate[chosen]
            trial = {
                "alpha": float(alpha),
                **weight_stats(w, chosen),
                **marginal_stats(sample, target_mean),
            }
            if (
                trial["ess_fraction"] >= 0.20
                and trial["unique_source_fraction"] >= 0.80
                and marginal_ok(trial)
            ):
                trial["correlation_relative_frobenius"] = corr_error(
                    sample, target_corr, corr_rank
                )
                safe_target = "".join(
                    c if (c.isalnum() or c in "-_") else "_" for c in a.target
                )
                alpha_tag = f"{alpha:.2f}".replace(".", "p")
                production_path = out / (
                    f"{safe_target}_residual_PRODUCTION_alpha{alpha_tag}_n{a.n_out}.npy"
                )
                np.save(production_path, sample)
                trial["sample"] = production_path.name
                production = trial
                break
    if production is None:
        w = np.ones_like(score)
        chosen = systematic_resample(w, a.n_out, np.random.default_rng(a.seed + 7077))
        sample = candidate[chosen]
        safe_target = "".join(
            c if (c.isalnum() or c in "-_") else "_" for c in a.target
        )
        production_path = out / f"{safe_target}_residual_PRODUCTION_noop_n{a.n_out}.npy"
        np.save(production_path, sample)
        production = {
            "alpha": 0.0,
            **weight_stats(w, chosen),
            **marginal_stats(sample, target_mean),
            "correlation_relative_frobenius": corr_error(sample, target_corr, corr_rank),
            "sample": production_path.name,
        }
    candidate_offset_path = out / "production_candidate_offsets.npy"
    proposal_source_path = out / "production_proposal_source_indices.npy"
    np.save(candidate_offset_path, chosen)
    np.save(proposal_source_path, q_cand_i[chosen])
    meta = {
        "method": "clean_endpoint_target_vs_actual_proposal_classifier",
        "weight": "bounded_target_probability_power",
        "target": a.target,
        "proposal": str(proposal_path.resolve()),
        "output_dir": str(out.resolve()),
        "seed": a.seed,
        "dtype_input": str(q.dtype),
        "shape_proposal": list(q.shape),
        "splits": {
            "proposal_fit": len(q_fit_i),
            "proposal_validation": len(q_val_i),
            "proposal_candidate": len(q_cand_i),
            "target_fit": len(p_fit_i),
            "target_validation": len(p_val_i),
        },
        "classifier": {
            "type": type(clf).__name__,
            "n_features": len(feat),
            "features": [str(genes[i]) for i in feat],
            "heldout_roc_auc": auc,
            "auc_gate": a.auc_gate,
            "gate_passed": passed,
            "n_iter": int(clf.n_iter_),
        },
        "candidate_stats": candidate_stats,
        "arms": records,
        "primary_marginal_gate_passed": marginal_gate,
        "production_rule": {
            "selection_blind_to_celltypist": True,
            "alpha_grid": "1.00, 0.95, ..., 0.05",
            "min_ess_fraction": 0.20,
            "min_unique_source_fraction": 0.80,
            "marginal_preservation_required": True,
            "selected_alpha": production["alpha"],
            "selected_sample": production["sample"],
            "selected_stats": production,
            "candidate_offsets": candidate_offset_path.name,
            "proposal_source_indices": proposal_source_path.name,
        },
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    (out / "residual_log.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[gate] primary marginal preservation passed={marginal_gate}")
    print(
        f"[production] selected alpha={production['alpha']:g} "
        f"sample={production['sample']}"
    )
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
