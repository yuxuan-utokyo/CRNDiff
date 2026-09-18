# -*- coding: utf-8 -*-
"""Best-of-N delivery on the frozen generator. CPU only.

No scoring logic is written here; both selectors already exist:

    S1  the one-versus-rest selection classifier fitted on real cells against real cells
    S2  the existing UNTILTED density-ratio discriminator, fitted against the base pool on a
        different seed, so there is no leakage into the pools drawn here

Scores go through the sampler's own reward function, so the selectors are scored exactly as the
particle layer scores its candidates.

    python scripts/sample_best_of_n.py --help
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.io import sha256_file                                       # noqa: E402
from crndiff.sampler import PortableHGBT, classifier_log_reward          # noqa: E402

CHUNK = 10000                       # scored in chunks; rows are independent so the values do not change
FLOOR = C.DEFAULT_REWARD_FLOOR      # the same reward floor as the deployed arm
DERIVED = C.OUT / "summary" / "200_derived_constants.json"
# a record with metadata but no delivered array must NOT go into the run directory: the
# loader treats every json there as a run and the scorer would then look for an array that
META_ROOT = C.OUT / "summary" / "200_bestofn_meta"


def normalize_log1p(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float32)
    lib = z.sum(axis=1, dtype=np.float64)
    scale = (1.0e4 / np.maximum(lib, 1.0)).astype(np.float32)
    return np.log1p(z * scale[:, None]).astype(np.float32, copy=False)


def load_selector(kind: str, t: str):
    """Return a scoring function and its metadata; the function maps raw counts to a score that is monotone in the classifier probability."""
    if kind == "S1":
        d = C.MODELS / "selection" / t
        clf = PortableHGBT(C.require(d / "selection_classifier_portable.npz", f"S1 {t}"))
        feat = np.load(d / "features.npy")
        log = json.loads((d / "log.json").read_text(encoding="utf-8"))

        def score(block: np.ndarray) -> np.ndarray:
            return np.asarray(clf.predict_probability(normalize_log1p(block)[:, feat]),
                              dtype=np.float64)

        meta = {"selector": "S1", "what": "one-vs-rest HGBT on real train (real vs real)",
                "dir": str(d), "portable_sha256": sha256_file(d / "selection_classifier_portable.npz"),
                "val_auroc": log["val_auroc"], "n_features": int(len(feat)),
                "score_meaning": "p(type | n)", "accept_rule": "p >= 0.5"}
        return score, meta, 0.5

    if kind in ("S2", "RHO"):
        # the untilted discriminator, fitted against the untilted base pool, used by best-of-N
        # the deployed residual discriminator, fitted against the tilted proposal, used by the
        p = C.residual_paths(t if kind == "RHO" else f"UNTILTED_{t}")
        clf = PortableHGBT(C.require(p["portable"], f"S2 UNTILTED_{t}"))
        split = np.load(p["split"])
        feat = split["classifier_features"].astype(np.int64)
        log = json.loads(p["log"].read_text(encoding="utf-8"))

        def score(block: np.ndarray) -> np.ndarray:
            # the existing function, in odds mode
            return np.asarray(classifier_log_reward(clf, feat, block, FLOOR, "odds"),
                              dtype=np.float64)

        meta = {"selector": kind,
                "what": ("the DEPLOYED residual discriminator rho (fitted against the TILTED "
                         "proposal) -- the endpoint filter of the tilt+filter arm"
                         if kind == "RHO" else
                         "UNTILTED residual density-ratio discriminator (Neyman-Pearson optimal)"),
                "dir": str(p["dir"]), "portable_sha256": sha256_file(p["portable"]),
                "heldout_roc_auc": log["classifier"]["heldout_roc_auc"],
                "n_features": int(len(feat)), "reward_floor": FLOOR,
                "score_meaning": "log odds = log r - log(1-r)", "accept_rule": "odds >= 1 (log odds >= 0)"}
        return score, meta, 0.0

    raise SystemExit(f"unknown selector {kind}")


def score_pool(pool: np.ndarray, score_fn) -> tuple[np.ndarray, int]:
    n = pool.shape[0]
    out = np.empty(n, dtype=np.float64)
    calls = 0
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        out[lo:hi] = score_fn(np.asarray(pool[lo:hi]))
        calls += hi - lo
    return out, calls


def write_delivery(out_root: Path, arm: str, stem: str, pool: np.ndarray,
                   idx: np.ndarray, payload: dict) -> dict:
    d = out_root / arm
    d.mkdir(parents=True, exist_ok=True)
    npy, ij, jsn = d / f"{stem}.npy", d / f"{stem}.idx.npy", d / f"{stem}.json"
    for p in (npy, ij, jsn):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    np.save(ij, idx.astype(np.int64))
    mat = np.asarray(pool[np.sort(idx)] if payload.get("_sorted_ok") else pool[idx])
    np.save(npy, mat)
    payload.pop("_sorted_ok", None)
    payload["npy"] = str(npy)
    payload["idx_npy"] = str(ij)
    payload["npy_sha256"] = sha256_file(npy)
    payload["idx_sha256"] = sha256_file(ij)
    payload["shape"] = list(mat.shape)
    payload["dtype"] = str(mat.dtype)
    jsn.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--generator", default="OURS", help="which generator produced the pool")
    ap.add_argument("--device-class", default="A")
    ap.add_argument("--types", nargs="+", default=["Endothelial", "Myeloid", "Neuronal"])
    ap.add_argument("--selectors", nargs="+", default=["S2", "S1"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--candidate-counts", type=int, nargs="+", default=None,
                    help="M1-5: the generator axis uses a candidate count C rather than an NFE multiple b")
    ap.add_argument("--out-root", default=str(C.RUNS / "bestofn"))
    ap.add_argument("--scores-dir", default=str(C.RUNS / "bestofn" / "_scores"))
    ap.add_argument("--meta-root", default=str(META_ROOT),
                    help="generate-until-N / NOT_SCORABLE these meta-only records go here; M1-5 uses its own directory so it cannot collide with M1-2's P-A3 records")
    a = ap.parse_args()

    pool_path = Path(a.pool)
    C.require(pool_path, "unconditional pool")
    D = json.loads(C.require(DERIVED, "derived constants (M1-0.5)").read_text(encoding="utf-8"))
    nfe_chain = int(D["NFE_chain"])
    n_pool_cap = int(D["N_pool"])

    pool = np.load(pool_path, mmap_mode="r")
    n_pool = int(pool.shape[0])
    pool_sha = sha256_file(pool_path)
    print(f"[pool] {pool_path.name}  shape={pool.shape} dtype={pool.dtype}  sha256={pool_sha[:16]}…")

    # the arm name and the stem must carry the generator label. The same best-of-N is applied
    # to several generators, and without a label two generators' stems COLLIDE, which the
    # overwrite guard turns into an outright failure. Our own arm keeps its original name so
    # that the artefacts already on disk are unaffected.
    gtag = "" if a.generator == "OURS" else f"_{a.generator}"
    meta_root = Path(a.meta_root)
    out_root = Path(a.out_root)
    sdir = Path(a.scores_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.generator}_seed{a.seed}_n{n_pool}"
    summary = {"ticket": 200, "block": "M1-2/M1-5", "generator": a.generator,
               "pool": str(pool_path), "pool_sha256": pool_sha, "n_pool": n_pool,
               "seed": a.seed, "device_class": a.device_class, "NFE_chain": nfe_chain,
               "rows": []}

    for t in a.types:
        N = int(D["types"][t]["N"]) if t in D.get("types", {}) else C.n_out_for(t)
        B1 = int(D["types"][t]["B1"]) if t in D.get("types", {}) else None
        for sel in a.selectors:
            score_fn, smeta, thr = load_selector(sel, t)
            spath = sdir / f"{tag}_{t}_{sel}.npy"
            t0 = time.time()
            if spath.exists():
                s = np.load(spath)
                aux_calls = 0
                print(f"[score] reuse {spath.name}")
            else:
                s, aux_calls = score_pool(pool, score_fn)
                np.save(spath, s)
            score_wall = time.time() - t0
            acc_all = s >= thr
            print(f"[{t}/{sel}] scored {n_pool} in {score_wall:.1f}s  "
                  f"accept_rate={acc_all.mean():.6f}  "
                  f"q=[{np.quantile(s,0.5):.4g},{np.quantile(s,0.9):.4g},{np.quantile(s,0.99):.4g}]")

            # generate-until-N: no GPU, the whole pool, no budget levels
            cum = np.cumsum(acc_all)
            enough = np.flatnonzero(cum >= N)
            if len(enough):
                istar = int(enough[0]) + 1
                censored = False
            else:
                istar = n_pool
                censored = True
            gu = {"ticket": 200, "arm": f"{t}_gunN_{sel}{gtag}", "experiment": "bestofn",
                  "stem": f"{t}_gunN_{sel}{gtag}_seed{a.seed}", "request": t, "target": t,
                  "seed": a.seed, "generator": a.generator, "selector": smeta,
                  "method": "generate-until-N", "N": N,
                  "n_candidates_consumed": istar, "right_censored": censored,
                  "NFE_needed": istar * nfe_chain,
                  "NFE_needed_ge": censored, "NFE_chain": nfe_chain,
                  "NFE_fk": int(D["types"][t]["NFE_fk"]) if B1 else None,
                  "NFE_needed_over_NFE_fk": (istar * nfe_chain / D["types"][t]["NFE_fk"]) if B1 else None,
                  "accepted_in_pool": int(cum[-1]), "accept_rate": float(acc_all.mean()),
                  "aux_calls": {"model": smeta["selector"], "n_calls": int(n_pool),
                                "device": "cpu"},
                  "walltime_s": score_wall, "pool": str(pool_path), "pool_sha256": pool_sha,
                  "device_class": a.device_class, "lineage": 1.0,
                  "lineage_note": "iid_by_construction"}
            gdir = meta_root / gu["arm"]
            gdir.mkdir(parents=True, exist_ok=True)
            gp = gdir / f"{gu['stem']}.meta.json"
            if gp.exists():
                raise SystemExit(f"refusing to overwrite {gp}")
            gp.write_text(json.dumps(gu, indent=1, ensure_ascii=False), encoding="utf-8")
            summary["rows"].append(gu)
            print(f"[{t}/{sel}] generate-until-N: i*={istar}{' (RIGHT-CENSORED)' if censored else ''} "
                  f"NFE_needed={istar*nfe_chain}  ratio={gu['NFE_needed_over_NFE_fk']}")

            # the budget ladder
            if a.candidate_counts is not None:
                grid = [("C", c, c) for c in a.candidate_counts]
            else:
                grid = [("b", b, b * B1) for b in a.budgets]
            for axis, level, m in grid:
                if m > n_pool:
                    row = {"ticket": 200, "request": t, "seed": a.seed, "axis": axis,
                           "level": level, "n_candidates": int(m), "N": N,
                           "selector": smeta["selector"], "generator": a.generator,
                           "status": ("not_run_budget_cap" if m > n_pool_cap
                                      else "not_run_pool_too_small"),
                           "n_pool": n_pool, "N_pool_cap": n_pool_cap}
                    summary["rows"].append(row)
                    print(f"[{t}/{sel}] {axis}={level}: {row['status']} (needs {m} > pool {n_pool})")
                    continue
                sm = s[:m]
                order = np.argsort(-sm, kind="stable")
                base = {"ticket": 200, "experiment": "bestofn", "request": t, "target": t,
                        "seed": a.seed, "generator": a.generator, "device_class": a.device_class,
                        "selector": smeta, "axis": axis, "level": int(level),
                        "n_candidates": int(m), "N": int(N), "NFE_chain": nfe_chain,
                        "NFE_total": int(m * nfe_chain),
                        "NFE_fk": int(D["types"][t]["NFE_fk"]) if B1 else None,
                        "B1": B1, "pool": str(pool_path), "pool_sha256": pool_sha,
                        "aux_calls": {"model": smeta["selector"], "n_calls": int(m),
                                      "device": "cpu"},
                        "walltime_s": score_wall,
                        "score_quantiles": {q: float(np.quantile(sm, q))
                                            for q in (0.5, 0.9, 0.99, 0.999)},
                        "lineage": 1.0, "unique_lineage_fraction": 1.0,
                        "lineage_note": "iid_by_construction",
                        "sampler": "best_of_N", "alpha": None, "tau": None,
                        "n_intermediate_resamples": 0}

                # --- forced-top-N ---
                take = order[:N] if N <= m else order
                idx = np.sort(take)
                p = dict(base)
                p.update({"variant": "forced_top_N", "arm": f"{t}_bestofn_{sel}{gtag}_forced_{axis}{level}",
                          "stem": f"{t}_bestofn_{sel}{gtag}_forced_{axis}{level}_seed{a.seed}",
                          "n_delivered": int(len(idx)),
                          "unique_cell_fraction": float(len(np.unique(idx)) / max(len(idx), 1)),
                          "selected_score_min": float(sm[take].min()),
                          "selected_score_median": float(np.median(sm[take])),
                          "note": "forced: the top N by score regardless of how low that score is"})
                write_delivery(out_root, p["arm"], p["stem"], pool, idx, p)
                summary["rows"].append(p)

                # --- threshold ---
                acc = np.flatnonzero(sm >= thr)
                n_take = min(N, len(acc))
                q = dict(base)
                q.update({"variant": "threshold", "arm": f"{t}_bestofn_{sel}{gtag}_thresh_{axis}{level}",
                          "stem": f"{t}_bestofn_{sel}{gtag}_thresh_{axis}{level}_seed{a.seed}",
                          "accept_rule": smeta["accept_rule"], "accepted_count": int(len(acc)),
                          "shortfall": int(max(0, N - len(acc))),
                          "n_delivered": int(n_take),
                          "purity_scorable": bool(n_take >= 2500),
                          "distance_scorable": bool(n_take >= N)})
                if n_take < 2500:
                    # the purity ruler needs a full block; below that nothing is materialised and only the
                    q["status"] = ("NOT_SCORABLE_no_accepted" if n_take == 0
                                   else "NOT_SCORABLE_below_matched_block")
                    q["purity"] = "NOT_SCORABLE"
                    q["distance"] = "NOT_RUN"
                    if n_take:
                        q["accepted_indices_head"] = acc[:min(n_take, 64)].tolist()
                    dq = meta_root / q["arm"]
                    dq.mkdir(parents=True, exist_ok=True)
                    dqp = dq / f"{q['stem']}.meta.json"
                    if dqp.exists():
                        raise SystemExit(f"refusing to overwrite {dqp}")
                    dqp.write_text(json.dumps(q, indent=1, ensure_ascii=False), encoding="utf-8")
                else:
                    idx2 = acc[:n_take]          # generation order, which is the rejection sampler itself
                    q["unique_cell_fraction"] = float(len(np.unique(idx2)) / len(idx2))
                    if n_take < N:
                        q["distance"] = "NOT_RUN"         # biased in n
                    write_delivery(out_root, q["arm"], q["stem"], pool, idx2, q)
                summary["rows"].append(q)
                print(f"[{t}/{sel}] {axis}={level}: forced N={len(idx)}  "
                      f"threshold accepted={len(acc)} delivered={n_take}"
                      f"{'  (<2500 -> purity NOT_SCORABLE)' if 0 < n_take < 2500 else ''}")

    sp = C.OUT / "summary" / f"200_bestofn_{a.generator}_seed{a.seed}.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    if sp.exists():
        sp = sp.with_name(sp.stem + f"_{int(time.time())}.json")
    sp.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
