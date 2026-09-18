# -*- coding: utf-8 -*-
"""Score the deliveries under out/runs for purity, W1, MMD^2, SCC and PCC.

No scoring logic is written here. All three rulers are imported from the existing modules and
are word for word the ones behind the published table:

    purity   evaluation/score_purity.py, with its own ruler check and block size
    W1       evaluation/score_wasserstein_raw_counts.py, on raw counts
    MMD^2, correlations   evaluation/score_distances_frozen_ruler.py, on the frozen ruler

    python scripts/score_deliveries.py --help
"""
from __future__ import annotations

# the BLAS thread count must be pinned BEFORE numpy is imported.
# Measured on one machine, one set of bytes, one command:
#   multi-threaded run 1: the floor row was bitwise identical on all three requests
#   multi-threaded run 2: one sliced-W1 differed by 7e-10, failing the mandatory self-check
#   single-threaded runs 1 and 2: every difference was exactly zero, twice
# So the ruler's bitwise reproducibility depends on the reduction order inside one matrix
# product, not on the data. Every scoring process is therefore pinned to one thread. That
# is not 'changing the ruler': it pins the reduction order back to the single order the
# archived floor row was computed in, which both single-threaded runs reproduce exactly.
import os                                                            # noqa: E402

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"        # FORCED, not defaulted: the queue sets these to eight, and this ruler's bitwise
                                # reproducibility requires one thread. Correctness comes before speed.

import argparse                                                      # noqa: E402
import json                                                          # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.io import load_runs                                         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="bestofn")
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--skip", nargs="*", default=[], choices=["purity", "w1", "e21"])
    ap.add_argument("--only-stems", nargs="*", default=None)
    ap.add_argument("--e21-device", choices=("cpu", "cuda", "auto"), default="cpu",
                    help="where to compute the E21 MMD^2. The default is cpu: there is one GPU and the "
                         "generation queue needs it, and the cpu path still passes the frozen ruler's "
                         "own float32 tolerances, measured well inside every one of them.")
    a = ap.parse_args()
    tag = a.out_tag or f"200_{a.experiment}"

    runs = load_runs(a.experiment)
    if a.only_stems:
        runs = [r for r in runs if r["stem"] in set(a.only_stems)]
    if not runs:
        raise SystemExit(f"no runs under out/runs/{a.experiment}")
    runs = sorted(runs, key=lambda r: r["stem"])
    # the delivered count: this round's run jsons carry one field name and older experiments
    # another. If neither exists the array's row count is read directly; it must NEVER default
    # to zero, which would silently mark every row unscorable.
    for r in runs:
        nd = r.get("n_delivered") or r.get("n_out") or r.get("n_cells")
        if nd is None:
            p = Path(r["_path"]).with_suffix(".npy")
            nd = int(np.load(p, mmap_mode="r").shape[0]) if p.exists() else 0
        r["n_delivered"] = int(nd)
    print(f"[runs] {len(runs)} deliveries under out/runs/{a.experiment}")

    out = {"ticket": 200, "experiment": a.experiment,
           "written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rulers": {}, "rows": {}}
    for r in runs:
        out["rows"][r["stem"]] = {
            "stem": r["stem"], "arm": r.get("arm"), "request": r.get("request"),
            "seed": r.get("seed"), "generator": r.get("generator"),
            "variant": r.get("variant"), "selector": (r.get("selector") or {}).get("selector"),
            "axis": r.get("axis"), "level": r.get("level"),
            "n_candidates": r.get("n_candidates"), "n_delivered": r.get("n_delivered"),
            "NFE_total": r.get("NFE_total"), "NFE_fk": r.get("NFE_fk"),
            "aux_calls": r.get("aux_calls"), "walltime_s": r.get("walltime_s"),
            "lineage": r.get("lineage"), "lineage_note": r.get("lineage_note"),
            "unique_cell_fraction": r.get("unique_cell_fraction"),
            "device_class": r.get("device_class"),
            "accepted_count": r.get("accepted_count"), "shortfall": r.get("shortfall"),
            "gamma": r.get("gamma"), "scorer": r.get("scorer"),
        }

    # ------------------------------------------------------------------ purity
    if "purity" not in a.skip:
        from evaluation.score_purity import BLOCK, ruler_check, score_run   # noqa: PLC0415
        from crndiff.metrics import load_fixed_label_sets                     # noqa: PLC0415
        rl = ruler_check()
        label_sets = load_fixed_label_sets(C.MASTER_PURITY_V7)
        genes = json.loads((C.LEGACY_DATA / "all" / "meta.json")
                           .read_text(encoding="utf-8"))["genes"]
        out["rulers"]["purity"] = {"scorer": str(ROOT / "evaluation" / "score_purity.py"),
                                   "matched_block": BLOCK, "celltypist": rl,
                                   "label_sets": str(C.MASTER_PURITY_V7)}
        print(f"[ruler] purity celltypist {rl['celltypist_version']} model {rl['model_sha256'][:16]}")
        for r in runs:
            t = r["request"]
            row = out["rows"][r["stem"]]
            n = int(r.get("n_delivered") or 0)
            if t not in label_sets:
                row["purity"] = "NO_LABEL_SET"
                continue
            if n < BLOCK:
                row["purity"] = "NOT_SCORABLE"
                row["purity_why"] = f"{n} < matched block {BLOCK}"
                continue
            t0 = time.time()
            pr = score_run(r, genes, label_sets[t], t)
            row["purity"] = pr["purity_matched_n"]
            row["purity_whole_set"] = pr["purity_whole_set"]
            row["purity_n_blocks"] = pr["n_blocks"]
            row["purity_per_block"] = pr["purity_per_block"]
            row["purity_block_spread_NOT_run_to_run"] = pr["block_spread_NOT_run_to_run"]
            row["canonical_ceiling"] = pr["canonical_ceiling"]
            row["purity_over_ceiling"] = pr["purity_over_ceiling"]
            row["label_dist_whole"] = pr["label_dist_whole"]
            row["purity_walltime_s"] = time.time() - t0
            print(f"[purity] {r['stem']:<62} n={n:<6} {pr['purity_matched_n']:.4f} "
                  f"({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------ W1 / sliced-W1
    if "w1" not in a.skip:
        import evaluation.score_wasserstein_raw_counts as SW                               # noqa: PLC0415
        M, sw1, W = SW.load_frozen()
        D = C.LEGACY_DATA / "all"
        xv = np.load(D / "val.npy", mmap_mode="r")
        yv = np.load(D / "val_celltype.npy", allow_pickle=True).astype(str)
        xt = np.load(D / "train.npy", mmap_mode="r")
        yt = np.load(D / "train_celltype.npy", allow_pickle=True).astype(str)
        proj = M.make_projections(xv.shape[1], n_proj=W.N_PROJ, seed=W.W1_PROJ_SEED)
        arch = json.loads(SW.COND_W1.read_text(encoding="utf-8"))
        sc = {}
        for t in W.TYPES:
            got, _ = W.w1_real_floor(t, xv, yv, xt, yt, M, sw1, proj)
            want = arch["types"][t]["rows"][SW.FLOOR_KEY]
            d = {"W1": got["W1"] - want["marginal_w1_mean"],
                 "W1_median": got["W1_median"] - want["marginal_w1_median"],
                 "W1_p90": got["W1_p90"] - want["marginal_w1_p90"],
                 "sliced_W1": got["sliced_W1"] - want["sliced_w1"]}
            ok = got["n"] == want["n"] and all(v == 0.0 for v in d.values())
            sc[t] = {"bit_identical": ok, "diffs": d}
            print(f"[selfcheck] W1 {t:<12} {'BIT-IDENTICAL' if ok else 'DIFFERS ' + str(d)}")
            if not ok:
                raise SystemExit(f"W1 ruler no longer reproduces the archived floor for {t}. STOP.")
        out["rulers"]["w1"] = {
            "scale": "raw counts", "scorer": str(SW.FROZEN / "e30plus_eval_cpu_worker.py"),
            "scorer_sha256": SW.FROZEN_SHA["e30plus_eval_cpu_worker.py"],
            "N_MATCH": W.N_MATCH, "eval_seed": W.EVAL_SEED, "n_proj": W.N_PROJ,
            "proj_seed": W.W1_PROJ_SEED, "selfcheck_vs_cond_w1_floor": sc,
            "identity": "the SAME ruler as Table 1 (score_w1_paper), not score_distribution",
            "blas_threads_pinned_to_1": True,
            "blas_thread_note":
                "multi-threaded BLAS makes the sliced-W1 projection reduction order "
                "non-deterministic across processes (measured: one run bit-identical, the "
                "next off by 6.8e-10 on Endothelial). Single-threaded reproduces the "
                "archived cond_w1 floor bit-exactly, twice. Precedent: "
                "HVG2K/ENV_LOG_single_thread_mitigation.md",
            "env": {v: os.environ.get(v) for v in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
        for r in runs:
            t, row = r["request"], out["rows"][r["stem"]]
            n = int(r.get("n_delivered") or 0)
            if t not in W.N_MATCH:
                row["W1"] = "NO_N_MATCH"
                continue
            if n < W.N_MATCH[t]:
                row["W1"] = "NOT_RUN"
                row["W1_why"] = f"{n} delivered < N_MATCH {W.N_MATCH[t]}"
                continue
            raw = np.load(Path(r["_path"]).with_suffix(".npy"), mmap_mode="r")
            res, sel = W.w1_rows(t, "OURS", raw, xv, yv, xt, yt, M, sw1, proj)
            g, f = res["generated"], res["floor"]
            row.update({"W1": g["W1"], "W1_median": g["W1_median"], "W1_p90": g["W1_p90"],
                        "sliced_W1": g["sliced_W1"], "floor_W1": f["W1"],
                        "floor_sliced_W1": f["sliced_W1"],
                        "W1_over_floor": g["W1"] / f["W1"],
                        "sliced_W1_over_floor": g["sliced_W1"] / f["sliced_W1"],
                        "W1_matched_n": W.N_MATCH[t]})
            print(f"[w1] {r['stem']:<62} W1 {g['W1']:.4f} ({g['W1']/f['W1']:.2f}x) "
                  f"sW1 {g['sliced_W1']:.4f} ({g['sliced_W1']/f['sliced_W1']:.2f}x)")

    # ------------------------------------------------------------------ SCC / PCC / MMD^2
    if "e21" not in a.skip:
        import torch                                                       # noqa: PLC0415
        torch.set_num_threads(1)        # as above: the frozen ruler's floor requires bitwise equality
        import evaluation.score_distances_frozen_ruler as SE                                    # noqa: PLC0415
        e21 = SE.frozen()
        types = e21.TYPES
        archived = json.loads(SE.ARCHIVED_E21.read_text(encoding="utf-8"))
        D = C.LEGACY_DATA / "all"
        Xv = np.load(D / "val.npy", mmap_mode="r")
        yv = np.load(D / "val_celltype.npy", allow_pickle=True).astype(str)
        Xt = np.load(D / "train.npy", mmap_mode="r")
        yt = np.load(D / "train_celltype.npy", allow_pickle=True).astype(str)
        device = ("cuda" if (a.e21_device == "cuda" or
                             (a.e21_device == "auto" and torch.cuda.is_available()))
                  else "cpu")
        out["rulers"]["e21"] = {"scorer": str(C.FROZEN_E21), "matched_n": types,
                                "seed": e21.SEED, "device": device,
                                "device_choice": a.e21_device,
                                "device_note":
                                    "there is one GPU and the generation queue needs it. The E21 self-check "
                                    "tolerances are the float32 precision the ruler declares for itself "
                                    "(sigma 2 ulp / MMD^2 1e-6 / SCC bitwise), and the cpu path measured "
                                    "sigma <= 0.66 ulp, SCC bitwise equal and MMD^2 1.5e-7, all well inside "
                                    "them; the self-check and the scoring happen in **one process**.",
                                "selfcheck": {}}
        for t, n in types.items():
            rs = [r for r in runs if r["request"] == t]
            if not rs:
                continue
            vi = np.flatnonzero(yv == t)
            rng = np.random.default_rng(e21.SEED)
            ref_np, _ = e21.cp10k_log1p(np.asarray(Xv[vi]))
            ref_gpu = torch.from_numpy(ref_np).to(device)
            sigma = e21.exact_bandwidth(ref_gpu)
            k_rr = e21.kernel_mean(ref_gpu, ref_gpu, sigma)
            ti = np.flatnonzero(yt == t)
            floor = SE.score_one(e21, ref_np, ref_gpu, e21.select_rows(Xt[ti], n, rng), sigma, k_rr)
            arch = archived.get("types", {}).get(t, {})
            af = (arch.get("rows") or {}).get("floor (real train vs real val)", {})
            rel = lambda x, y: abs(x - y) / max(abs(x), 1e-30)              # noqa: E731
            ok = True
            chk = {"archived_sigma": arch.get("bandwidth_sigma"), "recomputed_sigma": sigma}
            if chk["archived_sigma"] is not None:
                chk["sigma_rel_diff"] = rel(chk["archived_sigma"], sigma)
                ok &= chk["sigma_rel_diff"] <= SE.SIGMA_REL_TOL
            if af:
                chk["floor_scc_exact"] = (af["scc"] == floor["scc"])
                ok &= chk["floor_scc_exact"]
                chk["floor_mmd2_rel_diff"] = rel(af["mmd2_rbf_biased"], floor["mmd2_rbf_biased"])
                ok &= chk["floor_mmd2_rel_diff"] <= SE.MMD2_REL_TOL
            chk["passed"] = bool(ok)
            out["rulers"]["e21"]["selfcheck"][t] = chk
            print(f"[selfcheck] E21 {t:<12} sigma {sigma:.8f} -> {'MATCH' if ok else 'MISMATCH'}")
            if not ok:
                raise SystemExit(f"E21 ruler drifted on {t}. STOP.")
            for r in rs:
                row = out["rows"][r["stem"]]
                nd = int(r.get("n_delivered") or 0)
                if nd < n:
                    row["mmd2_rbf_biased"] = "NOT_RUN"
                    row["e21_why"] = f"{nd} delivered < matched n {n} (B6)"
                    continue
                raw = np.load(Path(r["_path"]).with_suffix(".npy"), mmap_mode="r")
                rr = np.random.default_rng(e21.SEED)
                got = SE.score_one(e21, ref_np, ref_gpu, e21.select_rows(raw, n, rr), sigma, k_rr)
                row.update({"scc": got["scc"], "pcc": got["pcc"],
                            "mmd2_rbf_biased": got["mmd2_rbf_biased"],
                            "e21_matched_n": n,
                            "floor_scc": floor["scc"], "floor_pcc": floor["pcc"],
                            "floor_mmd2": floor["mmd2_rbf_biased"]})
                print(f"[e21] {r['stem']:<62} scc {got['scc']:.4f} pcc {got['pcc']:.4f} "
                      f"mmd2 {got['mmd2_rbf_biased']:.4g}")
            del ref_gpu
            torch.cuda.empty_cache() if device == "cuda" else None

    out["rows"] = [out["rows"][k] for k in sorted(out["rows"])]
    p = C.OUT / "summary" / f"{tag}_scored.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p = p.with_name(f"{tag}_scored_{int(time.time())}.json")
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
