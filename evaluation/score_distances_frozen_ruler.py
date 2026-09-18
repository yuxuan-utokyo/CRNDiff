# -*- coding: utf-8 -*-
"""Table 6's SCC / PCC / MMD^2 for this round's OURS rows, on the ARCHIVED ruler.
Purpose: the three distance columns. No new ruler is written: the metric functions are imported
from the verbatim frozen copies, and PCC follows the frozen definition (Pearson between the same
pair of population mean vectors). The baseline rows are NOT re-run: they are carried across from
the archived scoring file unchanged, and only our own row is newly computed.

Self-check, which is half the reason this file exists: before computing our row, the same code
recomputes the archived kernel bandwidth and the floor row (real train against real val) and
requires them to agree value by value. Both depend only on real data and the metric definition,
never on generated samples, so agreement proves this call path is the archived ruler. A
mismatch stops the run: the ruler has moved and the rows would not be comparable.

The matched n comes from the frozen module's own TYPES table and is never transcribed.
A run with fewer delivered cells than the matched n is skipped and recorded; the matched n is

    python -m evaluation.score_distances_frozen_ruler --experiment table1_ours
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                       # noqa: E402
from crndiff.io import load_runs                      # noqa: E402

ARCHIVED_E21 = (C.ARCHIVE
                / "results" / "diagnostics" / "E21_scc_mmd.json")
# ---- self-check tolerances, derived from the arithmetic, NOT from the observed values -------
# The whole E21 pipeline is float32: `cp10k_log1p` returns float32 and `exact_bandwidth` uses
# matmul-based `cdist`. So agreement can only be expected to float32 precision, and a tolerance
# tighter than that tests the GPU's reduction order, not the ruler.
#
# My first pass used 1e-9 for both -- a float64 tolerance on a float32 quantity. It "failed" on
# Myeloid's sigma by 3.815e-06 at magnitude 55.27, which is 55.27 * 2**-24 = 3.3e-06, i.e.
# EXACTLY ONE float32 ulp: the median landed on the adjacent representable value. That was my
# specification error, not ruler drift. Evidence it is not drift: Endothelial and Neuronal sigma
# reproduce BIT-EXACTLY, and all three floor SCCs reproduce BIT-EXACTLY. A ruler that had really
# changed could not do that.
F32_EPS = 2.0 ** -23              # float32 relative spacing
SIGMA_REL_TOL = 2 * F32_EPS       # a median over float32 distances: allow 2 ulp
MMD2_REL_TOL = 1e-6               # MMD^2 sums ~n^2 float32 kernel terms; observed max 2.8e-7
# SCC gets NO slack: it is a rank statistic, so last-bit differences cannot move it. If SCC ever
# stops matching exactly, something real changed and this must stop.


def frozen():
    """Import the byte-identical archived scorer and hand back its metric functions."""
    C.require(C.FROZEN_E21, "frozen E21 scorer")
    spec = importlib.util.spec_from_file_location("_frozen_e21", C.FROZEN_E21)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def score_one(e21, ref_np, ref_gpu, gen_raw, sigma, k_rr) -> dict:
    """SCC + MMD^2 exactly as `_frozen/e21_scc_mmd.score`, plus PCC per `_frozen/e21c_pcc`."""
    from scipy.stats import pearsonr, spearmanr
    import torch
    transformed, zero = e21.cp10k_log1p(gen_raw)
    ref_mean = ref_np.mean(0, dtype=np.float64)
    gen_mean = transformed.mean(0, dtype=np.float64)
    scc = float(spearmanr(ref_mean, gen_mean).statistic)
    pcc = float(pearsonr(ref_mean, gen_mean).statistic)
    if not (math.isfinite(scc) and math.isfinite(pcc)):
        raise SystemExit("non-finite SCC/PCC")
    gen_gpu = torch.from_numpy(transformed).to(ref_gpu.device)
    mmd2, terms = e21.mmd2_biased(ref_gpu, gen_gpu, sigma, k_rr)
    del gen_gpu, transformed
    torch.cuda.empty_cache()
    return {"scc": scc, "pcc": pcc, "mmd2_rbf_biased": mmd2,
            "zero_library_rows": zero, "kernel_terms": terms}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="table1_ours")
    ap.add_argument("--validate-only", action="store_true",
                    help="only run the ruler self-check (sigma + floor), score nothing")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import torch
    e21 = frozen()
    if not torch.cuda.is_available():
        raise SystemExit("the exact matched-n E21 evaluator requires CUDA (frozen contract)")
    device = torch.device("cuda:0")
    types = dict(e21.TYPES)
    D = C.LEGACY_DATA / "all"
    Xv = np.load(D / "val.npy", mmap_mode="r")
    yv = np.load(D / "val_celltype.npy", allow_pickle=True).astype(str)
    Xt = np.load(D / "train.npy", mmap_mode="r")
    yt = np.load(D / "train_celltype.npy", allow_pickle=True).astype(str)
    archived = json.loads(ARCHIVED_E21.read_text(encoding="utf-8")) if ARCHIVED_E21.exists() \
        else {"types": {}}

    runs = load_runs(a.experiment) if not a.validate_only else []
    out = {"experiment": a.experiment,
           "ruler": {"source": str(C.FROZEN_E21), "seed": e21.SEED,
                     "target_sum": e21.TARGET_SUM, "block": e21.BLOCK,
                     "matched_n": types,
                     "definitions": {
                         "representation": "library-size normalize to 10000, then log1p",
                         "scc": "Spearman of the 2000-gene population mean vectors",
                         "pcc": "Pearson of the same two mean vectors (_frozen/e21c_pcc)",
                         "kernel": "exp(-squared_euclidean/(2*sigma^2))",
                         "bandwidth": "exact median of non-diagonal real-real distances",
                         "mmd2": "biased empirical Gaussian-kernel MMD squared, exact all pairs"}},
           "baselines_inherited_not_rerun": True,
           "types": {}}

    for cell_type, n in types.items():
        vi = np.flatnonzero(yv == cell_type)
        if len(vi) != n:
            raise SystemExit(f"{cell_type}: expected {n} validation cells, found {len(vi)}")
        rng = np.random.default_rng(e21.SEED)
        ref_np, ref_zero = e21.cp10k_log1p(np.asarray(Xv[vi]))
        ref_gpu = torch.from_numpy(ref_np).to(device)
        sigma = e21.exact_bandwidth(ref_gpu)
        k_rr = e21.kernel_mean(ref_gpu, ref_gpu, sigma)
        ti = np.flatnonzero(yt == cell_type)
        floor = score_one(e21, ref_np, ref_gpu, e21.select_rows(Xt[ti], n, rng), sigma, k_rr)

        # ---- the self-check: sigma and floor must reproduce the archived values ----
        arch = archived.get("types", {}).get(cell_type, {})
        chk = {"archived_sigma": arch.get("bandwidth_sigma"), "recomputed_sigma": sigma}
        af = (arch.get("rows") or {}).get("floor (real train vs real val)", {})
        chk.update({"archived_floor_scc": af.get("scc"), "recomputed_floor_scc": floor["scc"],
                    "archived_floor_mmd2": af.get("mmd2_rbf_biased"),
                    "recomputed_floor_mmd2": floor["mmd2_rbf_biased"]})
        ok = True

        def rel(a_, b_):
            return abs(a_ - b_) / max(abs(a_), 1e-30)

        if chk["archived_sigma"] is not None:
            chk["sigma_rel_diff"] = rel(chk["archived_sigma"], sigma)
            chk["sigma_ulps"] = chk["sigma_rel_diff"] / F32_EPS
            ok &= chk["sigma_rel_diff"] <= SIGMA_REL_TOL
        if af:
            chk["floor_scc_exact"] = (af["scc"] == floor["scc"])
            ok &= chk["floor_scc_exact"]                       # rank statistic: no slack
            chk["floor_mmd2_rel_diff"] = rel(af["mmd2_rbf_biased"], floor["mmd2_rbf_biased"])
            ok &= chk["floor_mmd2_rel_diff"] <= MMD2_REL_TOL
        chk["tolerances"] = {"sigma_rel": SIGMA_REL_TOL, "mmd2_rel": MMD2_REL_TOL,
                             "scc": "exact equality required",
                             "why": "the E21 pipeline is float32 throughout; these are its "
                                    "precision, not values chosen to make the check pass"}
        chk["passed"] = bool(ok)
        print(f"[ruler] {cell_type:12s} sigma {sigma:.10f} (archived "
              f"{chk['archived_sigma']}) floor scc {floor['scc']:.6f}/"
              f"{af.get('scc')} mmd2 {floor['mmd2_rbf_biased']:.6g}/"
              f"{af.get('mmd2_rbf_biased')}  -> {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(
                f"{cell_type}: the distance ruler does not reproduce the archived sigma/floor. "
                "STOP -- OURS rows computed on a drifted ruler are not comparable to the "
                "archived baseline rows. Do not adjust tolerances to make this pass.")

        rows = {"floor (real train vs real val)": {**floor, "n": n, "source": "recomputed"}}
        for name, r in (arch.get("rows") or {}).items():     # baselines carried over verbatim
            if name != "floor (real train vs real val)":
                rows[name] = {**r, "inherited_from_archive": True}

        ours, skipped = [], []
        for run in sorted((r for r in runs if r["request"] == cell_type),
                          key=lambda r: r["seed"]):
            if run["n_out"] < n:                              # gate B6
                skipped.append({"stem": run["stem"], "cell_type": cell_type,
                                "n_out": run["n_out"], "matched_n": n,
                                "reason": "B6: n_out < matched n; matched n is NOT reduced"})
                print(f"[B6 SKIP] {run['stem'][:60]} n_out={run['n_out']} < {n}")
                continue
            gen = np.load(Path(run["_path"]).with_suffix(".npy"), mmap_mode="r")
            rr = np.random.default_rng(e21.SEED)
            res = score_one(e21, ref_np, ref_gpu,
                            e21.select_rows(gen, n, rr), sigma, k_rr)
            res.update({"n": n, "seed": run["seed"], "stem": run["stem"],
                        "n_generated": int(gen.shape[0])})
            ours.append(res)
            print(f"[OURS] {cell_type:12s} seed {run['seed']}  scc={res['scc']:.6f}  "
                  f"pcc={res['pcc']:.6f}  mmd2={res['mmd2_rbf_biased']:.6g}")
        if ours:
            agg = {}
            for k in ("scc", "pcc", "mmd2_rbf_biased"):
                v = [o[k] for o in ours]
                agg[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))
                          if len(v) > 1 else 0.0, "n_seeds": len(v), "per_seed": v}
            rows["OURS (tilt+FK, this round)"] = {"aggregate": agg, "per_run": ours}
        out["types"][cell_type] = {
            "n": n, "bandwidth_sigma": sigma, "kernel_gamma": 1.0 / (2.0 * sigma * sigma),
            "reference_zero_library_rows": ref_zero,
            "shared_bandwidth_across_all_methods": True,
            "ruler_self_check": chk, "rows": rows,
            "skipped_B6": skipped}
        del ref_gpu, ref_np
        torch.cuda.empty_cache()

    dest = Path(a.out) if a.out else (C.OUT / "summary" / f"e21_{a.experiment}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[e21] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
