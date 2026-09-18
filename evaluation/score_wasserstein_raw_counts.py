# -*- coding: utf-8 -*-
"""W1 and sliced-W1 on RAW COUNTS, which is the ruler the paper reports.

Background, from the caliber gate: the other distance scorer computes W1 and sliced-W1 on
CP10K + log1p, which is NOT that ruler. Measured side by side on the same pairs of cells:
(paper ruler, then the log1p scorer)

        Endothelial   0.011002 / 0.040859     log1p scorer 0.006784 / 0.014080
        Myeloid       0.036177 / 0.158929     log1p scorer 0.018389 / 0.034664
        Neuronal      0.035466 / 0.124662     log1p scorer 0.035444 / 0.069251

So those two columns come from this file, and the log1p scorer's W1 is kept only as an auxiliary
reading on the log1p scale and does not enter a table.

The ruler is the ORIGINAL module, not a reimplementation: this file calls `w1_rows()` of
`evaluation/frozen/e30plus_eval_cpu_worker.py`, the verbatim copy, which is the same function
that produced the published numbers. It fixes all of the following itself:

    * the reference is that type's real VALIDATION cells at the matched n;
    * n cells are drawn from the delivery on a fixed evaluation stream, replayed in the archived
        order (per type: reference, floor, then the arms);
    * 512 projections on a fixed seed, with `make_projections` and the sliced-W1 implementation
        both taken from the frozen modules;
    * every run also yields its own floor (real train against real val at the same stream
        position), so the multiple-of-floor column is self-contained.

Mandatory self-check: `w1_real_floor()` reproduces the archived floor row and must match it
bitwise, otherwise the run stops. The gate already checks this once; it is checked again here
because the scoring and its self-check must happen in one process on one set of bytes.

    python -m analysis.score_w1_paper --experiment table1_ours
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.io import load_runs, sha256_file                            # noqa: E402

FROZEN = ROOT / "evaluation" / "frozen"
FROZEN_SHA = {
    "metrics_exp08.py": "bdb35b4592e44cd2ac2443971d1317cf061846592b5edb39d3aba656d01f56de",
    "sliced_w1.py": "6224428255b9f97cf577e668aa16bd62c263bd5c07cf0ce4c2a0bde9ef9a63e0",
    "e30plus_eval_cpu_worker.py":
        "b5ca3313ed7e540328d542afd8f5b7fb799321a13e83589cca4ab23da99af7bc",
}
ARCHIVE = C.ARCHIVE
COND_W1 = ARCHIVE / "cond_w1" / "out" / "cond_w1.json"
FLOOR_KEY = "floor (real train vs real val, same type)"


def load_frozen():
    mods = {}
    for name in FROZEN_SHA:
        p = FROZEN / name
        if not p.exists():
            raise SystemExit(f"missing frozen scorer {p}. Do not write a replacement: a new "
                             "ruler makes the row incomparable with the published table.")
        got = sha256_file(p)
        if got != FROZEN_SHA[name]:
            raise SystemExit(f"{p} changed: expected {FROZEN_SHA[name]}, got {got}. STOP.")
        spec = importlib.util.spec_from_file_location(f"_frozen_{p.stem}", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        mods[p.stem] = m
    return mods["metrics_exp08"], mods["sliced_w1"].sliced_w1, mods["e30plus_eval_cpu_worker"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="table1_ours")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    M, sw1, W = load_frozen()
    print(f"[ruler] w1_rows / w1_real_floor <- {FROZEN/'e30plus_eval_cpu_worker.py'}")
    print(f"[ruler] N_MATCH {W.N_MATCH} | eval seed {W.EVAL_SEED} | {W.N_PROJ} projections "
          f"@ {W.W1_PROJ_SEED} | raw counts")

    D = C.LEGACY_DATA / "all"
    xv = np.load(D / "val.npy", mmap_mode="r")
    yv = np.load(D / "val_celltype.npy", allow_pickle=True).astype(str)
    xt = np.load(D / "train.npy", mmap_mode="r")
    yt = np.load(D / "train_celltype.npy", allow_pickle=True).astype(str)
    proj = M.make_projections(xv.shape[1], n_proj=W.N_PROJ, seed=W.W1_PROJ_SEED)

    # ---- the mandatory identity check, in this very process ------------------------
    arch = json.loads(COND_W1.read_text(encoding="utf-8"))
    selfcheck = {}
    for t in W.TYPES:
        got, _ = W.w1_real_floor(t, xv, yv, xt, yt, M, sw1, proj)
        want = arch["types"][t]["rows"][FLOOR_KEY]
        d = {"W1": got["W1"] - want["marginal_w1_mean"],
             "W1_median": got["W1_median"] - want["marginal_w1_median"],
             "W1_p90": got["W1_p90"] - want["marginal_w1_p90"],
             "sliced_W1": got["sliced_W1"] - want["sliced_w1"]}
        ok = got["n"] == want["n"] and all(v == 0.0 for v in d.values())
        selfcheck[t] = {"recomputed": got, "archived_cond_w1": want, "diffs": d,
                        "bit_identical": ok}
        print(f"[selfcheck] {t:<12} floor mW1 {got['W1']:.6f} sliced {got['sliced_W1']:.6f} -> "
              f"{'BIT-IDENTICAL to cond_w1.json' if ok else 'DIFFERS ' + str(d)}")
        if not ok:
            raise SystemExit(
                f"the frozen W1 ruler no longer reproduces the archived floor row for {t}.\n"
                "Stop: any number produced now would not be comparable with the published\n"
                "table. Do not loosen this check.")

    # ---- the runs -------------------------------------------------------------------
    runs = load_runs(a.experiment)
    if not runs:
        raise SystemExit(f"no runs under out/runs/{a.experiment}")
    rows = []
    for r in sorted(runs, key=lambda r: r["stem"]):
        target = r["request"]
        if target not in W.N_MATCH:
            print(f"[skip] {r['stem']}: {target!r} has no N_MATCH entry in the frozen ruler "
                  "(this ruler is Table 1's three single-type targets only)")
            continue
        raw = np.load(Path(r["_path"]).with_suffix(".npy"), mmap_mode="r")
        res, sel = W.w1_rows(target, "OURS", raw, xv, yv, xt, yt, M, sw1, proj)
        g, f = res["generated"], res["floor"]
        row = {"stem": r["stem"], "arm": r.get("arm"), "seed": r.get("seed"),
               "target": target, "alpha": r.get("alpha"), "sampler": r.get("sampler"),
               "n_delivered": int(raw.shape[0]), "matched_n": W.N_MATCH[target],
               "W1": g["W1"], "W1_median": g["W1_median"], "W1_p90": g["W1_p90"],
               "sliced_W1": g["sliced_W1"],
               "floor_W1": f["W1"], "floor_W1_median": f["W1_median"],
               "floor_sliced_W1": f["sliced_W1"],
               "W1_over_floor": g["W1"] / f["W1"],
               "sliced_W1_over_floor": g["sliced_W1"] / f["sliced_W1"],
               "n_generated_indices": len(sel.get("generated_indices", [])),
               "unique_cell_fraction": r.get("unique_cell_fraction"),
               "samples_sha256": r.get("samples_sha256")}
        rows.append(row)
        print(f"[score] {row['arm']:<14} seed {row['seed']}  W1 {row['W1']:.4f} "
              f"({row['W1_over_floor']:.2f}x floor)  sliced-W1 {row['sliced_W1']:.4f} "
              f"({row['sliced_W1_over_floor']:.2f}x floor)  n={row['matched_n']}")

    per_target = {}
    for t in sorted({r["target"] for r in rows}):
        rs_ = [r for r in rows if r["target"] == t]
        agg = {"n_seeds": len(rs_), "matched_n": rs_[0]["matched_n"],
               "floor_W1": rs_[0]["floor_W1"], "floor_sliced_W1": rs_[0]["floor_sliced_W1"]}
        for k in ("W1", "W1_median", "sliced_W1", "W1_over_floor", "sliced_W1_over_floor"):
            vals = [x[k] for x in rs_]
            agg[k] = {"mean": float(statistics.fmean(vals)),
                      "sd_run_to_run": (float(statistics.stdev(vals)) if len(vals) > 1 else None),
                      "values": vals}
        per_target[t] = agg
        print(f"[summary] {t:<12} W1 {agg['W1']['mean']:.4f} +- "
              f"{(agg['W1']['sd_run_to_run'] or 0):.4f}   sliced-W1 "
              f"{agg['sliced_W1']['mean']:.4f} +- {(agg['sliced_W1']['sd_run_to_run'] or 0):.4f}"
              f"   (floor {agg['floor_W1']:.4f} / {agg['floor_sliced_W1']:.4f})")

    blob = {"experiment": a.experiment,
            "caliber": {"scale": "raw counts",
                        "scorer": str(FROZEN / "e30plus_eval_cpu_worker.py"),
                        "scorer_sha256": FROZEN_SHA["e30plus_eval_cpu_worker.py"],
                        "functions_called": ["w1_rows", "w1_real_floor"],
                        "metrics": str(FROZEN / "metrics_exp08.py"),
                        "metrics_sha256": FROZEN_SHA["metrics_exp08.py"],
                        "sliced_w1": str(FROZEN / "sliced_w1.py"),
                        "sliced_w1_sha256": FROZEN_SHA["sliced_w1.py"],
                        "N_MATCH": W.N_MATCH, "eval_seed": W.EVAL_SEED,
                        "n_proj": W.N_PROJ, "proj_seed": W.W1_PROJ_SEED,
                        "reference": "that type's real val cells",
                        "why_not_score_distribution":
                            "score_distribution.py measures on CP10K+log1p with its own "
                            "projections (seed 20260906); gate_w1_caliber measured both on the "
                            "same cells and they are different rulers. See out/gates/"
                            "W1_CALIBER.json."},
            "selfcheck_vs_archived_cond_w1_floor": selfcheck,
            "rows": rows, "per_target": per_target}
    dest = Path(a.out) if a.out else (C.OUT / "summary" / f"w1_paper_{a.experiment}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(blob, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[score] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
