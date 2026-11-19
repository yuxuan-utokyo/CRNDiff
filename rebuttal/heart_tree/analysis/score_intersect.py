# -*- coding: utf-8 -*-
"""Stage 3 (Table 3 / 9) readings: number of identities and distribution distance for the intersection request endothelial ∩ UAP1+.

★ **no new ruler is written.** Table 3/9 have their own archived definitions, recorded in
`HVG2K/results/tables/intersection_metrics.md` in it, verbatim:

> **Caliber**: the Table 2 path verbatim — `metrics.py` `w1_1d`, `sliced_w1` with **512**
> projections at seed **20261070**, subsample seed **20261071**. Not a new estimator.

so here we do **not** go through `analysis/score_distribution.py`(that ruler is for Tables 1/6 per-gene W1
used; projection seed, number of directions, reference set and matched n all differ, so the two sets of numbers are not comparable). Here we use:

* `analysis/_frozen/metrics_exp08.py` —— `w1_1d` / `make_projections` original.
  the archived driver script says `Experiments/Toy/_shared/code/metrics.py`, **that path was retired together with the X tree
  and disappears**;`metrics_exp08.py` file header itself records moved from cc/experiments/EXP08_toy_countsdiff/
  metrics.py(whole file) ... every other byte unchanged (GTReference / make_projections / swd /
  w1_1d / mmd_rbf / median_heuristic_gamma / marginal_gammas all unchanged), and
  `sliced_w1.py` file header records `metrics.py::swd` function body was copied verbatim: identical to the one here
  `metrics_exp08.py::swd` identical (including `projections.shape[1]` that bug is present in both). Two independent pieces of evidence
  points to the same file, so this **recovers the original**, not a rewrite.
* `analysis/_frozen/sliced_w1.py` —— `HVG2K/cond_w1/code/sliced_w1.py` verbatim copy
  (sha256 src == dst); this fixes exactly the issue above `shape[1]` loop bounds.
* `analysis/_frozen/intersection_metrics.py` — a verbatim copy of the archived driver script, kept only as a record of definitions,
  **not executed** (it is bound to the file names of retired arms and the retired X path).

reference set and floor, built exactly as in the archived script:
  reference = in the held-out set (val) `celltype == Endothelial` and `gene 128 (UAP1) > 0` cells, n = 853;
  shuffled with SPLIT_SEED=20261072 and split in half into disjoint A / B; A is the reference, B the floor;
  matched n = min(|A|, |B|, the delivered amount per arm); this round every arm has N=2500 ≫ 426, so n is still 426,
  reference and floor are two **bit-identical** draws (draws 1 and 2 of the same SUB_SEED=20261071 stream).

★ ****mandatory self-check**: the recomputed floor row must match the archived `intersection_metrics.json` of
`floor (held-out UAP1+ endo, other half)` **exactly equal** (mW1 0.08409037558685446 /
jSWD 0.27505956356632155). Stop if unequal: it means the ruler or reference set is not the same, and this table cannot be compared with the archived rows.

☆ **one deviation that must be disclosed**: the archived script draws the 3 arms **consecutively from the same rng stream**, so the numbers in a row
depends on its position among those 3. This round Stage 3 has 6 runs (3 seeds × 2 arms); consecutive draws would make the same run
numbers vary with ordering. So **each run starts its own `default_rng(SUB_SEED)`**(same seed, independent stream).
reference and floor are still drawn in the archived order (draws 1 and 2 of the stream), so the floor row stays bit-comparable with the archive; the self-check above is the evidence.

    python -m analysis.score_intersect --experiment intersect_ladder
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
from hvg import config as C                                          # noqa: E402
from hvg.io import load_runs, sha256_file                            # noqa: E402

FROZEN = ROOT / "analysis" / "_frozen"
# sha256 of the verbatim copies, recorded when they were taken. If a frozen file ever stops
# matching, STOP -- a silently different ruler is exactly what makes a table incomparable.
FROZEN_SHA = {
    "metrics_exp08.py": "6319272ff7d1440271773afb25fd8bbc6e383a230a604f3000c142334aadb0a5",
    "sliced_w1.py": "6224428255b9f97cf577e668aa16bd62c263bd5c07cf0ce4c2a0bde9ef9a63e0",
    "intersection_metrics.py": "465a9d48dab51a4a4d2d0b80373fc9bc87dea71205556e88908bd077738123ee",
}

# every constant below is READ OFF the archived driver, not chosen here
GENE = 128                                  # UAP1
PROJ_SEED, SUB_SEED, N_PROJ = 20261070, 20261071, 512
SPLIT_SEED = 20261072
REQUEST = "INTERSECT_Endothelial_g128"

# the archived floor row, quoted at full precision from
# HVG2K/results/tables/intersection_metrics.json
ARCHIVED_FLOOR = {"marginal_w1_mean": 0.08409037558685446,
                  "marginal_w1_median": 0.009389671361502348,
                  "sliced_w1": 0.27505956356632155,
                  "n": 426}
ARCHIVED_TABLE = (C.ROOT.parent.parent / "Conditional generation experiments" / "HVG2K"
                  / "results" / "tables" / "intersection_metrics.json")


def load_frozen():
    """Import the two frozen scorers by path, after checking their bytes."""
    mods = {}
    for name in ("metrics_exp08.py", "sliced_w1.py"):
        p = FROZEN / name
        if not p.exists():
            raise SystemExit(f"missing frozen scorer {p}. Do not write a replacement: the "
                             "archived caliber is the only thing that makes Table 3/9 "
                             "comparable to its published row.")
        got = sha256_file(p)
        if got != FROZEN_SHA[name]:
            raise SystemExit(f"{p} changed: expected {FROZEN_SHA[name]}, got {got}. STOP.")
        spec = importlib.util.spec_from_file_location(f"_frozen_{p.stem}", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        mods[p.stem] = m
    # the third file is archived for the record only; it is never executed (it binds the old
    # arms' filenames and the retired X paths), but its bytes are still checked.
    p3 = FROZEN / "intersection_metrics.py"
    if p3.exists():
        got = sha256_file(p3)
        if got != FROZEN_SHA["intersection_metrics.py"]:
            raise SystemExit(f"{p3} changed: expected {FROZEN_SHA['intersection_metrics.py']}, "
                             f"got {got}. STOP.")
    return mods["metrics_exp08"], mods["sliced_w1"].sliced_w1


def build_reference(arm_sizes: list[int]) -> dict:
    """Reference / floor exactly as `_frozen/intersection_metrics.py` builds them."""
    D = C.LEGACY_DATA / "all"
    Xv = np.load(D / "val.npy", mmap_mode="r")
    yv = np.load(D / "val_celltype.npy", allow_pickle=True).astype(str)
    sel = np.flatnonzero((yv == "Endothelial") & (np.asarray(Xv)[:, GENE] > 0))

    rs = np.random.default_rng(SPLIT_SEED)
    perm = rs.permutation(len(sel))
    half = len(sel) // 2
    idx_A, idx_B = np.sort(sel[perm[:half]]), np.sort(sel[perm[half:2 * half]])

    n = int(min([len(idx_A), len(idx_B)] + list(arm_sizes)))
    rng = np.random.default_rng(SUB_SEED)
    ref = np.asarray(Xv[idx_A[np.sort(rng.choice(len(idx_A), n, replace=False))]],
                     dtype=np.float64)
    flo = np.asarray(Xv[idx_B[np.sort(rng.choice(len(idx_B), n, replace=False))]],
                     dtype=np.float64)
    return {"ref": ref, "flo": flo, "n": n, "n_total": int(len(sel)),
            "half_sizes": [int(len(idx_A)), int(len(idx_B))],
            "data": str((D / "val.npy").resolve())}


def score(G, ref, proj, M, sw1) -> dict:
    """The archived `score()` verbatim: mean/median per-gene w1_1d + sliced_w1."""
    G = np.asarray(G, dtype=np.float64)
    w = np.array([M.w1_1d(G[:, g], ref[:, g]) for g in range(ref.shape[1])])
    return {"marginal_w1_mean": float(w.mean()),
            "marginal_w1_median": float(np.median(w)),
            "sliced_w1": float(sw1(G, ref, proj, M.w1_1d)),
            "n": int(len(G))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="intersect_ladder")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    M, sw1 = load_frozen()
    print(f"[ruler] w1_1d/make_projections <- {FROZEN/'metrics_exp08.py'}")
    print(f"[ruler] sliced_w1              <- {FROZEN/'sliced_w1.py'}")

    runs = [r for r in load_runs(a.experiment) if r.get("request") == REQUEST]
    if not runs:
        raise SystemExit(f"no {REQUEST} runs under out/runs/{a.experiment}")
    runs.sort(key=lambda r: r["stem"])
    paths = {r["stem"]: Path(r["_path"]).with_suffix(".npy") for r in runs}
    sizes = [int(np.load(paths[r["stem"]], mmap_mode="r").shape[0]) for r in runs]

    R = build_reference(sizes)
    ref, flo, n = R["ref"], R["flo"], R["n"]
    print(f"[ref] held-out endothelial with UAP1 > 0: n = {R['n_total']} "
          f"(halves {R['half_sizes']})")
    print(f"[matched n] {n}   (arms {sizes})")
    proj = M.make_projections(ref.shape[1], n_proj=N_PROJ, seed=PROJ_SEED)

    floor = {**score(flo, ref, proj, M, sw1),
             "source": "val.npy, disjoint half of the reference", "uap1_pos_frac": 1.0}

    # ---- the mandatory identity check against the archived floor row ------------------
    diffs = {k: floor[k] - ARCHIVED_FLOOR[k] for k in ("marginal_w1_mean",
                                                       "marginal_w1_median", "sliced_w1")}
    identical = floor["n"] == ARCHIVED_FLOOR["n"] and all(d == 0.0 for d in diffs.values())
    print(f"[floor] mW1 {floor['marginal_w1_mean']:.10f}  jSWD {floor['sliced_w1']:.10f}")
    print(f"[floor] archived mW1 {ARCHIVED_FLOOR['marginal_w1_mean']:.10f}  "
          f"jSWD {ARCHIVED_FLOOR['sliced_w1']:.10f}")
    if not identical:
        raise SystemExit(
            "the recomputed floor does NOT reproduce the archived floor row bit for bit:\n"
            f"  diffs {diffs}\n"
            "That means the reference set, the matched n, or the estimator is not the archived\n"
            "one, so Table 3/9 would not be comparable to its published row. STOP -- do not\n"
            "loosen this check and do not substitute a different scorer.")
    print("[floor] reproduces the archived row bit for bit -> the caliber is the archived one")

    # ---- the arms --------------------------------------------------------------------
    table = json.loads((C.TABLES / f"logc_{REQUEST}.json").read_text(encoding="utf-8"))
    real_rate = table["provenance"]["detection_rate_in_type"]
    print(f"[real] UAP1+ rate among {table['provenance']['type']} cells "
          f"= {real_rate:.10f}  (read from logc_{REQUEST}.json, not typed in)")

    rows = []
    for r in runs:
        p = paths[r["stem"]]
        arr = np.load(p, mmap_mode="r")
        rng = np.random.default_rng(SUB_SEED)          # fresh stream per run, see the docstring
        s = np.sort(rng.choice(arr.shape[0], n, replace=False))
        g = np.asarray(arr[s], dtype=np.float64)
        full = np.asarray(arr[:, GENE])
        row = {"stem": r["stem"], "arm": r.get("arm"), "seed": r.get("seed"),
               "sampler": r.get("sampler"), "alpha": r.get("alpha"), "tau": r.get("tau"),
               "n_particles": r.get("n_particles"), "n_out": r.get("n_out"),
               "draws": (r["n_particles"] / r["n_out"]) if r.get("n_out") else None,
               "uap1_pos_frac_delivered": float((full > 0).mean()),
               "uap1_pos_frac_in_sample": float((g[:, GENE] > 0).mean()),
               "uap1_pos_frac_real_in_type": real_rate,
               "enrichment_over_real": float((full > 0).mean() / real_rate),
               **score(g, ref, proj, M, sw1),
               "mW1_over_floor": None, "jSWD_over_floor": None,
               "unique_cell_fraction": r.get("unique_cell_fraction"),
               "unique_lineage_fraction": r.get("unique_lineage_fraction"),
               "ess_min_fraction": r.get("ess_min_fraction"),
               "ess_pre_final_fraction": r.get("ess_pre_final_fraction"),
               "n_intermediate_resamples": r.get("n_intermediate_resamples"),
               "tilt_table_sha256": r.get("tilt_table_sha256"),
               "samples_sha256": r.get("samples_sha256"),
               "source": p.name}
        row["mW1_over_floor"] = row["marginal_w1_mean"] / floor["marginal_w1_mean"]
        row["jSWD_over_floor"] = row["sliced_w1"] / floor["sliced_w1"]
        rows.append(row)
        print(f"[score] {row['arm']:<10s} seed {row['seed']}  UAP1+ "
              f"{row['uap1_pos_frac_delivered']:.4f} ({row['enrichment_over_real']:.1f}x real)  "
              f"mW1 {row['marginal_w1_mean']:.4f} ({row['mW1_over_floor']:.2f}x floor)  "
              f"jSWD {row['sliced_w1']:.4f} ({row['jSWD_over_floor']:.2f}x floor)")

    # per-arm run-to-run summary across the three seeds
    by_arm = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    summary = {}
    for arm, rs_ in sorted(by_arm.items()):
        agg = {}
        for k in ("uap1_pos_frac_delivered", "marginal_w1_mean", "sliced_w1",
                  "mW1_over_floor", "jSWD_over_floor", "unique_cell_fraction",
                  "unique_lineage_fraction", "ess_min_fraction"):
            vals = [x[k] for x in rs_ if x.get(k) is not None]
            if not vals:
                continue
            agg[k] = {"mean": float(statistics.fmean(vals)),
                      "sd_run_to_run": (float(statistics.stdev(vals)) if len(vals) > 1 else None),
                      "n_seeds": len(vals), "values": vals}
        summary[arm] = agg

    blob = {
        "experiment": a.experiment, "request": REQUEST,
        "caliber": {
            "source": "HVG2K/results/tables/intersection_metrics.md -- 'the Table 2 path "
                      "verbatim ... Not a new estimator'",
            "w1_1d_and_make_projections": {"file": str(FROZEN / "metrics_exp08.py"),
                                           "sha256": FROZEN_SHA["metrics_exp08.py"]},
            "sliced_w1": {"file": str(FROZEN / "sliced_w1.py"),
                          "sha256": FROZEN_SHA["sliced_w1.py"]},
            "archived_driver_kept_for_the_record": {
                "file": str(FROZEN / "intersection_metrics.py"),
                "sha256": FROZEN_SHA["intersection_metrics.py"], "executed": False},
            "n_proj": N_PROJ, "proj_seed": PROJ_SEED, "sub_seed": SUB_SEED,
            "split_seed": SPLIT_SEED, "gene": GENE,
            "dead_path_in_the_archived_driver":
                "Experiments/Toy/_shared/code/metrics.py (retired with the X tree); recovered "
                "as metrics_exp08.py, whose own header records it is that file verbatim and "
                "whose swd() body matches the copy quoted inside sliced_w1.py",
            "declared_deviation":
                "the archived driver drew all arm subsamples from ONE continuing rng stream, so "
                "a row's number depended on its position in a 3-arm list. With 6 runs each run "
                "gets its own default_rng(SUB_SEED) instead. Reference and floor are still the "
                "first two draws of that stream, which is why the floor row below reproduces "
                "the archived floor bit for bit.",
        },
        "reference": {"definition": "held-out (val) endothelial with UAP1 (gene 128) > 0",
                      "file": R["data"], "n_total": R["n_total"],
                      "half_sizes": R["half_sizes"], "matched_n": n,
                      "floor": "the reference's disjoint other half, same matched n"},
        "floor_identity_check": {
            "recomputed": {k: floor[k] for k in ("marginal_w1_mean", "marginal_w1_median",
                                                 "sliced_w1", "n")},
            "archived": ARCHIVED_FLOOR,
            "archived_source": str(ARCHIVED_TABLE),
            "diffs": diffs, "bit_identical": identical},
        "real_uap1_rate_in_type": {
            "value": real_rate,
            "source": str(C.TABLES / f"logc_{REQUEST}.json") + " :: provenance."
                                                               "detection_rate_in_type",
            "n_ref_cells": table["provenance"]["n_ref_cells"],
            "n_cells_type": table["provenance"]["n_cells_type"]},
        "floor_row": floor,
        "rows": rows,
        "per_arm": summary,
    }
    dest = Path(a.out) if a.out else (C.OUT / "summary" / f"intersect_{a.experiment}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(blob, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[score] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
