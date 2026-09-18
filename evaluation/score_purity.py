# -*- coding: utf-8 -*-
"""Score a delivered run's purity on the CANONICAL ruler, at matched n = 2500.
Purpose: score a run's delivered cells for purity ON THE SAME RULER as every published row.
Three things must hold, and none of them may be relaxed:

1. THE SAME CELLTYPIST MODEL. Every scoring records the model sha256, the celltypist version
      and the model date, and checks them against what gate G5 recorded. A mismatch stops the run.
2. THE SAME FROZEN LABEL SET: the `fixed_labels` of the ruler file, READ rather than
      recomputed. Recomputing it would quietly move published numbers as the directory changes.
3. THE SAME MATCHED N: the delivery is cut into non-overlapping blocks of that size, purity is
      computed within a block and then averaged over the run, because the other rows use that n.

With a single seed there is NO run-to-run error bar. The block-to-block spread is reported, but
`block_spread_NOT_run_to_run` / "this is a LOWER BOUND on run-to-run variability":
its field name says so explicitly, so it cannot be quoted as one.

    python -m analysis.score_purity --experiment main_single_type
    python -m analysis.score_purity --run out/runs/main_single_type/<arm>/<stem>.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                  # noqa: E402
from crndiff.io import load_runs, sha256_file                    # noqa: E402
from crndiff.metrics import (CANONICAL_CEILING, CEILING_SOURCE,   # noqa: E402
                         ceiling_from_archive, load_fixed_label_sets, purity_from_hist)

# A request whose canonical purity is scored against a BARE type's label set. INHERITED
# from the archive, not chosen here: HVG2K/results/diagnostics/E21b_intersect_purity.json
# scores the intersection request with "target": "Endothelial", "scoring_n": 2500 and the
# bare-Endothelial fixed_labels (the 8 EC* labels) of master_purity_v7.json. Scoring it
# against any other label set would put Table 3/9's purity column on a different ruler
# from the archived row, i.e. would make the two incomparable.
LABEL_SET_ALIAS = {"INTERSECT_Endothelial_g128": "Endothelial"}

BLOCK = 2500          # matched n: the size every other row in the table was scored at
MODEL_PKL = (Path(os.path.expanduser("~")) / ".celltypist" / "data" / "models"
             / "Healthy_Adult_Heart.pkl")


def ruler_check() -> dict:
    """The ruler must be the one gate G5 passed on. Different ruler => stop, do not score."""
    g5p = C.GATES / "CELLTYPIST.json"
    if not g5p.exists():
        raise SystemExit(f"{g5p} not found -- run `python -m evaluation.gate_celltypist` first; "
                         "scoring on an unverified ruler is exactly what G5 exists to prevent")
    g5 = json.loads(g5p.read_text(encoding="utf-8"))
    if not g5.get("passed"):
        raise SystemExit("gate G5 did not pass; refusing to score purity on an unverified ruler")
    import celltypist
    now = {"celltypist_version": str(celltypist.__version__),
           "model_sha256": sha256_file(MODEL_PKL) if MODEL_PKL.exists() else None}
    was = {"celltypist_version": g5["identity"].get("celltypist_version"),
           "model_sha256": g5["identity"].get("model_sha256")}
    if now != was:
        raise SystemExit(f"the CellTypist ruler changed since gate G5 passed:\n"
                         f"  at G5: {was}\n  now:   {now}\nStop and report (do not re-run G5 "
                         "with a different model to make this go away).")
    return {**now, "model_date": g5["identity"].get("model_date"),
            "model_version": g5["identity"].get("model_version"),
            "verified_by": "gate G5", "g5_reference": g5.get("reference")}


def annotate(x: np.ndarray, genes: list) -> np.ndarray:
    """CellTypist's own input caliber: normalize_total(1e4) + log1p, majority_voting=False.
    Copied from the frozen `celltypist_eval.py::to_adata` / `annotate` -- do not 'improve'."""
    import anndata as ad
    import celltypist
    import scanpy as sc
    a = ad.AnnData(np.asarray(x, dtype=np.float32))
    a.var_names = list(genes)
    a.var_names_make_unique()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    res = celltypist.annotate(a, model="Healthy_Adult_Heart.pkl", majority_voting=False)
    return res.predicted_labels["predicted_labels"].to_numpy().astype(str)


def score_run(run: dict, genes: list, labels: set, target: str) -> dict:
    x = np.load(Path(run["_path"]).with_suffix(".npy"), mmap_mode="r")
    n = x.shape[0]
    n_blocks = n // BLOCK
    if n_blocks < 1:
        raise SystemExit(f"{run['stem']}: only {n} cells, cannot form one matched block of {BLOCK}")
    used = n_blocks * BLOCK
    lab = annotate(np.asarray(x[:used], dtype=np.float64), genes)
    per_block, hists = [], []
    for b in range(n_blocks):
        h = lab[b * BLOCK:(b + 1) * BLOCK]
        u, c = np.unique(h, return_counts=True)
        hist = {str(k): int(v) for k, v in zip(u, c)}
        p, _ = purity_from_hist(hist, labels)
        per_block.append(p)
        hists.append(hist)
    u, c = np.unique(lab, return_counts=True)
    whole = {str(k): int(v) for k, v in zip(u, c)}
    p_whole, _ = purity_from_hist(whole, labels)
    ceiling = CANONICAL_CEILING.get(target)
    mean = float(statistics.fmean(per_block))
    spread = float(statistics.pstdev(per_block)) if n_blocks > 1 else 0.0
    return {
        "stem": run["stem"], "arm": run.get("arm"),
        "request": run.get("request"), "target": target,
        "label_set_alias": (None if run.get("request") == target
                            else f"{run.get('request')} -> {target} "
                                 "(E21b_intersect_purity.json)"),
        "alpha": run.get("alpha"), "sampler": run.get("sampler"), "tau": run.get("tau"),
        "n_delivered": int(n), "n_scored": int(used),
        "matched_n": BLOCK, "n_blocks": n_blocks,
        "purity_matched_n": mean,
        "purity_per_block": per_block,
        "block_spread_NOT_run_to_run": spread,
        "block_spread_note": "dispersion ACROSS BLOCKS of one run. This is NOT a run-to-run "
                             "error bar and is a LOWER BOUND on it: the blocks share one seed, "
                             "one chain and one resampling history. Only seeds 20260988/20260989 "
                             "can give a run-to-run bar.",
        "purity_whole_set": p_whole,
        "canonical_ceiling": ceiling,
        "canonical_ceiling_source": CEILING_SOURCE,
        "purity_over_ceiling": (mean / ceiling) if ceiling else None,
        "label_dist_whole": whole,
        "label_dists_per_block": hists,
        # the diversity / sampler readings Q asked for alongside purity
        "unique_cell_fraction": run.get("unique_cell_fraction"),
        "unique_lineage_fraction": run.get("unique_lineage_fraction"),
        "ess_min_fraction": run.get("ess_min_fraction"),
        "ess_pre_final_fraction": run.get("ess_pre_final_fraction"),
        "n_intermediate_resamples": run.get("n_intermediate_resamples"),
        "resample_steps": run.get("resample_steps"),
        # the ruler this row is on, carried per row so a table can be audited line by line
        "tilt_table_sha256": run.get("tilt_table_sha256"),
        "celltypist_model_sha256": run.get("celltypist_model_sha256"),
        "celltypist_version": run.get("celltypist_version"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="main_single_type")
    ap.add_argument("--run", default=None, help="score a single run json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    ruler = ruler_check()
    print(f"[ruler] celltypist {ruler['celltypist_version']} | model "
          f"{ruler['model_sha256'][:16]} | verified by {ruler['verified_by']}")
    label_sets = load_fixed_label_sets(C.MASTER_PURITY_V7)
    genes = json.loads((C.LEGACY_DATA / "all" / "meta.json").read_text(encoding="utf-8"))["genes"]

    if a.run:
        r = json.loads(Path(a.run).read_text(encoding="utf-8"))
        r["_path"] = str(Path(a.run))
        runs = [r]
    else:
        runs = load_runs(a.experiment)
    if not runs:
        raise SystemExit(f"no runs under out/runs/{a.experiment}")

    rows = []
    for r in sorted(runs, key=lambda r: r["stem"]):
        target = LABEL_SET_ALIAS.get(r["request"], r["request"])
        if target not in label_sets:
            print(f"[skip] {r['stem']}: no canonical label set for {target!r}")
            continue
        print(f"[score] {r['stem']}")
        row = score_run(r, genes, label_sets[target], target)
        rows.append(row)
        print(f"    purity@n{BLOCK} {row['purity_matched_n']:.4f} "
              f"(blocks {row['n_blocks']}, spread {row['block_spread_NOT_run_to_run']:.4f}) "
              f"| ceiling {row['canonical_ceiling']} "
              f"-> {row['purity_over_ceiling']:.1%} | uniq_cell "
              f"{row['unique_cell_fraction']} | ess_min {row['ess_min_fraction']} "
              f"| fires {row['n_intermediate_resamples']}")

    blob = {"experiment": a.experiment, "matched_n": BLOCK,
            "label_set_source": str(C.MASTER_PURITY_V7),
            "label_set_rule": "master_purity_v7 fixed_labels, READ not recomputed",
            "ruler": ruler,
            "single_seed_warning": "one seed only: no run-to-run error bars anywhere in this file",
            "rows": rows}
    dest = Path(a.out) if a.out else (C.OUT / "summary" / f"purity_{a.experiment}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(blob, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[score] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
