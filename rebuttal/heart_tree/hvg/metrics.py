# -*- coding: utf-8 -*-
"""The canonical readings on delivered HVG cells.
purpose: reading definition for delivered cells. **The purity definition comes entirely from `_frozen/canonical_purity.py` exported**,
unchanged: purity = mass of the CellTypist label histogram that falls in the **frozen label set**; the label set comes from
`master_purity_v7.json`("labels strong in >= 50% of the calls scoring this type").
frozen means: **the label set is never recomputed here**; recomputing would silently shift published
OURS numbers (verbatim from the frozen copy).** The ceiling uses `CANONICAL_CEILING`, not `PUBLISHED_CEILING`**
(two rulers, see the comment block below).

the two kinds of readings are clearly separated:

* `canonical_purity(...)`  needs label histograms scored by CellTypist. CellTypist is **not** in this directory
  in assets (`tools/check_assets.py` reports it together with the expected path), so this function only applies when the label
  histogram **already exists**; without labels it raises; never substitute anything else for purity.
* `distribution_readings(...)`  uses only the delivered counts: mean / max / zero fraction / library size
  quantiles / per-gene detection rates. These need no missing assets and are the basis of gate G3 and summarize.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# constants of the frozen canonical_purity.py, exported verbatim
TYPES = ["Endothelial", "Myeloid", "Neuronal"]

# ★ two ceilings, do not mix them up (I got it wrong the first time on 2026-09-06, see receipt 106 section 10):
#
# PUBLISHED_CEILING — **the old, non-canonical** one. It is what each baseline table **published separately** at the time
#   [real cells] row; each table uses the RNG-dependent label set of its own call, so they are not on the same scale.
#   in the frozen copy it only serves the delta column and **must not be used as the denominator of canonical purity**.
# CANONICAL_CEILING — the real val cells row under the v7 frozen label set, i.e. that of main_table_v2.md
#   Real validation those three rows (verbatim from the frozen copy comment: 0.9372 / 0.8836 / 0.7045).
#   the other rows kept in the table are all numbers on this ruler, so **the new OURS row must be divided by this one**.
#
# verifiable: compute directly from the archived label_dists in assets/celltypist/ with the v7 fixed_labels,
# Myeloid 9 calls all give 0.8836 (n=2302), Neuronal all 9 give 0.7045 (n=396);
# Endothelial n=2500 subsamples fluctuate between 0.9348–0.9380; the full val (n=10057) gives 0.9374,
# consistent with 0.9372 in main_table_v2.`ceiling_from_archive()` is what this is for.
PUBLISHED_CEILING = {"Endothelial": 0.9428, "Myeloid": 0.9457, "Neuronal": 0.7045}
CANONICAL_CEILING = {"Endothelial": 0.9372, "Myeloid": 0.8836, "Neuronal": 0.7045}
CEILING_SOURCE = ("main_table_v2.md 'Real validation' rows, i.e. the v7 canonical scale; "
                  "NOT the pre-canonical PUBLISHED_CEILING the baseline tables quoted")
# OURS, verbatim from `main_table_v2.md` (uncond, cond, sd_of_cond)
OURS_MAIN = {
    "Endothelial": (0.2240, 0.9115, 0.0007),
    "Myeloid": (0.0420, 0.7840, 0.0065),
    "Neuronal": (0.0048, 0.5504, None),
}


def purity_from_hist(dist: dict, labels: set) -> tuple[float, int]:
    """verbatim from `_frozen/canonical_purity.py`. Returns (purity, n)."""
    tot = sum(dist.values())
    hit = sum(v for k, v in dist.items() if k in labels)
    return (hit / tot if tot else 0.0), tot


def load_fixed_label_sets(v7_path: Path) -> dict:
    """The FROZEN label sets: {type: set(labels)}. Never recomputed here (see the module note)."""
    p = Path(v7_path)
    if not p.exists():
        raise SystemExit(
            f"missing required input (canonical label sets): {p}\n"
            "  -> master_purity_v7.json is the frozen label set every published purity is "
            "built on. It was NOT part of the HVG migration; see tools/check_assets.py.")
    blob = json.loads(p.read_text(encoding="utf-8"))
    return {t: set(v["fixed_labels"]) for t, v in blob["types"].items()}


CEILING_ROW = "real val cells"      # the "real val cells" row in every archived call


def ceiling_from_archive(celltypist_dir: Path, target: str, v7_path: Path) -> dict:
    """Recompute the canonical ceiling from the ARCHIVED label histograms, so the constant is
    checkable rather than trusted. Returns every distinct value found and how often, because
    the n=2500 subsample rows legitimately differ from the whole-val row."""
    labels = load_fixed_label_sets(v7_path)[target]
    found: dict = {}
    for p in sorted(Path(celltypist_dir).glob(f"celltypist_{target}_*.json")):
        hist = json.loads(p.read_text(encoding="utf-8")).get("label_dists", {}).get(CEILING_ROW)
        if not hist:
            continue
        pur, n = purity_from_hist(hist, labels)
        key = f"{pur:.4f}@n{n}"
        found.setdefault(key, []).append(p.name)
    return {"target": target, "constant": CANONICAL_CEILING.get(target),
            "recomputed_from_archive": {k: len(v) for k, v in found.items()},
            "calls": found, "row": CEILING_ROW}


def canonical_purity(label_hist: dict, target: str, v7_path: Path) -> dict:
    """Purity of one arm's CellTypist label histogram on the canonical scale."""
    labels = load_fixed_label_sets(v7_path)
    if target not in labels:
        raise SystemExit(f"no canonical label set for target {target!r}; have {sorted(labels)}")
    pur, n = purity_from_hist(label_hist, labels[target])
    ceiling = CANONICAL_CEILING.get(target)
    return {"purity": pur, "n": n, "target": target,
            "canonical_ceiling": ceiling,
            "purity_over_ceiling": (pur / ceiling if ceiling else None),
            "label_set_size": len(labels[target]),
            "scale": "master_purity_v7 fixed_labels (frozen; never recomputed)"}


def distribution_readings(x: np.ndarray, reference: np.ndarray | None = None) -> dict:
    """Label-free readings on delivered counts. These need no missing asset.

    reference  optional real counts (same gene order) for the paired columns; None leaves
               every `*_reference` field null rather than inventing a comparison.
    """
    a = np.asarray(x)
    lib = a.sum(axis=1, dtype=np.float64)
    detect = (a > 0).mean(axis=0)
    out = {
        "n_cells": int(a.shape[0]), "n_genes": int(a.shape[1]),
        "mean": float(a.mean()), "max": int(a.max()),
        "zero_fraction": float(np.mean(a == 0)),
        "library_size_mean": float(lib.mean()),
        "library_size_quantiles": {str(q): float(np.quantile(lib, q))
                                   for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "gene_detection_rate_mean": float(detect.mean()),
        "n_genes_never_detected": int((detect == 0).sum()),
        "unique_cell_fraction": float(np.unique(a, axis=0).shape[0] / a.shape[0]),
    }
    if reference is not None:
        r = np.asarray(reference)
        rlib = r.sum(axis=1, dtype=np.float64)
        rdetect = (r > 0).mean(axis=0)
        out.update({
            "mean_reference": float(r.mean()),
            "zero_fraction_reference": float(np.mean(r == 0)),
            "library_size_mean_reference": float(rlib.mean()),
            "gene_detection_rate_mean_reference": float(rdetect.mean()),
            "gene_detection_rate_max_abs_error": float(np.abs(detect - rdetect).max()),
            "gene_mean_log1p_corr": float(np.corrcoef(
                np.log1p(a.mean(axis=0)), np.log1p(r.mean(axis=0)))[0, 1]),
        })
    return out
