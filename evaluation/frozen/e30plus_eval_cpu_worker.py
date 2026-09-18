# -*- coding: utf-8 -*-
"""Ticket 124: evaluate one manifest item on frozen non-MMD rulers.

No estimator is reimplemented here.  W1 and sliced-W1 call the exact functions
loaded by the registered W1 scripts; CP10K/SCC/PCC use E21/E21c functions; Fano
uses the current matched-n F8 function; purity uses celltypist_eval.to_adata and
the frozen v7 label sets.  RNG calls before the scored row are replayed so the
selected rows are identical to the registered Table-1 scripts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


BASE = Path(__file__).resolve().parents[3]
HVG = BASE / "HVG2K"
XROOT = BASE.parent / "Experiments" / "X"
DATA = XROOT / "_shared" / "data" / "all"
PLUS = HVG / "universality_fk" / "out" / "e30plus"
# The manifest and output directory can be pointed elsewhere so the
# SAME evaluator scores ticket 125's arms.  That is deliberate: 125's
# arms only mean anything against the control arm, so they have to be
# measured with the same reference rows, projections, bandwidth and
# matched n, which is what reusing this file guarantees.
MANIFEST = Path(os.environ.get("E30PLUS_MANIFEST",
                                str(PLUS / "E30plus_manifest.json")))
OUT = Path(os.environ.get("E30PLUS_METRICS_CPU",
                           str(PLUS / "metrics_cpu")))
METRICS = BASE.parent / "Experiments" / "Toy" / "_shared" / "code" / "metrics.py"
W1_CODE = HVG / "cond_w1" / "code"
E21_CODE = HVG / "code" / "eval"
F8 = HVG / "viz" / "make_F8_dispersion_baselines.py"
V7 = HVG / "results" / "tables" / "master_purity_v7.json"
E21_JSON = HVG / "results" / "diagnostics" / "E21_scc_mmd.json"

TYPES = ("Endothelial", "Myeloid", "Neuronal")
N_MATCH = {"Endothelial": 10057, "Myeloid": 2302, "Neuronal": 396}
W1_PROJ_SEED = 20261070
EVAL_SEED = 20261071
N_PROJ = 512
CEILING = {"Endothelial": 0.9372, "Myeloid": 0.8836, "Neuronal": 0.7045}
E21_ORDER = {"scVI": 0, "CFGen": 1, "scANVI": 2, "OURS": 4}
W1_ARM_COUNT = {"scVI": 2, "scANVI": 2, "CFGen": 2, "OURS": 3}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def choose_indices(rng: np.random.Generator, population: np.ndarray | int,
                   n: int) -> np.ndarray:
    return np.sort(rng.choice(population, n, replace=False))


def w1_rows(target: str, method: str, raw: np.ndarray, xv, yv, xt, yt,
            metric_mod, sliced_fn, projections) -> tuple[dict, dict]:
    """Replay the registered method script through this target and score it."""
    rng = np.random.default_rng(EVAL_SEED)
    arm_count = W1_ARM_COUNT[method]
    ref = floor = selected = None
    selection = {}
    for cell_type in TYPES:
        n = N_MATCH[cell_type]
        vi = np.flatnonzero(yv == cell_type)
        ti = np.flatnonzero(yt == cell_type)
        ri = choose_indices(rng, vi, n)
        fi = choose_indices(rng, ti, n)
        if cell_type == target:
            ref = np.asarray(xv[ri], dtype=np.float64)
            floor = np.asarray(xt[fi], dtype=np.float64)
            selection["reference_indices"] = ri.tolist()
            selection["floor_indices"] = fi.tolist()
        for arm_index in range(arm_count):
            source_n = int(raw.shape[0]) if cell_type == target and arm_index == arm_count - 1 else 20000
            gi = choose_indices(rng, source_n, n)
            if cell_type == target and arm_index == arm_count - 1:
                selected = np.asarray(raw[gi], dtype=np.float64)
                selection["generated_indices"] = gi.tolist()
        if cell_type == target:
            break
    assert ref is not None and floor is not None and selected is not None

    def score(array: np.ndarray) -> dict:
        per_gene = np.array([metric_mod.w1_1d(array[:, gene], ref[:, gene])
                             for gene in range(ref.shape[1])])
        return {
            "W1": float(per_gene.mean()),
            "W1_median": float(np.median(per_gene)),
            "W1_p90": float(np.percentile(per_gene, 90)),
            "sliced_W1": float(sliced_fn(array, ref, projections, metric_mod.w1_1d)),
            "n": int(array.shape[0]),
        }
    return {"generated": score(selected), "floor": score(floor)}, selection


def w1_real_floor(target: str, xv, yv, xt, yt, metric_mod, sliced_fn,
                  projections) -> tuple[dict, dict]:
    """Reproduce cond_w1's Table-1 real-train vs real-val row."""
    rng = np.random.default_rng(EVAL_SEED)
    for cell_type in TYPES:
        n = N_MATCH[cell_type]
        vi = np.flatnonzero(yv == cell_type)
        ti = np.flatnonzero(yt == cell_type)
        ri = choose_indices(rng, vi, n)
        fi = choose_indices(rng, ti, n)
        if cell_type == target:
            ref = np.asarray(xv[ri], dtype=np.float64)
            floor = np.asarray(xt[fi], dtype=np.float64)
            per_gene = np.array([metric_mod.w1_1d(floor[:, gene], ref[:, gene])
                                 for gene in range(ref.shape[1])])
            return {
                "W1": float(per_gene.mean()),
                "W1_median": float(np.median(per_gene)),
                "W1_p90": float(np.percentile(per_gene, 90)),
                "sliced_W1": float(sliced_fn(floor, ref, projections,
                                              metric_mod.w1_1d)),
                "n": int(n),
            }, {"reference_indices": ri.tolist(), "floor_indices": fi.tolist()}
        # cond_w1 consumes unconditional, product-tilt and OURS draws before
        # moving to the next type.  All three registered pools have 20000 rows.
        for _ in range(3):
            choose_indices(rng, 20000, n)
    raise AssertionError(target)


def e21_selected(target: str, method: str, raw: np.ndarray, xv, yv, xt, yt):
    """Replay E21's per-type RNG order and return its exact scored rows."""
    n = N_MATCH[target]
    rng = np.random.default_rng(EVAL_SEED)
    vi = np.flatnonzero(yv == target)
    ti = np.flatnonzero(yt == target)
    ref_raw = np.asarray(xv[vi])
    floor_i = choose_indices(rng, ti, n)
    floor_raw = np.asarray(xt[floor_i])
    if method == "REAL":
        return ref_raw, floor_raw, {"reference_indices": vi.tolist(),
                                    "floor_indices": floor_i.tolist()}
    pos = E21_ORDER[method]
    selected_i = None
    for slot in range(pos + 1):
        source_n = int(raw.shape[0]) if slot == pos else 20000
        selected_i = choose_indices(rng, source_n, n)
    assert selected_i is not None
    return ref_raw, np.asarray(raw[selected_i]), {
        "reference_indices": vi.tolist(), "generated_indices": selected_i.tolist()}


def correlations(ref_raw, other_raw, e21_mod, f8_mod, fano_raw=None) -> dict:
    ref_np, ref_zero = e21_mod.cp10k_log1p(ref_raw)
    other_np, other_zero = e21_mod.cp10k_log1p(other_raw)
    real_mean = ref_np.mean(0, dtype=np.float64)
    other_mean = other_np.mean(0, dtype=np.float64)
    scc = float(spearmanr(real_mean, other_mean).statistic)
    pcc = float(pearsonr(real_mean, other_mean).statistic)
    if not math.isfinite(scc) or not math.isfinite(pcc):
        raise SystemExit(f"nonfinite correlation SCC={scc}, PCC={pcc}")
    fano = f8_mod.dispersion(other_raw if fano_raw is None else fano_raw)[0]
    return {"SCC": scc, "PCC": pcc, "median_Fano": fano["fano_median"],
            "fano_q25": fano["fano_q25"], "fano_q75": fano["fano_q75"],
            "zero_library_rows_reference": ref_zero,
            "zero_library_rows_other": other_zero}


def label_sets(target: str) -> dict:
    """The three canonical label sets, read out of the FROZEN v7 table.

    `master_purity_v7.json` is read as data and never recomputed
    (`results/tables/ceiling_canonical.md`): its `fixed_labels` is the set that
    was strong in >= 50% of the calls that scored this type, and
    `borderline_labels` records what fell below that bar.

    Ticket 124 s9.2 asks for `purity` and `purity_strict` side by side.  On the
    CANONICAL scale those two collapse, and it is worth saying why rather than
    printing the same column twice: v7's fixed set is ALREADY the weak-label-
    excluded set, so `fixed \\ borderline == fixed` for all three types (the one
    borderline entry that exists, Myeloid's `DC` with support 4, is not in the
    fixed set to begin with).  The genuinely different second column on this
    scale is the INCLUSIVE one, so all three are emitted and named for what
    they are.  The 0.9412 / 0.9292 pair quoted in the ticket comes from a
    single `celltypist_eval.py` call's own per-call label sets, which is the
    per-call scale that ceiling_canonical.md replaced.
    """
    v7 = json.loads(V7.read_text(encoding="utf-8"))[ "types" ][target]
    fixed = set(v7["fixed_labels"])
    borderline = {b[0] if isinstance(b, (list, tuple)) else b
                  for b in v7.get("borderline_labels", [])}
    return {"fixed": fixed, "strict": fixed - borderline,
            "inclusive": fixed | borderline, "borderline": borderline}


def purity_from_dist(dist: dict, target: str) -> dict:
    """Purity from a label histogram, so it does not depend on RNG order."""
    sets = label_sets(target)
    total = int(sum(dist.values()))
    def frac(keys):
        return sum(v for k, v in dist.items() if k in keys) / total
    value = frac(sets["fixed"])
    ceiling = CEILING[target]
    over = value / ceiling
    out = {
        "purity": value,
        "purity_strict": frac(sets["strict"]),
        "purity_inclusive": frac(sets["inclusive"]),
        # s9.1: the ceiling is its own reported field, not just a divisor
        "purity_ceiling": ceiling,
        "purity_over_ceiling": over,
        "purity_n": total,
        # s9.1: the identity is checked in-process and the residual reported
        "purity_identity_residual": abs(over - value / ceiling),
        "purity_scale": "frozen master_purity_v7 fixed label set + "
                        "ceiling_canonical ceiling; computed from the label "
                        "histogram, so independent of how many arms a "
                        "CellTypist call scored",
        "label_sets": {k: sorted(v) for k, v in sets.items()},
        "label_dist": dist,
    }
    return out


def purity(raw: np.ndarray, target: str, ct_mod) -> dict:
    import celltypist
    from celltypist import models
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    genes = meta["genes"]
    if isinstance(genes, str):
        genes = eval(genes)
    model = "Healthy_Adult_Heart.pkl"
    models.download_models(model=[model], force_update=False)
    result = celltypist.annotate(ct_mod.to_adata(raw, genes), model=model,
                                majority_voting=False)
    labels = result.predicted_labels["predicted_labels"].to_numpy().astype(str)
    names, counts = np.unique(labels, return_counts=True)
    dist = {str(name): int(count) for name, count in zip(names, counts)}
    return purity_from_dist(dist, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    hits = [row for row in manifest["items"] if row["key"] == args.key]
    if len(hits) != 1:
        raise SystemExit(f"manifest key {args.key!r} matched {len(hits)} rows")
    item = hits[0]
    output = OUT / f"{args.key}.json"
    if output.exists():
        print(f"[skip] {output}")
        return

    sys.path.insert(0, str(W1_CODE))
    from sliced_w1 import sliced_w1
    metric_mod = load_module("_ticket124_toy_metrics", METRICS)
    e21_mod = load_module("_ticket124_e21", E21_CODE / "e21_scc_mmd.py")
    f8_mod = load_module("_ticket124_f8", F8)
    ct_mod = load_module("_ticket124_ct", E21_CODE / "celltypist_eval.py")

    xv = np.load(DATA / "val.npy", mmap_mode="r")
    yv = np.load(DATA / "val_celltype.npy", allow_pickle=True).astype(str)
    xt = np.load(DATA / "train.npy", mmap_mode="r")
    yt = np.load(DATA / "train_celltype.npy", allow_pickle=True).astype(str)
    projections = metric_mod.make_projections(xv.shape[1], n_proj=N_PROJ,
                                               seed=W1_PROJ_SEED)
    started = time.time()
    target, method = item["target"], item["method"]
    # `ruler_slot` decides WHICH registered script's RNG stream is replayed and
    # which slot in it this array is scored at; `method` is only the label.
    # Ticket 125's arms are new, so no frozen script has a slot of their own --
    # they are scored at the OURS slot, which means the same reference rows,
    # the same projection set, the same matched n and the same bandwidth as the
    # control arm they exist to be compared against.  Only the array changes.
    slot = item.get("ruler_slot", method)
    if method == "REAL":
        vi = np.flatnonzero(yv == target)
        raw_for_purity = np.asarray(xv[vi])
        # The reference row's purity IS the ceiling: `ceiling_canonical.md`
        # freezes it and forbids recomputation, because re-running CellTypist
        # over a different set of arms moves it (0.9412-0.9448 measured for
        # Endothelial).  So the frozen value is what the row reports.  The
        # same held-out cells are ALSO scored here as a drift diagnostic, and
        # it is stored beside the frozen number, never used as a divisor.
        real_pur = purity(raw_for_purity, target, ct_mod)
        real_pur["purity_measured_this_run_DIAGNOSTIC_ONLY"] = real_pur["purity"]
        real_pur["purity"] = CEILING[target]
        real_pur["purity_over_ceiling"] = 1.0
        real_pur["purity_identity_residual"] = abs(
            1.0 - CEILING[target] / CEILING[target])
        real_pur["purity_drift_vs_frozen_ceiling"] = (
            real_pur["purity_measured_this_run_DIAGNOSTIC_ONLY"] - CEILING[target])
        ref_raw, other_raw, e21_sel = e21_selected(target, slot, raw_for_purity,
                                                   xv, yv, xt, yt)
        real_w1, w1_sel = w1_real_floor(target, xv, yv, xt, yt, metric_mod,
                                        sliced_w1, projections)
        corr = correlations(ref_raw, other_raw, e21_mod, f8_mod,
                            fano_raw=ref_raw)
        diversity = {
            "unique_cell_fraction": float(np.unique(raw_for_purity, axis=0).shape[0]
                                          / raw_for_purity.shape[0]),
            "unique_lineage_fraction": None,
            "n_intermediate_resamples": None,
            "ess_min_fraction": None,
            "source": "real held-out rows; lineage/SMC diagnostics not defined",
        }
        result = {
            "ticket": item.get("ticket", 124), "key": args.key, "method": method,
            "ruler_slot": slot,
            "target": target, "seed": item["seed"], "source": item,
            "source_sha256": None,
            "rulers": {
                "W1": "cond_w1 Table-1 real-train vs real-val floor",
                "SCC_PCC": "E21/E21c real-train vs real-val floor",
                "Fano": "make_F8_dispersion_baselines.py full held-out real rows",
                "purity": "frozen canonical CellTypist ceiling",
            },
            "metrics": {**real_pur, **real_w1, **corr, **diversity},
            "selections": {"W1": w1_sel, "E21": e21_sel},
            "walltime_s": time.time() - started,
        }
        OUT.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[out] {output}", flush=True)
    else:
        raw = np.load(Path(item["npy"]), mmap_mode="r")
        w1, w1_sel = w1_rows(target, slot, raw, xv, yv, xt, yt,
                             metric_mod, sliced_w1, projections)
        ref_raw, other_raw, e21_sel = e21_selected(target, slot, raw,
                                                   xv, yv, xt, yt)
        corr = correlations(ref_raw, other_raw, e21_mod, f8_mod)
        pur = purity(np.asarray(raw), target, ct_mod)
        if item.get("run_json"):
            run = json.loads(Path(item["run_json"]).read_text(encoding="utf-8"))
            diversity = {
                "unique_cell_fraction": run["unique_cell_fraction"],
                "unique_lineage_fraction": run["unique_lineage_fraction"],
                "n_intermediate_resamples": run["n_intermediate_resamples"],
                "ess_min_fraction": run["ess_min_fraction"],
                "source": "run_json",
            }
        else:
            diversity = {
                "unique_cell_fraction": float(np.unique(np.asarray(raw), axis=0).shape[0]
                                              / raw.shape[0]),
                "unique_lineage_fraction": None,
                "n_intermediate_resamples": None,
                "ess_min_fraction": None,
                "source": "np.unique on generated rows; lineage/SMC diagnostics not defined for this baseline",
            }
        result = {
            "ticket": item.get("ticket", 124), "key": args.key, "method": method,
            "ruler_slot": slot,
            "target": target, "seed": item["seed"], "source": item,
            "source_sha256": sha256_file(Path(item["npy"])),
            "rulers": {
                "W1": "registered method-specific W1 script; projection seed 20261070, 512 projections, subsample seed 20261071",
                "SCC_PCC": "e21_scc_mmd.py + e21c_pcc.py CP10K-log1p population means",
                "Fano": "make_F8_dispersion_baselines.py matched-n E21 cell set",
                "purity": "celltypist_eval.to_adata + frozen master_purity_v7 labels and canonical ceiling",
            },
            "metrics": {**pur, **w1["generated"], **corr, **diversity},
            "selections": {"W1": w1_sel, "E21": e21_sel},
            "walltime_s": time.time() - started,
        }
        OUT.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[out] {output}", flush=True)


if __name__ == "__main__":
    main()
