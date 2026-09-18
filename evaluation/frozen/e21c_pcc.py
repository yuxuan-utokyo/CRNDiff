# -*- coding: utf-8 -*-
"""E21c: PCC on the frozen E21 population-mean vectors.

This evaluator deliberately reuses the E21 transformation, source discovery,
matched-n selection, seed, and method order.  It recomputes all 18 SCC values
in the same process and requires IEEE-754 byte equality with the frozen E21
authority before writing any PCC output.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import struct
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import pearsonr, spearmanr

import e21_scc_mmd as e21


HVG = Path(__file__).resolve().parents[2]
BASE = HVG.parent
FROZEN = HVG / "results" / "diagnostics" / "E21_scc_mmd.json"
OUT = HVG / "results" / "diagnostics" / "E21c_pcc.json"
RECEIPT = BASE / "Paper" / "adversarial" / "comms" / "cc_to_cloud" / "052_E21c_pcc_delivered.md"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def float_bytes(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def corr_pair(real_mean: np.ndarray, other_mean: np.ndarray) -> tuple[float, float]:
    scc = float(spearmanr(real_mean, other_mean).statistic)
    pcc = float(pearsonr(real_mean, other_mean).statistic)
    if not math.isfinite(scc) or not math.isfinite(pcc):
        raise SystemExit(f"non-finite correlation: SCC={scc}, PCC={pcc}")
    return scc, pcc


def source_record(path: Path, cache: dict[str, dict]) -> dict:
    resolved = str(path.resolve())
    if resolved not in cache:
        print(f"[md5] {resolved}", flush=True)
        cache[resolved] = {
            "path": resolved,
            "md5": md5(path),
            "bytes": path.stat().st_size,
        }
    return dict(cache[resolved])


def write_failure_receipt(script_hash: str, frozen_hash: str, gate: dict) -> None:
    lines = [
        "# 052 — E21c PCC gate failure (2026-08-25)",
        "",
        "## Status",
        "",
        "**BLOCKED** — the frozen-SCC self-check did not reproduce all 18 values byte-for-byte.",
        "Per ticket 100, execution stopped and no `E21c_pcc.json` was written.",
        "",
        f"- evaluator: `HVG2K/code/eval/e21c_pcc.py`",
        f"- evaluator md5: `{script_hash}`",
        f"- frozen authority: `HVG2K/results/diagnostics/E21_scc_mmd.json`",
        f"- frozen authority md5: `{frozen_hash}`",
        f"- exact SCC matches: `{gate['matched_count']}/{gate['expected_count']}`",
        "",
        "## Mismatches",
        "",
    ]
    for item in gate["mismatches"]:
        lines.append(
            f"- {item['type']} / {item['row']}: expected `{item['expected']!r}` "
            f"(`{item['expected_ieee754']}`), observed `{item['observed']!r}` "
            f"(`{item['observed_ieee754']}`)"
        )
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_success_receipt(data: dict, script_hash: str, json_hash: str) -> None:
    gate = data["self_check_gate"]
    lines = [
        "# 052 — E21c PCC delivered (2026-08-25)",
        "",
        "## Status",
        "",
        "**PASS** — PCC was computed on the frozen E21 population-mean vectors. "
        "All 18 SCC self-check values matched the frozen authority byte-for-byte.",
        "",
        "- evaluator: `HVG2K/code/eval/e21c_pcc.py`",
        f"- evaluator md5: `{script_hash}`",
        "- authoritative output: `HVG2K/results/diagnostics/E21c_pcc.json`",
        f"- authoritative output md5: `{json_hash}`",
        "- frozen SCC/MMD authority: `HVG2K/results/diagnostics/E21_scc_mmd.json`",
        f"- frozen SCC/MMD authority md5: `{gate['frozen_md5']}`",
        f"- seed: `{data['seed']}`",
        "- matched n: `10057 / 2302 / 396` (Endothelial / Myeloid / Neuronal)",
        f"- SCC gate: `{gate['matched_count']}/{gate['expected_count']}` exact IEEE-754 matches; "
        f"mismatches: `{len(gate['mismatches'])}`",
        "",
        "## PCC (all 2000 genes)",
        "",
        "| type | row | PCC |",
        "|---|---|---:|",
    ]
    for cell_type, type_data in data["types"].items():
        for row, result in type_data["rows"].items():
            lines.append(f"| {cell_type} | {row} | {result['pcc']:.10f} |")
    lines.extend([
        "",
        "## Diagnostic subset",
        "",
        "The JSON `diagnostics` section contains the symmetric 5 methods × 3 types "
        "SCC/PCC calculation restricted to genes whose real validation mean is "
        "strictly above that type's median. These values are diagnostic only and "
        "do not replace the all-2000-gene SCC/PCC table values.",
        "",
        "## Frozen-metric integrity",
        "",
        "`E21_scc_mmd.json` was read only. No SCC, MMD, bandwidth, or TeX value was modified.",
        "Ticket 100 contained no prefilled PCC values, so there is no prefilled-number discrepancy to report.",
    ])
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite authoritative output: {OUT}")
    if RECEIPT.exists():
        raise SystemExit(f"refusing to overwrite receipt: {RECEIPT}")
    if not FROZEN.exists():
        raise SystemExit(f"missing frozen E21 authority: {FROZEN}")

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen.get("seed") != e21.SEED:
        raise SystemExit(f"seed mismatch: frozen={frozen.get('seed')} code={e21.SEED}")

    script_hash = md5(Path(__file__))
    frozen_hash = md5(FROZEN)
    Xv = np.load(e21.DATA / "val.npy", mmap_mode="r")
    yv = np.load(e21.DATA / "val_celltype.npy", allow_pickle=True).astype(str)
    Xt = np.load(e21.DATA / "train.npy", mmap_mode="r")
    yt = np.load(e21.DATA / "train_celltype.npy", allow_pickle=True).astype(str)

    source_cache: dict[str, dict] = {}
    output_types: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}
    comparisons: list[dict] = []

    for cell_type, n in e21.TYPES.items():
        print(f"\n=== {cell_type}: matched n={n} ===", flush=True)
        vi = np.flatnonzero(yv == cell_type)
        ti = np.flatnonzero(yt == cell_type)
        if len(vi) != n:
            raise SystemExit(f"{cell_type}: expected {n} validation cells, found {len(vi)}")

        rng = np.random.default_rng(e21.SEED)
        ref_raw = np.asarray(Xv[vi])
        floor_raw = e21.select_rows(Xt[ti], n, rng)
        ref_np, ref_zero = e21.cp10k_log1p(ref_raw)
        real_mean = ref_np.mean(0, dtype=np.float64)
        real_median = float(np.median(real_mean))
        high_mask = real_mean > real_median
        n_high = int(np.count_nonzero(high_mask))
        if n_high == 0:
            raise SystemExit(f"{cell_type}: empty diagnostic gene subset")

        ref_source = source_record(e21.DATA / "val.npy", source_cache)
        floor_source = source_record(e21.DATA / "train.npy", source_cache)
        rows: dict[str, dict] = {}
        diagnostic_rows: dict[str, dict] = {}

        floor_np, floor_zero = e21.cp10k_log1p(floor_raw)
        floor_mean = floor_np.mean(0, dtype=np.float64)
        floor_scc, floor_pcc = corr_pair(real_mean, floor_mean)
        floor_name = "floor (real train vs real val)"
        expected_floor = float(frozen["types"][cell_type]["rows"][floor_name]["scc"])
        comparisons.append({
            "type": cell_type,
            "row": floor_name,
            "expected": expected_floor,
            "observed": floor_scc,
            "expected_ieee754": float_bytes(expected_floor),
            "observed_ieee754": float_bytes(floor_scc),
            "exact": float_bytes(expected_floor) == float_bytes(floor_scc),
        })
        rows[floor_name] = {
            "pcc": floor_pcc,
            "n": n,
            "n_genes": int(real_mean.size),
            "zero_library_rows": floor_zero,
            "source": floor_source,
            "source_selection": f"cell_type={cell_type}, without replacement seed {e21.SEED}",
        }
        del floor_np, floor_mean

        sources = e21.method_sources(cell_type)
        for method, path in sources.items():
            if not path.exists():
                raise SystemExit(f"missing {method} source: {path}")
            arr = np.load(path, mmap_mode="r")
            raw = e21.select_rows(arr, n, rng)
            transformed, zero = e21.cp10k_log1p(raw)
            other_mean = transformed.mean(0, dtype=np.float64)
            scc, pcc = corr_pair(real_mean, other_mean)
            expected = float(frozen["types"][cell_type]["rows"][method]["scc"])
            comparisons.append({
                "type": cell_type,
                "row": method,
                "expected": expected,
                "observed": scc,
                "expected_ieee754": float_bytes(expected),
                "observed_ieee754": float_bytes(scc),
                "exact": float_bytes(expected) == float_bytes(scc),
            })
            src = source_record(path, source_cache)
            rows[method] = {
                "pcc": pcc,
                "n": n,
                "n_source": int(arr.shape[0]),
                "n_genes": int(real_mean.size),
                "zero_library_rows": zero,
                "source": src,
            }
            diag_scc, diag_pcc = corr_pair(real_mean[high_mask], other_mean[high_mask])
            diagnostic_rows[method] = {
                "scc": diag_scc,
                "pcc": diag_pcc,
                "n_genes": n_high,
                "source": src,
            }
            print(f"[{method}] SCC={scc:.17g} PCC={pcc:.10f} diag_n={n_high}", flush=True)
            del transformed, other_mean

        output_types[cell_type] = {
            "n": n,
            "n_val": int(len(vi)),
            "n_train": int(len(ti)),
            "reference": ref_source,
            "reference_zero_library_rows": ref_zero,
            "rows": rows,
        }
        diagnostics[cell_type] = {
            "selection": "real_validation_population_mean > median(real_validation_population_mean)",
            "real_mean_median": real_median,
            "n_genes_selected": n_high,
            "n_genes_total": int(real_mean.size),
            "rows": diagnostic_rows,
        }
        del ref_np, real_mean, ref_raw, floor_raw

    mismatches = [x for x in comparisons if not x["exact"]]
    gate = {
        "comparison": "IEEE-754 binary64 byte equality",
        "frozen_path": str(FROZEN.relative_to(BASE)),
        "frozen_md5": frozen_hash,
        "expected_count": 18,
        "matched_count": len(comparisons) - len(mismatches),
        "all_exact": len(comparisons) == 18 and not mismatches,
        "mismatches": mismatches,
        "comparisons": comparisons,
    }
    if not gate["all_exact"]:
        write_failure_receipt(script_hash, frozen_hash, gate)
        print(f"[STOP] SCC gate failed; receipt: {RECEIPT}", flush=True)
        raise SystemExit(2)

    out = {
        "ticket": "100 E21c",
        "status": "PASS",
        "definitions": {
            "representation": "library-size normalize to 10000, then log1p",
            "population_vector": "mean expression per gene across the matched cells",
            "pcc": "standard Pearson correlation of the two 2000-gene population mean vectors",
            "scc_gate": "Spearman correlation of the same two 2000-gene population mean vectors",
            "diagnostic_subset": "genes with real validation population mean strictly above the within-type median",
        },
        "seed": e21.SEED,
        "target_sum": e21.TARGET_SUM,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "self_check_gate": gate,
        "types": output_types,
        "diagnostics": diagnostics,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    json_hash = md5(OUT)
    delivered = json.loads(OUT.read_text(encoding="utf-8"))
    write_success_receipt(delivered, script_hash, json_hash)
    print(f"\n[out] {OUT}", flush=True)
    print(f"[md5] {json_hash}", flush=True)
    print(f"[receipt] {RECEIPT}", flush=True)


if __name__ == "__main__":
    main()
