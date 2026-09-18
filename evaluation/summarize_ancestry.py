# -*- coding: utf-8 -*-
"""Aggregate the per-run ancestry traces into mean and standard deviation by request, arm and step.

Nothing sampled is recomputed here. Every number comes from the recorded ancestry arrays and
their run jsons; this file only takes per-step statistics across the three seeds.

Both normalisations are reported, because they are different quantities:

    * `anc_frac_all_particles`: distinct ancestors among the M live particles, divided by M
    * `anc_frac_delivered`: the same taken over the lineages of the N delivered particles only,
                                   traced back through the delivery index. Its final value must equal the recorded unique
                                   lineage fraction, which is one of the gates.
    * `unique_frac_all_particles`: distinct particle rows divided by M. The population is M, the
                                   live particles, not the delivered N. Before the terminal resampling the M particles are
                                   almost all distinct, so this sits near one; the duplicates in a delivery are introduced by
                                   the final draw with replacement. Do not read it as the delivered uniqueness.

The four gate values measured on each run are written out as well.

    python -m analysis.summarize_ancestry
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff.config import OUT, RUNS                            # noqa: E402
from crndiff.io import sha256_array                             # noqa: E402

SUMMARY = OUT / "summary"
TRACE = RUNS / "ancestry_trace"
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
SEEDS = [20260987, 20260988, 20260989]
# arm -> (archived experiment directory, archived arm directory, the name used in figures/tables)
ARMS = [("a1", "table1_ours", "{t}_a1", "OURS"),
        ("fk_only", "ablation_table10", "{t}_fk_only", "Residual Correction")]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def one_run(new_json: Path, arch_json: Path) -> dict:
    new = json.loads(new_json.read_text(encoding="utf-8"))
    arch = json.loads(arch_json.read_text(encoding="utf-8"))
    stem = new["stem"]
    npz = np.load(new_json.with_name(f"{stem}_ancestry.npz"))
    cells = np.load(new_json.with_suffix(".npy"))
    anc, didx = npz["anc"], npz["delivered_idx"]
    M, N = int(arch["n_particles"]), int(arch["n_out"])

    got = sha256_array(np.asarray(cells, dtype=np.int16))
    lin_last = float(np.unique(anc[-1][didx]).size / N)
    uniq_delivered = float(np.unique(np.ascontiguousarray(cells), axis=0).shape[0] / N)

    gates = {
        "gate1_samples_bitwise": {
            # 182b s1: `samples_sha256` is the ARRAY hash a.tobytes(), not the file hash.
            # Both readings are checked; the file hash is the stricter of the two.
            "archived_array_sha256": arch["samples_sha256"], "recomputed_array_sha256": got,
            "archived_file_sha256": sha256_file(arch_json.with_suffix(".npy")),
            "new_file_sha256": sha256_file(new_json.with_suffix(".npy")),
            "array_hash_identical": got == arch["samples_sha256"],
            "file_bytes_identical": (sha256_file(arch_json.with_suffix(".npy"))
                                     == sha256_file(new_json.with_suffix(".npy"))),
            "passed": (got == arch["samples_sha256"]
                       and sha256_file(arch_json.with_suffix(".npy"))
                       == sha256_file(new_json.with_suffix(".npy")))},
        "gate2_final_ancestors": {
            "archived_unique_lineage_fraction": arch["unique_lineage_fraction"],
            "recomputed_from_trace": lin_last,
            "passed": lin_last == arch["unique_lineage_fraction"]},
        # 182b s3: the ORIGINAL gate 3 (unique_per_step[-1]/M == unique_cell_fraction) was
        # withdrawn -- it compares two different populations and cannot be satisfied when
        # M != N. The replacement counts distinct rows in the delivered array over N.
        "gate3_final_unique": {
            "criterion": "182b s3 (replaces the withdrawn one): distinct rows of the new "
                         ".npy / N == archived unique_cell_fraction",
            "archived_unique_cell_fraction": arch["unique_cell_fraction"],
            "recomputed_from_delivered_cells": uniq_delivered,
            "passed": uniq_delivered == arch["unique_cell_fraction"]},
        "gate4_walltime": {
            "archived_s": arch["walltime_s"], "new_s": new["walltime_s"],
            "ratio": new["walltime_s"] / arch["walltime_s"],
            "passed": new["walltime_s"] < 2.0 * arch["walltime_s"]},
    }
    return {
        "stem": stem, "seed": int(arch["seed"]), "M": M, "N": N,
        "new_json": str(new_json), "archived_json": str(arch_json),
        "npz": str(new_json.with_name(f"{stem}_ancestry.npz")),
        "npz_sha256": sha256_file(new_json.with_name(f"{stem}_ancestry.npz")),
        "resample_steps": arch["resample_steps"],
        "gates": gates,
        "all_gates_passed": all(g["passed"] for g in gates.values()),
        "curves": {
            "step": [int(v) for v in npz["step"]],
            "t": [float(v) for v in npz["t"]],
            "anc_frac_all_particles": [float(np.unique(a).size / M) for a in anc],
            "anc_frac_delivered": [float(np.unique(a[didx]).size / N) for a in anc],
            "unique_frac_all_particles": [float(v / M) for v in npz["unique_per_step"]],
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(SUMMARY / "ancestry_trace.json"))
    a = ap.parse_args(argv)

    cells_out, missing, all_pass = [], [], True
    for arm_key, arch_exp, armpat, label in ARMS:
        for t in TYPES:
            arm = armpat.format(t=t)
            runs = []
            for seed in SEEDS:
                hits = [p for p in sorted((TRACE / arm).glob("*.json"))
                        if p.stem.endswith(f"seed{seed}")]
                arch = [p for p in sorted((RUNS / arch_exp / arm).glob("*.json"))
                        if p.stem.endswith(f"seed{seed}")]
                if len(hits) != 1 or len(arch) != 1:
                    missing.append({"arm": arm, "seed": seed,
                                    "n_new": len(hits), "n_archived": len(arch)})
                    continue
                runs.append(one_run(hits[0], arch[0]))
            if not runs:
                continue
            all_pass &= all(r["all_gates_passed"] for r in runs)
            steps = runs[0]["curves"]["step"]
            for r in runs:
                if r["curves"]["step"] != steps:
                    raise SystemExit(f"{arm}: step grids differ between seeds")
            agg = {"step": steps, "t": runs[0]["curves"]["t"]}
            for m in ("anc_frac_all_particles", "anc_frac_delivered",
                      "unique_frac_all_particles"):
                v = np.asarray([r["curves"][m] for r in runs], dtype=np.float64)
                agg[f"{m}_mean"] = [float(x) for x in v.mean(axis=0)]
                agg[f"{m}_sd"] = [float(x) for x in v.std(axis=0, ddof=1)]
            rs = {tuple(r["resample_steps"]) for r in runs}
            cells_out.append({
                "target": t, "arm": arm_key, "label": label,
                "archived_experiment": arch_exp, "arm_dir": arm,
                "n_seeds": len(runs), "seeds": [r["seed"] for r in runs],
                "M": runs[0]["M"], "N": runs[0]["N"],
                "resample_steps": sorted(rs)[0] if len(rs) == 1 else None,
                "resample_steps_identical_across_seeds": len(rs) == 1,
                "resample_steps_per_seed": [r["resample_steps"] for r in runs],
                "samples_sha256_match": all(
                    r["gates"]["gate1_samples_bitwise"]["passed"] for r in runs),
                "all_gates_passed": all(r["all_gates_passed"] for r in runs),
                "curves": agg,
                "runs": runs,
            })

    blob = {
        "produced_by": "analysis/summarize_ancestry.py",
        "sd": "sample standard deviation over the sampling seeds (ddof=1)",
        "normalisations": {
            "anc_frac_all_particles": "distinct ancestors among the M live particles / M",
            "anc_frac_delivered": "distinct ancestors among the N delivered particles / N "
                                  "(traced back through delivered_idx); its last step equals "
                                  "the archived unique_lineage_fraction -- gate 2",
            "unique_frac_all_particles": "distinct particle rows among the M LIVE PARTICLES / M "
                                         "(population = M, renamed per 182b s3). This is NOT the "
                                         "archived unique_cell_fraction, which counts distinct rows "
                                         "among the N DELIVERED cells; the two differ whenever "
                                         "M != N. Before the terminal resampling the M particles are "
                                         "typically all distinct, so this series sits at ~1.0; the "
                                         "duplicates in the delivery are introduced BY that "
                                         "with-replacement terminal draw.",
        },
        "gates": {
            "definition": "ticket 182 s5",
            "all_runs_passed": bool(all_pass and not missing),
            "n_cells": len(cells_out),
            "n_runs": sum(c["n_seeds"] for c in cells_out),
            "missing": missing},
        "tilt_only_arm_not_run": "it never resamples, so its ancestor fraction is identically "
                                 "1 and the curve is flat (ticket 182 s3)",
        "cells": cells_out,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(blob, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    for c in cells_out:
        cur = c["curves"]
        print(f"  {c['target']:<13} {c['label']:<22} n={c['n_seeds']}  "
              f"anc_delivered {cur['anc_frac_delivered_mean'][0]:.4f} -> "
              f"{cur['anc_frac_delivered_mean'][-1]:.4f}   "
              f"uniqM {cur['unique_frac_all_particles_mean'][-1]:.4f}   "
              f"gates {'PASS' if c['all_gates_passed'] else 'FAIL'}")
    if missing:
        print(f"[warn] missing runs: {missing}")
    print(f"[summary] all gates passed: {blob['gates']['all_runs_passed']}")
    print(f"[summary] -> {p}")
    print(f"[summary] sha256 {sha256_file(p)}")
    return 0 if blob["gates"]["all_runs_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
