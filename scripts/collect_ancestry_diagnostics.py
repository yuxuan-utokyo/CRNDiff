# -*- coding: utf-8 -*-
"""Collect the lineage fields of the deployed arm from the run sidecars. CPU only, read only.

For the deployed arm and its three sampling seeds, four fields are taken: the unique lineage
fraction, the minimum ESS fraction, the pre-resampling ESS at the end of the chain, and the
number of intermediate resampling events. Nothing is resampled, no GPU is touched and no
queue file or lock is opened. Runs with a complete set of fields are rolled up across seeds;
runs missing a field are recorded as missing rather than estimated.

    python scripts/collect_ancestry_diagnostics.py
"""
from __future__ import annotations

import os                                                            # noqa: E402

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json                                                          # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
from collections import defaultdict                                  # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402

FIELDS = ["unique_lineage_fraction", "ess_min_fraction",
          "ess_pre_final_fraction", "n_intermediate_resamples"]
SEEDS = [20260987, 20260988, 20260989]
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
OUT = C.OUT / "summary" / "200_deployed_lineage.json"
MISSING = C.OUT / "summary" / "200_LINEAGE_MISSING.md"


def idle_priority() -> str:
    try:
        import psutil                                                 # noqa: PLC0415
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
        return "IDLE_PRIORITY_CLASS"
    except Exception as e:                                            # noqa: BLE001
        return f"unchanged ({e})"


def main() -> int:
    prio = idle_priority()
    print(f"[prio] {prio}", flush=True)

    rows: list[dict] = []
    for p in sorted((C.RUNS).glob("*/*_a1/*.json")):
        seed = None
        for s in SEEDS:
            if p.stem.endswith(f"seed{s}"):
                seed = s
                break
        if seed is None:
            continue
        t = p.parent.name[:-3]
        if t not in TYPES:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "type": t, "experiment": p.parent.parent.name, "arm": p.parent.name,
            "seed": seed, "stem": p.stem, "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "n_delivered": d.get("n_delivered", d.get("n_out")),
            "samples_sha256": d.get("samples_sha256", d.get("npy_sha256")),
            "tau": d.get("tau"), "J": d.get("J"), "K": d.get("K"),
            "resample_interval": d.get("resample_interval"),
            "ess_threshold_fraction": d.get("ess_threshold_fraction"),
            "resample_trigger": d.get("resample_trigger"),
            **{f: d.get(f) for f in FIELDS},
            "_missing": [f for f in FIELDS if f not in d]})

    if not rows:
        MISSING.parent.mkdir(parents=True, exist_ok=True)
        MISSING.write_text(
            "# 200 M1-8 STEP 1 -- deployed-arm lineage fields: **no run sidecar found**\n\n"
            f"written at {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
            f"no json under `out/runs/*/<type>_a1/` matches the seeds {SEEDS}.\n\n"
            "**By instruction, nothing is resampled. Stopping here.**\n", encoding="utf-8")
        print(f"[STOP] no sidecars -> {MISSING}")
        return 1

    bad = [r for r in rows if r["_missing"]]
    if bad:
        MISSING.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"- `{r['path']}` is missing {r['_missing']}" for r in bad]
        MISSING.write_text(
            "# 200 M1-8 STEP 1 -- deployed-arm lineage fields: **fields missing**\n\n"
            f"written at {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
            f"the four required fields: {FIELDS}\n\n"
            f"{len(rows)} sidecars found, of which **{len(bad)}** are missing fields:\n\n"
            + "\n".join(lines)
            + "\n\nBy instruction, nothing is resampled. Stopping here.\n", encoding="utf-8")
        print(f"[STOP] {len(bad)}/{len(rows)} sidecars missing fields -> {MISSING}")
        return 1

    # deduplicate by sha256: one delivery stored under several experiment directories is one
    by_sha: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sha[r["samples_sha256"] or f"__nosha__{r['path']}"].append(r)
    dups = {k: [x["path"] for x in v] for k, v in by_sha.items() if len(v) > 1}

    # roll up over the three seeds, by request and experiment
    roll = {}
    for t in TYPES:
        for exp in sorted({r["experiment"] for r in rows if r["type"] == t}):
            sel = sorted((r for r in rows if r["type"] == t and r["experiment"] == exp),
                         key=lambda r: r["seed"])
            agg = {}
            for f in FIELDS:
                v = np.array([float(r[f]) for r in sel], dtype=np.float64)
                agg[f] = {"per_seed": {str(r["seed"]): r[f] for r in sel},
                          "mean": float(v.mean()),
                          "sd_ddof1": (float(v.std(ddof=1)) if len(v) > 1 else None),
                          "min": float(v.min()), "max": float(v.max())}
            roll[f"{t}|{exp}"] = {
                "type": t, "experiment": exp, "n_seeds": len(sel),
                "seeds": [r["seed"] for r in sel],
                "three_seed_rollup": bool(sorted(r["seed"] for r in sel) == SEEDS),
                "n_delivered": sorted({r["n_delivered"] for r in sel}),
                "config": {k: sel[0][k] for k in
                           ("tau", "J", "K", "resample_interval",
                            "ess_threshold_fraction", "resample_trigger")},
                "fields": agg}

    blob = {
        "ticket": 200, "block": "M1-8 STEP 1 -- deployed-arm lineage fields",
        "scope": "read-only join of existing run sidecars; no sampling was re-run, "
                 "no GPU touched, no VGR queue file or lockfile touched",
        "arm": "<type>_a1 (deployed arm)", "seeds_requested": SEEDS,
        "fields": FIELDS,
        "n_sidecars_found": len(rows),
        "all_fields_present_in_every_sidecar": True,
        "duplicate_deliveries_by_samples_sha256": {
            "note": "one delivery is stored once under each of several experiment directories. Duplicates "
                    "are detected by sha256 rather than by path string (E42). The roll-up is still "
                    "grouped by experiment, so a duplicated delivery appears once in each group. "
                    "That is deliberate: it is what lets a reader see which two groups are in fact "
                    "the same data.",
            "groups": dups},
        "rows": sorted(rows, key=lambda r: (r["type"], r["experiment"], r["seed"])),
        "rollup": roll,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = OUT.parent / "_superseded"
        old.mkdir(parents=True, exist_ok=True)
        OUT.rename(old / f"200_deployed_lineage_{int(time.time())}.json")
    OUT.write_text(json.dumps(blob, indent=1, ensure_ascii=False), encoding="utf-8")

    for k, v in roll.items():
        f = v["fields"]
        sd = f["unique_lineage_fraction"]["sd_ddof1"]
        sdtxt = f"{sd:.5f}" if sd is not None else "  n/a"
        print(f"[{k:<34}] n={v['n_seeds']}  uniq_lineage={f['unique_lineage_fraction']['mean']:.5f}"
              f" +-{sdtxt}  ess_min={f['ess_min_fraction']['mean']:.5f}"
              f"  ess_pre_final={f['ess_pre_final_fraction']['mean']:.5f}"
              f"  n_resamp={f['n_intermediate_resamples']['mean']:.2f}")
    if dups:
        print(f"\n[dup] {len(dups)} delivery/deliveries appear under more than one experiment "
              f"dir with an identical samples_sha256")
    print(f"\n[out] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
