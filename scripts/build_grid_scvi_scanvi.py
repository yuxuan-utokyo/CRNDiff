# -*- coding: utf-8 -*-
"""Fill the conditional-fidelity table out to a full training-by-sampling grid, serially.

Four steps, unattended:

1. our first training seed already has its nine deliveries on disk, so they are only scored,
   never resampled or retrained, which keeps every cell of the grid on one frozen ruler;
2. the scVI and scANVI arms gain two more sampling runs per training seed. They are only
   REGENERATED, never retrained, since a checkpoint does not depend on the sampling seed;
3. our other two training seeds gain the missing sampling runs;
4. everything is scored by the unmodified scoring driver and assembled into the grid.

    python scripts/build_grid_scvi_scanvi.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402

LEGACY = ROOT / "assets" / "legacy_root"
CODE = LEGACY / "Baseline" / "OURS" / "code"
MODELS = LEGACY / "Baseline" / "OURS" / "models"
LOGS = C.OUT / "ticket214_logs"
SUMMARY = C.OUT / "summary"
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
PY = sys.executable
ENV = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
       "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
       "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

OURS_SAMPS = [20260987, 20260988, 20260989]
OURS_TRAIN = [20260625, 20260626, 20260627]
STEP3_TRAIN = (20260626, 20260627)     # the training seeds of the extra sampling step; overwritten by the caller
FK = {"Endothelial": (0.4, 20114, 10057), "Myeloid": (0.2, 5000, 2500),
      "Neuronal": (0.2, 5000, 2500)}
HVG2K = C.ARCHIVE
BASE = {
    "scvi": {"gen": C.BASELINES / "scvi" / "code" / "generate_scvi.py",
             "trains": [20261050, 20261053, 20261054],
             "samps": [20261051, 20261055, 20261056],
             "seed1_root": HVG2K / "baseline_scvi"},
    "scanvi": {"gen": C.BASELINES / "scanvi" / "code" / "generate_scanvi.py",
               "trains": [20261060, 20261063, 20261064],
               "samps": [20261061, 20261065, 20261066],
               "seed1_root": HVG2K / "baseline_scanvi"},
}


def sha(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, cwd, log, what, retries=1):
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {what} ===", flush=True)
    t0 = time.time()
    with Path(log).open("a", encoding="utf-8") as fh:
        fh.write(f"\n\n##### {what} @ {time.strftime('%H:%M:%S')}\n"
                 f"{' '.join(str(c) for c in cmd)}\n")
        fh.flush()
        rc = subprocess.call([str(c) for c in cmd], cwd=str(cwd), stdout=fh,
                             stderr=subprocess.STDOUT, env=ENV)
    print(f"    rc={rc}  {(time.time()-t0)/60:.1f} min", flush=True)
    if rc != 0:
        if retries > 0:
            print(f"    [RETRY] {what}", flush=True)
            return run(cmd, cwd, log, what + " (retry)", retries - 1)
        raise SystemExit(f"[STOP] {what} rc={rc}; see {log}")


def place(src: Path, exp: str, t: str, train: int, samp: int, meta: dict) -> Path:
    """Lay one delivery out in the shape the scorer addresses, named by both seeds."""
    arm = C.RUNS / exp / t
    arm.mkdir(parents=True, exist_ok=True)
    stem = f"{meta['generator']}_{t}_train{train}_samp{samp}"
    dst = arm / f"{stem}.npy"
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            dst.write_bytes(src.read_bytes())
    (arm / f"{stem}.json").write_text(json.dumps(
        {"ticket": "214-A#4", "experiment": exp, "arm": t, "stem": stem,
         "request": t, "target": t, "train_seed": train, "sample_seed": samp,
         "seed": samp, "source": str(src), "samples_sha256": sha(dst),
         **meta, "written_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return dst


def step1(log):
    """The first seed's nine existing deliveries: laid out and scored only, never resampled."""
    for samp in OURS_SAMPS:
        exp = f"table2_ours_s20260625_samp{samp}"
        n = 0
        for t in TYPES:
            d = C.RUNS / "table1_ours" / f"{t}_a1"
            hits = sorted(d.glob(f"*seed{samp}.npy"))
            if len(hits) != 1:
                raise SystemExit(f"seed1 {t} samp{samp}: found {len(hits)} npy")
            place(hits[0], exp, t, 20260625, samp,
                  {"generator": "ours", "n_delivered": FK[t][2],
                   "note": "pre-existing table1_ours delivery, re-scored only; "
                           "no sampling, no training"})
            n += 1
        if not (SUMMARY / f"214g_ours_s20260625_samp{samp}_scored.json").exists():
            run([PY, "-m", "scripts.score_deliveries", "--experiment", exp,
                 "--out-tag", f"214g_ours_s20260625_samp{samp}"], ROOT, log,
                f"step1 score OURS train20260625 samp{samp} ({n} deliveries)")


def step2(log):
    """Extra baseline sampling: REGENERATED only, never retrained."""
    for meth, sp in BASE.items():
        gm = None
        for train in sp["trains"]:
            # the first training seed also gets its own root here: NOTHING is ever written into the
            # beside it, so those are copied in (the archive is read, never written)
            # archived baseline directories. The generation script needs the model and the training log
            root = C.OUT / "baseline_214" / f"{meth}_s{train}"
            if train == sp["trains"][0]:
                root.mkdir(parents=True, exist_ok=True)
                (root / "logs").mkdir(exist_ok=True)
                (root / "out").mkdir(exist_ok=True)
                src_model = sp["seed1_root"] / "out" / "model"
                if src_model.is_dir() and not (root / "out" / "model").is_dir():
                    shutil.copytree(src_model, root / "out" / "model")
                lg = sp["seed1_root"] / "logs" / f"train_{meth}.json"
                if lg.exists() and not (root / "logs" / lg.name).exists():
                    shutil.copy2(lg, root / "logs" / lg.name)
            for samp in sp["samps"]:
                need = [root / "out" / "counts" /
                        f"{meth}_cond_{t}_n20000_seed{samp}.npy" for t in TYPES]
                if not all(q.exists() for q in need):
                    if gm is None:
                        spec = importlib.util.spec_from_file_location(
                            f"_g_{meth}", sp["gen"])
                        gm = importlib.util.module_from_spec(spec)
                        sys.modules[f"_g_{meth}"] = gm
                        sys.path.insert(0, str(sp["gen"].parent))
                        spec.loader.exec_module(gm)
                    gm.HERE, gm.DATA = root, C.LEGACY_DATA / "all"
                    gm.SAMPLE_SEED = samp
                    print(f"\n=== step2 {meth} train{train} samp{samp} ===", flush=True)
                    t0 = time.time()
                    try:
                        gm.main()
                    except SystemExit as e:
                        # the generation script signals 'refusing to overwrite' with SystemExit, which would kill
                        # the whole driver if main() were called directly. It is recorded here instead, and the
                        # missing-file check below is what reports a genuinely absent input
                        print(f"    [note] generate raised SystemExit: {e}", flush=True)
                    print(f"    {(time.time()-t0)/60:.1f} min", flush=True)
                miss = [str(q) for q in need if not q.exists()]
                if miss:
                    raise SystemExit(f"{meth} train{train} samp{samp} missing: {miss}")
                exp = f"table2_{meth}_s{train}_samp{samp}"
                for t, q in zip(TYPES, need):
                    place(q, exp, t, train, samp,
                          {"generator": meth, "n_delivered": 20000,
                           "lib_seed": 20261052})
                tag = f"214g_{meth}_s{train}_samp{samp}"
                if not (SUMMARY / f"{tag}_scored.json").exists():
                    run([PY, "-m", "scripts.score_deliveries", "--experiment", exp,
                         "--out-tag", tag], ROOT, log,
                        f"step2 score {meth} train{train} samp{samp}")


def step3(log):
    """Extra sampling for our arm: only the final sampling stage is re-run.

        The training seeds are a module constant, with the same default as before, so the caller
        can reuse this code rather than copying it.
    """
    for train in STEP3_TRAIN:
        ck = MODELS / f"pilot_run214_s{train}.pt"
        tiltdir = ROOT / "assets" / "tables" / f"tables_kz_s{train}"
        resroot = ROOT / "assets" / "models" / "residual"
        for samp in (20260988, 20260989):
            exp = f"table2_ours_s{train}_samp{samp}"
            for t in TYPES:
                tau, m, n = FK[t]
                arm = C.RUNS / exp / f"{t}_fk"
                if arm.is_dir() and any(arm.glob("*.npy")):
                    continue
                run([PY, "-m", "experiments.run_one",
                     "--experiment", exp, "--arm", f"{t}_fk", "--request", t,
                     "--seed", str(samp),
                     "--sampler", "twist", "--tilt", "on", "--tilt-mode", "every",
                     "--trigger", "any", "--reward-mode", "odds",
                     "--reward-model", "residual",
                     "--residual-dir", resroot / f"{t}_s{train}",
                     "--ckpt", ck, "--logpi", tiltdir / f"logc_{t}.npy",
                     "--K", "32", "--J", "16", "--alpha", "1.0", "--tau", str(tau),
                     "--n-particles", str(m), "--n-out", str(n),
                     "--cell-chunk", "256", "--ess-frac", "0.5",
                     "--resample-interval", "8", "--min-resample-gap", "2"],
                    ROOT, log, f"step3 OURS FK train{train} samp{samp} {t}")
            # lay it out under the two-seed name, then score
            for t in TYPES:
                hits = sorted((C.RUNS / exp / f"{t}_fk").glob("*.npy"))
                if len(hits) != 1:
                    raise SystemExit(f"{exp}/{t}: {len(hits)} npy")
                place(hits[0], exp, t, train, samp,
                      {"generator": "ours", "n_delivered": FK[t][2]})
            tag = f"214g_ours_s{train}_samp{samp}"
            if not (SUMMARY / f"{tag}_scored.json").exists():
                run([PY, "-m", "scripts.score_deliveries", "--experiment", exp,
                     "--out-tag", tag], ROOT, log,
                    f"step3 score OURS train{train} samp{samp}")


def step4(log):
    """Assemble the grid."""
    grid: dict = {"ticket": "214-A#4", "types": TYPES,
                  "ours": {"train_seeds": OURS_TRAIN, "sample_seeds": OURS_SAMPS,
                           "cells": {}},
                  "baselines": {}}
    for train in OURS_TRAIN:
        for samp in OURS_SAMPS:
            p = SUMMARY / f"214g_ours_s{train}_samp{samp}_scored.json"
            src_note = "214g"
            if not p.exists():
                # some cells were run earlier under the older naming; the data is there, so it is collected
                # rather than re-run
                alt = SUMMARY / f"214_ours_s{train}_scored.json"
                if samp == 20260987 and alt.exists():
                    p, src_note = alt, "pre-existing (2026-09-16 run)"
                else:
                    continue
            d = json.loads(p.read_text(encoding="utf-8"))
            # one delivery was scored twice, through its own directory and through the two-seed hard
            # link; the values are bitwise identical, so the two-seed row is kept and the other dropped
            best: dict = {}
            for r in d["rows"]:
                t = r.get("request") or r.get("target")
                prefer = str(r.get("stem", "")).startswith("ours_")
                if t not in best or prefer:
                    best[t] = r
            for t, r in best.items():
                grid["ours"]["cells"][f"{t}|{train}|{samp}"] = {
                    "train_seed": train, "sample_seed": samp, "request": t,
                    "purity_rel": r.get("purity_over_ceiling"),
                    "purity_abs": r.get("purity"), "W1": r.get("W1"),
                    "mmd2": r.get("mmd2_rbf_biased"), "pcc": r.get("pcc"),
                    "source": src_note}
    for meth, sp in BASE.items():
        grid["baselines"][meth] = {"train_seeds": sp["trains"],
                                   "sample_seeds": sp["samps"], "cells": {}}
        for train in sp["trains"]:
            for samp in sp["samps"]:
                p = SUMMARY / f"214g_{meth}_s{train}_samp{samp}_scored.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text(encoding="utf-8"))
                for r in d["rows"]:
                    t = r.get("request") or r.get("target")
                    grid["baselines"][meth]["cells"][f"{t}|{train}|{samp}"] = {
                        "train_seed": train, "sample_seed": samp, "request": t,
                        "purity_rel": r.get("purity_over_ceiling"),
                        "purity_abs": r.get("purity"), "W1": r.get("W1"),
                        "mmd2": r.get("mmd2_rbf_biased"), "pcc": r.get("pcc")}
    grid["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p = SUMMARY / "214_table2_grid_3x3.json"
    p.write_text(json.dumps(grid, indent=1, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    print(f"\n[out] {p}")
    print(f"  ours cells: {len(grid['ours']['cells'])} / 27 expected")
    for m in grid["baselines"]:
        print(f"  {m} cells: {len(grid['baselines'][m]['cells'])} / 27 expected")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=["1", "2", "3", "4"])
    a = ap.parse_args()
    log = LOGS / "GRID.log"
    t0 = time.time()
    for s in a.steps:
        {"1": step1, "2": step2, "3": step3, "4": step4}[s](log)
    print(f"\n[done] {(time.time()-t0)/3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
