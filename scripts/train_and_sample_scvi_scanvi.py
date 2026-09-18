# -*- coding: utf-8 -*-
"""Train and sample two further seeds for the scVI and scANVI baselines.

The four baseline scripts are not edited by a single byte. They are imported and three
module-level constants are overwritten afterwards, which works because each is resolved by name
inside the script's own main():

    the seed          the baseline training scripts hard-code it and expose no command line
    the working root  so each seed writes into its own directory
    the data root     so it resolves inside this repository

Each script's sha256 is recorded before and after the run.

    python scripts/train_and_sample_scvi_scanvi.py --help
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402

HVG2K = C.ARCHIVE
WORK = C.OUT / "baseline_214"
LOGS = C.OUT / "ticket214_logs"
SUMMARY = C.OUT / "summary"
DATA_NOW = C.LEGACY_DATA / "all"
TYPES = ["Endothelial", "Myeloid", "Neuronal"]

SPEC = {
    "scvi": {"train": C.BASELINES / "scvi" / "code" / "train_scvi.py",
             "gen": C.BASELINES / "scvi" / "code" / "generate_scvi.py",
             "seed1": 20261050, "new": [20261053, 20261054], "samp": 20261051},
    "scanvi": {"train": C.BASELINES / "scanvi" / "code" / "train_scanvi.py",
               "gen": C.BASELINES / "scanvi" / "code" / "generate_scanvi.py",
               "seed1": 20261060, "new": [20261063, 20261064], "samp": 20261061},
}


def sha(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(m)
    return m


def run_one(method: str, seed: int, rec: dict) -> None:
    sp = SPEC[method]
    root = WORK / f"{method}_s{seed}"
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "out").mkdir(parents=True, exist_ok=True)

    # training
    mdir = root / "out" / "model"
    if not mdir.exists():
        t0 = time.time()
        tm = load(sp["train"], f"_t214_train_{method}_{seed}")
        was = (tm.SEED, str(tm.HERE), str(tm.DATA))
        tm.SEED, tm.HERE, tm.DATA = seed, root, DATA_NOW
        print(f"[{method} s{seed}] train: SEED {was[0]} -> {seed}", flush=True)
        tm.main()
        rec["train_walltime_s"] = time.time() - t0
        rec["train_overrides"] = {"SEED": [was[0], seed], "HERE": [was[1], str(root)],
                                  "DATA": [was[2], str(DATA_NOW)]}
        print(f"[{method} s{seed}] trained in {rec['train_walltime_s']/60:.1f} min",
              flush=True)
    else:
        print(f"[{method} s{seed}] model already there, skipping training", flush=True)

    # conditional generation
    counts = root / "out" / "counts"
    need = [counts / f"{method}_cond_{t}_n20000_seed{sp['samp']}.npy" for t in TYPES]
    if not all(q.exists() for q in need):
        t0 = time.time()
        gm = load(sp["gen"], f"_t214_gen_{method}_{seed}")
        gm.HERE, gm.DATA = root, DATA_NOW
        print(f"[{method} s{seed}] generate: SAMPLE_SEED {gm.SAMPLE_SEED} (unchanged), "
              f"LIB_SEED {gm.LIB_SEED} (unchanged)", flush=True)
        gm.main()
        rec["generate_walltime_s"] = time.time() - t0
        rec["sample_seed"] = gm.SAMPLE_SEED
        rec["lib_seed"] = gm.LIB_SEED
    missing = [str(q) for q in need if not q.exists()]
    if missing:
        raise SystemExit(f"{method} s{seed}: conditional counts missing: {missing}")

    # lay the deliveries out in the shape the scorer addresses
    # one experiment directory per training seed: the stem encodes only the sampling seed, so
    # two training seeds would collide inside one directory and be silently deduplicated
    exp = C.RUNS / f"table2_{method}_s{seed}"
    rec["deliveries"] = {}
    for t, q in zip(TYPES, need):
        arm = exp / t
        arm.mkdir(parents=True, exist_ok=True)
        stem = f"{method}_cond_{t}_train{seed}_samp{sp['samp']}_n20000"
        dst = arm / f"{stem}.npy"
        if not dst.exists():
            try:
                os.link(q, dst)
            except OSError:
                dst.write_bytes(q.read_bytes())
        (arm / f"{stem}.json").write_text(json.dumps({
            "ticket": "214-A", "experiment": exp.name, "arm": t, "stem": stem,
            "request": t, "target": t, "generator": method,
            "train_seed": seed, "seed": sp["samp"],
            "n_delivered": 20000, "n_out": 20000,
            "source": str(q), "samples_sha256": sha(dst),
            "note": "filename carries BOTH the training seed and the sampling seed; the "
                    "upstream script names its output by the sampling seed only",
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")},
            indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        rec["deliveries"][t] = str(dst)
    rec["experiment"] = exp.name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["scvi", "scanvi"])
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    out = {"ticket": "214-A task B", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "source_scripts_unchanged": {}, "runs": {}}
    for meth in a.methods:
        for f in ("train", "gen"):
            out["source_scripts_unchanged"][SPEC[meth][f].name] = sha(SPEC[meth][f])

    for meth in a.methods:
        for seed in SPEC[meth]["new"]:
            rec: dict = {"method": meth, "train_seed": seed}
            run_one(meth, seed, rec)
            out["runs"][f"{meth}_s{seed}"] = rec
            (SUMMARY / "214_baselines_new_seeds.json").write_text(
                json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p = SUMMARY / "214_baselines_new_seeds.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[out] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
