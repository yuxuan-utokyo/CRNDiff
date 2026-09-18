# -*- coding: utf-8 -*-
"""Generate the CFGen conditional deliveries for three sampling seeds.

No sampling logic is copied. This calls the very script that produced the published CFGen
delivery, unmodified, and overwrites only its module constants after import: the sampling seed,
the single request type (each type has its own delivered count, so they run one at a time), and
the data root. The script's sha256 is recorded before and after.

    python scripts/sample_baseline_cfgen.py --help
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402

GEN_PY = C.BASELINES / "cfgen" / "code" / "generate_cfgen.py"
ARCHIVE_CFGEN = (C.ARCHIVE /
                 "baseline_cfgen")
STATE = ARCHIVE_CFGEN / "out" / "cfgen_sampling_state_full.pt"
ARCHIVED_META = ARCHIVE_CFGEN / "out" / "cfgen_generate_meta_full.json"
WORK = C.OUT / "cfgen_210"
POOLS = C.OUT / "cond_pools"
SEEDS = [20260931, 20260932, 20260933]
N_OUT = {"Endothelial": 10057, "Myeloid": 2500, "Neuronal": 2500}
TAG = "full"


def sha(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_gen():
    spec = importlib.util.spec_from_file_location("_cfgen_gen", GEN_PY)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def link_state(root: Path) -> Path:
    """Hard-link the archived sampling state into the new root: the same bytes, not copied, not moved."""
    d = root / "out"
    d.mkdir(parents=True, exist_ok=True)
    q = d / STATE.name
    if not q.exists():
        try:
            os.link(STATE, q)
        except OSError:
            shutil.copy2(STATE, q)
    return q


def run_one(G, ctype: str, seed: int, rec: dict) -> dict:
    n = N_OUT[ctype]
    root = WORK / f"{ctype}_s{seed}"
    link_state(root)

    G.CFGEN = root
    G.DATA = C.LEGACY_DATA / "all"
    G.SAMPLE_SEED = seed
    G.TYPES = [ctype]

    t0 = time.time()
    argv = sys.argv
    sys.argv = ["generate_cfgen.py", "--tag", TAG, "--n", str(n)]
    try:
        G.main()
    finally:
        sys.argv = argv

    src = root / "out" / "counts" / f"cfgen_cond_{ctype}_n{n}_seed{seed}.npy"
    if not src.exists():
        raise SystemExit(f"expected output missing: {src}")
    POOLS.mkdir(parents=True, exist_ok=True)
    dst = POOLS / f"cfgen_cond_{ctype}_seed{seed}_n{n}.npy"
    if dst.exists():
        raise SystemExit(f"refusing to overwrite an existing product: {dst}")
    shutil.copy2(src, dst)

    import numpy as np
    a = np.load(dst, mmap_mode="r")
    meta = json.loads((root / "out" / f"cfgen_generate_meta_{TAG}.json").read_text(
        encoding="utf-8"))
    rec[f"{ctype}|{seed}"] = {
        "type": ctype, "sample_seed": seed, "n_out": n,
        "npy": str(dst), "sha256": sha(dst),
        "shape": [int(a.shape[0]), int(a.shape[1])],
        "dtype": str(a.dtype),
        "min": int(a.min()), "max": int(a.max()),
        "any_negative": bool((a[:] < 0).any()),
        "generator_meta": meta["arms"][src.name],
        "library_seed": meta["library_seed"],
        "n_sample_steps": meta["n_sample_steps"],
        "fm_ckpt": meta["fm_ckpt"], "encoder_ckpt": meta["encoder_ckpt"],
        "walltime_s": time.time() - t0}
    del a
    print(f"[done] {ctype:<13} seed {seed}  n {n}  -> {dst.name}  "
          f"({rec[f'{ctype}|{seed}']['walltime_s']:.0f}s)", flush=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    a = ap.parse_args(argv)
    types = a.only or list(N_OUT)
    seeds = a.seeds or SEEDS

    archived_meta_sha = sha(ARCHIVED_META)
    state_sha = sha(STATE)
    print(f"[archive] sampling state sha256 {state_sha[:16]}")
    print(f"[archive] archived meta sha256  {archived_meta_sha[:16]} (must not change)")

    G = load_gen()
    print(f"[script] {GEN_PY.name} sha256 {sha(GEN_PY)[:16]} (UNCHANGED)")
    print(f"[const ] SAMPLE_SEED was {G.SAMPLE_SEED}, LIB_SEED {G.LIB_SEED} (kept), "
          f"N_SAMPLE_STEPS {G.N_SAMPLE_STEPS} (kept), BATCH {G.BATCH} (kept)")

    rec = {}
    p = C.OUT / "summary" / "210_cfgen_pools.json"
    for ctype in types:
        for seed in seeds:
            run_one(G, ctype, seed, rec)
            p.write_text(json.dumps(
                {"ticket": 210, "stage": "Stage 2 -- CFGen conditional, three sampling seeds",
                 "script": str(GEN_PY), "script_sha256": sha(GEN_PY),
                 "overridden_constants": {
                     "SAMPLE_SEED": "20260931 / 20260932 / 20260933 (the ticket's change)",
                     "TYPES": "one type per run (n_out differs per type)",
                     "DATA": f"{C.LEGACY_DATA / 'all'} (the X tree it pointed at was retired)",
                     "CFGEN": f"{WORK} (so the archived cfgen_generate_meta_full.json is "
                              f"never overwritten)"},
                 "unchanged": {"fm_ckpt": "from the archived sampling state",
                               "encoder_ckpt": "from the archived sampling state",
                               "LIB_SEED": G.LIB_SEED, "N_SAMPLE_STEPS": G.N_SAMPLE_STEPS,
                               "BATCH": G.BATCH},
                 "sampling_state_sha256": state_sha,
                 "archived_meta_sha256_before": archived_meta_sha,
                 "pools": rec,
                 "written_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    after = sha(ARCHIVED_META)
    assert after == archived_meta_sha, "the archived CFGen meta was overwritten -- STOP"
    print(f"[archive] archived meta sha256 after: {after[:16]}  UNCHANGED")
    print(f"[out] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
