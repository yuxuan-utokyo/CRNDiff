# -*- coding: utf-8 -*-
"""Differential-expression row for CFGen alone, as a single-arm wrapper.

The original multi-seed wrapper is not edited. Three module constants are overwritten after
import: the arm list becomes CFGen only, that arm's pool is repointed in memory (no manifest
copy is written), and the output directory is redirected so the archived artefact is not
touched. No statistics are copied: `block_b`, `logfc`, `top_overlap` and `load_manifest` all
come from the unmodified archived module, whose sha256 is checked before and after, and the
seed loop and the K sweep reuse the multi-seed wrapper.

    python -m evaluation.downstream_cfgen_only
"""
from __future__ import annotations

import os                                                            # noqa: E402

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"

import argparse                                                      # noqa: E402
import json                                                          # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from evaluation import downstream_multiseed as MS                             # noqa: E402
from evaluation import downstream_deployed_arm as DA                          # noqa: E402

WORK = C.OUT / "e33_212"
OUT_JSON = WORK / "e33_cfgen_only.json"
POOL_SEED = 20260931
N_OUT = {"Endothelial": 10057, "Myeloid": 2500, "Neuronal": 2500}


def new_pools() -> dict:
    out = {}
    for t, n in N_OUT.items():
        p = C.OUT / "cond_pools" / f"cfgen_cond_{t}_seed{POOL_SEED}_n{n}.npy"
        if not p.exists():
            raise SystemExit(f"missing CFGen pool: {p}")
        out[t] = p
    return out


def low_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return "BELOW_NORMAL_PRIORITY_CLASS"
    except Exception as e:                                           # noqa: BLE001
        return f"not set ({type(e).__name__})"


def repoint(by: dict, arrays: dict) -> tuple[dict, dict]:
    """Repoint the CFGen entry at the new pool, the same way the deployed-arm swap does."""
    out = {k: list(v) for k, v in by.items()}
    rec = {}
    for t, q in arrays.items():
        q = Path(q)
        was = out.get(("CFGen", t), [{}])[0].get("npy", "<none>")
        a = np.load(q, mmap_mode="r")
        out[("CFGen", t)] = [{"method": "CFGen", "target": t, "seed": POOL_SEED,
                              "npy": str(q)}]
        rec[t] = {"was": Path(was).name, "was_path": str(was),
                  "now": q.name, "now_path": str(q),
                  "now_sha256": MS.sha256_file(q),
                  "n_rows": int(a.shape[0]), "n_genes": int(a.shape[1])}
        del a
    return out, rec


def prepare():
    m, prov = MS.load_body(WORK / "body")
    body_sha = MS.sha256_file(MS.SRC)
    arms_was, out_was = list(m.ARMS), str(m.OUT)
    m.ARMS = ["CFGen"]
    print(f"[sha] body {body_sha}", flush=True)
    print(f"[override] ARMS {arms_was} -> {m.ARMS}", flush=True)
    print(f"[override] OUT  {out_was} -> {m.OUT}", flush=True)

    by = DA.manifest_local(m)                   # the archived manifest, unmodified
    by, rec = repoint(by, new_pools())
    for t, r in rec.items():
        print(f"[repoint] {t:12s} {r['was']}\n              -> {r['now']} "
              f"({r['n_rows']} rows, sha {r['now_sha256'][:16]})", flush=True)
    return m, prov, by, rec, body_sha, {"ARMS_was": arms_was, "OUT_was": out_was}


def run_one(seed: int) -> int:
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    prio = low_priority()
    m, prov, by, rec, body_sha, ovr = prepare()

    print(f"\n=== seed {seed} : block_b ===", flush=True)
    bb, tk, fcs, chk = MS.run_block_b_with_k(m, by, seed)

    blob = {"ticket": 212, "arms_mode": "single_arm", "ARMS": list(m.ARMS),
            "pool_seed": POOL_SEED, "cfgen_repoint": rec, "seed": seed,
            "overrides": ovr, "body": prov,
            "wrapper_sha256": MS.sha256_file(Path(__file__)),
            "multiseed_sha256": MS.sha256_file(Path(MS.__file__)),
            "deployed_arm_sha256": MS.sha256_file(Path(DA.__file__)),
            "no_statistics_here": "block_b / logfc / top_overlap / load_manifest come "
                                  "from the untouched body; the K sweep is "
                                  "e33_multiseed.run_block_b_with_k, called not copied",
            "top_overlap_interception": {"seed": seed, **chk},
            "per_seed_block_b": {str(seed): bb},
            "per_seed_topk": {str(seed): tk},
            "real_logfc_stats": fcs,
            "cpu_threads": 2, "priority": prio,
            "walltime_s": time.time() - t0,
            "sha256_after_run": MS.sha256_file(MS.SRC),
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if blob["sha256_after_run"] != body_sha:
        raise SystemExit("body changed during the run -- STOP")
    q = WORK / f"raw_cfgen_seed_{seed}.json"
    q.write_text(json.dumps(blob, ensure_ascii=False, indent=1, default=float) + "\n",
                 encoding="utf-8")
    print(f"\n[out] {q}")
    return 0


def aggregate() -> int:
    parts = {}
    for s in MS.SEEDS:
        q = WORK / f"raw_cfgen_seed_{s}.json"
        if not q.exists():
            raise SystemExit(f"missing {q.name}; run --seed {s} first")
        parts[s] = json.loads(q.read_text(encoding="utf-8"))
    reps = {json.dumps(p["cfgen_repoint"], sort_keys=True) for p in parts.values()}
    if len(reps) != 1:
        raise SystemExit("the per-seed runs disagree on the CFGen re-point -- STOP")
    first = parts[MS.SEEDS[0]]
    blob = {k: first[k] for k in ("ticket", "arms_mode", "ARMS", "pool_seed",
                                  "cfgen_repoint", "overrides", "body",
                                  "wrapper_sha256", "multiseed_sha256",
                                  "deployed_arm_sha256", "no_statistics_here",
                                  "real_logfc_stats", "cpu_threads", "priority",
                                  "sha256_after_run")}
    blob["eval_seeds"] = list(MS.SEEDS)
    blob["K_registered"] = list(MS.KS)
    blob["assembled_from"] = {str(s): {"file": f"raw_cfgen_seed_{s}.json",
                                       "sha256": MS.sha256_file(
                                           WORK / f"raw_cfgen_seed_{s}.json")}
                              for s in MS.SEEDS}
    blob["top_overlap_interception"] = [parts[s]["top_overlap_interception"]
                                        for s in MS.SEEDS]
    for field in ("per_seed_block_b", "per_seed_topk"):
        blob[field] = {str(s): parts[s][field][str(s)] for s in MS.SEEDS}
    blob["walltime_s"] = sum(parts[s]["walltime_s"] for s in MS.SEEDS)
    blob["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    OUT_JSON.write_text(json.dumps(blob, ensure_ascii=False, indent=1, default=float)
                        + "\n", encoding="utf-8")
    print(f"[out] {OUT_JSON}")
    print(f"[sha] body {MS.sha256_file(MS.SRC)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args(argv)
    if a.aggregate:
        return aggregate()
    if a.seed is None or a.seed not in MS.SEEDS:
        raise SystemExit(f"--seed must be one of {MS.SEEDS}")
    return run_one(a.seed)


if __name__ == "__main__":
    raise SystemExit(main())
