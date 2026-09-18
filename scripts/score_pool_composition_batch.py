# -*- coding: utf-8 -*-
"""The composition of every pool, in one pass.

No scorer is written here: each pool goes through the unmodified single-pool script, so this is
the same code path as the individual composition artefacts, and the results are then collected
into one summary json.

    python scripts/score_pool_composition_batch.py
"""
from __future__ import annotations

import os                                                            # noqa: E402

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import json                                                          # noqa: E402
import re                                                            # noqa: E402
import subprocess                                                    # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
from pathlib import Path                                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402

OUT = C.OUT / "summary" / "202_pool_composition_mdlm_blackout.json"
LOG = C.OUT / "ticket201_logs" / "202_POOLCOMP.log"
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
SELECTED_T = {"mdlm": 128, "blackout": 32}


def main() -> int:
    pools = []
    for p in sorted((C.OUT / "uncond_pools").glob("*_uncond_T*_n20000.npy")):
        m = re.match(r"(mdlm|blackout)_uncond_T(\d+)_seed(\d+)_n20000\.npy$", p.name)
        if m:
            pools.append((m.group(1), int(m.group(2)), int(m.group(3)), p))
    if len(pools) != 13:
        raise SystemExit(f"expected 13 pools, found {len(pools)}")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with open(LOG, "w", encoding="utf-8") as lf:
        for gen, T, seed, p in pools:
            tag = f"{gen}T{T}"
            dst = C.OUT / "summary" / f"200_pool_composition_{tag}_seed{seed}.json"
            if dst.exists():
                print(f"[skip] {tag} seed {seed}: already scored", flush=True)
            else:
                print(f"[run ] {tag} seed {seed}", flush=True)
                r = subprocess.run(
                    [sys.executable, "-m", "scripts.score_pool_composition",
                     "--pool", p.as_posix(), "--seed", str(seed), "--generator", tag],
                    cwd=str(ROOT), capture_output=True, text=True,
                    env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
                if r.returncode != 0:
                    raise SystemExit(f"pool_composition failed for {tag} seed {seed}\n"
                                     f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            d = json.loads(dst.read_text(encoding="utf-8"))
            row = {"generator": tag, "method_family": gen, "T_sampling_steps": T,
                   "seed": seed, "pool": d["pool"], "pool_sha256": d["pool_sha256"],
                   "n_pool": d["n_pool"],
                   "selected_for_main_table": bool(T == SELECTED_T[gen]),
                   "arm_role": ("main_table" if T == SELECTED_T[gen]
                                else ("additional_arms_only" if (gen == "blackout" and T == 256)
                                      else "additional_arms")),
                   "source_json": dst.name,
                   "types": {t: {k: d["types"][t].get(k) for k in
                                 ("n_labelled_as_type", "pi_hat_pool", "N", "B1",
                                  "ceil_by_budget")} for t in TYPES},
                   "celltypist": d.get("celltypist")}
            rows.append(row)
            lf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {tag:<14} seed={seed}  "
                     f"n_pool={d['n_pool']}  "
                     + "  ".join(f"pi_{t[:4]}={d['types'][t]['pi_hat_pool']:.6f}"
                                 for t in TYPES) + "\n")
            lf.flush()
            print(f"       " + "  ".join(
                f"{t}: pi={d['types'][t]['pi_hat_pool']:.5f} "
                f"(n={d['types'][t]['n_labelled_as_type']})" for t in TYPES), flush=True)

    blob = {
        "ticket": 202, "block": "Stage 4 -- pool composition pi_hat for the 13 new pools",
        "code_path": "tools/t200_pool_composition.py, unmodified -- the same path that "
                     "produced 200_pool_composition_*.json; this file only aggregates",
        "prereg": str(C.OUT / "queue" / "202_MDLM_BLACKOUT_PREREG.md"),
        "diagnostic_not_ruler": True,
        "selected_T_per_method": SELECTED_T,
        "n_pools": len(rows),
        "ceil_by_budget_caveat":
            "ceil_by_budget comes from the unmodified scorer, which compares b*B1 against the "
            "GLOBAL constant 200_derived_constants.json::N_pool (the 100000-cell pool size), "
            "not against this pool's own n_pool of 20000. For these 13 pools the achievable "
            "budgets are therefore narrower than ceil_by_budget suggests. The computation was "
            "NOT altered; the C-ladder restriction is handled separately by PREREG R3 "
            "(these generators enter only C in {6250, 12500}).",
        "rows": rows,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "walltime_s": time.time() - t0}

    if OUT.exists():
        OUT.rename(OUT.with_name(f"202_pool_composition_mdlm_blackout__preexisting_"
                                 f"{int(time.time())}.json"))
    OUT.write_text(json.dumps(blob, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
