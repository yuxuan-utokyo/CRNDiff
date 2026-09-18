# -*- coding: utf-8 -*-
"""Train one replacement seed for our arm and take it through the whole chain, unattended.

The pre-registration is on disk before training starts and its sha256 is asserted here. Only
this one seed is run, and it enters the table WHATEVER its purity turns out to be. The only
circumstance that moves to the next seed is a divergence abort during training, and that fact
is recorded in the provenance rather than quietly absorbed.

The recipe is not copied from the previous seed: this script CALLS the previous seed's drivers,
so there is no second implementation that could drift.

    python scripts/run_ours_training_seed.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
import scripts.build_grid_scvi_scanvi as G                                          # noqa: E402
from scripts.run_ours_downstream_chain import CODE, ENV, LOGS, MODELS, run  # noqa: E402

PY = sys.executable
PREREG = C.OUT / "summary" / "217_prereg.md"
PREREG_SHA = "82468ac317dff0be04708c837ca35e68a11e818fefd015583140b1633782f2ab"
# Amendment 1: the first two seeds of the chain both diverged, exhausting the original
# fallback chain, which was extended by two more seeds WITHOUT changing the recipe.
AMEND1 = C.OUT / "summary" / "217_AMEND_1.md"
AMEND1_SHA = "d44822a474ae26c2d6c72ace40734a78255a2386f56152669fad519aa1a7c17f"
# Amendment 2: if the third seed fails the rare-request criterion it is replaced once.
AMEND2 = C.OUT / "summary" / "217_AMEND_2.md"
AMEND2_SHA = "7852fd6b81ae65daa6610f9fcc7e1704b1c8c9dba244a7553b3f17fc47d63272"
# Amendment 3: that criterion reads the lineage ruler; the threshold is unchanged.
AMEND3 = C.OUT / "summary" / "217_AMEND_3.md"
AMEND3_SHA = "821ff36e74b18d24960c27a0fe0914de40a8d39447978fbfc363ab9adb5f0f1c"
PRIMARY_SEED = 20260628
# the fallback chain: the first seed to pass the convergence threshold enters the table and
# the chain stops at once; if every seed diverges the run stops and reports rather than
# extending further. The seed argument says where on the chain to start; only seeds after
SEED_CHAIN = [20260628, 20260629, 20260630, 20260631]
EXCLUDE = 20260626
BEST_VAL_MAX = 0.125                 # the convergence threshold
TRAIN_BATCH, TRAIN_STEPS = 384, 20000
TRAIN_NAME = "pilot_run214"
PROV = C.OUT / "summary" / "217_PROVENANCE.json"


def sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def train(seed: int, log: Path) -> tuple[int, dict]:
    """Return the return code and the training json; a specific code means a divergence abort."""
    ck = MODELS / f"{TRAIN_NAME}_s{seed}.pt"
    js = MODELS / f"{TRAIN_NAME}_s{seed}.json"
    if ck.exists() and js.exists():
        print(f"[217] {ck.name} already exists: training is skipped, nothing is retrained", flush=True)
        return 0, json.loads(js.read_text(encoding="utf-8"))
    cmd = [PY, "train_pilot.py", "--seed", str(seed), "--name", TRAIN_NAME,
           "--batch", str(TRAIN_BATCH), "--steps", str(TRAIN_STEPS)]
    print(f"\n=== s{seed} stage0 train ===\n    {' '.join(cmd)}", flush=True)
    t0 = time.time()
    LOGS.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n\n########## s{seed} stage0 train @ {time.strftime('%H:%M:%S')}\n"
                 f"{' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.call(cmd, cwd=str(CODE), stdout=fh,
                             stderr=subprocess.STDOUT, env=ENV)
    print(f"    rc={rc}  {(time.time()-t0)/60:.1f} min", flush=True)
    tj = json.loads(js.read_text(encoding="utf-8")) if js.exists() else {}
    return rc, tj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=PRIMARY_SEED)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--promotion-json", default=None,
                    help="when training was interrupted and an intermediate checkpoint is promoted "
                         "instead, point at the promotion record; its contents are embedded "
                         "verbatim in PROVENANCE")
    a = ap.parse_args()

    for p, want, nm in ((PREREG, PREREG_SHA, "pre-registration 1"),
                        (AMEND2, AMEND2_SHA, "amendment 2"), (AMEND3, AMEND3_SHA, "amendment 3")):
        if not p.exists() or sha256(p) != want:
            print(f"{nm} {p.name} is missing or has been modified -> stop. A criterion must be "
                  f"frozen before the numbers exist.")
            return 2
        print(f"[217] {nm} {p.name} sha256 {want[:16]}… OK")

    t_all = time.time()
    prov: dict = {"ticket": 217, "prereg": PREREG.name, "prereg_sha256": PREREG_SHA,
                  "primary_seed": PRIMARY_SEED, "seed_chain": SEED_CHAIN,
                  "amendments": [{"file": AMEND1.name, "sha256": AMEND1_SHA,
                                  "what": "the fallback chain is extended, with the recipe unchanged"},
                                 {"file": AMEND2.name, "sha256": AMEND2_SHA,
                                  "what": "630 the third seed is replaced once if it fails the rare-request criterion 631"},
                                 {"file": AMEND3.name, "sha256": AMEND3_SHA,
                                  "what": "AMEND-2 that criterion reads the lineage ruler; the threshold is unchanged"}],
                  "diverged_before_this_run": [20260628, 20260629],
                  "excluded_train_seed": EXCLUDE,
                  "exclusion_reason":
                      ("By the supervisor's decision, the OURS row of Table 2 no longer uses training "
                       "seed 20260626. That seed trained successfully (best_val 0.11207, no "
                       "divergence). The defensible grounds for excluding it are out-of-lineage "
                       "contamination (14-17% of its neuronal deliveries fall outside the NC "
                       "lineage, against 1.4-1.8% for 20260625 and 2.8-4.2% for 20260627) "
                       "together with worse W1, MMD^2 and PCC. It must NOT be described as "
                       "collapsing onto NC2: under the lineage ruler the paper uses, NC2 is a "
                       "hit and its neuronal rel_v8 is 0.848-0.876 (AMEND-3 4a). Its artefacts "
                       "are excluded only at aggregation time, through "
                       "exclude_train_seeds=[20260626]."),
                  "recipe_source":
                      ("stages 1-5 (samp 20260987) and 6 call scripts.run_ours_downstream_chain "
                       "directly; stage 5b (samp 20260988/20260989) calls "
                       "scripts.build_grid_scvi_scanvi.step3 with STEP3_TRAIN overwritten. The "
                       "recipe is not copied, it is the same code reused."),
                  "stage0_recipe": {"name": TRAIN_NAME, "batch": TRAIN_BATCH,
                                    "steps": TRAIN_STEPS,
                                    "source": "read from pilot_run214_s20260626/27.json"},
                  "convergence_gate": {"best_val_max": BEST_VAL_MAX,
                                       "source": "214-A, inherited; nothing new is introduced here"},
                  "stages": [], "written_at": None}

    def save():
        prov["walltime_s"] = round(time.time() - t_all, 1)
        prov["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        PROV.write_text(json.dumps(prov, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    # stage 0: training
    seed = a.seed
    log = LOGS / f"T217_s{seed}.log"
    if not a.skip_train:
        rc, tj = train(seed, log)
        prov["stages"].append({"stage": "0_train", "seed": seed, "rc": rc,
                               "best_val": tj.get("best_val"),
                               "stopped_at": tj.get("stopped_at"),
                               "walltime_s": tj.get("walltime_s")})
        # on a divergence abort the chain moves to the next seed; the first to pass the threshold
        # enters the table and the chain stops; if all diverge the run stops and reports
        if seed not in SEED_CHAIN:
            print(f"{seed} is not on the pre-registered fallback chain: stopping")
            return 2
        chain = SEED_CHAIN[SEED_CHAIN.index(seed) + 1:]
        while rc == 3:
            prov.setdefault("breaker_fired", []).append(
                {"seed": seed, "step": tj.get("stopped_at"),
                 "best_val_before_collapse": tj.get("best_val")})
            if not chain:
                print("[217] the fallback chain is exhausted and every seed diverged: stopping and reporting")
                prov["stopped"] = "all seeds in the pre-registered chain diverged"
                save()
                return 3
            nxt = chain.pop(0)
            print(f"[217] divergence abort at {nxt}")
            seed = nxt
            log = LOGS / f"T217_s{seed}.log"
            rc, tj = train(seed, log)
            prov["stages"].append({"stage": "0_train_fallback", "seed": seed, "rc": rc,
                                   "best_val": tj.get("best_val"),
                                   "stopped_at": tj.get("stopped_at")})
        if rc != 0:
            print(f"[217] training exited non-zero:")
            save()
            return rc
        bv = tj.get("best_val")
        print(f"[217] best_val = {bv}  threshold <= <= {BEST_VAL_MAX}")
        if bv is None or bv > BEST_VAL_MAX:
            print("[217] best_val the best validation loss is above the convergence threshold: stopping and reporting")
            prov["stopped"] = f"best_val {bv} > {BEST_VAL_MAX}"
            save()
            return 4
        prov["train_seed_used"] = seed
        save()
    else:
        prov["train_seed_used"] = seed
        ck = MODELS / f"{TRAIN_NAME}_s{seed}.pt"
        if not ck.exists():
            print(f"--skip-train but the checkpoint does not exist: stopping")
            return 2
        if a.promotion_json:
            pj = Path(a.promotion_json)
            promo = json.loads(pj.read_text(encoding="utf-8"))
            if promo.get("promoted_sha256") != sha256(ck):
                print("the promotion record's sha256 does not match the checkpoint on disk: stopping")
                return 2
            if not promo.get("convergence_gate", {}).get("passed"):
                print("the promoted checkpoint does not pass the convergence threshold: stopping")
                return 4
            prov["checkpoint_promotion"] = {
                k: promo.get(k) for k in
                ("source", "source_sha256", "promoted_sha256", "step", "steps_requested",
                 "steps_missing", "best_val_at_step", "lr_at_step", "lr_fraction_of_peak",
                 "last_1000_steps_on_other_seeds", "why", "why_not_resume",
                 "why_not_retrain", "caption_declaration")}
            prov["checkpoint_promotion"]["record"] = str(pj)
            print(f"[217] using the promoted checkpoint: step {promo['step']}/"
                  f"{promo['steps_requested']}, best_val {promo['best_val_at_step']}")
        save()

    # stages 1 to 5 and scoring
    t0 = time.time()
    print(f"\n{'='*72}\n[217] stages 1 to 5 and scoring, by calling tools.t214_ours_downstream --seeds {seed}"
          f"\n{'='*72}", flush=True)
    # its scoring stage rescans the whole experiment directory under one tag and writes a file
    # of the same name; an existing artefact is archived rather than overwritten
    old = C.OUT / "summary" / "214_ours_table2_seeds_scored.json"
    if old.exists():
        arch = old.with_name(f"{old.stem}__preexisting_{int(time.time())}.json")
        old.rename(arch)
        print(f"[archive] {old.name} -> {arch.name}")
        prov["archived"] = prov.get("archived", []) + [arch.name]
    rc = subprocess.call([PY, "-m", "scripts.run_ours_downstream_chain", "--seeds", str(seed)],
                         cwd=str(ROOT), env=ENV)
    prov["stages"].append({"stage": "1-5+6_via_t214_ours_downstream", "rc": rc,
                           "walltime_s": round(time.time() - t0, 1)})
    save()
    if rc != 0:
        print(f"[217] the downstream stages exited non-zero:")
        return rc

    # stage 5b: the remaining sampling seeds
    t0 = time.time()
    print(f"\n{'='*72}\n[217] stage 5b: sampling seeds via build_grid_scvi_scanvi.step3"
          f"\n{'='*72}", flush=True)
    G.STEP3_TRAIN = (seed,)
    G.step3(LOGS / f"T217_s{seed}_step3.log")
    prov["stages"].append({"stage": "5b_extra_sampling_seeds",
                           "sampling_seeds": [20260988, 20260989],
                           "walltime_s": round(time.time() - t0, 1)})
    save()

    print(f"\n[217] whole chain finished in {(time.time()-t_all)/3600:.2f} h")
    print(f"[217] PROVENANCE -> {PROV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
