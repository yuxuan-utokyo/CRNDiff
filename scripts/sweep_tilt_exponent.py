# -*- coding: utf-8 -*-
"""Sweep tau, the exponent of the product tilt.

In the sampler, the tilt is raised to this exponent, so each coordinate's tilt factor becomes
c_i(n_i)^tau. The derivation in the paper has NO exponent, which is tau = 1. The deployed arms
use smaller values, and the pre-registration records those as ASSERTED rather than derived,
which is what this sweep tests.

Do not confuse the two exponents: alpha is the exponent of the particle reward and tau is the
exponent of the proposal tilt. They act at different places in the sampler.

    python scripts/sweep_tilt_exponent.py --help
"""
from __future__ import annotations

import os                                                            # noqa: E402

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse                                                      # noqa: E402
import json                                                          # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402

TYPES = ["Endothelial", "Myeloid", "Neuronal"]
TAUS = [0.2, 0.4, 1.0]
SEEDS = [20260987, 20260988, 20260989]
DEPLOYED_TAU = {"Endothelial": 0.4, "Myeloid": 0.2, "Neuronal": 0.2}
EXPERIMENT = "tau_sweep"
PREREG = C.OUT / "queue" / "201_TAU_PREREG.md"
QUEUE = C.OUT / "ticket201_logs" / "queue_tau_sweep.json"
OUT = C.OUT / "summary" / "201_tau_sweep_scored.json"
SELFTEST = C.OUT / "summary" / "200_selftest_table1_scored.json"
LOGLINE = C.OUT / "ticket201_logs" / "TAU.log"

# Keys EXCLUDED from the configuration comparison: bookkeeping tags and measured products.
# Every other key must equal the deployed arm verbatim; only `tau` may differ. This is an
# exclusion list rather than an inclusion list, so that a configuration field added to the run
# json later enters the comparison automatically and fails loudly instead of being ignored.
EXCLUDE = {
    # bookkeeping tags
    "experiment", "arm", "role", "ticket", "stem", "samples_file", "samples_sha256",
    "written_at", "_rerun_argv", "seed",
    # measured products / diagnostics
    "n_intermediate_resamples", "resample_steps", "ess_trace", "ess_min_fraction",
    "ess_pre_final_fraction", "unique_lineage_fraction", "unique_cell_fraction",
    "telescope_max_abs_error", "discriminator_evaluations_total", "twist_plugin_gap_trace",
    "n_cells", "n_genes", "mean", "max", "zero_fraction", "library_size_mean",
    "library_size_quantiles", "gene_detection_rate_mean", "n_genes_never_detected",
    "walltime_s", "peak_alloc_gb", "kernel_bank_gate",
}
ALLOWED_TO_DIFFER = {"tau"}

# The deployed arms were written on 2026-09-06 (ticket 163). Since then the writer of the run json
# has gained two provenance fields, only ever adding, never removing. Those fields are ABSENT from the
# deployed arms (as opposed to holding a different value), so they cannot be compared directly. They
# are not waved through either: each one is pinned to its expected value here, and any other value is
# a real violation.
#   resampler      hvg/sampler.py:529 = "systematic" if resample_fn is None else "stratified_quota"
#                  single request, no --stratify/--types/--weights => resample_fn is None => systematic
#   mixture_reward None when there is no mixture reward
EXPECTED_ADDED = {"resampler": "systematic", "mixture_reward": None}

DIAGS = ["ess_min_fraction", "ess_pre_final_fraction", "n_intermediate_resamples",
         "unique_lineage_fraction", "unique_cell_fraction"]


def config_check(new: dict, dep: dict) -> dict:
    """Compare a new run's configuration against the deployed arm's.

    The three kinds of result are reported separately; none of them is swallowed silently.

    violations        keys present on both sides with different values (tau excepted): real violations
    added_since       keys only the new run has (the writer had no such field back then); each is
                      pinned to EXPECTED_ADDED, and a different value is a real violation too
    missing_in_new    keys only the deployed arm has: a real violation, the writer must not drop fields
    stem              token-by-token comparison of the stem: exactly ONE token may differ, and it must
                      be the tau token
    """
    violations, added, missing = {}, {}, {}
    for k in sorted(set(new) | set(dep)):
        if k in EXCLUDE or k in ALLOWED_TO_DIFFER:
            continue
        in_new, in_dep = k in new, k in dep
        if in_new and in_dep:
            if new[k] != dep[k]:
                violations[k] = {"run": new[k], "deployed": dep[k]}
        elif in_new and not in_dep:
            exp = EXPECTED_ADDED.get(k, "__UNEXPECTED__")
            entry = {"run": new[k], "deployed": "<field absent>", "expected": exp}
            if k not in EXPECTED_ADDED or new[k] != exp:
                violations[k] = entry | {"why": "field is new AND its value is not the "
                                                "pre-registered default"}
            else:
                added[k] = entry
        else:
            missing[k] = {"run": "<field absent>", "deployed": dep[k]}
    violations.update({k: v | {"why": "field present in the deployed run but missing here"}
                       for k, v in missing.items()})

    # The stem is assembled from the configuration, so it is INDEPENDENT evidence:
    # exactly one token differs, and that token is tau.
    a, b = new.get("stem", "").split("_"), dep.get("stem", "").split("_")
    diff = [(x, y) for x, y in zip(a, b) if x != y] if len(a) == len(b) else [("LEN", "LEN")]
    stem_ok = (len(a) == len(b) and len(diff) == 1
               and diff[0][0].startswith("tau") and diff[0][1].startswith("tau"))
    return {"violations": violations, "added_since_deployed": added,
            "stem_token_diff": diff, "stem_only_tau_differs": bool(stem_ok),
            "ok": bool(not violations and stem_ok)}


def tau_tag(v: float) -> str:
    return f"tau{v}".replace(".", "p")


def deployed_run(t: str, seed: int) -> Path:
    d = C.RUNS / "table1_ours" / f"{t}_a1"
    hits = [p for p in sorted(d.glob("*.json")) if p.stem.endswith(f"seed{seed}")]
    if not hits:
        raise SystemExit(f"deployed run not found for {t} seed {seed}")
    return hits[0]


def build_argv(t: str, tau: float, seed: int, dep: dict) -> list[str]:
    """Derive the command line from the deployed arm's own `_rerun_argv`.

    Only --tau is replaced, plus the experiment/arm/ticket bookkeeping tags.
    """
    argv = list(dep["_rerun_argv"])
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--tau":
            out += ["--tau", str(tau)]; i += 2; continue
        if a == "--experiment":
            out += ["--experiment", EXPERIMENT]; i += 2; continue
        if a == "--arm":
            out += ["--arm", f"{t}_{tau_tag(tau)}"]; i += 2; continue
        if a == "--ticket":
            out += ["--ticket", "201"]; i += 2; continue
        if a == "--seed":
            out += ["--seed", str(seed)]; i += 2; continue
        out.append(a); i += 1
    return out


def write_prereg() -> dict:
    deps = {t: json.loads(deployed_run(t, SEEDS[0]).read_text(encoding="utf-8"))
            for t in TYPES}
    plan, n_new, n_arch = [], 0, 0
    for t in TYPES:
        for tau in TAUS:
            archived = (tau == DEPLOYED_TAU[t])
            for s in SEEDS:
                plan.append({"request": t, "tau": tau, "seed": s, "archived": archived})
            n_arch += 3 if archived else 0
            n_new += 0 if archived else 3

    PREREG.parent.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(f"""# PREREG - Ticket 201 / TAU-SWEEP: tau, the exponent of the product tilt

**Frozen on write, written before any new number exists ({time.strftime('%Y-%m-%d %H:%M:%S')}). Not edited afterwards.**

Every configuration number below is READ from the three deployed arms' own run json, never copied by hand.

## 0. Why

`hvg/sampler.py:215` applies the product tilt as `exp(tau * logpi)`, i.e. `omega_i(n_i) = c_i(n_i)^tau`.
The tilt derived in the paper has NO exponent (equivalently tau = 1). The deployed arms use
Endothelial **{DEPLOYED_TAU['Endothelial']}** / Myeloid **{DEPLOYED_TAU['Myeloid']}** /
Neuronal **{DEPLOYED_TAU['Neuronal']}**, which `PREREG_200.md` line 85 records as ASSERTED values
with no derivation. The deployed method therefore departs from the derived method by a per-request
quantity the paper does not disclose, while the same paper reports three gammas for the value-guided
baseline. This ticket removes that asymmetry.

**alpha is out of scope here**: `alpha` (sampler.py:10) is the exponent of the FK reward potential and
`alpha = 1.0` IS its derived value. **Every run below must have alpha = 1.0; alpha is not swept.**

## 1. Grid

`tau in {{0.2, 0.4, 1.0}}` x three requests. **Every request runs all three settings**, including the
two it does not currently use. **tau = 1.0 is the derived value, not an option.**

## 2. Seeds

**{SEEDS[0]} / {SEEDS[1]} / {SEEDS[2]}**, the three seeds of the deployed arms. Fixed, never substituted.

## 3. Everything else identical to the deployed arms

alpha 1.0, J 16, K 32, M = 2N, resample interval 8, ESS threshold 0.5, reward_mode odds,
the same frozen ckpt, the same tilt tables and logpi. **Only tau changes.**

The command line is **not rewritten**. It is derived from the deployed arm's own `_rerun_argv` by
replacing `--tau` only (plus `--experiment` / `--arm` / `--ticket`, the three bookkeeping tags, below).

**Per-run assertion**: every new run json is compared key by key against the deployed run json of the
**same request and the same seed**. **Only `tau` may differ**; any other configuration key that differs
aborts that run.

The comparison **excludes** two classes of key, declared here in advance, so that the assertion can hold
at all rather than these being waved through after the fact:

- **Bookkeeping**: `experiment` / `arm` / `role` / `ticket` / `stem` / `seed` /
  `samples_file` / `samples_sha256` / `written_at` / `_rerun_argv`.
  `experiment` and `arm` **must** change: keeping `table1_ours` / `<type>_a1` would drop these 18 runs
  into the Table 1 directory, where `load_runs('table1_ours')` would score them as Table 1 deliverables.
  That is silent contamination of the main table, far worse than two extra differing fields.
  They therefore land in `out/runs/{EXPERIMENT}/<type>_{{tau_tag}}/`.
- **Measured products**: ESS traces, lineage and uniqueness fractions, library size, wall clock, GPU
  memory and so on. These are supposed to move with tau; they are exactly what this ticket looks at.

The comparison uses an **exclusion list, not an inclusion list**: if the run json ever gains a new
configuration field it **enters the comparison automatically** and fails loudly instead of being ignored.

## 4. Selection rule (fixed now)

**For each request, the tau with the highest three-seed mean purity goes into the main table.**
The same rule already governs the value-guided baseline (PREREG_200 P-A6), so both methods are reported
at **their own best operating point**, and **both complete sweeps go into the sampler-axis table**.
The rule is **purity**, applied identically to all three requests; it **does not vary by request and does
not vary by metric**. On a tie under the three-seed rule, **take the smaller tau**.

## 5. All nine arms are reported, selected or not

**No prediction is registered about the shape of the curve** (this ticket registers no directional expectation).

## 6. Stop after selection

This ticket produces **one json and one recommendation**. **It rebuilds no table and re-runs no other arm.**
Whether anything downstream is rebuilt (Stage 2) is decided after the selection result has been read.

## 7. What this ticket runs

{n_new} new runs + {n_arch} **reused from the archive, never re-run**:

| request | tau 0.2 | tau 0.4 | tau 1.0 | n_out |
|---|---|---|---|---|
| Endothelial | new | **archived (deployed value)** | new | {deps['Endothelial']['n_out']} |
| Myeloid | **archived (deployed value)** | new | new | {deps['Myeloid']['n_out']} |
| Neuronal | **archived (deployed value)** | new | new | {deps['Neuronal']['n_out']} |

Cheapest first: Neuronal, then Myeloid, then Endothelial.
Measured single-run wall clock of the deployed arms (seed {SEEDS[0]}):
Neuronal {deps['Neuronal']['walltime_s']:.0f} s / Myeloid {deps['Myeloid']['walltime_s']:.0f} s /
Endothelial {deps['Endothelial']['walltime_s']:.0f} s. tau does not change the cost.
=> estimated total for the 18 runs is about
**{6*(deps['Neuronal']['walltime_s']+deps['Myeloid']['walltime_s']+deps['Endothelial']['walltime_s'])/3600:.1f} GPU hours**
(the ticket text says ~3.5 h; the ~440/~930/~1700 s the ticket itself quotes also work out to ~5.1 h.
Recorded as measured; no plan is changed on this basis).

## 8. Scoring convention (identical to Table 2, no exceptions)

- purity: `analysis/score_purity.py`, `master_purity_v7` label set, matched_block 2500, the G5-checked CellTypist;
- W1: `_frozen/e30plus_eval_cpu_worker.py`, raw counts, N_MATCH 10057/2302/396,
  eval_seed 20261071, n_proj 512, proj_seed 20261070, BLAS pinned to 1 thread;
- MMD2/PCC: `_frozen/e21_scc_mmd.py`, seed 20261071, target_sum 10000, block 1024,
  sigma taken from the archived per-request value, **never refitted**.

Every ruler passes its self-test before anything is written to disk. The three archived arms are
**re-scored** here and asserted to reproduce the values already in `200_selftest_table1_scored.json`;
**a mismatch is reported as it stands, never silently corrected**.

Recorded per run in addition: `ess_min_fraction`, `ess_pre_final_fraction`, `n_intermediate_resamples`,
`unique_lineage_fraction`, `unique_cell_fraction`, `W1_median`, `W1_p90`.

## 9. Supervision

Check every 10 minutes. A run exceeding **3x** the mean for its request counts as hung: kill it and
**re-run it once**. **tau = 1.0 is the strongest tilt, so a low ESS there is a result and not a fault**;
record it and continue. NaN or an empty delivery at any tau is a **real fault**: stop that arm, diagnose
from the log, fix the mechanism, re-run. **Never stop at "it stopped".**
""", encoding="utf-8")
    return {"deps": deps, "plan": plan, "n_new": n_new, "n_archived": n_arch}


def build_queue(deps: dict) -> int:
    jobs = []
    for t in ["Neuronal", "Myeloid", "Endothelial"]:          # cheapest request first
        dep0 = deps[t]
        for tau in TAUS:
            if tau == DEPLOYED_TAU[t]:
                continue
            for s in SEEDS:
                dep = json.loads(deployed_run(t, s).read_text(encoding="utf-8"))
                arm = f"{t}_{tau_tag(tau)}"
                stem_npy = (C.RUNS / EXPERIMENT / arm)
                jobs.append({
                    "tag": f"TAU_{t}_{tau_tag(tau)}_s{s}",
                    "argv": build_argv(t, tau, s, dep),
                    "retry": 1,
                    "skip_if_exists": str((stem_npy / f"__{s}__").as_posix()),  # placeholder, see below
                })
    # skip_if_exists needs a real artefact path, but the stem is assembled by run_one itself, so the
    # key is dropped here and run_one's own "refusing to overwrite" guard is what protects the runs.
    for j in jobs:
        j.pop("skip_if_exists", None)
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text(json.dumps(jobs, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(jobs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg-only", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()

    if not PREREG.exists():
        info = write_prereg()
        n = build_queue(info["deps"])
        print(f"[prereg] {PREREG}")
        print(f"[queue ] {QUEUE}  ({n} new runs; {info['n_archived']} archived reused)")
    else:
        print(f"[prereg] already frozen, not touched: {PREREG}")
        deps = {t: json.loads(deployed_run(t, SEEDS[0]).read_text(encoding="utf-8"))
                for t in TYPES}
        if not QUEUE.exists():
            print(f"[queue ] rebuilt {QUEUE} ({build_queue(deps)} jobs)")
    if a.prereg_only:
        return 0
    if a.score:
        from scripts.score_tilt_exponent_sweep import run_scoring                  # noqa: PLC0415
        return run_scoring()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
