# -*- coding: utf-8 -*-
"""Ticket 201 / TAU-SWEEP: the product-tilt exponent tau is a hyperparameter; sweep it.

`hvg/sampler.py:215`:`self.logpi = exp(tau * logpi)`, i.e. `omega_i(n_i) = c_i(n_i)^tau`.
the tilt derived in the paper **has no exponent** (equivalent to tau = 1). The deployed arm uses tau = 0.4 (Endothelial)
and 0.2 (Myeloid / Neuronal),`PREREG_200.md` line 85 records them as **asserted values without derivation**.

**do not confuse the two exponents**:`alpha`(sampler.py:10) is the exponent of the FK reward potential,
`alpha = 1.0` **is **its derived value; this ticket **does not sweep alpha**; alpha must be 1.0 in every run.

    python -m tools.t201_tau_sweep --prereg-only   # write criteria + generation queue (not run)
    python -m tools.t201_tau_sweep --score         # scoring + aggregation + choose by rule 4 tau
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
from hvg import config as C                                          # noqa: E402

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

# keys **excluded** from the configuration comparison: bookmarks and measured outputs. Every other key must equal the deployed arm verbatim,
# only allows `tau` different. Use"exclusion table"rather than"includes table": when a run json later gains a new config field,
# it automatically enters the comparison and fails loudly instead of being silently ignored.
EXCLUDE = {
    # bookmark
    "experiment", "arm", "role", "ticket", "stem", "samples_file", "samples_sha256",
    "written_at", "_rerun_argv", "seed",
    # measured outputs / diagnostics
    "n_intermediate_resamples", "resample_steps", "ess_trace", "ess_min_fraction",
    "ess_pre_final_fraction", "unique_lineage_fraction", "unique_cell_fraction",
    "telescope_max_abs_error", "discriminator_evaluations_total", "twist_plugin_gap_trace",
    "n_cells", "n_genes", "mean", "max", "zero_fraction", "library_size_mean",
    "library_size_quantiles", "gene_detection_rate_mean", "n_genes_never_detected",
    "walltime_s", "peak_alloc_gb", "kernel_bank_gate",
}
ALLOWED_TO_DIFFER = {"tau"}

# the deployed arm was written on 2026-09-06 (ticket 163); since then the run json writer has **only gained, never lost** two
# provenance fields. They do **not exist** in the deployed arm at all (rather than having different values), so they are not comparable:
# but it also **cannot** just be let through: every one is pinned to an expected value; a mismatch is a real violation.
#   resampler      hvg/sampler.py:529 = "systematic" if resample_fn is None else "stratified_quota"
#                  single request, no --stratify/--types/--weights ⇒ resample_fn is None ⇒ systematic
#   mixture_reward without mixture reward it is None
EXPECTED_ADDED = {"resampler": "systematic", "mixture_reward": None}

DIAGS = ["ess_min_fraction", "ess_pre_final_fraction", "n_intermediate_resamples",
         "unique_lineage_fraction", "unique_cell_fraction"]


def config_check(new: dict, dep: dict) -> dict:
    """configuration comparison. The three kinds of results are reported separately; **none is silently swallowed**.

    violations        keys present on both sides with different values (except tau): a real violation
    added_since       keys only the new run has (the deployed-arm writer did not have this field at the time):
                      pinned to EXPECTED_ADDED; a mismatch is also a real violation
    missing_in_new    keys only the deployed arm has: a real violation (the writer should not drop fields)
    stem              stem token-by-token comparison: must differ by **exactly one token**, and that token is tau
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

    # stem is assembled from the config, so it is an **independent** piece of evidence: exactly one token differs, and it is tau.
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
    """from the deployed arm own `_rerun_argv` derived: **only** --tau is replaced, plus the experiment/arm labels."""
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
    PREREG.write_text(f"""# PREREG · ticket 201 / TAU-SWEEP: product-tilt exponent tau

**frozen once saved; written before any new number exists ({time.strftime('%Y-%m-%d %H:%M:%S')}). Not changed after writing.**

all config numbers are **read** from the run json of the three deployed arms, not copied by hand.

## 0. why

`hvg/sampler.py:215` applies the product tilt as `exp(tau * logpi)`, i.e. `omega_i(n_i) = c_i(n_i)^tau`.
the tilt derived in the paper **has no exponent** (equivalent to tau = 1). The deployed arm uses
Endothelial **{DEPLOYED_TAU['Endothelial']}** / Myeloid **{DEPLOYED_TAU['Myeloid']}** /
Neuronal **{DEPLOYED_TAU['Neuronal']}**,`PREREG_200.md` line 85 records it as an **asserted value, no derivation**.
so the deployed method deviates from the derived method by **a per-type quantity not disclosed in the paper**,
while the same paper reports three gammas for the value-guided baseline. This ticket removes that asymmetry.

**alpha out of scope for this ticket**:`alpha`(sampler.py:10) is the exponent of the FK reward potential,
`alpha = 1.0` **is **its derived value.** alpha must be 1.0 in every run below; it is not swept.**

## 1. grid

`tau ∈ {{0.2, 0.4, 1.0}}` × three requests, **each type runs all three levels**, including the two it does not currently use.
**tau = 1.0 is a derived value, not an option.**

## 2. seed

**{SEEDS[0]} / {SEEDS[1]} / {SEEDS[2]}**, i.e. the three of the deployed arm. Fixed, not replaced.

## 3. everything else identical to the deployed arm

alpha 1.0, J 16, K 32, M = 2N, resample interval 8, ESS threshold 0.5, reward_mode odds,
the same frozen ckpt, the same tilt tables and logpi. **Only tau changes.**

the command line **is not rewritten**; it comes from the deployed arm own `_rerun_argv` derived, replacing only `--tau`
(and `--experiment` / `--arm` / `--ticket` these three **bookmarks**, see below).

**per-run assertion**: compare the new run json key by key with the deployed-arm run json of **the same type and seed**,
**only allows `tau` different**; any other differing config key aborts that entry.

the comparison **excludes** two kinds of keys, declared here first (otherwise the assertion cannot hold; not because they are let through):

- **bookmark**:`experiment` / `arm` / `role` / `ticket` / `stem` / `seed` /
  `samples_file` / `samples_sha256` / `written_at` / `_rerun_argv`.
  of which `experiment` and `arm` ****must** change: if we kept `table1_ours` / `<type>_a1`,
  these 18 land in the Table 1 directory and would be `load_runs('table1_ours')` treated as the Table 1 delivery
  scored together: that would silently contaminate the main table, much worse than two extra differing fields.
  so it falls in `out/runs/{EXPERIMENT}/<type>_{{tau_tag}}/`.
- **measured outputs**: ESS trajectories, lineage and uniqueness ratios, library size, wall clock, memory, etc.:
  they are supposed to change with tau; that is exactly what this ticket looks at.

the comparison uses an **exclusion table** rather than an inclusion table: if a run json later gains a new config field, it **automatically enters the comparison**
and fails loudly instead of being silently ignored.

## 4. selection rule (fixed now)

**for each request, the tau with the highest purity averaged over three seeds goes into the main table.**
the same rule already governs the value-guided baseline (PREREG_200 P-A6), so both methods are at **their own best
reported at the **operating point**, and **both full scans go into the sampler-axis table**.
the rule is **purity**, the same for all three requests, **not varying by request or by metric**.
when the three seed rules tie, **take the smaller tau**.

## 5. all nine arms are reported whether selected or not

**no prediction is made about the shape of the curve** (this ticket registers no directional expectation).

## 6. stop once selected

this ticket only produces **one json and one recommendation**. **No table is rebuilt and no other arm is rerun.**
whether to rebuild downstream (Stage 2) is decided after reading the selection results.

## 7. what this ticket runs

{n_new} new runs + {n_arch} that **reuse the archive and must not be rerun**:

| request | tau 0.2 | tau 0.4 | tau 1.0 | n_out |
|---|---|---|---|---|
| Endothelial | new | **archived (deployed value)** | new | {deps['Endothelial']['n_out']} |
| Myeloid | **archived (deployed value)** | new | new | {deps['Myeloid']['n_out']} |
| Neuronal | **archived (deployed value)** | new | new | {deps['Neuronal']['n_out']} |

cheap ones first: Neuronal → Myeloid → Endothelial.
measured wall clock of a single deployed-arm run (seed {SEEDS[0]}):
Neuronal {deps['Neuronal']['walltime_s']:.0f} s / Myeloid {deps['Myeloid']['walltime_s']:.0f} s /
Endothelial {deps['Endothelial']['walltime_s']:.0f} s. tau does not change the cost.
⇒ 18 expected total of about
**{6*(deps['Neuronal']['walltime_s']+deps['Myeloid']['walltime_s']+deps['Endothelial']['walltime_s'])/3600:.1f} GPU hours**
(the ticket text says ~3.5 h; converting the ticket own ~440/~930/~1700 s also gives ~5.1 h.
recorded as is; no plan changed because of it).

## 8. scoring definition (identical to Table 2, no exceptions)

- purity:`analysis/score_purity.py`,`master_purity_v7` label set, matched_block 2500, CellTypist checked by G5;
- W1:`_frozen/e30plus_eval_cpu_worker.py`, raw counts, N_MATCH 10057/2302/396,
  eval_seed 20261071, n_proj 512, proj_seed 20261070, BLAS pinned to 1 thread;
- MMD²/PCC:`_frozen/e21_scc_mmd.py`, seed 20261071, target_sum 10000, block 1024,
  sigma take the archived value per type, **no refit**.

every ruler passes a self-check before writing. The three archived arms are **rescored** here and asserted to reproduce
`200_selftest_table1_scored.json` existing values; **if they do not match, report as is; never silently correct**.

recorded separately per run:`ess_min_fraction`,`ess_pre_final_fraction`,`n_intermediate_resamples`,
`unique_lineage_fraction`,`unique_cell_fraction`,`W1_median`,`W1_p90`.

## 9. supervision

check every 10 minutes. A single job exceeding **3 times** the type mean is considered hung: kill and **rerun once**.
**tau = 1.0 is the strongest tilt; low ESS is an outcome, not a failure**; record and continue.
NaN or an empty delivery at any tau is a **real failure**: stop that arm, diagnose from the logs, fix the mechanism, rerun.
**never stops at"stopped".**
""", encoding="utf-8")
    return {"deps": deps, "plan": plan, "n_new": n_new, "n_archived": n_arch}


def build_queue(deps: dict) -> int:
    jobs = []
    for t in ["Neuronal", "Myeloid", "Endothelial"]:          # run the cheap ones first
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
    # skip_if_exists needs the real output path; the stem is built by run_one itself, so it is left empty here,
    # by run_one own "refusing to overwrite" fallback (it already has one).
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
        from tools.t201_tau_score import run_scoring                  # noqa: PLC0415
        return run_scoring()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
