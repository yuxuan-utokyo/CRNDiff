# -*- coding: utf-8 -*-
"""Score the tilt-exponent sweep, aggregate it, and choose tau by the pre-registered rule.

All four rulers go through the UNMODIFIED scoring driver, so the convention is identical to the
main table's. The aggregation and the 'indistinguishable' verdict import the sampler-axis
table's own functions rather than reimplementing the statistics.

It writes one summary json and touches no table and no figure.

    python scripts/score_tilt_exponent_sweep.py
"""
from __future__ import annotations

import os                                                            # noqa: E402

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json                                                          # noqa: E402
import subprocess                                                    # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
from pathlib import Path                                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from scripts.build_sampler_axis_tables import agg, three_v_three                     # noqa: E402
from scripts.sweep_tilt_exponent import (ALLOWED_TO_DIFFER, DEPLOYED_TAU,    # noqa: E402
                                  EXCLUDE, EXPECTED_ADDED, EXPERIMENT, SEEDS,
                                  TAUS, TYPES, config_check, deployed_run)

OUT = C.OUT / "summary" / "201_tau_sweep_scored.json"
SELFTEST = C.OUT / "summary" / "200_selftest_table1_scored.json"
DIAGS = ["ess_min_fraction", "ess_pre_final_fraction", "n_intermediate_resamples",
         "unique_lineage_fraction", "unique_cell_fraction"]
METRICS = ["purity", "sliced_W1", "W1", "mmd2_rbf_biased", "pcc", "scc"]
FLOORS = ["floor_W1", "floor_sliced_W1", "floor_scc", "floor_pcc", "floor_mmd2"]


def _score(experiment: str, tag: str, only_stems: list[str] | None = None) -> dict:
    argv = [sys.executable, "-m", "scripts.score_deliveries",
            "--experiment", experiment, "--out-tag", tag, "--e21-device", "cpu"]
    if only_stems:
        argv += ["--only-stems", *only_stems]
    print(f"[score] {experiment} -> {tag}"
          + (f" ({len(only_stems)} stems)" if only_stems else ""), flush=True)
    r = subprocess.run(argv, cwd=str(ROOT),
                       env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
    if r.returncode != 0:
        raise SystemExit(f"scorer failed on {experiment} rc={r.returncode}")
    cands = sorted((C.OUT / "summary").glob(f"{tag}_scored*.json"),
                   key=lambda p: p.stat().st_mtime)
    return json.loads(cands[-1].read_text(encoding="utf-8"))


def run_scoring() -> int:
    t0 = time.time()

    archived_stems = {}
    for t in TYPES:
        for s in SEEDS:
            p = deployed_run(t, s)
            archived_stems[p.stem] = {"type": t, "seed": s, "tau": DEPLOYED_TAU[t],
                                      "path": p}

    sw = _score(EXPERIMENT, "201_tau_sweep_raw")
    ar = _score("table1_ours", "201_tau_archived_raw", sorted(archived_stems))

    # every ruler self-check must be present
    rulers = {}
    for name, blob in (("sweep", sw), ("archived", ar)):
        rl = blob.get("rulers", {})
        rulers[name] = {
            "purity_matched_block": rl.get("purity", {}).get("matched_block"),
            "purity_celltypist": rl.get("purity", {}).get("celltypist"),
            "w1": {k: rl.get("w1", {}).get(k) for k in
                   ("N_MATCH", "eval_seed", "n_proj", "proj_seed",
                    "blas_threads_pinned_to_1", "selfcheck_vs_cond_w1_floor")},
            "e21": {k: rl.get("e21", {}).get(k) for k in ("matched_n", "selfcheck")}}
    w1ok = all((rulers[n]["w1"]["selfcheck_vs_cond_w1_floor"] or {}).get(t, {})
               .get("bit_identical") for n in rulers for t in TYPES)
    e21ok = all((rulers[n]["e21"]["selfcheck"] or {}).get(t, {}).get("passed")
                for n in rulers for t in TYPES)
    print(f"[ruler] W1 bit-identical on every type/both passes: {w1ok}")
    print(f"[ruler] E21 self-check passed on every type/both passes: {e21ok}")

    # rows
    rows, cfg_violations = [], []
    for blob, archived in ((sw, False), (ar, True)):
        for r in blob["rows"]:
            stem = r["stem"]
            if archived:
                meta = archived_stems.get(stem)
                if meta is None:
                    continue
                rp = meta["path"]
            else:
                hits = list((C.RUNS / EXPERIMENT).glob(f"*/{stem}.json"))
                if not hits:
                    continue
                rp = hits[0]
            d = json.loads(rp.read_text(encoding="utf-8"))
            t, seed, tau = d["request"], int(d["seed"]), float(d["tau"])

            chk = (None if archived else config_check(
                d, json.loads(deployed_run(t, seed).read_text(encoding="utf-8"))))
            bad = {} if archived else chk["violations"]
            if chk is not None and (bad or not chk["stem_only_tau_differs"]):
                cfg_violations.append({"stem": stem, "fields": bad,
                                       "stem_token_diff": chk["stem_token_diff"]})

            row = {"stem": stem, "seed": seed, "request": t, "tau": tau,
                   "alpha": d.get("alpha"), "n_delivered": r.get("n_delivered"),
                   "archived_reused": bool(archived),
                   "run_json": str(rp.relative_to(ROOT)).replace("\\", "/")}
            for k in METRICS:
                row[k] = r.get(k)
            for k in ("W1_median", "W1_p90"):
                row[k] = r.get(k)
            for k in FLOORS:
                row[k] = r.get(k)
            for k in DIAGS:
                row[k] = d.get(k)
            row["config_only_tau_differs"] = (None if archived else chk["ok"])
            row["config_fields_added_since_deployed"] = (
                None if archived else chk["added_since_deployed"])
            row["stem_token_diff"] = None if archived else chk["stem_token_diff"]
            rows.append(row)

    # the archived arms must reproduce the existing scored values
    repro = []
    if SELFTEST.exists():
        ref = {x["stem"]: x for x in json.loads(SELFTEST.read_text(encoding="utf-8"))["rows"]}
        for row in rows:
            if not row["archived_reused"]:
                continue
            o = ref.get(row["stem"])
            if not o:
                repro.append({"stem": row["stem"], "status": "NOT_IN_SELFTEST"})
                continue
            item = {"stem": row["stem"]}
            for k in ("purity", "sliced_W1", "W1"):
                a, b = row.get(k), o.get(k)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    item[k] = {"now": a, "archived": b, "exact": bool(a == b),
                               "abs_diff": abs(a - b)}
                else:
                    item[k] = {"now": a, "archived": b, "exact": None,
                               "note": "absent in the selftest file"}
            for k in ("mmd2_rbf_biased", "pcc"):
                item[k] = {"now": row.get(k), "archived": o.get(k),
                           "exact": None,
                           "note": "the selftest file carries no E21 value for this arm, "
                                   "so no reproduction assertion is possible for it"}
            item["all_asserted_exact"] = all(
                item[k].get("exact") for k in ("purity", "sliced_W1", "W1")
                if item[k].get("exact") is not None)
            repro.append(item)
    not_repro = [x for x in repro if x.get("all_asserted_exact") is False]

    # ---- rollup ----
    roll = {}
    for t in TYPES:
        for tau in TAUS:
            sel = sorted([r for r in rows if r["request"] == t and r["tau"] == tau],
                         key=lambda r: r["seed"])
            if not sel:
                continue
            key = f"{t}|tau{tau}"
            m = {k: agg([r.get(k) for r in sel]) for k in METRICS}
            dg = {k: agg([r.get(k) for r in sel]) for k in DIAGS}
            roll[key] = {"request": t, "tau": tau, "n_seeds": len(sel),
                         "seeds": [r["seed"] for r in sel],
                         "archived_reused": bool(sel[0]["archived_reused"]),
                         "is_deployed_tau": bool(tau == DEPLOYED_TAU[t]),
                         "n_delivered": sorted({r["n_delivered"] for r in sel}),
                         "metrics": m, "diagnostics": dg}

    # the rule: the tau with the highest three-seed mean purity wins; a tie goes to the smaller tau
    selected = {}
    for t in TYPES:
        cands = [(tau, roll[f"{t}|tau{tau}"]) for tau in TAUS if f"{t}|tau{tau}" in roll]
        scored = [(c["metrics"]["purity"]["mean"], tau, c) for tau, c in cands
                  if c["metrics"]["purity"]["mean"] is not None]
        if not scored:
            selected[t] = {"status": "NOT_SCORABLE"}
            continue
        best = max(s[0] for s in scored)
        tied = sorted(tau for mean_, tau, _ in scored if mean_ == best)
        win = tied[0]                                   # a tie goes to the smaller tau
        rest = sorted((m_, tau) for m_, tau, _ in scored if tau != win)
        runner = max(rest)[1] if rest else None
        cmp_ = (three_v_three(roll[f"{t}|tau{win}"]["metrics"]["purity"],
                              roll[f"{t}|tau{runner}"]["metrics"]["purity"])
                if runner is not None else None)
        selected[t] = {
            "selected_tau": win, "deployed_tau": DEPLOYED_TAU[t],
            "changes_from_deployed": bool(win != DEPLOYED_TAU[t]),
            "rule": "highest 3-seed mean purity; ties broken towards the smaller tau "
                    "(PREREG 201 rule 4, frozen before any number existed)",
            "tie": len(tied) > 1, "tied_taus": tied,
            "purity_mean_by_tau": {str(tau): roll[f'{t}|tau{tau}']["metrics"]["purity"]["mean"]
                                   for tau in TAUS if f"{t}|tau{tau}" in roll},
            "purity_per_seed_by_tau": {
                str(tau): roll[f'{t}|tau{tau}']["metrics"]["purity"]["per_seed"]
                for tau in TAUS if f"{t}|tau{tau}" in roll},
            "runner_up_tau": runner,
            "three_seed_comparison_vs_runner_up": cmp_}

    blob = {
        "ticket": 201, "block": "TAU-SWEEP -- product tilt exponent",
        "prereg": str(C.OUT / "queue" / "201_TAU_PREREG.md"),
        "scope": "Stage 1 only: scored and selected. No table, figure or manuscript "
                 "file was touched; no other arm was re-run; Stage 2 not started.",
        "alpha_note": "alpha is the FK reward exponent and is NOT swept; alpha = 1.0 is "
                      "its derived value and is 1.0 in every run here",
        "alpha_values_seen": sorted({r["alpha"] for r in rows}),
        "grid": {"taus": TAUS, "seeds": SEEDS, "requests": TYPES},
        "deployed_tau": DEPLOYED_TAU,
        "selected_tau": {t: selected[t].get("selected_tau") for t in TYPES},
        "selection": selected,
        "any_change_from_deployed": any(selected[t].get("changes_from_deployed")
                                        for t in TYPES),
        "rulers": rulers,
        "ruler_selfchecks_all_pass": {"w1_bit_identical": w1ok, "e21_passed": e21ok},
        "config_assertion": {
            "rule": "every sweep run json is diffed against the SAME-TYPE SAME-SEED "
                    "deployed run json; only `tau` may differ. Bookkeeping and measured "
                    "fields are excluded and listed in the prereg.",
            "excluded_keys": sorted(EXCLUDE), "allowed_to_differ": sorted(ALLOWED_TO_DIFFER),
            "fields_added_to_the_writer_since_the_deployed_run": EXPECTED_ADDED,
            "fields_added_note":
                "the deployed arms were written 2026-09-06 (ticket 163); the run-json "
                "writer has since gained resampler and mixture_reward. They are ABSENT "
                "in the deployed jsons, not different, so a plain key diff cannot "
                "compare them. Each is pinned to the value the deployed arm provably "
                "used -- resampler=systematic (hvg/sampler.py:529 sets systematic iff "
                "resample_fn is None; the deployed argv carries no "
                "--stratify/--types/--weights, and neither stem carries the quota token "
                "from hvg/io.py:62) and mixture_reward=None. A different value counts "
                "as a real violation, not an exemption.",
            "independent_stem_check":
                "the stem is built from the configuration, so it is separate evidence: "
                "every sweep stem must differ from its deployed counterpart in exactly "
                "one token, and that token must be the tau token.",
            "n_violations": len(cfg_violations), "violations": cfg_violations},
        "archived_reproduction": {
            "reference": str(SELFTEST.relative_to(ROOT)).replace("\\", "/"),
            "rule": "the three archived arms are re-scored here and must reproduce the "
                    "values already on disk; any that do not are reported, not fixed",
            "n_not_reproduced": len(not_repro), "rows": repro},
        "rows": sorted(rows, key=lambda r: (r["request"], r["tau"], r["seed"])),
        "rollup_key": "<type>|tau<value>", "rollup": roll,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "walltime_s": time.time() - t0}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = OUT.parent / "_superseded"
        old.mkdir(parents=True, exist_ok=True)
        OUT.rename(old / f"201_tau_sweep_scored_{int(time.time())}.json")
    OUT.write_text(json.dumps(blob, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\n=== rollup (purity) ===")
    for t in TYPES:
        for tau in TAUS:
            k = f"{t}|tau{tau}"
            if k not in roll:
                continue
            p = roll[k]["metrics"]["purity"]
            e = roll[k]["diagnostics"]["ess_min_fraction"]
            mark = " <- deployed" if roll[k]["is_deployed_tau"] else ""
            print(f"  {k:<24} n={p['n_seeds']}  purity {p['mean']:.4f} +- {p['sd']:.4f}"
                  f"   ess_min {e['mean']:.4f}{mark}")
    print("\n=== selection (prereg rule 4) ===")
    for t in TYPES:
        s = selected[t]
        print(f"  {t:<12} selected tau={s.get('selected_tau')}  "
              f"deployed tau={s.get('deployed_tau')}  "
              f"changes={s.get('changes_from_deployed')}  runner-up={s.get('runner_up_tau')}")
    print(f"\n[config] violations: {len(cfg_violations)}")
    print(f"[archived] not reproduced: {len(not_repro)}")
    print(f"[out] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_scoring())
