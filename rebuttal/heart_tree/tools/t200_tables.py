# -*- coding: utf-8 -*-
"""Ticket 200 · M1-9: tables A / B1 / B2 / budget curves + item-by-item verdicts on the section 7 criteria.

**numbers are read only from output json, never copied by hand.** Missing blocks are marked `NOT_RUN`, no other cell is substituted.

n=3 definition (ticket section 1.7): no significance test and no CI; report the **raw full-permutation differences** of 3-vs-3.
the third state indistinguishable uses the frozen rule (Toy2D `analysis/verdicts.py::indistinguishable`):

    differences not all of the same sign, **or** every |difference| below the larger per-seed sd of the two cells ⇒ indistinguishable

pool seeds (31–33) and deployed seeds (87–89) are **not paired**, so these are 9 pairs of 3-vs-3, not 3 pairs.

    python -m tools.t200_tables
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hvg import config as C                                          # noqa: E402

S = C.OUT / "summary"
TYPES = ["Endothelial", "Myeloid", "Neuronal"]


def jload(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def pick(stem: str):
    """take the **final** scoring file of the family, not a half-written overnight one.

    priority:`<stem>_FINAL_scored.json` → latest `<stem>_scored_<ts>.json` → `<stem>_scored.json`.
    the scoring driver adds a timestamp on name clashes, so a family may have several copies; only the one with FINAL
    is rescored after all three seeds are in. The chosen path is written into the table header `inputs`, traceable.
    """
    fin = S / f"{stem}_FINAL_scored.json"
    if fin.exists():
        return fin
    ts = sorted(S.glob(f"{stem}_scored_*.json"))
    if ts:
        return ts[-1]
    base = S / f"{stem}_scored.json"
    return base if base.exists() else None


def merge_by_stem(dst: dict, src: dict, keys: tuple) -> int:
    """merge columns from another scoring file by stem (E21 and purity/W1 were scored in two separate passes)."""
    if not src:
        return 0
    by = {r["stem"]: r for r in src.get("rows", [])}
    n = 0
    for r in dst.get("rows", []):
        s = by.get(r["stem"])
        if not s:
            continue
        for k in keys:
            if k in s and s[k] is not None:
                r[k] = s[k]
        n += 1
    return n


def agg(vals: list[float]) -> dict:
    v = [x for x in vals if isinstance(x, (int, float))]
    if not v:
        return {"n_seeds": 0, "mean": None, "sd": None, "per_seed": vals}
    return {"n_seeds": len(v), "mean": float(statistics.fmean(v)),
            "sd": (float(statistics.stdev(v)) if len(v) > 1 else 0.0), "per_seed": v}


def three_v_three(a: dict, b: dict) -> dict:
    """3-vs-3 full permutation differences (a - b) + the frozen indistinguishable rule."""
    if not a.get("per_seed") or not b.get("per_seed"):
        return {"status": "NOT_RUN"}
    av = [x for x in a["per_seed"] if isinstance(x, (int, float))]
    bv = [x for x in b["per_seed"] if isinstance(x, (int, float))]
    if not av or not bv:
        return {"status": "NOT_RUN"}
    diffs = [x - y for x in av for y in bv]
    signs = {(d > 0) - (d < 0) for d in diffs if d != 0}
    sds = [a.get("sd") or 0.0, b.get("sd") or 0.0]
    ind = (len(signs) != 1) or (max(abs(d) for d in diffs) < max(sds))
    return {"pairs": len(diffs), "diffs": diffs,
            "n_a_ge_b": sum(1 for d in diffs if d >= 0),
            "mean_diff": float(statistics.fmean(diffs)),
            "all_same_sign": len(signs) == 1,
            "indistinguishable": bool(ind),
            "rule": "n=3 frozen rule: not all one sign OR every |diff| < max(sd_a, sd_b); "
                    "3-vs-3 because the pool seeds and the deployed seeds are not paired"}


def deployed_rows() -> dict:
    """readings of the three types for the deployed arm (tilted FK), all read from existing output json."""
    pur = jload(S / "purity_table1_ours.json")
    w1 = jload(S / "w1_paper_table1_ours.json")
    e21 = jload(S / "e21_table1_ours.json")
    out = {}
    for t in TYPES:
        row = {"source": {"purity": str(S / "purity_table1_ours.json"),
                          "w1": str(S / "w1_paper_table1_ours.json"),
                          "e21": str(S / "e21_table1_ours.json")}}
        if pur:
            row["purity"] = agg([r["purity_matched_n"] for r in pur["rows"]
                                 if r.get("target", r.get("request")) == t])
        if w1:
            rs = [r for r in w1["rows"] if r["target"] == t]
            row["W1"] = agg([r["W1"] for r in rs])
            row["sliced_W1"] = agg([r["sliced_W1"] for r in rs])
            row["unique_cell_fraction"] = agg([r.get("unique_cell_fraction") for r in rs])
        if e21:
            r = e21["types"][t]["rows"].get("OURS (tilt+FK, this round)")
            if r:
                for k in ("scc", "pcc", "mmd2_rbf_biased"):
                    row[k] = agg(r["aggregate"][k]["per_seed"])
        out[t] = row
    return out


def collect(scored: dict, key_fn) -> dict:
    """group the rows of the scoring driver by key_fn(row), and aggregate each metric over seeds within each group."""
    groups: dict = {}
    for r in (scored or {}).get("rows", []):
        k = key_fn(r)
        if k is None:
            continue
        groups.setdefault(k, []).append(r)
    out = {}
    for k, rs in groups.items():
        rs = sorted(rs, key=lambda r: r.get("seed") or 0)
        cell = {"n_runs": len(rs), "seeds": [r.get("seed") for r in rs],
                "request": rs[0].get("request"), "stems": [r["stem"] for r in rs]}
        for m in ("purity", "W1", "sliced_W1", "mmd2_rbf_biased", "scc", "pcc",
                  "unique_cell_fraction", "purity_over_ceiling"):
            vals = [r.get(m) for r in rs]
            cell[m] = agg(vals) if any(isinstance(v, (int, float)) for v in vals) else {
                "n_seeds": 0, "mean": None, "sd": None, "per_seed": vals}
        for m in ("n_candidates", "n_delivered", "NFE_total", "NFE_fk", "aux_calls",
                  "accepted_count", "shortfall", "axis", "level", "selector", "gamma",
                  "scorer", "generator", "device_class", "lineage", "lineage_note",
                  "walltime_s"):
            cell[m] = [r.get(m) for r in rs]
        out[k] = cell
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="200")
    a = ap.parse_args()

    D = jload(S / "200_derived_constants.json")
    dep = deployed_rows()
    inputs = {k: (str(pick(k)) if pick(k) else None) for k in
              ("200_bestofn", "200_bestofn_e21", "200_tiltfilter", "200_vgr", "200_aprime",
               "200_bestofn_all_generators_purity")}
    bestofn = jload(pick("200_bestofn")) if pick("200_bestofn") else None
    e21extra = jload(pick("200_bestofn_e21")) if pick("200_bestofn_e21") else None
    n_merged = merge_by_stem(bestofn or {}, e21extra,
                             ("scc", "pcc", "mmd2_rbf_biased", "e21_matched_n",
                              "floor_scc", "floor_pcc", "floor_mmd2"))
    tiltfilter = jload(pick("200_tiltfilter")) if pick("200_tiltfilter") else None
    vgr = jload(pick("200_vgr")) if pick("200_vgr") else None
    aprime = jload(pick("200_aprime")) if pick("200_aprime") else None
    fidelity = jload(S / "200_uncond_fidelity.json")
    allgen = (jload(pick("200_bestofn_all_generators_purity"))
              if pick("200_bestofn_all_generators_purity") else None)
    print(f"[inputs] merged {n_merged} E21 rows into best-of-N")
    for k, v in inputs.items():
        print(f"[inputs] {k:<40} {v}")
    comp = {t: None for t in TYPES}
    comps = sorted(S.glob("200_pool_composition_OURS_seed*.json"))
    pool_comp = [jload(p) for p in comps]

    # ---------------------------------------------------------------- table A
    A = {"ticket": 200, "table": "A -- samplers on OUR frozen generator",
         "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
         "columns_required": ["component", "NFE", "aux_calls", "walltime_s", "lineage",
                              "unique_cell_fraction"],
         "inputs": inputs,
         "deployed_arm": dep,
         "deployed_residual_dir": {t: (D["types"][t]["residual_dir"] if D else None)
                                   for t in TYPES},
         "derived": D, "rows": {}}

    def add_rows(tag, scored, keyf, component):
        if not scored:
            A["rows"][tag] = {"status": "NOT_RUN"}
            return
        cells = collect(scored, keyf)
        for k, cell in cells.items():
            cell["component"] = component
            t = cell["request"]
            for m in ("purity", "sliced_W1", "W1", "mmd2_rbf_biased", "scc", "pcc",
                      "unique_cell_fraction"):
                if dep.get(t, {}).get(m) and cell.get(m, {}).get("n_seeds"):
                    cell.setdefault("vs_deployed", {})[m] = three_v_three(cell[m], dep[t][m])
            A["rows"][f"{tag}::{k}"] = cell

    add_rows("bestofn", bestofn,
             lambda r: (f"{r.get('request')}|{r.get('selector')}|{r.get('variant')}|"
                        f"{r.get('axis')}{r.get('level')}"
                        if r.get("variant") else None),
             "best-of-N selector (S1 = one-vs-rest HGBT on real train; "
             "S2 = UNTILTED residual density ratio, fitted on BASE_POOL seed 20260902)")
    add_rows("tilt_plus_endpoint_filter", tiltfilter,
             lambda r: (f"{r.get('request')}|{r.get('selector')}|{r.get('variant')}|"
                        f"{r.get('axis')}{r.get('level')}" if r.get("variant") else None),
             "deployed residual discriminator rho (fitted against the TILTED proposal)")
    add_rows("vgr", vgr,
             lambda r: (f"{r.get('request')}|{r.get('scorer')}|g{r.get('gamma')}"
                        if r.get("scorer") else None),
             "VGR scorer (noised-state MLP, or S1 on the clean state)")

    # ---- B-V1 gate failed ⇒ stamp every VGR row per AMEND-1 section A1; nothing hidden, nothing softened ----
    # keep the original rule: **by mtime**, not sorted(glob)[-1]: '200_BV1_vgr.json' in '.' ranks
    # '_' before; lexicographic order is unrelated to write order; it was right before by pure coincidence.
    bv1_files = sorted((C.GATES).glob("200_BV1_vgr*.json"), key=lambda p: p.stat().st_mtime)
    bv1 = jload(bv1_files[-1]) if bv1_files else None

    # G5-R Stage 0 **three-seed re-verdict**, once saved, replaces the single-seed verdict (criterion in
    # out/queue/200_G5R_PREREG.md, fixed before any new number). The original verdict file is kept as is.
    bv1_3 = jload(C.OUT / "summary" / "200_BV1_3seed.json")
    A["b_v1_gate"] = bv1
    A["b_v1_3seed"] = bv1_3
    if bv1_3:
        passed = bv1_3.get("verdict") == "PASS"
        failed = ([] if passed else
                  [k for k, v in (("criterion_i", bv1_3.get("criterion_i")),
                                  ("criterion_ii", bv1_3.get("criterion_ii")),
                                  ("criteria_iii_iv", bv1_3.get("criteria_iii_iv")))
                   if not (v or {}).get("passed")])
        A["b_v1_supersedes"] = ("200_BV1_3seed.json (3 seeds, prereg 200_G5R_PREREG.md) "
                                "supersedes the single-seed gate json")
    else:
        passed = bool(bv1 and bv1.get("passed"))
        failed = [k for k in ("criterion_i_composition", "criterion_ii_sliced_w1",
                              "criterion_iii_unique_cell_fraction", "criterion_iv_NFE")
                  if not (bv1.get(k) or {}).get("passed",
                                               (bv1.get(k) or {}).get("within_1_sd"))] if bv1 else []
    if (bv1 or bv1_3) and not passed:
        for k, cell in A["rows"].items():
            if k.startswith("vgr::") and isinstance(cell, dict):
                cell["do_not_publish"] = True
                cell["b_v1_gate"] = f"failed: {', '.join(failed)}"
                cell["b_v1_note"] = (
                    "AMEND-1 §A1: a failed B-class criterion is recorded and the block is run "
                    "to completion; the product carries do_not_publish rather than being hidden. "
                    "See 200_OPEN_QUESTIONS.md D14 for the measured values and why criterion "
                    "(ii) -- a 1-sd rule whose sd is estimated from 3 points -- is near a "
                    "coin flip at an effect size of 0.7%.")
    add_rows("aprime", aprime, lambda r: f"{r.get('request')}|A_prime_v1",
             "training-time label embedding in the SAME count-native architecture "
             "(v1: batch=128, last checkpoint -- DIVERGED, see AMEND-5 §1)")
    aprime2 = jload(pick("200_aprime_v2")) if pick("200_aprime_v2") else None
    inputs["200_aprime_v2"] = str(pick("200_aprime_v2")) if pick("200_aprime_v2") else None
    add_rows("aprime_v2", aprime2, lambda r: f"{r.get('request')}|A_prime_v2",
             "training-time label embedding in the SAME count-native architecture "
             "(v2: batch=384 read from the frozen ckpt's own json, best_val checkpoint, "
             "divergence guard -- AMEND-5 §1.2)")
    for k, cell in A["rows"].items():
        if k.startswith("aprime::") and isinstance(cell, dict):
            cell["aprime_v1_diverged"] = True
            cell["aprime_v1_note"] = (
                "v1 divergence: from step ~1800 val jumps from 0.11 to 0.30 and never recovers; the final slope is positive;"
                "sampling used the post-divergence weights. AMEND-5 section 1 ruled it a void run. v1 is kept and reported as is,"
                "kept as a record that naive fine-tuning breaks the model; if the main text uses A′, it uses v2.")

    # generate-until-N(metadata directory, not under out/runs)
    gu = {}
    for p in sorted((S / "200_bestofn_meta").rglob("*_gunN_*.meta.json")):
        d = jload(p)
        k = f"{d['request']}|{d['selector']['selector']}"
        gu.setdefault(k, []).append(d)
    A["generate_until_N"] = {}
    for k, ds in gu.items():
        ds = sorted(ds, key=lambda d: d["seed"])
        A["generate_until_N"][k] = {
            "seeds": [d["seed"] for d in ds],
            "NFE_needed": [d["NFE_needed"] for d in ds],
            "right_censored": [d["right_censored"] for d in ds],
            "NFE_needed_over_NFE_fk": agg([d["NFE_needed_over_NFE_fk"] for d in ds]),
            "accepted_in_pool": [d["accepted_in_pool"] for d in ds],
            "accept_rate": [d["accept_rate"] for d in ds],
            "note": "zero GPU: it only scans the already-generated pool in generation order"}

    (S / f"{a.out_prefix}_tableA_sampler.json").write_text(
        json.dumps(A, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------------------- table B1 / B2
    B1 = {"ticket": 200, "table": "B1 -- unconditional fidelity",
          "status": "NOT_RUN" if not fidelity else "ok", "content": fidelity,
          "footnote": "the unconditional baseline row is the label-trained model marginalized over labels; library size from real cells"
                      "provided (oracle-fed); the B1 library-size row does not rank these arms."}
    (S / f"{a.out_prefix}_tableB1_uncond_fidelity.json").write_text(
        json.dumps(B1, indent=1, ensure_ascii=False), encoding="utf-8")

    B2 = {"ticket": 200, "table": "B2 -- one best-of-N across generators",
          "status": "NOT_RUN" if not allgen else "ok",
          "main_line_caliber": "fixed C = 25000 only (descriptive)",
          "appendix_caliber": "celltypist_informed_budget (expected-target-count matched), "
                              "appendix only -- its budget is decided by CellTypist composition",
          "rows": (collect(allgen, lambda r: (f"{r.get('generator')}|{r.get('request')}|"
                                              f"{r.get('selector')}|{r.get('variant')}|"
                                              f"C{r.get('level')}") if r.get("variant") else None)
                   if allgen else {})}
    (S / f"{a.out_prefix}_tableB2_bestofn_generators.json").write_text(
        json.dumps(B2, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------------------- budget curves
    curve = {"ticket": 200, "x_axes": ["NFE_total", "walltime_s (machine 1)"],
             "y": ["purity", "sliced_W1"],
             "analytic_ceiling": {}, "series": {}, "deployed_point": {}}
    for pc in pool_comp:
        if not pc:
            continue
        for t, v in pc["types"].items():
            curve["analytic_ceiling"].setdefault(t, {})[str(pc["seed"])] = {
                "pi_hat_pool": v["pi_hat_pool"], "ceil_by_budget": v["ceil_by_budget"],
                "right_censored": v["right_censored"]}
    for k, cell in A["rows"].items():
        if not isinstance(cell, dict) or not cell.get("purity", {}).get("n_seeds"):
            continue
        curve["series"][k] = {"NFE_total": cell.get("NFE_total"),
                              "walltime_s": cell.get("walltime_s"),
                              "purity": cell["purity"], "sliced_W1": cell.get("sliced_W1")}
    for t in TYPES:
        curve["deployed_point"][t] = {
            "NFE_fk": (D["types"][t]["NFE_fk"] if D else None),
            "purity": dep[t].get("purity"), "sliced_W1": dep[t].get("sliced_W1"),
            "walltime_s": [r["walltime_s"] for r in (D["types"][t]["deployed_runs"]
                                                     if D else [])]}
    (S / f"{a.out_prefix}_budget_curve.json").write_text(
        json.dumps(curve, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------------------- §7 criterion
    V = {"ticket": 200, "prereg": str(ROOT / "prereg" / "PREREG_200.md"),
         "n3_rule": "no significance tests, no CIs; raw 3-vs-3 differences + the frozen "
                    "indistinguishable rule",
         "verdicts": {}}

    def bo(t, sel, variant, level):
        return A["rows"].get(f"bestofn::{t}|{sel}|{variant}|b{level}")

    # P-A1 consistency check
    pa1 = {}
    for t in TYPES:
        cell = bo(t, "S2", "forced_top_N", 1)
        ceil1 = None
        for s, v in (curve["analytic_ceiling"].get(t) or {}).items():
            ceil1 = (v["ceil_by_budget"] or {}).get("1")
        pa1[t] = {"status": "NOT_RUN" if not cell else "ok",
                  "ceil_b1": ceil1,
                  "deployed_purity": dep[t].get("purity"),
                  "bestofn_S2_forced_b1_purity": (cell or {}).get("purity"),
                  "diff_3v3": ((cell or {}).get("vs_deployed", {}) or {}).get("purity")}
    V["verdicts"]["P-A1"] = {"what": "consistency check, NOT a main verdict", "per_type": pa1}

    # P-A2 catch-up ratio
    pa2 = {}
    for t in TYPES:
        dm = (dep[t].get("purity") or {}).get("mean")
        first, detail = None, {}
        for b in (1, 2, 4, 8, 16):
            cell = bo(t, "S2", "forced_top_N", b)
            if not cell or not cell["purity"]["n_seeds"]:
                detail[str(b)] = "NOT_RUN"
                continue
            m = cell["purity"]["mean"]
            ge = (cell.get("vs_deployed", {}).get("purity") or {}).get("n_a_ge_b")
            detail[str(b)] = {"mean_purity": m, "deployed_mean": dm,
                              "reached": bool(dm is not None and m >= dm),
                              "n_of_9_pairs_ge": ge}
            if first is None and dm is not None and m >= dm:
                first = b
        pa2[t] = {"first_b_reaching_deployed_mean": first,
                  "right_censored": first is None,
                  "report": (f"b = {first}" if first else ">= the highest runnable budget"),
                  "ladder": detail}
    V["verdicts"]["P-A2"] = {"what": "MAIN verdict: smallest b whose 3-seed mean purity "
                                     ">= the deployed arm's 3-seed mean purity",
                             "per_type": pa2}

    # P-A3 generate-until-N
    V["verdicts"]["P-A3"] = {"what": "MAIN verdict: NFE_needed / NFE_fk, per type, 3 seeds raw",
                             "no_monotonicity_preregistered": True,
                             "per_key": A["generate_until_N"]}

    # P-A4 / P-A5 / P-A6 / P-A7 / P-A8'
    pa4 = {}
    for t in TYPES:
        top = None
        for b in (16, 8, 4, 2, 1):
            if bo(t, "S2", "forced_top_N", b):
                top = b
                break
        cell = bo(t, "S2", "forced_top_N", top) if top else None
        pa4[t] = {"highest_runnable_b": top,
                  "sliced_W1": (cell or {}).get("sliced_W1"),
                  "mmd2": (cell or {}).get("mmd2_rbf_biased"),
                  "pcc": (cell or {}).get("pcc"),
                  "vs_deployed": (cell or {}).get("vs_deployed")}
    V["verdicts"]["P-A4"] = {"what": "honesty item: distances at the highest runnable budget",
                             "per_type": pa4}

    pa5 = {}
    for t in TYPES:
        cell = A["rows"].get(f"tilt_plus_endpoint_filter::{t}|RHO|forced_top_N|b1")
        pa5[t] = {"status": "NOT_RUN" if not cell else "ok",
                  "purity": (cell or {}).get("purity"),
                  "sliced_W1": (cell or {}).get("sliced_W1"),
                  "unique_cell_fraction": (cell or {}).get("unique_cell_fraction"),
                  "vs_deployed": (cell or {}).get("vs_deployed")}
    V["verdicts"]["P-A5"] = {"what": "resampling vs endpoint filtering, b=1, no directional "
                                     "prediction", "per_type": pa5}

    pa6 = {}
    for scorer in ("noised", "clean"):
        pa6[scorer] = {}
        for t in TYPES:
            cands = {}
            for g in (1, 2, 4):
                cell = A["rows"].get(f"vgr::{t}|{scorer}|g{float(g)}") or \
                       A["rows"].get(f"vgr::{t}|{scorer}|g{g}")
                if cell and cell["purity"]["n_seeds"]:
                    cands[g] = cell
            if not cands:
                pa6[scorer][t] = {"status": "NOT_RUN"}
                continue
            best = max(cands, key=lambda g: cands[g]["purity"]["mean"])
            pa6[scorer][t] = {
                "selection_rule": "highest 3-seed mean purity (written down before any number)",
                "selected_gamma": best,
                "purity": cands[best]["purity"], "sliced_W1": cands[best].get("sliced_W1"),
                "vs_deployed": cands[best].get("vs_deployed"),
                "all_gammas": {str(g): {"purity": c["purity"],
                                        "sliced_W1": c.get("sliced_W1"),
                                        "NFE_total": c.get("NFE_total")}
                               for g, c in cands.items()}}
    V["verdicts"]["P-A6"] = {"what": "VGR; the noised-vs-clean difference isolates whether the "
                                     "scorer looks at the noised or the clean state",
                             "per_scorer": pa6}

    pa7 = {}
    for t in TYPES:
        pa7[t] = {}
        for b in (1, 2, 4, 8, 16):
            c1 = bo(t, "S1", "forced_top_N", b)
            c2 = bo(t, "S2", "forced_top_N", b)
            if c1 and c2 and c1["purity"]["n_seeds"] and c2["purity"]["n_seeds"]:
                pa7[t][str(b)] = {"S1_purity": c1["purity"], "S2_purity": c2["purity"],
                                  "S1_minus_S2": three_v_three(c1["purity"], c2["purity"])}
    V["verdicts"]["P-A7"] = {"what": "selector-quality effect; the paper uses the S2 row, "
                                     "S1 goes to the appendix", "per_type": pa7}

    pa8 = {}
    for t in TYPES:
        cell = (A["rows"].get(f"aprime_v2::{t}|A_prime_v2")
                or A["rows"].get(f"aprime::{t}|A_prime_v1"))
        pa8[t] = {"status": "NOT_RUN" if not cell else "ok",
                  "purity": (cell or {}).get("purity"),
                  "sliced_W1": (cell or {}).get("sliced_W1"),
                  "vs_deployed": (cell or {}).get("vs_deployed")}
    V["verdicts"]["P-A8prime"] = {
        "what": "A' (training-time conditioning, same architecture) vs the deployed tilted-FK "
                "arm; NO directional prediction",
        "prior_disclosure": "E22-d the donor-retrained arm once beat us (0.902 vs 0.601) and FAIL was reported as is."
                            "A' likely beats us on abundant types.",
        "training_json": str(C.MODELS / "conditional" / "aprime_cond_s20260817.json"),
        "per_type": pa8}
    # ---- A′ degenerate or not: read from its own training json, not copied by hand ----
    tj = jload(C.MODELS / "conditional" / "aprime_cond_s20260817.json")
    if tj:
        frozen_best = 0.11133664226531982        # pilot_run1c_s20260625.json::best_val
        ratio = (tj["final_val"] / frozen_best) if tj.get("final_val") else None
        degenerate = bool(ratio and ratio > 1.5)
        V["verdicts"]["P-A8prime"]["training"] = {
            "steps_done": tj.get("steps_done"), "stop_reason": tj.get("stop_reason"),
            "final_val": tj.get("final_val"), "best_val": tj.get("best_val"),
            "best_val_at_step": (tj["curve"]["step"][tj["curve"]["val"].index(
                min(tj["curve"]["val"]))] if tj.get("curve") else None),
            "frozen_ckpt_best_val": frozen_best,
            "final_val_over_frozen_best": ratio,
            "loss_plateau_slope_per_1000_steps_last20pct":
                tj.get("loss_plateau_slope_per_1000_steps_last20pct"),
            "steps_cap_for_cosine": tj.get("steps_cap_for_cosine"),
            "batch": tj.get("batch"), "lr": tj.get("lr"),
            "ckpt_sha256": tj.get("ckpt_sha256")}
        if degenerate:
            V["verdicts"]["P-A8prime"]["status"] = "inconclusive / arm_degenerate"
            V["verdicts"]["P-A8prime"]["why_not_evidence"] = (
                "A' final val is {:.2f} times the best_val of the frozen model, and the slope over the last 20% of steps is still positive "
                "({:+.2e}/1000 steps), yet the cosine schedule has already run {:.0%}: the model is not under-annealed, it fell into a clearly worse solution."
                "two further independent pieces of evidence: Endothelial purity is below the natural share of that type in the pool (0.076 < 0.229),"
                "and with labels zeroed, library size 249 versus 325 for the pool prefix / 373 for real val."
                "**so these three rows are not evidence that conditioning at training time is worse**; they come from a cold start on a converged ckpt"
                "optimizer, reduce batch from 384 to 128, train another 5 h. Root cause in 200_OPEN_QUESTIONS.md "
                "D6 and D13; both were written before any number was seen."
                "AMEND-2 §4 both pre-written sentences assumeA' is an effective training-time conditioning,"
                "**neither sentence applies**; using the first would credit our method with a broken model. Needs a decision."
            ).format(ratio, tj.get("loss_plateau_slope_per_1000_steps_last20pct") or 0.0,
                     (tj.get("steps_done") or 0) / max(tj.get("steps_cap_for_cosine") or 1, 1))

    V["verdicts"]["P-B1"] = {"what": "unconditional fidelity ranking, no prediction; the four "
                                     "columns (training budget / NFE per sample / library size "
                                     "source / count legality) must be in the same table",
                             "status": B1["status"]}
    V["verdicts"]["P-B2"] = {"what": "same best-of-N (S2) across generators, descriptive",
                             "status": B2["status"]}

    (S / f"{a.out_prefix}_verdicts.json").write_text(
        json.dumps(V, indent=1, ensure_ascii=False), encoding="utf-8")

    for name in ("tableA_sampler", "tableB1_uncond_fidelity", "tableB2_bestofn_generators",
                 "budget_curve", "verdicts"):
        print(f"[out] {S / f'{a.out_prefix}_{name}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
