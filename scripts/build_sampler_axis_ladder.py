# -*- coding: utf-8 -*-
"""Build the best-of-N budget ladder of the sampler-axis table.

The gate is checked first: its pass flag must be true, or the script exits without writing a
table, because a single inconsistent value among the sixty stops the delivery. Numbers are then
read from two scored jsons and never recomputed: the four arms from the published per-run
values and MDLM from the file scored for this round on the frozen ruler.

    python scripts/build_sampler_axis_ladder.py
"""
from __future__ import annotations

import os                                                            # noqa: E402

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"

import hashlib                                                       # noqa: E402
import json                                                          # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402

SUMMARY = C.OUT / "summary"
GATE = SUMMARY / "213_ladder_gate.json"
PUB = SUMMARY / "200_bestofn_all_generators_v3_scored.json"
MDLM = SUMMARY / "213_mdlm_scored.json"
ARMS = [("OURS", r"\textbf{Ours}"), ("scanvi", "scANVI"), ("cfgen", "CFGen"),
        ("scvi", "scVI"), ("mdlm", "MDLM")]
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
BUDGETS = [6250, 12500, 25000, 50000, 100000]
SEEDS = [20260931, 20260932, 20260933]
N_OUT_EXPECT = {"Endothelial": 10057, "Myeloid": 2500, "Neuronal": 2500}
BLOCK = {"Endothelial": r"\emph{abundant}, $80{,}463$ training cells; $n_{\mathrm{out}}=10{,}057$",
         "Myeloid": r"$18{,}422$ training cells; $n_{\mathrm{out}}=2{,}500$",
         "Neuronal": r"\emph{rare}, $3{,}168$ training cells; $n_{\mathrm{out}}=2{,}500$"}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def index(rows):
    out = {}
    for x in rows:
        if (x.get("variant") == "forced_top_N" and x.get("selector") == "S1"
                and x.get("axis") == "C"):
            out[(x["generator"], x["request"], x["level"], x["seed"])] = x
    return out


def stats(v):
    a = np.asarray(v, dtype=np.float64)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "n_seeds": int(a.size),
            "per_seed": [float(x) for x in a]}


def main() -> int:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    if not gate["passed"]:
        raise SystemExit("213_ladder_gate.json says the 60-number gate did NOT pass -- "
                         "per the ticket: stop, no table.")
    src = index(json.loads(PUB.read_text(encoding="utf-8"))["rows"])
    src.update(index(json.loads(MDLM.read_text(encoding="utf-8"))["rows"]))

    METRICS = [("purity", "purity"), ("W1", "W1"), ("MMD2", "mmd2_rbf_biased"),
               ("PCC", "pcc")]
    ladder, n_out, missing = {}, {}, []
    for g, _ in ARMS:
        ladder[g] = {}
        n_out[g] = {}
        for t in TYPES:
            ladder[g][t] = {}
            n_out[g][t] = {}
            for c in BUDGETS:
                rows = [src.get((g, t, c, s)) for s in SEEDS]
                if any(r is None for r in rows):
                    missing.append({"generator": g, "request": t, "budget": c})
                    continue
                cell = {}
                for lab, key in METRICS:
                    vals = [r.get(key) for r in rows]
                    cell[lab] = stats(vals) if all(
                        isinstance(v, (int, float)) for v in vals) else {
                        "mean": None, "sd": None, "per_seed": vals,
                        "note": "not scorable at this rung"}
                cell["stems"] = [r["stem"] for r in rows]
                ladder[g][t][str(c)] = cell
                nd = [int(r["n_delivered"]) for r in rows]
                n_out[g][t][str(c)] = {
                    "n_delivered_per_seed": nd,
                    "all_seeds_equal": len(set(nd)) == 1,
                    "n_out": nd[0] if len(set(nd)) == 1 else None,
                    "expected": N_OUT_EXPECT[t],
                    "matches_expected": len(set(nd)) == 1 and nd[0] == N_OUT_EXPECT[t]}
    if missing:
        raise SystemExit(f"missing ladder cells, refusing to build the table: {missing[:8]}")

    no = {"ticket": 213, "what": "actual n_out at every rung, from the run jsons",
          "expected": N_OUT_EXPECT, "arms": [g for g, _ in ARMS],
          "budgets": BUDGETS, "seeds": SEEDS, "per_arm": n_out,
          "all_rungs_match_expected": all(
              n_out[g][t][str(c)]["matches_expected"]
              for g, _ in ARMS for t in TYPES for c in BUDGETS),
          "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    p_no = SUMMARY / "213_ladder_n_out.json"
    if p_no.exists():
        raise SystemExit(f"refusing to overwrite {p_no}")
    p_no.write_text(json.dumps(no, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    out = {"ticket": 213, "block": 3,
           "what": "five-arm best-of-N ladder, S1 selector, forced-top-N, candidate "
                   "budgets C; the four published arms are the existing scored values, "
                   "MDLM is new in this ticket",
           "selector": "S1", "variant": "forced_top_N", "axis": "C",
           "arms": [g for g, _ in ARMS], "arm_order_note":
               "MDLM is appended LAST (ticket 213 block 3 rule 1)",
           "requests": TYPES, "budgets": BUDGETS, "eval_seeds": SEEDS,
           "sd_convention": "ddof=1 over the three sampling seeds",
           "sources": {"four_published_arms": str(PUB),
                       "four_published_arms_sha256": sha(PUB),
                       "MDLM": str(MDLM), "MDLM_sha256": sha(MDLM)},
           "mdlm_pools": json.loads(
               (SUMMARY / "213_pool_manifest.json").read_text(encoding="utf-8"))["pools"],
           "gate": {"file": str(GATE), "sha256": sha(GATE),
                    "n_compared": gate["n_compared"], "n_mismatch": gate["n_mismatch"],
                    "passed": gate["passed"],
                    "per_run": gate["per_run"]},
           "ladder": ladder,
           "n_out": no["per_arm"],
           "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    p_l = SUMMARY / "213_bestofn_ladder.json"
    if p_l.exists():
        raise SystemExit(f"refusing to overwrite {p_l}")
    p_l.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # emit the table
    def bold_set(t, c):
        v = {g: ladder[g][t][str(c)]["purity"]["mean"] for g, _ in ARMS}
        v = {g: x for g, x in v.items() if x is not None}
        if not v:
            return set()
        top = max(v.values())
        return {g for g in v
                if v[g] == top
                or (top - v[g]) <= (ladder[g][t][str(c)]["purity"]["sd"] or 0.0)}

    lines = []
    lines.append(f"% source: 213_bestofn_ladder.json  sha256={sha(p_l)[:16]}")
    lines.append(f"% selector S1, forced-top-N, candidate budgets C; means over seeds "
                 f"{SEEDS}, s.d. ddof=1")
    lines.append(f"% MDLM appended LAST in the arm list (ticket 213 block 3 rule 1); "
                 f"its pools are 3 x 100,000 with disjoint seed lineages "
                 f"(213_pool_manifest.json)")
    lines.append(f"% rng-coupling gate: {gate['per_run']['n_compared']} per-run and "
                 f"{gate['n_compared']} per-cell numbers of the four published arms, "
                 f"{gate['n_mismatch']} mismatch, passed={gate['passed']}")
    lines.append("% bold: best in the column, plus any arm within its own s.d. of the best")
    lines.append(r"\begin{tabular}{l" + "c" * len(BUDGETS) + "}")
    lines.append(r"\toprule")
    lines.append("Generator & " + " & ".join(
        f"$C={c:,}$".replace(",", "{,}") for c in BUDGETS) + r" \\")
    for t in TYPES:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{len(BUDGETS) + 1}}}{{l}}"
                     rf"{{\textbf{{{t}}} \; {BLOCK[t]}}} \\")
        for g, lab in ARMS:
            cells = []
            bs = {c: bold_set(t, c) for c in BUDGETS}
            for c in BUDGETS:
                e = ladder[g][t][str(c)]["purity"]
                if e["mean"] is None:
                    cells.append("--")
                    continue
                s = f"{e['mean']:.3f}"
                if g in bs[c]:
                    s = rf"\mathbf{{{s}}}"
                cells.append(f"${s} \\pm {e['sd']:.3f}$")
            lines.append(f"{lab:16s} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    tex = "\n".join(lines) + "\n"
    p_t = SUMMARY / "213_table8_rows.tex"
    if p_t.exists():
        raise SystemExit(f"refusing to overwrite {p_t}")
    p_t.write_text(tex, encoding="utf-8")

    print(f"[n_out] every rung matches the expected n_out: {no['all_rungs_match_expected']}")
    print(f"[out] {p_no}\n[out] {p_l}\n[out] {p_t}\n")
    print(tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
