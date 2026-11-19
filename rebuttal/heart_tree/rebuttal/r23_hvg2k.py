# -*- coding: utf-8 -*-
"""R23: HVG2K sampling steps K=128 vs the deployed K=32, val only.

Three runs, one at a time, in this order: Myeloid (tau 0.2), Neuronal (tau 0.2), Endothelial (tau 0.4). Each is the paper's Table 1
run 1 for that type: generator 20260625 (the frozen default checkpoint, sha asserted by run_one) x sampling seed 20260989, i.e. the
deployed run out/runs/table1_ours/<t>_a1/<stem>_seed20260989.json. Its own `_rerun_argv` is used with EXACTLY two tokens changed:
  --K 32 -> --K 128,   --experiment table1_ours -> --experiment rebuttal_R23   (nothing is written into the Table 1 folder).
--arm <t>_a1 and --ticket 163 are kept. Nothing in tools/, experiments/ or hvg/ is edited; this file only imports from them.

Config check (ticket 201's config_check logic with K in place of tau): every key of the new run json must equal the deployed one,
except the t201 EXCLUDE bookkeeping/measurement keys and the K-derived keys, which are pinned to their expected K=128 values:
  K 128, NFE 128, reward_evaluations_per_particle 128, generator_forward_passes_per_particle 255 (2K-1; deployed 63 at K=32).
Fields that only the new writer has must match t201 EXPECTED_ADDED. The stem must differ in exactly one token, K32 -> K128.
v8 purity: the R17 path (rebuttal/r17_fill.py score_v8 logic): tools.t200_score_deliveries --experiment X --out-tag TAG
--e21-device cpu --only-stems STEM (no --skip), then the t215_rescore_v8 row (ceilings() + purity_from_hist, gate G2).
Gate before any FK: the three deployed K=32 runs (table1_ours, seed 20260989) are re-scored on the same path and must reproduce
their recorded row in out/summary/215_purity_v8_rescore.json (file 200_selftest_table1_scored.json) exactly in purity_v8,
purity_v7, n_cells, rel_v8; else STOP. Outputs: rebuttal/out/R23/v8/<stem>_v8.json (K128 and K32), rebuttal/out/R23/r23_summary.json
(per type: K32/K128 purity_v8, wall time of this K128 run vs the deployed K32 walltime_s, peak_alloc_gb), log rebuttal/logs/r23.log.
Re-launchable: a type whose K128 v8 json exists is skipped; a K128 run json that already exists and passes the config check is
scored without re-running the FK. rebuttal/logs/PAUSE_NIGHT (night guard) is waited on before each FK.
    python -u rebuttal/r23_hvg2k.py
"""
from __future__ import annotations

import json, subprocess, sys, time
from pathlib import Path

HVG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HVG))
from hvg import config as C                                                    # noqa: E402
from hvg.metrics import purity_from_hist                                       # noqa: E402
from tools.t201_tau_sweep import EXCLUDE, EXPECTED_ADDED                       # noqa: E402
from tools.t215_rescore_v8 import ceilings                                     # noqa: E402

R = HVG / "rebuttal"; LOG = R / "logs"; OUT = R / "out" / "R23"; V8 = OUT / "v8"
EXP_NEW, EXP_DEP, SEED, K_NEW = "rebuttal_R23", "table1_ours", 20260989, 128
ORDER = ["Myeloid", "Neuronal", "Endothelial"]
TAU = {"Endothelial": 0.4, "Myeloid": 0.2, "Neuronal": 0.2}
K_PINNED = {"K": K_NEW, "NFE": K_NEW, "reward_evaluations_per_particle": K_NEW, "generator_forward_passes_per_particle": 2 * K_NEW - 1}
KEYS = ("purity_v8", "purity_v7", "n_cells", "rel_v8")


def note(m: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] R23 {m}"
    print(line, flush=True)
    with open(LOG / "r23.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def deployed(t: str) -> tuple[Path, dict]:
    hits = sorted((C.RUNS / EXP_DEP / f"{t}_a1").glob(f"*seed{SEED}.json"))
    if len(hits) != 1:
        raise SystemExit(f"deployed run for {t} seed {SEED}: {len(hits)} hits")
    return hits[0], json.loads(hits[0].read_text(encoding="utf-8"))


def build_argv(dep: dict) -> list[str]:
    argv, out, i, seen = list(dep["_rerun_argv"]), [], 0, set()
    while i < len(argv):
        a = argv[i]
        if a == "--K":
            assert argv[i + 1] == "32", argv[i + 1]; out += ["--K", str(K_NEW)]; seen.add(a); i += 2; continue
        if a == "--experiment":
            assert argv[i + 1] == EXP_DEP, argv[i + 1]; out += ["--experiment", EXP_NEW]; seen.add(a); i += 2; continue
        out.append(a); i += 1
    assert seen == {"--K", "--experiment"}, seen
    return out


def config_check(new: dict, dep: dict) -> dict:
    viol = {}
    for k in sorted(set(new) | set(dep)):
        if k in EXCLUDE:
            continue
        if k in K_PINNED:
            if new.get(k) != K_PINNED[k]:
                viol[k] = {"run": new.get(k), "expected": K_PINNED[k], "deployed": dep.get(k)}
            continue
        if k in new and k in dep:
            if new[k] != dep[k]:
                viol[k] = {"run": new[k], "deployed": dep[k]}
        elif k in new:
            if k not in EXPECTED_ADDED or new[k] != EXPECTED_ADDED[k]:
                viol[k] = {"run": new[k], "deployed": "<field absent>", "expected": EXPECTED_ADDED.get(k, "__UNEXPECTED__")}
        else:
            viol[k] = {"run": "<field absent>", "deployed": dep[k]}
    a, b = new.get("stem", "").split("_"), dep.get("stem", "").split("_")
    diff = [(x, y) for x, y in zip(a, b) if x != y] if len(a) == len(b) else [("LEN", "LEN")]
    stem_ok = len(a) == len(b) and diff == [(f"K{K_NEW}", "K32")]
    return {"violations": viol, "stem_token_diff": diff, "stem_only_K_differs": stem_ok, "ok": bool(not viol and stem_ok)}


def score_v8(experiment: str, t: str, stem: str, tag: str, cal: dict):
    """t200_score_deliveries as t201_tau_score._score / r17_fill.score_v8 call it, then the t215_rescore_v8 row."""
    sc = [sys.executable, "-u", "-m", "tools.t200_score_deliveries", "--experiment", experiment, "--out-tag", tag,
          "--e21-device", "cpu", "--only-stems", stem]
    with open(LOG / f"{tag}_score.log", "w", encoding="utf-8") as f:
        rc = subprocess.run(sc, cwd=HVG, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        note(f"{tag} score exit {rc}"); return None, None
    sp = sorted((C.OUT / "summary").glob(f"{tag}_scored*.json"), key=lambda p: p.stat().st_mtime)[-1]
    rows = [r for r in json.loads(sp.read_text(encoding="utf-8")).get("rows", []) if r.get("stem") == stem]
    if len(rows) != 1 or not rows[0].get("label_dist_whole"):
        note(f"{tag} scored rows {len(rows)} / no label_dist_whole in {sp.name}"); return None, None
    r = rows[0]; e = cal[t]; dist = r["label_dist_whole"]
    p7, n = purity_from_hist(dist, e["labels_v7"]); p8, _ = purity_from_hist(dist, e["labels_v8"])
    ref = r.get("purity_whole_set");  ref = r.get("purity", r.get("purity_matched_n")) if ref is None else ref
    if p8 < p7:
        note(f"{tag} G2 FAIL v8 {p8} < v7 {p7}"); return None, None
    return ({"file": sp.name, "stem": stem, "request": t, "generator": r.get("generator"), "seed": r.get("seed"),
             "n_cells": n, "purity_v7": p7, "purity_v7_stored": float(ref), "purity_v8": p8,
             "rel_v7": p7 / e["ceiling_v7_published"], "rel_v8": p8 / e["ceiling_v8_derived"],
             "rel_v7_stored": float(ref) / e["ceiling_v7_published"],
             "g1_v7_recompute_abs_delta": abs(p7 - float(ref))}, sp)


def k32_gate(cal: dict) -> dict | None:
    resc = json.loads((C.OUT / "summary" / "215_purity_v8_rescore.json").read_text(encoding="utf-8"))
    res = {}
    for t in ORDER:
        p = V8 / f"{deployed(t)[1]['stem']}_v8.json"
        if p.exists():
            res[t] = json.loads(p.read_text(encoding="utf-8")); continue
        dep_p, dep = deployed(t)
        rec = [r for r in resc["rows"] if r.get("file") == "200_selftest_table1_scored.json" and r.get("stem") == dep["stem"]]
        row, sp = score_v8(EXP_DEP, t, dep["stem"], f"r23_k32_{t}_s{SEED}", cal)
        ok = len(rec) == 1 and row is not None and all(row[k] == rec[0][k] for k in KEYS)
        note(f"K32 gate {t}: {'OK' if ok else 'FAILED'} recorded {({k: rec[0][k] for k in KEYS}) if len(rec) == 1 else len(rec)} "
             f"rescored {({k: row[k] for k in KEYS}) if row else None}")
        if not ok:
            return None
        row |= {"K": 32, "r23": {"run_json": str(dep_p), "scored_json": str(sp), "recorded_in": "out/summary/215_purity_v8_rescore.json "
                "(file 200_selftest_table1_scored.json)", "compared_exactly": list(KEYS), "walltime_s": dep.get("walltime_s"),
                "peak_alloc_gb": dep.get("peak_alloc_gb")}}
        p.write_text(json.dumps(row, indent=1, ensure_ascii=False), encoding="utf-8"); res[t] = row
    return res


def main() -> int:
    V8.mkdir(parents=True, exist_ok=True)
    cal = ceilings()
    k32 = k32_gate(cal)
    if k32 is None:
        note("K32 re-score gate not ok, STOP (no FK started)"); return 1
    summ = {}
    for t in ORDER:
        dep_p, dep = deployed(t)
        stem_new = dep["stem"].replace("_K32_", f"_K{K_NEW}_")
        vp = V8 / f"{stem_new}_v8.json"
        if vp.exists():
            summ[t] = json.loads(vp.read_text(encoding="utf-8")); continue
        arm_dir = C.RUNS / EXP_NEW / f"{t}_a1"
        hits = sorted(arm_dir.glob(f"*K{K_NEW}*seed{SEED}.json")) if arm_dir.exists() else []
        wall = None
        if not hits:
            while (LOG / "PAUSE_NIGHT").exists():
                time.sleep(30)
            argv = build_argv(dep)
            note(f"{t} FK start: python {' '.join(argv)}")
            t0 = time.time()
            with open(LOG / f"r23_{t}_K{K_NEW}.log", "w", encoding="utf-8") as f:
                rc = subprocess.run([sys.executable, "-u", *argv], cwd=HVG, stdout=f, stderr=subprocess.STDOUT).returncode
            wall = time.time() - t0
            if rc != 0:
                note(f"{t} FK exit {rc} after {wall:.0f}s, STOP"); return 1
            hits = sorted(arm_dir.glob(f"*K{K_NEW}*seed{SEED}.json"))
        if len(hits) != 1:
            note(f"{t} expected 1 K{K_NEW} run json, found {len(hits)}, STOP"); return 1
        new = json.loads(hits[0].read_text(encoding="utf-8")); chk = config_check(new, dep)
        if not chk["ok"]:
            note(f"{t} config_check FAILED {json.dumps(chk, ensure_ascii=False)[:600]}, STOP"); return 1
        note(f"{t} FK ok wall {wall if wall is None else round(wall)}s walltime_s {new.get('walltime_s'):.0f} "
             f"(K32 deployed {dep.get('walltime_s'):.0f}) peak_alloc_gb {new.get('peak_alloc_gb')} config_check ok")
        row, sp = score_v8(EXP_NEW, t, new["stem"], f"r23_k128_{t}_s{SEED}", cal)
        if row is None:
            note(f"{t} scoring failed, STOP"); return 1
        row |= {"K": K_NEW, "tau": TAU[t], "r23": {"run_json": str(hits[0]), "deployed_run_json": str(dep_p), "config_check": chk,
                "scored_json": str(sp), "script": "rebuttal/r23_hvg2k.py", "walltime_s": new.get("walltime_s"),
                "walltime_s_K32_deployed": dep.get("walltime_s"), "peak_alloc_gb": new.get("peak_alloc_gb"),
                "peak_alloc_gb_K32_deployed": dep.get("peak_alloc_gb")}}
        vp.write_text(json.dumps(row, indent=1, ensure_ascii=False), encoding="utf-8"); summ[t] = row
        note(f"{t} K{K_NEW} v8 ok purity_v8 {row['purity_v8']:.4f} (K32 {k32[t]['purity_v8']:.4f}) rel_v8 {row['rel_v8']:.4f}")
    out = {t: {"tau": TAU[t], "generator": 20260625, "sampling_seed": SEED,
               "purity_v8_K32": k32[t]["purity_v8"], "purity_v8_K128": summ[t]["purity_v8"],
               "rel_v8_K32": k32[t]["rel_v8"], "rel_v8_K128": summ[t]["rel_v8"],
               "walltime_s_K32": summ[t]["r23"]["walltime_s_K32_deployed"], "walltime_s_K128": summ[t]["r23"]["walltime_s"],
               "peak_alloc_gb_K32": summ[t]["r23"]["peak_alloc_gb_K32_deployed"], "peak_alloc_gb_K128": summ[t]["r23"]["peak_alloc_gb"],
               "run_json_K128": summ[t]["r23"]["run_json"]} for t in ORDER}
    (OUT / "r23_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    note("all 3 done; summary rebuttal/out/R23/r23_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
