# -*- coding: utf-8 -*-
"""ticket 185 B: MMD² / PCC / SCC with the intersection definition. **Thin wrapper, no metric logic.**

all metric functions are imported from the frozen copies, not a single line copied:

  * `_frozen/e21_scc_mmd.py`:`cp10k_log1p`,`exact_bandwidth`,`kernel_mean`,
    `mmd2_biased`,`select_rows`,`score`
  * `_frozen/e21c_pcc.py`:`corr_pair`
  * `analysis/score_intersect.py`:`build_reference` — reference/floor **archived split**
    (SPLIT_SEED 20261072, gene 128, 853 → A/B 426 each), not re-split here.

this file is responsible for only three things: replacing the reference population, **computing σ from the reference itself**, and feeding samples.

### σ definition (sections 3.1–3.2)

`mmd2_biased(real=ref, ...)` in `real` is the reference used for scoring (half A, 426 cells),
so the σ used in production must come from **the same cells**:`sigma = exact_bandwidth(ref_gpu)`.
**no generated cell is used for σ**, and **the σ of bare endothelial cells is not borrowed**.
also compute σ over all 853 reference cells and store it in the json, for reference only, not used in scoring.

substitute gate (no reproducible archived σ, so it is a self-consistency gate):
  1. the σ of half A and half B, relative difference < 5e-2;
  2. use half B against half A to compute real–real MMD²/PCC/SCC as the floor; require MMD² > 0 and
     far below any generated arm, PCC > 0.99.

    python -m analysis.score_e21_intersect
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hvg import config as C                                          # noqa: E402
from hvg.io import load_runs                                         # noqa: E402
from analysis.score_intersect import build_reference, GENE, SPLIT_SEED, SUB_SEED  # noqa: E402

FROZEN = ROOT / "analysis" / "_frozen"
SCORE_E21 = ROOT / "analysis" / "score_e21.py"
FROZEN_SHA = {
    "score_e21.py": "09c0636950abb2d76089ed5c8ec407fe9dfce24303e11481b3ee4dc5f57e1b0a",
    "e21_scc_mmd.py": "e1de4039c8a657dbc025a2e64b5f9439b88f5672a805fe8fd67bbc15a0d8f9d4",
    "e21c_pcc.py": "83b19b11871875d977c8c81bd22e5cbcc5108db83fd5a7d052a92b3dc46a40dc",
}
PANEL_A = C.OUT / "summary" / "intersect_panelA_three_arms.json"
E50 = (ROOT.parent.parent / "Conditional generation experiments" / "HVG2K"
       / "universality_fk" / "out" / "e50" / "E50_intersection_three_routes.json")
OUR_ARMS = [("tilt_only", "Tilt Sampler"), ("fk_only", "Residual Correction"),
            ("a1", "Tilt Sampler with Residual Correction")]
ROUTES = ["A:scVI", "A:scANVI", "A:CFGen", "B:scVI", "B:scANVI", "B:CFGen"]
SIGMA_REL_TOL = 5.0e-2
PCC_FLOOR_MIN = 0.99


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def guard() -> dict:
    got = {"score_e21.py": sha256_file(SCORE_E21),
           "e21_scc_mmd.py": sha256_file(FROZEN / "e21_scc_mmd.py"),
           "e21c_pcc.py": sha256_file(FROZEN / "e21c_pcc.py")}
    bad = [k for k, v in got.items() if v != FROZEN_SHA[k]]
    if bad:
        raise SystemExit(f"frozen files changed: {bad} -- STOP")
    return got


def load_frozen():
    # A class:`e21c_pcc.py` contains a sibling statement `import e21_scc_mmd as e21`, so `_frozen/` must
    # on sys.path, and the module must be registered under its own name. Neither file is modified.
    if str(FROZEN) not in sys.path:
        sys.path.insert(0, str(FROZEN))
    mods = {}
    for name in ("e21_scc_mmd", "e21c_pcc"):
        spec = importlib.util.spec_from_file_location(name, FROZEN / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        mods[name] = m
    return mods["e21_scc_mmd"], mods["e21c_pcc"]


def ms(v):
    a = np.asarray(v, dtype=np.float64)
    return {"mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "n": int(a.size), "per_seed": [float(x) for x in a]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(C.OUT / "summary" / "e21_intersect.json"))
    a = ap.parse_args(argv)
    import torch

    sha = guard()
    E, PCC = load_frozen()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    panel = json.loads(PANEL_A.read_text(encoding="utf-8"))
    matched_n = int(panel["reference"]["matched_n"])

    # ---- collect all rows to score (sample sizes first; build_reference uses them to fix n)--------
    jobs = []          # (group, route, arm, seed, npy_path, n_rows, extra)
    for arm, label in OUR_ARMS:
        for p in sorted((C.RUNS / "intersect_ladder" / arm).glob("*.json")):
            npy = p.with_suffix(".npy")
            d = json.loads(p.read_text(encoding="utf-8"))
            jobs.append({"group": "ours", "route": None, "arm": arm, "label": label,
                         "seed": int(d["seed"]), "npy": npy,
                         "n_rows": int(np.load(npy, mmap_mode="r").shape[0]),
                         "stem": d["stem"]})
    e50 = json.loads(E50.read_text(encoding="utf-8"))
    for key in ROUTES:
        members = [k for k in e50["rows"] if k == key or k.startswith(key + "_seed")]
        for m in sorted(members):
            r = e50["rows"][m]
            npy = Path(r["npy"])
            job = {"group": "baseline_route", "route": key, "arm": r["arm"],
                   "label": key, "seed": (int(m.split("seed")[-1]) if "seed" in m else None),
                   "npy": npy, "stem": m,
                   "archived_npy_sha256": r.get("npy_sha256"),
                   "archived_n": r.get("n")}
            if not npy.is_file():
                job["status"] = "MISSING delivered sample"
                job["n_rows"] = 0
            else:
                job["n_rows"] = int(np.load(npy, mmap_mode="r").shape[0])
                got = sha256_file(npy)
                job["actual_npy_sha256"] = got
                job["npy_sha256_matches_archive"] = (got == r.get("npy_sha256"))
            jobs.append(job)

    usable = [j for j in jobs if j.get("status") is None and j["n_rows"] >= matched_n]
    skipped = [j for j in jobs if j not in usable]

    # ---- reference and floor: archived split (build_reference), n fixed to that of Panel A matched_n ----
    R = build_reference([matched_n])
    if R["n"] != matched_n:
        raise SystemExit(f"reference n {R['n']} != Panel A matched_n {matched_n}")
    ref_np, _z = E.cp10k_log1p(R["ref"])
    flo_raw = R["flo"]
    ref_gpu = torch.from_numpy(ref_np).to(dev)

    # ---- σ: computed from the reference itself, never including generated cells ----------------------------------
    sigma = E.exact_bandwidth(ref_gpu)
    flo_np, _ = E.cp10k_log1p(flo_raw)
    flo_gpu = torch.from_numpy(flo_np).to(dev)
    sigma_B = E.exact_bandwidth(flo_gpu)
    rel = abs(sigma - sigma_B) / max(sigma, sigma_B)
    # reference value: σ over all 853 reference cells (not used in scoring)
    D = C.LEGACY_DATA / "all"
    Xv = np.load(D / "val.npy", mmap_mode="r")
    yv = np.load(D / "val_celltype.npy", allow_pickle=True).astype(str)
    sel = np.flatnonzero((yv == "Endothelial") & (np.asarray(Xv)[:, GENE] > 0))
    all_np, _ = E.cp10k_log1p(np.asarray(Xv[sel], dtype=np.float64))
    sigma_all = E.exact_bandwidth(torch.from_numpy(all_np).to(dev))
    print(f"[sigma] half A (reference, n={matched_n}) = {sigma:.9f}")
    print(f"[sigma] half B (floor,     n={matched_n}) = {sigma_B:.9f}")
    print(f"[sigma] relative difference = {rel:.6e}   gate < {SIGMA_REL_TOL}  "
          f"{'PASS' if rel < SIGMA_REL_TOL else 'FAIL'}")
    print(f"[sigma] all {len(sel)} reference cells (reported, NOT used) = {sigma_all:.9f}")

    k_rr = E.kernel_mean(ref_gpu, ref_gpu, sigma)
    floor = E.score(ref_np, ref_gpu, flo_raw, sigma, k_rr)
    f_scc, f_pcc = PCC.corr_pair(ref_np.mean(0, dtype=np.float64),
                                 flo_np.mean(0, dtype=np.float64))
    floor.update({"pcc": f_pcc, "scc_from_corr_pair": f_scc, "n": matched_n})
    print(f"[floor] real-real  MMD2={floor['mmd2_rbf_biased']:.8f}  "
          f"PCC={f_pcc:.8f}  SCC={floor['scc']:.8f}")

    # ---- score row by row ---------------------------------------------------------
    rows = []
    for j in usable:
        rng = np.random.default_rng(E.SEED)     # every row is independent; a row number does not depend on its position in the list
        arr = np.load(j["npy"], mmap_mode="r")
        gen = E.select_rows(np.asarray(arr), matched_n, rng)
        s = E.score(ref_np, ref_gpu, gen, sigma, k_rr)
        gen_np, _ = E.cp10k_log1p(gen)
        scc2, pcc = PCC.corr_pair(ref_np.mean(0, dtype=np.float64),
                                  gen_np.mean(0, dtype=np.float64))
        if abs(scc2 - s["scc"]) > 1e-12:
            raise SystemExit(f"SCC disagrees between the two frozen paths for {j['stem']}: "
                             f"{s['scc']} vs {scc2}")
        row = {k: j.get(k) for k in
               ("group", "route", "arm", "label", "seed", "stem",
                "archived_npy_sha256", "actual_npy_sha256",
                "npy_sha256_matches_archive", "archived_n")}
        row.update({"npy": str(j["npy"]), "n_in": j["n_rows"], "matched_n": matched_n,
                    "mmd2_rbf_biased": s["mmd2_rbf_biased"], "pcc": pcc, "scc": s["scc"],
                    "zero_library_rows": s["zero_library_rows"],
                    "kernel_terms": s["kernel_terms"]})
        rows.append(row)
        print(f"  {str(j['route'] or j['arm']):<12} {str(j['seed']):>10}  "
              f"n_in={j['n_rows']:<6} MMD2={s['mmd2_rbf_biased']:.6f}  "
              f"PCC={pcc:.6f}  SCC={s['scc']:.6f}")

    # ---- aggregate by arm/route ----------------------------------------------------
    agg = []
    for arm, label in OUR_ARMS:
        rr = [r for r in rows if r["group"] == "ours" and r["arm"] == arm]
        if rr:
            agg.append({"group": "ours", "route": None, "arm": arm, "label": label,
                        "n_seeds": len(rr), "seeds": [r["seed"] for r in rr],
                        "matched_n": matched_n,
                        "mmd2": ms([r["mmd2_rbf_biased"] for r in rr]),
                        "pcc": ms([r["pcc"] for r in rr]),
                        "scc": ms([r["scc"] for r in rr])})
    for key in ROUTES:
        rr = [r for r in rows if r["route"] == key]
        if rr:
            agg.append({"group": "baseline_route", "route": key, "arm": rr[0]["arm"],
                        "label": key, "n_seeds": len(rr),
                        "seeds": [r["seed"] for r in rr], "matched_n": matched_n,
                        "mmd2": ms([r["mmd2_rbf_biased"] for r in rr]),
                        "pcc": ms([r["pcc"] for r in rr]),
                        "scc": ms([r["scc"] for r in rr]),
                        "any_npy_sha256_mismatch": any(
                            r.get("npy_sha256_matches_archive") is False for r in rr)})

    # ---- A block: archived mW1, read-only --------------------------------------------
    mw1 = {}
    for key in ROUTES:
        members = sorted(k for k in e50["rows"] if k == key or k.startswith(key + "_seed"))
        vals = []
        for m in members:
            dist = e50["rows"][m].get("distances") or {}
            vals.append({"row": m,
                         "marginal_w1_mean": dist.get("marginal_w1_mean"),
                         "marginal_w1_median": dist.get("marginal_w1_median"),
                         "matched_n": dist.get("matched_n")})
        present = [v["marginal_w1_mean"] for v in vals if v["marginal_w1_mean"] is not None]
        mw1[key] = {"members": vals,
                    "marginal_w1_mean_over_members": ms(present) if present else None,
                    "matched_n": sorted({v["matched_n"] for v in vals}),
                    "all_members_have_marginal_w1_mean": len(present) == len(vals)}

    gate1 = {"sigma_half_A": sigma, "sigma_half_B": sigma_B, "relative_difference": rel,
             "tolerance": SIGMA_REL_TOL, "passed": bool(rel < SIGMA_REL_TOL),
             "split": "the archived split (SPLIT_SEED %d, gene %d), taken from "
                      "score_intersect.build_reference; not re-cut here"
                      % (SPLIT_SEED, GENE)}
    min_gen = min((r["mmd2_rbf_biased"] for r in rows), default=None)
    gate2 = {"floor": floor, "min_generated_mmd2": min_gen,
             "floor_mmd2_positive": floor["mmd2_rbf_biased"] > 0,
             "floor_mmd2_below_every_arm": (min_gen is not None
                                            and floor["mmd2_rbf_biased"] < min_gen),
             "floor_pcc": f_pcc, "pcc_threshold": PCC_FLOOR_MIN,
             "floor_pcc_passed": f_pcc > PCC_FLOOR_MIN}
    gate2["passed"] = bool(gate2["floor_mmd2_positive"]
                           and gate2["floor_mmd2_below_every_arm"]
                           and gate2["floor_pcc_passed"])
    domain = [r["stem"] for r in rows
              if r["mmd2_rbf_biased"] < 0 or not (-1 <= r["pcc"] <= 1)
              or not (-1 <= r["scc"] <= 1)]

    out = {
        "produced_by": "analysis/score_e21_intersect.py",
        "wrapper_sha256": sha256_file(Path(__file__)),
        "frozen_sha256": sha,
        "no_metric_logic_here": "cp10k_log1p / exact_bandwidth / kernel_mean / mmd2_biased / "
                                "select_rows / score come from _frozen/e21_scc_mmd.py and "
                                "corr_pair from _frozen/e21c_pcc.py; build_reference comes "
                                "from analysis/score_intersect.py. Nothing is re-implemented.",
        "device": dev,
        "reference": {"definition": "held-out (val) endothelial with UAP1 (gene %d) > 0" % GENE,
                      "n_total": R["n_total"], "half_sizes": R["half_sizes"],
                      "matched_n": matched_n, "split_seed": SPLIT_SEED,
                      "subsample_seed": SUB_SEED, "data": R["data"],
                      "source_of_matched_n": str(PANEL_A)},
        "sigma_not_borrowed": True,
        "bandwidth": {
            "sigma_used": sigma,
            "computed_on": "the scoring reference itself (half A, n=%d), transformed with the "
                           "frozen cp10k_log1p; NO generated cell takes part, and the bare "
                           "Endothelial sigma is NOT borrowed" % matched_n,
            "definition": "_frozen/e21_scc_mmd.exact_bandwidth: exact median of the "
                          "off-diagonal real-real distances",
            "sigma_half_B": sigma_B,
            "sigma_all_reference_cells": sigma_all,
            "sigma_all_n": int(len(sel)),
            "sigma_all_note": "reported for context only; the scoring uses sigma_used, "
                              "because mmd2_biased's `real` argument is the same half-A "
                              "reference"},
        "gate_sigma_stability": gate1,
        "gate_floor_sanity": gate2,
        "domain_violations": domain,
        "row_rng": "each row gets its own default_rng(%d) for select_rows, so a row's value "
                   "does not depend on its position in the list" % E.SEED,
        "rows": rows,
        "by_arm": agg,
        "skipped": [{"stem": j.get("stem"), "route": j.get("route"), "arm": j.get("arm"),
                     "n_rows": j.get("n_rows"), "matched_n": matched_n,
                     "reason": j.get("status") or
                               f"n_rows {j.get('n_rows')} < matched_n {matched_n}"}
                    for j in skipped],
        "marginal_w1_from_archive": {
            "source": str(E50), "source_sha256": sha256_file(E50),
            "read_only": True,
            "note": "read from the archived table, not recomputed",
            "matched_n_note": "the archived mW1 is at matched_n 853; the MMD2/PCC above are "
                              "at matched_n %d. The two columns are NOT on the same n."
                              % matched_n,
            "routes": mw1},
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    p = Path(a.out)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n",
                 encoding="utf-8")
    print(f"\n[gate1 sigma stability] {'PASS' if gate1['passed'] else 'FAIL'}")
    print(f"[gate2 floor sanity]     {'PASS' if gate2['passed'] else 'FAIL'}")
    print(f"[domain violations]      {domain if domain else 'none'}")
    print(f"[skipped rows]           {len(out['skipped'])}")
    print(f"[out] {p}\n[out] sha256 {sha256_file(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
