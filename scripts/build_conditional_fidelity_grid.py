# -*- coding: utf-8 -*-
"""Rebuild the training-by-sampling-seed grid of the conditional-fidelity table on the v8 ruler.

The structure matches the earlier grid file exactly; only the relative purity changes ruler.
W1, MMD^2 and PCC do not depend on the purity ruler and are carried over unchanged.

Three conventions are written out side by side: three training seeds with one sampling seed
each, one training seed with three sampling seeds, and the full grid. The variance is then
decomposed into a within-training-seed part and a between-training-seed part (ddof=1).

    python scripts/build_conditional_fidelity_grid.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.metrics import CANONICAL_CEILING, purity_from_hist          # noqa: E402

S = C.OUT / "summary"
V7 = C.MASTER_PURITY_V7
V8 = C.MASTER_PURITY_V8
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
REAL_ROW = "real validation cells"
OURS_SAMP0 = 20260987


def load_rulers():
    v7 = {t: set(v["fixed_labels"])
          for t, v in json.loads(V7.read_text(encoding="utf-8"))["types"].items()}
    v8 = {t: set(v["fixed_labels"])
          for t, v in json.loads(V8.read_text(encoding="utf-8"))["types"].items()}
    ceil = {}
    for t in TYPES:
        d = json.loads((C.CELLTYPIST / f"celltypist_{t}_ceiling_canonical.json")
                       .read_text(encoding="utf-8"))
        real = d["label_dists"][REAL_ROW]
        r7, _ = purity_from_hist(real, v7[t])
        r8, _ = purity_from_hist(real, v8[t])
        ceil[t] = {"v7": CANONICAL_CEILING[t],
                   "v8": CANONICAL_CEILING[t] * r8 / r7}
    return v7, v8, ceil


def scored_for(meth, tr, sa, ours_first_samp):
    """Return (path, json). Two naming conventions exist, an early one and the current one."""
    p = S / f"214g_{meth}_s{tr}_samp{sa}_scored.json"
    if p.exists():
        return p
    if meth == "ours" and sa == ours_first_samp:
        q = S / f"214_ours_s{tr}_scored.json"
        if q.exists():
            return q
    return None


def rows_of(p, t):
    d = json.loads(p.read_text(encoding="utf-8"))
    out = [r for r in d.get("rows", [])
           if (r.get("request") or r.get("arm")) == t and r.get("label_dist_whole")]
    out.sort(key=lambda r: not str(r.get("stem", "")).startswith(("ours_", "scvi_",
                                                                  "scanvi_")))
    return out


def summarise(M):
    ok = ~np.isnan(M)
    out = {"n_cells": int(ok.sum()), "complete": bool(ok.all())}
    if not ok.all():
        return out
    a, b, c = M[:, 0], M[0, :], M.ravel()
    out["A_3train_1samp"] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1))}
    out["B_1train_3samp"] = {"mean": float(b.mean()), "sd": float(b.std(ddof=1))}
    out["C_3x3"] = {"mean": float(c.mean()), "sd": float(c.std(ddof=1)), "n": int(c.size)}
    within = float(np.mean([r.var(ddof=1) for r in M]))
    between = float(M.mean(axis=1).var(ddof=1))
    out["variance_decomposition"] = {
        "within_training_seed_sampling_var": within,
        "between_training_seed_var": between,
        "ratio_between_over_within": (between / within) if within > 0 else None}
    out["matrix_rows_train_cols_samp"] = [[float(x) for x in r] for r in M]
    return out


def main() -> int:
    if not V8.exists():
        print(f"missing {V8} -> run scripts.build_lineage_purity_ruler first")
        return 2
    v7, v8, ceil = load_rulers()
    old = json.loads((S / "214_table2_grid_3x3.json").read_text(encoding="utf-8"))

    arms = [("ours", old["ours"]["train_seeds"], old["ours"]["sample_seeds"])]
    for m, node in old["baselines"].items():
        arms.append((m, node["train_seeds"], node["sample_seeds"]))

    grid, gmax, miss = {}, 0.0, []
    for meth, trains, samps in arms:
        cells = {}
        for tr in trains:
            for sa in samps:
                p = scored_for(meth, tr, sa, samps[0])
                if p is None:
                    miss.append(f"{meth} {tr}/{sa}")
                    continue
                for t in TYPES:
                    rs = rows_of(p, t)
                    if not rs:
                        miss.append(f"{meth} {tr}/{sa} {t}")
                        continue
                    r = rs[0]
                    dist = r["label_dist_whole"]
                    p7, _ = purity_from_hist(dist, v7[t])
                    p8, _ = purity_from_hist(dist, v8[t])
                    ref = float(r.get("purity_whole_set", r["purity"]))
                    gmax = max(gmax, abs(p7 - ref))
                    okey = f"{t}|{tr}|{sa}"
                    src = (old["ours"] if meth == "ours"
                           else old["baselines"][meth])["cells"].get(okey, {})
                    cells[okey] = {
                        "train_seed": tr, "sample_seed": sa, "request": t,
                        "purity_rel": p8 / ceil[t]["v8"],
                        "purity_rel_v7": p7 / ceil[t]["v7"],
                        "purity_abs": p8, "purity_abs_v7": p7,
                        "W1": src.get("W1"), "mmd2": src.get("mmd2"),
                        "pcc": src.get("pcc"),
                    }
        grid[meth] = {"train_seeds": trains, "sample_seeds": samps, "cells": cells}

    print(f"[G] v7 recomputed v7 against the stored json: max |delta| = json: max |delta| = {gmax:.3e}")
    if gmax != 0.0:
        print("[G] FAIL")
        return 1
    print("[G] PASS (bitwise)")
    if miss:
        print(f"[!] missing cells {len(miss)}: {miss[:8]}")

    blob = {"ticket": "215", "ruler": "master_purity_v8_lineage",
            "ceilings": {t: ceil[t] for t in TYPES},
            "types": TYPES,
            "ours": grid["ours"],
            "baselines": {k: v for k, v in grid.items() if k != "ours"},
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    gp = S / "215_table2_grid_3x3_v8.json"
    gp.write_text(json.dumps(blob, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # the three conventions
    calib = {"ticket": "215", "ruler": "v8", "arms": {}}
    print(f"\n{'arm':<9}{'request':<13}{'A 3train x1samp':<20}{'B 1train x3samp':<20}"
          f"{'C 3x3 (v8)':<20}{'C 3x3 (v7)':<20}{'betw/with'}")
    for meth in grid:
        node = grid[meth]
        calib["arms"][meth] = {"train_seeds": node["train_seeds"],
                               "sample_seeds": node["sample_seeds"], "by": {}}
        for t in TYPES:
            M8 = np.full((len(node["train_seeds"]), len(node["sample_seeds"])), np.nan)
            M7 = M8.copy()
            for i, tr in enumerate(node["train_seeds"]):
                for j, sa in enumerate(node["sample_seeds"]):
                    c = node["cells"].get(f"{t}|{tr}|{sa}")
                    if c:
                        M8[i, j] = c["purity_rel"]
                        M7[i, j] = c["purity_rel_v7"]
            e8, e7 = summarise(M8), summarise(M7)
            calib["arms"][meth]["by"][t] = {"purity_rel_v8": e8, "purity_rel_v7": e7}
            if not e8.get("complete"):
                print(f"{meth:<9}{t:<13}incomplete ({e8['n_cells']}/9)")
                continue
            r = e8["variance_decomposition"]["ratio_between_over_within"]
            print(f"{meth:<9}{t:<13}"
                  f"{e8['A_3train_1samp']['mean']:.4f}+-{e8['A_3train_1samp']['sd']:.4f}     "
                  f"{e8['B_1train_3samp']['mean']:.4f}+-{e8['B_1train_3samp']['sd']:.4f}     "
                  f"{e8['C_3x3']['mean']:.4f}+-{e8['C_3x3']['sd']:.4f}     "
                  f"{e7['C_3x3']['mean']:.4f}+-{e7['C_3x3']['sd']:.4f}     "
                  + (f"{r:.1f}x" if r else "n/a"))
        print()

    calib["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cp = S / "215_table2_three_calibres_v8.json"
    cp.write_text(json.dumps(calib, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[out] {gp}\n[out] {cp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
