# -*- coding: utf-8 -*-
"""Assemble the MDLM conditional runs into the training-by-sampling grid (nine cells).

Numbers are read from json only: the first training seed from the existing scored artefact,
which is not rescored, and the other two from their own scored files. The three training seeds
share one set of sampling seeds. Both purity conventions are written out, the frozen v7 one for
the appendix and the lineage v8 one for the main text.

    python scripts/build_grid_mdlm.py
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
OUT = S / "216_mdlm_grid_3x3.json"
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
SAMP = [20260931, 20260932, 20260933]
REAL_ROW = "real validation cells"

SOURCES = [(20260625, "208_mdlm_cond_scored.json", "ticket 208 (seed 1, not re-scored)"),
           (20260934, "216_mdlm_cond_s20260934_scored.json", "ticket 216"),
           (20260935, "216_mdlm_cond_s20260935_scored.json", "ticket 216")]


def rulers():
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
        pub = CANONICAL_CEILING[t]
        ceil[t] = {"v7": pub, "v8": pub * r8 / r7}
    return v7, v8, ceil


def main() -> int:
    v7, v8, ceil = rulers()
    cells, gmax, missing = {}, 0.0, []
    for ts, fname, prov in SOURCES:
        p = S / fname
        if not p.exists():
            missing.append(fname)
            continue
        blob = json.loads(p.read_text(encoding="utf-8"))
        for r in blob.get("rows", []):
            t = r.get("request") or r.get("arm")
            if t not in TYPES:
                continue
            sa = r.get("seed")
            dist = r.get("label_dist_whole")
            if not dist:
                missing.append(f"{fname}:{r.get('stem')} no label_dist_whole")
                continue
            p7, _ = purity_from_hist(dist, v7[t])
            p8, _ = purity_from_hist(dist, v8[t])
            ref = float(r.get("purity_whole_set", r.get("purity")))
            gmax = max(gmax, abs(p7 - ref))
            cells[f"{t}|{ts}|{sa}"] = {
                "request": t, "train_seed": ts, "sample_seed": sa,
                "provenance": prov, "source_file": fname, "stem": r.get("stem"),
                "purity_abs_v7": p7, "purity_abs_v8": p8,
                "purity_rel_v7": p7 / ceil[t]["v7"], "purity_rel_v8": p8 / ceil[t]["v8"],
                "W1": r.get("W1"), "mmd2": r.get("mmd2_rbf_biased"), "pcc": r.get("pcc"),
                "sliced_W1": r.get("sliced_W1"), "scc": r.get("scc"),
                "n_delivered": r.get("n_delivered")}
    if missing:
        print("missing inputs:")
        for m in missing:
            print("   ", m)
    print(f"\n[G] v7 recomputed vs the value stored in each scored json: {len(cells)} cells, "
          f"max |delta| = {gmax:.3e}")
    if cells and gmax != 0.0:
        print("[G] FAIL")
        return 1
    if len(cells) != 27:
        # an incomplete grid is NOT written: a half-finished artefact is more dangerous than none,
        print(f"[!] cells only; finish the sampling and scoring first, nothing written this time {OUT.name}")
        return 3
    print("[G] PASS (bitwise)")

    trains = sorted({c["train_seed"] for c in cells.values()})
    summary = {}
    for t in TYPES:
        for key, field in [("purity_abs_v8", "purity_abs_v8"),
                           ("purity_rel_v8", "purity_rel_v8"),
                           ("purity_abs_v7", "purity_abs_v7"),
                           ("purity_rel_v7", "purity_rel_v7"),
                           ("W1", "W1"), ("mmd2", "mmd2"), ("pcc", "pcc")]:
            M = np.full((len(trains), len(SAMP)), np.nan)
            for i, tr in enumerate(trains):
                for j, sa in enumerate(SAMP):
                    c = cells.get(f"{t}|{tr}|{sa}")
                    if c and isinstance(c.get(field), (int, float)):
                        M[i, j] = c[field]
            ok = ~np.isnan(M)
            e = {"n_cells": int(ok.sum()), "complete": bool(ok.all()),
                 "matrix_rows_train_cols_samp": [[None if np.isnan(x) else float(x)
                                                  for x in row] for row in M]}
            if ok.all():
                flat = M.ravel()
                e["C_3x3"] = {"mean": float(flat.mean()), "sd": float(flat.std(ddof=1)),
                              "n": int(flat.size)}
                b = M[0, :]
                e["B_1train_3samp"] = {"mean": float(b.mean()),
                                       "sd": float(b.std(ddof=1)), "n": int(b.size)}
                within = float(np.mean([r.var(ddof=1) for r in M]))
                between = float(M.mean(axis=1).var(ddof=1))
                e["variance_decomposition"] = {
                    "within_training_seed_sampling_var": within,
                    "between_training_seed_var": between,
                    "ratio_between_over_within": (between / within) if within > 0 else None}
            summary.setdefault(t, {})[key] = e

    blob = {"ticket": 216, "arm": "MDLM (Sahoo et al., 2024), cell-type conditioned",
            "train_seeds": trains, "sample_seeds": SAMP,
            "shared_base": {"ckpt": "out/machine2/MDLM_Blackout/ckpt/mdlm_best.pt",
                            "sha256": "e39034ded1ccc577b1520e908151f8fb927d851a984dc1fb1"
                                      "1326ec2fa00252f",
                            "caveat": "MDLM retrains only its conditional layer, so its three training seeds share one "
                                      "unconditional base, so its 'training seed' does not mean "
                                      "other arms, and the caption must say so caption"},
            "ceilings": ceil, "sources": [{"train_seed": s, "file": f, "provenance": p}
                                          for s, f, p in SOURCES],
            "self_check": {"v7_recompute_max_abs_delta": gmax, "bitwise": gmax == 0.0,
                           "n_cells": len(cells)},
            "cells": cells, "by": summary,
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if OUT.exists():
        OUT.rename(OUT.with_name(f"216_mdlm_grid_3x3__preexisting_{int(time.time())}.json"))
    OUT.write_text(json.dumps(blob, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'request':<13}{'metric':<16}{'B 1train x3samp':<22}{'C 3x3':<22}betw/with")
    for t in TYPES:
        for k in ("purity_abs_v8", "purity_abs_v7", "W1", "mmd2", "pcc"):
            e = summary[t][k]
            if not e.get("complete"):
                print(f"{t:<13}{k:<16}incomplete ({e['n_cells']}/9)")
                continue
            r = e["variance_decomposition"]["ratio_between_over_within"]
            print(f"{t:<13}{k:<16}"
                  f"{e['B_1train_3samp']['mean']:.4f}+-{e['B_1train_3samp']['sd']:.4f}     "
                  f"{e['C_3x3']['mean']:.4f}+-{e['C_3x3']['sd']:.4f}     "
                  + (f"{r:.1f}x" if r else "n/a"))
        print()
    print(f"[out] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
