# -*- coding: utf-8 -*-
"""Rescore every existing delivery under the v8 lineage ruler, overwriting no artefact.

Purity is just the mass of a label histogram inside a label set, and every scored json already
stores its full histogram. So this step re-runs no classifier, resamples nothing and retrains
nothing: it reads json.

The ceiling convention is what keeps every existing v7 number unchanged: the v7 relative purity
keeps its own ceiling untouched, and the v8 relative purity divides by that ceiling scaled by
the ratio the two rulers give ON THE SAME ROW of archived REAL cells.

    python scripts/rescore_purity_lineage_ruler.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.metrics import CANONICAL_CEILING, purity_from_hist          # noqa: E402

S = C.OUT / "summary"
V7 = C.MASTER_PURITY_V7
V8 = C.MASTER_PURITY_V8
OUT = S / "215_purity_v8_rescore.json"
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
# Row label as it appears inside the archived scoring files. It is DATA, not text:
# changing it would stop the lookup from matching. It reads "real validation cells".
REAL_ROW = "\u771f\u5b9e val \u7ec6\u80de"


def ceilings():
    """Both rulers' hit rates on the SAME archived histogram; returns the two rates and ceilings."""
    v7 = {t: set(v["fixed_labels"])
          for t, v in json.loads(V7.read_text(encoding="utf-8"))["types"].items()}
    v8 = {t: set(v["fixed_labels"])
          for t, v in json.loads(V8.read_text(encoding="utf-8"))["types"].items()}
    out = {}
    for t in TYPES:
        d = json.loads((C.CELLTYPIST / f"celltypist_{t}_ceiling_canonical.json")
                       .read_text(encoding="utf-8"))
        real = d["label_dists"][REAL_ROW]
        r7, _ = purity_from_hist(real, v7[t])
        r8, _ = purity_from_hist(real, v8[t])
        pub = CANONICAL_CEILING[t]
        out[t] = {"labels_v7": v7[t], "labels_v8": v8[t],
                  "r7_from_archive": r7, "r8_from_archive": r8,
                  "ceiling_v7_published": pub,
                  "ceiling_v8_derived": pub * r8 / r7,
                  "v7_recompute_matches_published": round(r7, 4) == pub}
    return out


def _main() -> int:
    if not V8.exists():
        print(f"missing {V8} -> run scripts.build_lineage_purity_ruler first")
        return 2
    cal = ceilings()

    print("ceilings")
    print(f"  {'type':<13}{'v7 published':<15}{'v7 recomputed':<16}"
          f"{'v8 derived':<14}{'v8/v7'}")
    for t in TYPES:
        e = cal[t]
        print(f"  {t:<13}{e['ceiling_v7_published']:<15.4f}"
              f"{e['r7_from_archive']:<16.4f}{e['ceiling_v8_derived']:<14.4f}"
              f"{e['r8_from_archive']/e['r7_from_archive']:.4f}"
              + ("" if e["v7_recompute_matches_published"] else "   <-    <- the archive does not reproduce the published value published"))

    # globbing the scored files alone is not enough: one table's source is a differently named
    # file, and older files use other names again, all of them carrying purity rows.
    # the rule: any json in the summary directory whose first row has a purity field is
    # included, except this script's own output, which would otherwise feed on itself
    files = []
    for p in sorted(S.glob("*.json")):
        if p.name.startswith(("215_", "214_table2", "214_neuronal")):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                            # noqa: BLE001
            continue
        rs = d.get("rows")
        if (isinstance(rs, list) and rs and isinstance(rs[0], dict)
                and any("purity" in k for k in rs[0])):
            files.append(p)
    rows, skipped, gmax, bad = [], [], 0.0, []
    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:                                     # noqa: BLE001
            skipped.append({"file": p.name, "why": f"unreadable: {exc}"})
            continue
        for i, r in enumerate(d.get("rows", [])):
            t = r.get("request") or r.get("arm") or r.get("target")
            if t not in TYPES:
                continue
            dist = r.get("label_dist_whole")
            if not dist:
                skipped.append({"file": p.name, "row": i, "request": t,
                                "stem": r.get("stem"),
                                "why": "no label_dist_whole (needs a CellTypist re-run)"})
                continue
            e = cal[t]
            p7, n = purity_from_hist(dist, e["labels_v7"])
            p8, _ = purity_from_hist(dist, e["labels_v8"])
            # older files use one field name, newer ones another
            ref = r.get("purity_whole_set")
            if ref is None:
                ref = r.get("purity", r.get("purity_matched_n"))
            if ref is None:
                skipped.append({"file": p.name, "row": i, "request": t,
                                "why": "no stored purity to gate against"})
                continue
            delta = abs(p7 - float(ref))
            gmax = max(gmax, delta)
            if delta != 0.0:
                bad.append({"file": p.name, "stem": r.get("stem"), "delta": delta})
            if p8 < p7:                                              # G2
                print(f"G2 FAIL: v8 < v7 at {p.name} row {i}")
                return 1
            rows.append({
                "file": p.name, "stem": r.get("stem"), "request": t,
                "generator": r.get("generator"), "seed": r.get("seed"),
                "n_cells": n,
                "purity_v7": p7, "purity_v7_stored": float(ref),
                "purity_v8": p8,
                "rel_v7": p7 / e["ceiling_v7_published"],
                "rel_v8": p8 / e["ceiling_v8_derived"],
                "rel_v7_stored": r.get("purity_over_ceiling"),
            })

    print(f"\n[G1] v7 recomputed vs the existing json: {len(rows)} rows, max |delta| = {gmax:.3e}")
    if gmax != 0.0:
        print("[G1] FAIL -- not bitwise equal, so no v8 number is issued. The first few:")
        for x in bad[:5]:
            print("   ", x)
        return 1
    print("[G1] PASS (bitwise)")
    print(f"[G2] PASS: v8 >= v7 on all {len(rows)} rows")
    print(f"[G3] PASS: every existing json was read only; the sole file written is {OUT.name}")

    print(f"\n[G4] rows that cannot be recomputed: {len(skipped)}")
    for x in skipped[:20]:
        print("   ", json.dumps(x, ensure_ascii=False))
    if len(skipped) > 20:
        print(f"    …     ... and")

    blob = {"ticket": "215", "what": "v8 lineage-ruler rescore of every stored delivery",
            "method": "recomputed from stored label_dist_whole; no CellTypist re-run, "
                      "no re-sampling, no re-training",
            "ceilings": {t: {k: (sorted(v) if isinstance(v, set) else v)
                             for k, v in cal[t].items()} for t in TYPES},
            "self_check": {"v7_recompute_max_abs_delta": gmax, "bitwise": gmax == 0.0,
                           "n_rows": len(rows), "n_skipped": len(skipped)},
            "skipped": skipped, "rows": rows,
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if OUT.exists():                      # never overwrite an existing artefact: archive the old version first
        arch = OUT.with_name(f"{OUT.stem}__preexisting_{int(time.time())}.json")
        OUT.rename(arch)
        print(f"[archive] archived previous version -> -> {arch.name}")
    OUT.write_text(json.dumps(blob, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\ncoverage:")
    by = {}
    for r in rows:
        by.setdefault((r["request"], r["generator"]), []).append(r)
    print(f"  {'request':<13}{'generator':<12}{'n rows':<9}"
          f"{'rel_v7 range':<24}{'rel_v8 range'}")
    for (t, g), rs in sorted(by.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        a = [x["rel_v7"] for x in rs]
        b = [x["rel_v8"] for x in rs]
        print(f"  {t:<13}{str(g):<12}{len(rs):<9}"
              f"{min(a):.4f}-{max(a):.4f}{'':<8}{min(b):.4f}-{max(b):.4f}")
    print(f"\n[out] {OUT}")
    return 0


def main() -> int:
    # shared-summary write lock (tickets 217 / 219 may run concurrently)
    from scripts._summary_lock import summary_write_lock
    with summary_write_lock("t215_rescore_v8"):
        return _main()


if __name__ == "__main__":
    raise SystemExit(main())
