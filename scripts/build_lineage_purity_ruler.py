# -*- coding: utf-8 -*-
"""Build the v8 lineage purity ruler. It is ADDED alongside v7 and does not replace it.

Why. The v7 label set comes from one rule: a fine label counts for a target type if its majority
assignment on the validation set IS that type. On the neuronal request that rule degenerates to
a single label, because the reference annotation assigns two other glial labels to different
coarse classes. The consequence is that the frozen ruler misses a large share of REAL neuronal
cells: more than a quarter of them are called by one of the labels it excludes.

The v8 rule adds, to each v7 set, the fine labels that share the target's lineage stem and carry
enough support. v7 is untouched and remains a valid ruler; every artefact records which one it
used, and both are shipped in assets/rulers/.

    python scripts/build_lineage_purity_ruler.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.metrics import CANONICAL_CEILING, purity_from_hist          # noqa: E402

V7 = C.MASTER_PURITY_V7
V8 = C.MASTER_PURITY_V8
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
# Row label as it appears inside the archived scoring files. It is DATA, not text:
# changing it would stop the lookup from matching. It reads "the no-op (unconditional) row".
NOOP_ROW = "\u7a7a\u64cd\u4f5c(\u65e0\u6761\u4ef6)"
# Row label as it appears inside the archived scoring files. It is DATA, not text:
# changing it would stop the lookup from matching. It reads "real validation cells".
REAL_ROW = "\u771f\u5b9e val \u7ec6\u80de"
STEM = re.compile(r"^([A-Za-z]+)\d")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def stem_of(label: str):
    m = STEM.match(label)
    return m.group(1) if m else None


def shared_stem(labels: set):
    """Return the shared lineage stem if every label has one, otherwise None."""
    stems = {stem_of(x) for x in labels}
    return stems.pop() if len(stems) == 1 and None not in stems else None


def main() -> int:
    if V8.exists():
        print(f"refusing to overwrite an existing product: {V8}")
        print("  -> -> an existing artefact is never overwritten; move it aside first to rebuild _to_delete\\")
        return 2

    v7_sha_before = sha256(V7)
    v7 = json.loads(V7.read_text(encoding="utf-8"))
    fixed7 = {t: set(v["fixed_labels"]) for t, v in v7["types"].items()}
    border7 = {t: {x[0] if isinstance(x, list) else x
                   for x in v.get("borderline_labels", [])}
               for t, v in v7["types"].items()}

    # the archived file carries the same mapping in all three calls; the relevant one is taken
    ct = json.loads((C.CELLTYPIST / "celltypist_Neuronal_ceiling_canonical.json")
                    .read_text(encoding="utf-8"))
    f2c, sup = ct["fine_to_coarse"], ct["label_support"]

    out_types, notes = {}, []
    for t in v7["types"]:
        seeds = set(fixed7[t])
        st = shared_stem(seeds)
        added = {}
        if st:
            for L in f2c:
                if L not in seeds and stem_of(L) == st:
                    added[L] = "lineage-stem:" + st
        for L in border7.get(t, set()):
            if L not in seeds:
                added.setdefault(L, "v7-borderline")
        new = seeds | set(added)
        # G2
        if not (new >= seeds):
            print(f"G2 FAIL: v8({t}) is not a superset of v7({t})")
            return 1
        out_types[t] = {
            "fixed_labels": sorted(new),
            "inherited_from_v7": sorted(seeds),
            "added": {L: {"why": why,
                          "support_n": sup.get(L, {}).get("n"),
                          "support_major_frac": sup.get(L, {}).get("major_frac"),
                          "reference_coarse": f2c.get(L)}
                      for L, why in sorted(added.items())},
            "shared_stem": st,
        }
        if added:
            notes.append(f"{t}: +{sorted(added)}")

    # ceiling and floor
    calib = {}
    for t in TYPES:
        p = C.CELLTYPIST / f"celltypist_{t}_ceiling_canonical.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        real, noop = d["label_dists"][REAL_ROW], d["label_dists"][NOOP_ROW]
        e = {}
        for tag, lb in [("v7", fixed7[t]), ("v8", set(out_types[t]["fixed_labels"]))]:
            r, nr = purity_from_hist(real, lb)
            f, nf = purity_from_hist(noop, lb)
            e[tag] = {"ceiling": r, "n_real": nr, "noop_floor": f, "n_noop": nf,
                      "separation": r - f,
                      "real_over_floor": (r / f) if f else None}
        calib[t] = e

    # gate: the v7 ceiling must reproduce the recorded constant bitwise
    got = round(calib["Neuronal"]["v7"]["ceiling"], 4)
    if got != CANONICAL_CEILING["Neuronal"]:
        print(f"G5 FAIL: v7 Neuronal ceiling recompute {got} != "
              f"{CANONICAL_CEILING['Neuronal']}")
        return 1
    print(f"[G5] PASS: v7 Neuronal ceiling recompute = {got} "
          f"= CANONICAL_CEILING (same recipe)")

    blob = {
        "ruler": "master_purity_v8_lineage",
        "ticket": "215",
        "supersedes": None,
        "relationship_to_v7": ("ADDITIVE. v7 is untouched and remains a valid ruler; "
                               "v8(T) is a strict superset of v7(T) for every T. "
                               "Both are reported in the paper."),
        "rule": ("ruler_v8(T) = fixed_labels_v7(T) | {fine labels sharing the lineage "
                 "stem, only when all v7 seeds of T share one stem} | "
                 "borderline_labels_v7(T)"),
        "why": ("v7's rule 'majority coarse class == T' degenerates to a single label "
                "for Neuronal, because the reference assigns the sibling glial "
                "subtypes NC2_glial_NGF+ -> Fibroblast and NC3_glial -> Pericytes. "
                "That gives the v7 Neuronal ruler a 29.55% false-negative rate on "
                "REAL Neuronal cells."),
        "min_frac": v7.get("min_frac"),
        "built_from": {"v7": str(V7.name), "v7_sha256": v7_sha_before,
                       "celltypist_call": "celltypist_<type>_ceiling_canonical.json"},
        "types": out_types,
        "calibration": calib,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    V8.write_text(json.dumps(blob, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # G1
    v7_sha_after = sha256(V7)
    print(f"[G1] v7 sha256 before {v7_sha_before[:16]}… after {v7_sha_after[:16]}… "
          f"-> {'PASS' if v7_sha_before == v7_sha_after else 'FAIL'}")
    if v7_sha_before != v7_sha_after:
        return 1
    print("[G2] PASS: v8 is a strict superset of v7 for every type")

    print("\nwhich labels were added (G3)")
    for t in v7["types"]:
        a = out_types[t]["added"]
        if not a:
            print(f"  {t:<13} (no change; shared_stem={out_types[t]['shared_stem']})")
            continue
        print(f"  {t:<13} shared_stem={out_types[t]['shared_stem']}")
        for L, m in a.items():
            print(f"      + {L:<22} {m['why']:<20} support n={m['support_n']} "
                  f"major_frac={m['support_major_frac']} -> {m['reference_coarse']}")

    print("\nthe cost of changing ruler: how much the floor rises, how much signal is lost")
    print(f"  {'type':<13}{'ruler':<7}{'n_labels':<10}{'no-op floor':<14}"
          f"{'real val':<12}{'separation':<13}{'real/floor'}")
    for t in TYPES:
        for tag, lbs in [("v7", fixed7[t]), ("v8", set(out_types[t]["fixed_labels"]))]:
            e = calib[t][tag]
            rf = e["real_over_floor"]
            print(f"  {t:<13}{tag:<7}{len(lbs):<10}{e['noop_floor']:<14.4f}"
                  f"{e['ceiling']:<12.4f}{e['separation']:<13.4f}"
                  + (f"{rf:.1f}x" if rf else "inf"))
        print()

    print(f"[out] {V8}")
    print(f"[sha] v8 = {sha256(V8)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
