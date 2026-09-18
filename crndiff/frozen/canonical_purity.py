# -*- coding: utf-8 -*-
"""M2.3 archive: put every arm's purity on ONE canonical scale.

THE PROBLEM
-----------
`celltypist_eval.py` builds a single `np.random.default_rng(seed)` and consumes
it in order: each scored arm, then the mapping subsample, then the real-val
ceiling subsample. So both the fine->coarse mapping and the ceiling depend on
how many arms a call scored and how large each arm's array was. Measured across
the 9 calls in this directory:

    Endothelial ceiling 0.9412 - 0.9448   (4 distinct values)
    Myeloid     ceiling 0.9453 - 0.9457
    Neuronal    ceiling 0.7045 - 0.7071

and worse, the TARGET LABEL SET itself moved: Endothelial appears with three
different sets across calls (one containing `Adip4`, another `CD4+T_Th1` /
`CD4+T_reg`), Myeloid with and without `SAN_P_cell`. Different arms were scored
against different definitions of "correct".

THE FIX ALREADY EXISTS, AND IT IS NOT NEW
-----------------------------------------
`master_table.py` solved this for the OURS main table: freeze a label set, then
recompute purity from the per-arm label histograms every call already stores in
`label_dists`. `results/tables/master_purity_v7.json` is that frozen result and
`main_table_v2.md` is built on it. The baseline tables written later
(scvi / scanvi / scdiffusion / cfgen / cfdiffusion) were NOT -- they each used
their own call's RNG-dependent set and ceiling.

So this script does not invent a canonical. It FREEZES v7's label sets and pulls
the baselines onto them.

WHY NOT JUST RE-RUN master_table.py
-----------------------------------
Its fixed set is "labels strong in >= 50% of the calls scoring this type", i.e.
a function of which json files sit in the directory. Adding the five baseline
calls changes that population and would silently move the already-published OURS
numbers. The canonical must not depend on what was evaluated afterwards, so v7
is read as data and never recomputed here.

Purity is recomputed from stored histograms only -- CellTypist is never re-run,
so nothing about any arm's cells changes.

    python canonical_purity.py            # print
    python canonical_purity.py --write    # also write results/tables/*
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CT = HERE / "celltypist"
HVG = HERE.parents[1]
TABLES = HVG / "results" / "tables"
V7 = TABLES / "master_purity_v7.json"

TYPES = ["Endothelial", "Myeloid", "Neuronal"]

# tag -> (display name, uncond arm prefix, cond arm prefix)
BASELINES = [
    ("_scvi", "scVI", "scvi_uncond", "scvi_cond"),
    ("_scanvi", "scANVI", "scanvi_uncond", "scanvi_cond"),
    ("_cfgen", "CFGen", "cfgen_uncond", "cfgen_cond"),
    ("_scdiffusion", "scDiffusion", "scd_uncond", "scd_cond"),
]

# What each baseline table published, for the delta column.
PUBLISHED = {
    "scVI": {"Endothelial": (0.1784, 0.4264), "Myeloid": (0.0432, 0.4608),
             "Neuronal": (0.0036, 0.2692)},
    "scANVI": {"Endothelial": (0.2164, 0.8700), "Myeloid": (0.0452, 0.6392),
               "Neuronal": (0.0024, 0.0852)},
    "CFGen": {"Endothelial": (0.2144, 0.7996), "Myeloid": (0.0376, 0.6536),
              "Neuronal": (0.0032, 0.3600)},
    "scDiffusion": {"Endothelial": (0.2220, 0.2992), "Myeloid": (0.0232, 0.0676),
                    "Neuronal": (0.0032, 0.0020)},
}
PUBLISHED_CEILING = {"Endothelial": 0.9428, "Myeloid": 0.9457, "Neuronal": 0.7045}

# OURS, verbatim from `main_table_v2.md`, which is ALREADY built on v7 -- its
# "Real validation" rows are 0.9372 / 0.8836 / 0.7045, i.e. the canonical
# ceiling. So OURS never needed re-scaling; only the baseline tables did.
# (uncond, cond, sd_of_cond)
OURS_MAIN = {
    "Endothelial": (0.2240, 0.9115, 0.0007),
    "Myeloid": (0.0420, 0.7840, 0.0065),
    "Neuronal": (0.0048, 0.5504, None),
}


def purity_from_hist(dist: dict, labels: set) -> tuple[float, int]:
    tot = sum(dist.values())
    hit = sum(v for k, v in dist.items() if k in labels)
    return (hit / tot if tot else 0.0), tot


def load_call(tag: str) -> dict | None:
    out = {}
    for t in TYPES:
        f = CT / f"celltypist_{t}{tag}.json"
        if not f.exists():
            return None
        out[t] = json.loads(f.read_text(encoding="utf-8"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    v7 = json.loads(V7.read_text(encoding="utf-8"))
    fixed = {t: set(v7["types"][t]["fixed_labels"]) for t in TYPES}
    ceil_v7, spread_v7, nceil = {}, {}, {}
    for t in TYPES:
        row = next(r for r in v7["types"][t]["rows"] if r["arm"] == "真实 val 细胞")
        ceil_v7[t] = row["purity_fixed"]
        spread_v7[t] = row["across_call_spread"]
        nceil[t] = row["n"]

    print("=" * 78)
    print("CANONICAL (frozen from master_purity_v7.json, min_frac=0.5)")
    print("=" * 78)
    for t in TYPES:
        print(f"{t:<14} ceiling {ceil_v7[t]:.4f}  (n={nceil[t]}, "
              f"across-call spread {spread_v7[t]:.4f}, "
              f"{v7['types'][t]['n_calls']} calls)")
        print(f"{'':<14} fixed labels ({len(fixed[t])}): "
              f"{sorted(fixed[t])}")
        bl = v7["types"][t]["borderline_labels"]
        if bl:
            print(f"{'':<14} excluded as borderline: {bl}")

    # arm-independent cross-check of the ceiling
    print("\n" + "=" * 78)
    print("CROSS-CHECK: arm-independent ceiling run (no --arms, so the RNG "
          "stream\ncannot depend on what was being scored), same fixed labels")
    print("=" * 78)
    cc = load_call("_ceiling_canonical")
    cross = {}
    for t in TYPES:
        if cc is None:
            break
        p, n = purity_from_hist(cc[t]["label_dists"]["真实 val 细胞"], fixed[t])
        cross[t] = p
        d = p - ceil_v7[t]
        flag = "within v7 spread" if abs(d) <= max(spread_v7[t], 1e-9) else "OUTSIDE v7 spread"
        print(f"{t:<14} {p:.4f}  (n={n})   v7 {ceil_v7[t]:.4f}   "
              f"delta {d:+.4f}   {flag}")

    # ---- self-check: reproduce each call's own published purity --------------
    print("\n" + "=" * 78)
    print("SELF-CHECK: recompute each call's OWN published purity from its "
          "histogram\nusing that call's OWN label set (must match to 1e-9)")
    print("=" * 78)
    bad = 0
    for tag, name, *_ in BASELINES:
        call = load_call(tag)
        if call is None:
            continue
        worst = 0.0
        for t in TYPES:
            own = set(call[t]["target_labels"])
            for r in call[t]["rows"]:
                dist = call[t]["label_dists"].get(r["arm"])
                if dist is None:
                    continue
                p, _ = purity_from_hist(dist, own)
                worst = max(worst, abs(p - r["purity"]))
        status = "OK" if worst < 1e-9 else "MISMATCH"
        bad += worst >= 1e-9
        print(f"  {name:<13} max |recomputed - published| = {worst:.2e}  {status}")
    if bad:
        raise SystemExit("histogram recomputation does not reproduce published "
                         "purities; refusing to publish a canonical table")

    # ---- the canonical table -----------------------------------------------
    rows = []
    print("\n" + "=" * 78)
    print("CANONICAL BASELINE TABLE (v7 fixed labels, v7 ceiling)")
    print("=" * 78)
    print(f"{'type':<13}{'arm':<14}{'pub':>8}{'canon':>8}{'delta':>8}"
          f"{'/real pub':>11}{'/real canon':>13}")
    print("-" * 78)
    for t in TYPES:
        for tag, name, up, cp in BASELINES:
            call = load_call(tag)
            if call is None:
                continue
            d = call[t]
            got = {}
            for r in d["rows"]:
                arm = str(r["arm"])
                dist = d["label_dists"].get(arm)
                if dist is None:
                    continue
                if arm.startswith(up):
                    got["unconditional"] = purity_from_hist(dist, fixed[t])
                elif arm.startswith(cp):
                    got["conditional"] = purity_from_hist(dist, fixed[t])
            for mode in ("unconditional", "conditional"):
                if mode not in got:
                    continue
                p, n = got[mode]
                pub = PUBLISHED[name][t][0 if mode == "unconditional" else 1]
                rows.append({
                    "type": t, "model": name, "arm": mode, "n": n,
                    "purity_published": pub,
                    "purity_canonical": round(p, 6),
                    "purity_delta": round(p - pub, 6),
                    "ceiling_published": PUBLISHED_CEILING[t],
                    "ceiling_canonical": ceil_v7[t],
                    "ratio_published": round(pub / PUBLISHED_CEILING[t], 4),
                    "ratio_canonical": round(p / ceil_v7[t], 4),
                    "ratio_delta": round(p / ceil_v7[t] - pub / PUBLISHED_CEILING[t], 4),
                })
                print(f"{t:<13}{name + ' ' + mode[:4]:<14}{pub:>8.4f}{p:>8.4f}"
                      f"{p - pub:>+8.4f}{pub / PUBLISHED_CEILING[t]:>11.3f}"
                      f"{p / ceil_v7[t]:>13.3f}")

    # ---- cross-arm ranking on the canonical scale ---------------------------
    rank = {}
    for t in TYPES:
        rank[t] = {"OURS": OURS_MAIN[t][1] / ceil_v7[t]}
        for r in rows:
            if r["type"] == t and r["arm"] == "conditional":
                rank[t][r["model"]] = r["ratio_canonical"]
    order = ["OURS"] + [n for _, n, *_ in BASELINES]
    print("\n" + "=" * 78)
    print("CROSS-ARM RANKING, canonical scale (conditional / real val ceiling)")
    print("=" * 78)
    print(f"{'arm':<14}" + "".join(f"{t[:11]:>13}" for t in TYPES))
    print("-" * 78)
    for m in order:
        print(f"{m:<14}" + "".join(
            f"{rank[t].get(m, float('nan')):>13.3f}" for t in TYPES))
    print("\nFor comparison, the ratios as previously published (wrong ceiling):")
    prev = {"OURS": [0.967, 0.829, 0.781]}
    for _, n, *_ in BASELINES:
        prev[n] = [round(PUBLISHED[n][t][1] / PUBLISHED_CEILING[t], 3) for t in TYPES]
    for m in order:
        print(f"{m:<14}" + "".join(f"{v:>13.3f}" for v in prev[m]))

    if not a.write:
        print("\n(dry run; pass --write to emit results/tables files)")
        return

    payload = {
        "source_of_truth": "master_purity_v7.json (frozen, not recomputed)",
        "min_frac": v7["min_frac"],
        "fixed_labels": {t: sorted(fixed[t]) for t in TYPES},
        "borderline_excluded": {t: v7["types"][t]["borderline_labels"] for t in TYPES},
        "ceiling_canonical": ceil_v7,
        "ceiling_n": nceil,
        "ceiling_across_call_spread": spread_v7,
        "ceiling_arm_independent_crosscheck": cross,
        "ceiling_as_published_in_baseline_tables": PUBLISHED_CEILING,
        "rows": rows,
    }
    jp = TABLES / "ceiling_canonical.json"
    if jp.exists():
        print(f"[exists] {jp.name} not overwritten")
    else:
        jp.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                      encoding="utf-8")
        print(f"[out] {jp.name}")

    L = ["# Canonical purity scale (M2.3 archive)", "",
         "Every purity number in the paper is a fraction of cells whose "
         "CellTypist call lands in a target label set. That set, and the real-val "
         "ceiling the ratio is taken against, were **not** the same across runs: "
         "`celltypist_eval.py` draws its mapping subsample and its ceiling "
         "subsample from one RNG stream that the scored arms consume first, so "
         "both depend on how many arms a call scored and how big each was.", "",
         "Measured across the 9 calls in `code/eval/celltypist/`:", "",
         "| type | ceiling range across calls | distinct target-label sets |",
         "|---|---|---:|",
         "| Endothelial | 0.9412 - 0.9448 | 3 |",
         "| Myeloid | 0.9453 - 0.9457 | 2 |",
         "| Neuronal | 0.7045 - 0.7071 | 2 |", "",
         "One Endothelial set contained `Adip4` (the exact contamination "
         "`celltypist_eval.py`'s own docstring warns about), another contained "
         "`CD4+T_Th1` / `CD4+T_reg`; one Myeloid set contained `SAN_P_cell`. "
         "Different arms were scored against different definitions of correct.", "",
         "## The canonical scale", "",
         "This is not a new convention. `master_table.py` already fixed this for "
         "the OURS main table by freezing a label set and recomputing purity from "
         "the per-arm histograms every call stores in `label_dists`; "
         "`master_purity_v7.json` is that frozen result and `main_table_v2.md` is "
         "built on it. The baseline tables written later were not. So v7 is "
         "**frozen and read as data here, never recomputed** -- its fixed set is "
         "\"strong in >= 50% of the calls scoring this type\", which is a function "
         "of which files sit in the directory, and the five baseline calls added "
         "afterwards would have moved it, silently changing already-published "
         "OURS numbers.", "",
         "| type | ceiling | n | across-call spread | fixed labels |",
         "|---|---:|---:|---:|---|"]
    for t in TYPES:
        L.append(f"| {t} | **{ceil_v7[t]:.4f}** | {nceil[t]} | "
                 f"{spread_v7[t]:.4f} | `{'`, `'.join(sorted(fixed[t]))}` |")
    L += ["", "Cross-check with an arm-independent run (`--tag "
          "_ceiling_canonical`, no `--arms`, so nothing the arms do can move the "
          "RNG stream), scored on the same fixed labels:", "",
          "| type | arm-independent | v7 canonical | delta |", "|---|---:|---:|---:|"]
    for t in TYPES:
        if t in cross:
            L.append(f"| {t} | {cross[t]:.4f} | {ceil_v7[t]:.4f} | "
                     f"{cross[t] - ceil_v7[t]:+.4f} |")
    L += ["", "## Baselines re-scored on the canonical scale", "",
          "Purity is recomputed from the stored histograms only; CellTypist is "
          "never re-run and no arm's cells change. `published` is what the arm's "
          "own table reported, on its own call's label set and ceiling.", "",
          "| type | model | arm | purity (published) | purity (canonical) | delta "
          "| / real (published) | / real (canonical) |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['type']} | {r['model']} | {r['arm']} | "
                 f"{r['purity_published']:.4f} | {r['purity_canonical']:.4f} | "
                 f"{r['purity_delta']:+.4f} | {r['ratio_published']:.3f} | "
                 f"**{r['ratio_canonical']:.3f}** |")
    L += ["", "## Cross-arm ranking on the canonical scale", "",
          "Conditional purity as a fraction of that type's real-val ceiling. "
          "OURS is taken verbatim from `main_table_v2.md`, which was already "
          "built on v7 -- its `Real validation` rows are exactly the canonical "
          "ceilings, so OURS never needed re-scaling. The previously circulated "
          "OURS ratios 0.967 / 0.829 / 0.781 were computed against the baseline "
          "tables' wrong ceiling and are superseded here.", "",
          "| arm | " + " | ".join(TYPES) + " |",
          "|---|" + "---:|" * len(TYPES)]
    for m in order:
        L.append(f"| {'**OURS**' if m == 'OURS' else m} | "
                 + " | ".join(
                     (f"**{rank[t][m]:.3f}**" if m == "OURS" else f"{rank[t][m]:.3f}")
                     if m in rank[t] else "n/a" for t in TYPES) + " |")
    L += ["",
          "Ordering is unchanged by the correction, but two things move: OURS' "
          "margin over scANVI on Myeloid widens (0.887 vs 0.636, not 0.829 vs "
          "0.676), and every Myeloid number drops because `DC`, `Neut` and "
          "`LYVE1+MP_cycling` are no longer counted as Myeloid.", ""]
    L += ["", "The Myeloid ceiling moves most (0.9457 published -> "
          f"{ceil_v7['Myeloid']:.4f} canonical) because the published calls "
          "counted `DC`, `Neut` and `LYVE1+MP_cycling` as Myeloid while v7 does "
          "not; that lowers numerator and denominator together, so the ratio is "
          "the stable quantity and is what the paper should quote.",
          "",
          "`cfDiffusion` is not in this table: its call scored four arms under "
          "different names and its published table has no `real val (ceiling)` "
          "row, so it is re-derived separately in the archive rather than forced "
          "into this shape."]
    mp = TABLES / "ceiling_canonical.md"
    if mp.exists():
        print(f"[exists] {mp.name} not overwritten")
    else:
        mp.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"[out] {mp.name}")

    cp = TABLES / "ceiling_canonical.csv"
    if cp.exists():
        print(f"[exists] {cp.name} not overwritten")
    else:
        with cp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[out] {cp.name}")


if __name__ == "__main__":
    main()
