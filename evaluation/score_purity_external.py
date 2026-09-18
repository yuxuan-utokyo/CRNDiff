# -*- coding: utf-8 -*-
"""A thin wrapper that scores a bare `.npy` (a baseline pool, or real cells) into a label histogram.

There is no scoring logic here. The model, the input convention, the fixed labels, the matched-n
blocking rule and the G5 ruler gate are all imported from `evaluation/score_purity.py`, whose
sha256 is asserted below and which this file never modifies.
This file does one thing: it swaps the sample source from a run json to an explicit npy path

and a target type name. `score_run` derives the npy location from `run['_path']` alone and
reads every other field with `.get()`, so a synthetic run dict is enough and no scoring code

The real-cell row takes a different path, `annotate` plus `purity_from_hist`, also imported,
because the myeloid and neuronal held-out sets are smaller than one matched block and

    python -m analysis.score_purity_external            #     python -m evaluation.score_purity_external          # score only, write raw histograms
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.metrics import load_fixed_label_sets, purity_from_hist      # noqa: E402
# the scorer itself: imported, never edited, never copied
from evaluation.score_purity import (BLOCK, annotate, ruler_check,     # noqa: E402
                                   score_run)

SCORER = ROOT / "evaluation" / "score_purity.py"
# sha256 of the scorer as it stands in this repository. The scoring code itself is unchanged;
# the digest moved only because the module was renamed (hvg -> crndiff, analysis -> evaluation)
# and its comments were translated into English. Its value before the move was
# fe563c209f3697b31b5160eb4ff01f77d103bbcfa9ff7fadc19cf86cadf2f524.
SCORER_SHA = "35aa524ef70c286291d642b75a31279f17f9000e2fa000262645cd2e5235e654"
E21 = C.OUT / "summary" / "e21_table1_ours.json"
E30 = (C.ARCHIVE
       / "universality_fk" / "out" / "e30plus" / "E30plus_full_metrics.json")
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
BASELINES = ["scVI", "CFGen", "scANVI"]
CACHE = C.OUT / "summary" / "_subtype_label_dists_cache.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def guard_scorer() -> str:
    got = sha256_file(SCORER)
    if got != SCORER_SHA:
        raise SystemExit(f"score_purity.py changed: {got} != {SCORER_SHA} -- STOP (ticket "
                         f"183 s9: its sha256 must not move)")
    return got


def genes() -> list:
    return json.loads((C.LEGACY_DATA / "all" / "meta.json")
                      .read_text(encoding="utf-8"))["genes"]


def score_npy(npy: Path, target: str, labels: set, gene_list: list, tag: str) -> dict:
    """A bare npy turned into one full scored row; the synthetic run dict supplies path and metadata."""
    npy = Path(npy)
    if not npy.is_file():
        raise SystemExit(f"missing baseline array: {npy}")
    a = np.load(npy, mmap_mode="r")
    if a.ndim != 2 or a.shape != (20000, 2000):
        raise SystemExit(f"{npy.name}: shape {a.shape}, expected (20000, 2000) -- gate 4")
    run = {"_path": str(npy.with_suffix(".json")), "stem": npy.stem, "arm": tag,
           "request": target}
    row = score_run(run, gene_list, labels, target)
    row["source_npy"] = str(npy)
    row["source_sha256"] = sha256_file(npy)
    row["arm_label"] = tag
    return row


def score_real(target: str, labels: set, gene_list: list) -> dict:
    """The full histogram of the real validation cells; annotate and purity_from_hist are imported."""
    x = np.load(C.LEGACY_DATA / "all" / "val.npy", mmap_mode="r")
    y = np.load(C.LEGACY_DATA / "all" / "val_celltype.npy",
                allow_pickle=True).astype(str)
    idx = np.flatnonzero(y == target)
    lab = annotate(np.asarray(x[idx], dtype=np.float64), gene_list)
    u, c = np.unique(lab, return_counts=True)
    hist = {str(k): int(v) for k, v in zip(u, c)}
    pur, n = purity_from_hist(hist, labels)
    return {"arm_label": "real cells", "target": target, "n_cells": int(len(idx)),
            "purity_whole_set": float(pur), "n_scored": int(n),
            "label_dist_whole": hist,
            "source_npy": str(C.LEGACY_DATA / "all" / "val.npy"),
            "note": "the whole held-out set for the type; smaller than one matched block "
                    "for Myeloid (2302) and Neuronal (396), so score_run's block rule does "
                    "not apply and purity is taken on the whole set, which is exactly how "
                    "the archived ceiling calls did it"}


def ceiling_matched_n(target: str, labels: set, gene_list: list, tmpdir: Path) -> dict:
    """The ceiling is on the matched-n convention: if the held-out set fills at least one block it is

        blocked and averaged by `score_run` itself, with the same block size and the same rule; the
        blocking is not rewritten here. A type whose held-out set is smaller than one block has no
    """
    x = np.load(C.LEGACY_DATA / "all" / "val.npy", mmap_mode="r")
    y = np.load(C.LEGACY_DATA / "all" / "val_celltype.npy",
                allow_pickle=True).astype(str)
    idx = np.flatnonzero(y == target)
    if len(idx) < BLOCK:
        return {"target": target, "n_cells": int(len(idx)), "applicable": False,
                "why": f"{len(idx)} held-out cells < one matched block of {BLOCK}; the "
                       f"whole-set value is the only one that exists for this type"}
    tmpdir.mkdir(parents=True, exist_ok=True)
    tmp = tmpdir / f"_real_{target}.npy"
    np.save(tmp, np.asarray(x[idx], dtype=np.int16))
    run = {"_path": str(tmp.with_suffix(".json")), "stem": f"real_{target}",
           "arm": "real cells", "request": target}
    row = score_run(run, gene_list, labels, target)
    tmp.unlink()
    return {"target": target, "n_cells": int(len(idx)), "applicable": True,
            "n_blocks": row["n_blocks"], "n_scored": row["n_scored"],
            "purity_per_block": row["purity_per_block"],
            "purity_matched_n": row["purity_matched_n"],
            "purity_whole_set_of_the_scored_cells": row["purity_whole_set"],
            "label_dist_whole": row["label_dist_whole"]}


def baseline_seed_sources() -> list:
    """Every baseline seed: the REAL seed and its file are read from the archived manifest.

        The seed in the file name is a constant, and the three seeds share a name in different
        directories, so a row must be labelled by `training_or_run_seed`, never by the file name.
    """
    d = json.loads(E30.read_text(encoding="utf-8"))
    out = []
    for _k, it in sorted(d["items"].items()):
        if it.get("method") not in BASELINES:
            continue
        s = it["source"]
        out.append({"method": it["method"], "target": it["target"],
                    "training_or_run_seed": s.get("training_or_run_seed", s["seed"]),
                    "artifact_filename_seed": s.get("artifact_filename_seed"),
                    "npy": Path(s["npy"])})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours-check-stem", default=None,
                    help="an OURS run stem to re-score through this wrapper (gate 3)")
    ap.add_argument("--all-seeds", action="store_true",
                    help="score every baseline seed the E30plus manifest lists, not just "
                         "the one e21_table1_ours.json names (183b s2)")
    ap.add_argument("--ceiling-check", action="store_true",
                    help="183b s1: reproduce the matched-n ceiling from the real cells")
    ap.add_argument("--out", default=str(CACHE))
    a = ap.parse_args(argv)

    scorer_sha = guard_scorer()
    ruler = ruler_check()
    print(f"[ruler] celltypist {ruler['celltypist_version']} | model "
          f"{ruler['model_sha256'][:16]} | verified by {ruler['verified_by']}")
    label_sets = load_fixed_label_sets(C.MASTER_PURITY_V7)
    gene_list = genes()
    e21 = json.loads(E21.read_text(encoding="utf-8"))

    out = {"produced_by": "analysis/score_purity_external.py",
           "scorer": str(SCORER), "scorer_sha256": scorer_sha,
           "scorer_unchanged": True,
           "wrapper_note": "no scoring logic is defined here; annotate / score_run / "
                           "ruler_check / purity_from_hist / load_fixed_label_sets are all "
                           "imported from the frozen scorer and hvg.metrics",
           "ruler": ruler, "matched_n": BLOCK,
           "label_set_source": str(C.MASTER_PURITY_V7),
           "baseline_source_field": "e21_table1_ours.json :: types.<T>.rows.<method>.source",
           "rows": [], "real": {}, "ours_check": None,
           "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    if a.all_seeds:
        srcs = baseline_seed_sources()
        out["baseline_seed_rule"] = ("every seed the E30plus manifest lists; rows are "
                                     "labelled by training_or_run_seed, NEVER by the seed "
                                     "in the filename (that one is a constant)")
        for s in srcs:
            t, m = s["target"], s["method"]
            if t not in TYPES:
                continue
            print(f"[score] {t} / {m} / seed {s['training_or_run_seed']}  "
                  f"{s['npy'].parent.name}/{s['npy'].name}", flush=True)
            row = score_npy(s["npy"], t, label_sets[t], gene_list, m)
            row["training_or_run_seed"] = s["training_or_run_seed"]
            row["artifact_filename_seed"] = s["artifact_filename_seed"]
            print(f"    purity@n{BLOCK} {row['purity_matched_n']:.4f} "
                  f"(blocks {row['n_blocks']}, spread "
                  f"{row['block_spread_NOT_run_to_run']:.4f})", flush=True)
            out["rows"].append(row)
    else:
        for t in TYPES:
            for m in BASELINES:
                src = Path(e21["types"][t]["rows"][m]["source"])
                print(f"[score] {t} / {m}  {src.name}", flush=True)
                row = score_npy(src, t, label_sets[t], gene_list, m)
                print(f"    purity@n{BLOCK} {row['purity_matched_n']:.4f} "
                      f"(blocks {row['n_blocks']}, spread "
                      f"{row['block_spread_NOT_run_to_run']:.4f}) | whole "
                      f"{row['purity_whole_set']:.4f}", flush=True)
                out["rows"].append(row)

    for t in TYPES:
        print(f"[score] {t} / real cells", flush=True)
        out["real"][t] = score_real(t, label_sets[t], gene_list)
        print(f"    purity {out['real'][t]['purity_whole_set']:.6f} "
              f"(n={out['real'][t]['n_cells']})", flush=True)

    if a.ceiling_check:
        from crndiff.metrics import CANONICAL_CEILING
        tmp = C.OUT / "summary" / "_tmp_ceiling"
        out["ceiling_matched_n"] = {}
        for t in TYPES:
            r = ceiling_matched_n(t, label_sets[t], gene_list, tmp)
            if r["applicable"]:
                r["constant"] = CANONICAL_CEILING[t]
                r["reproduces_constant"] = (round(r["purity_matched_n"], 4)
                                            == CANONICAL_CEILING[t])
                print(f"[ceiling] {t}: block mean {r['purity_matched_n']:.6f} over "
                      f"{r['n_blocks']} blocks -> {round(r['purity_matched_n'], 4)}  "
                      f"vs constant {CANONICAL_CEILING[t]}  "
                      f"{'MATCH' if r['reproduces_constant'] else 'DIFF'}", flush=True)
            else:
                r["constant"] = CANONICAL_CEILING[t]
                r["whole_set_value"] = out["real"][t]["purity_whole_set"]
                r["reproduces_constant"] = (round(r["whole_set_value"], 4)
                                            == CANONICAL_CEILING[t])
                print(f"[ceiling] {t}: {r['why']}; whole-set "
                      f"{r['whole_set_value']:.6f} vs constant {CANONICAL_CEILING[t]}  "
                      f"{'MATCH' if r['reproduces_constant'] else 'DIFF'}", flush=True)
            out["ceiling_matched_n"][t] = r
        if tmp.exists():
            for p in tmp.iterdir():
                p.unlink()
            tmp.rmdir()

    if a.ours_check_stem:
        from crndiff.io import load_runs
        runs = [r for r in load_runs("table1_ours") if r["stem"] == a.ours_check_stem]
        if len(runs) != 1:
            raise SystemExit(f"gate 3: {len(runs)} runs match stem {a.ours_check_stem}")
        r = runs[0]
        npy = Path(r["_path"]).with_suffix(".npy")
        print(f"[gate3] re-scoring OURS through the wrapper: {r['stem']}", flush=True)
        run = {"_path": str(npy.with_suffix(".json")), "stem": npy.stem,
               "arm": "OURS", "request": r["request"]}
        row = score_run(run, gene_list, label_sets[r["request"]], r["request"])
        out["ours_check"] = {"stem": r["stem"], "target": r["request"],
                             "label_dist_whole": row["label_dist_whole"],
                             "purity_matched_n": row["purity_matched_n"],
                             "purity_whole_set": row["purity_whole_set"]}

    out["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[out] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
