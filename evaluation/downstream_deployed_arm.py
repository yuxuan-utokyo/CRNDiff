# -*- coding: utf-8 -*-
"""Re-run the downstream blocks with our arm replaced by the DEPLOYED sampler.

The main table and the atlas figures use the deployed arm, while the archived downstream
manifest registers a different configuration family for our row, and that family is not even
consistent across the three requests. The method is the same tilt-plus-Feynman-Kac sampler, but
the configuration is not, so a downstream number taken from the archived manifest would not
describe the delivery the rest of the paper reports. This file swaps that one arm in memory and
runs the blocks again; every other arm, the seeds and the statistics are untouched.

Manifest paths are stored as Windows absolute paths, so they are translated to this repository's
layout on read. The archived module is never edited.

    python -m evaluation.downstream_deployed_arm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                        # noqa: E402
from evaluation import downstream_multiseed as MS                           # noqa: E402

# Every array path recorded in the archived E30plus manifest is an absolute Windows path under
# the old HVG2K tree. That tree is mirrored here as archive/, so the prefix below is stripped
# and the remainder is resolved against archive/. The arrays themselves (~7.6 GB) are not part
# of the repository; restore them under archive/ to run this script.
REPO = C.ARCHIVE
WIN_PREFIX = ("D:\\research\\Git-backup\\crndiff\\CRNDIFF\\"
              "Conditional generation experiments\\HVG2K\\")
WORK = C.OUT / "e33_197"
SUMMARY = C.OUT / "summary"

SEEDS = MS.SEEDS                    # the five registered evaluation seeds, neither added to nor removed from
KS = MS.KS                          # the registered K values, likewise
DEPLOYED_SEED = 20260987            # the same round as the main-table figures
RUNS_T1 = C.OUT / "runs" / "table1_ours"
DEPLOYED = {
    "Endothelial": RUNS_T1 / "Endothelial_a1" / (
        "Endothelial_residual_hvgtwist_fk_tiltevery_trigany_K32_M20114_N10057_"
        f"alpha1p0_J16_int8_th0p5_tau0p4_cc256_odds_seed{DEPLOYED_SEED}.npy"),
    "Myeloid": RUNS_T1 / "Myeloid_a1" / (
        "Myeloid_residual_hvgtwist_fk_tiltevery_trigany_K32_M5000_N2500_"
        f"alpha1p0_J16_int8_th0p5_tau0p2_cc256_odds_seed{DEPLOYED_SEED}.npy"),
    "Neuronal": RUNS_T1 / "Neuronal_a1" / (
        "Neuronal_residual_hvgtwist_fk_tiltevery_trigany_K32_M5000_N2500_"
        f"alpha1p0_J16_int8_th0p5_tau0p2_cc256_odds_seed{DEPLOYED_SEED}.npy"),
}


def translate(p: str) -> Path:
    """Translate a Windows absolute path from the manifest; a non-matching prefix is returned as is."""
    return REPO / p[len(WIN_PREFIX):].replace("\\", "/") if p.startswith(WIN_PREFIX) else Path(p)


def manifest_local(m):
    """The module's own manifest loader, with every array path translated; a missing entry stops."""
    by = m.load_manifest()
    out, missing = {}, []
    for key, items in by.items():
        new = []
        for it in items:
            q = translate(it["npy"])
            if not q.exists():
                missing.append(str(q))
            d = dict(it)
            d["npy"] = str(q)
            new.append(d)
        out[key] = new
    if missing:
        raise SystemExit("manifest paths do not resolve on this machine:\n  "
                         + "\n  ".join(missing[:10]))
    return out


def swap_ours(by, arrays: dict) -> tuple[dict, dict]:
    """Repoint our arm at the deployed arrays; returns the new mapping and a record of the swap."""
    out = {k: list(v) for k, v in by.items()}
    record = {}
    for t, q in arrays.items():
        q = Path(q)
        if not q.exists():
            raise SystemExit(f"deployed array missing: {q}")
        was = out.get(("OURS", t), [{}])[0].get("npy", "<none>")
        a = np.load(q, mmap_mode="r")
        out[("OURS", t)] = [{"method": "OURS", "target": t, "seed": DEPLOYED_SEED,
                             "npy": str(q)}]
        record[t] = {"was": Path(was).name, "now": q.name,
                     "now_path": str(q), "now_sha256": MS.sha256_file(q),
                     "n_rows": int(a.shape[0]), "n_genes": int(a.shape[1])}
    return out, record


# --------------------------------------------------------------------- G0 --
def gate_g0(m, by_local) -> dict:
    """With the manifest unmodified, compare the block bitwise against the earlier record."""
    ref_p = SUMMARY / "e33_block_a_multiseed.json"
    ref = json.loads(ref_p.read_text(encoding="utf-8"))
    seed = SEEDS[0]
    got = m.block_a(by_local, np.random.default_rng(seed))
    rows, worst = [], 0.0
    for arm in ["REAL (ceiling)", "OURS", "scVI", "scANVI", "CFGen"]:
        for metric in ["macro_F1", "Endothelial", "Myeloid", "Neuronal"]:
            r = (ref["arms"][arm]["macro_F1"] if metric == "macro_F1"
                 else ref["arms"][arm]["per_type_F1"][metric])["per_seed"][0]
            g = (got["arms"][arm]["macro_F1"] if metric == "macro_F1"
                 else got["arms"][arm]["per_type_F1"][metric])
            d = abs(g - r)
            worst = max(worst, d)
            rows.append({"arm": arm, "metric": metric, "recorded": r, "here": g, "abs_diff": d})
    return {"criterion": "this machine reproduces e33_block_a_multiseed.json per-seed[0] "
                         "to 1e-12 on the UNCHANGED manifest",
            "reference": str(ref_p), "seed": seed,
            "worst_abs_diff": worst, "pass": bool(worst < 1e-12), "rows": rows}


def aggregate() -> int:
    """Merge the five single-seed files; a missing one stops, since four seeds may not make a table."""
    parts = {}
    for s in SEEDS:
        q = WORK / f"raw_seed_{s}.json"
        if not q.exists():
            raise SystemExit(f"missing {q.name}; run --mode one --seed {s} first")
        parts[s] = json.loads(q.read_text(encoding="utf-8"))
    g0s = {json.dumps(p["g0"]["pass"]) for p in parts.values()}
    if g0s != {"true"}:
        raise SystemExit("a per-seed run did not pass G0 -- STOP")
    swaps = {json.dumps(p["arm_swap"], sort_keys=True) for p in parts.values()}
    if len(swaps) != 1:
        raise SystemExit("the per-seed runs did not use the same arm swap -- STOP")
    first = parts[SEEDS[0]]
    blob = {k: first[k] for k in ("ticket", "produced_by", "what_changed", "deployed_seed",
                                  "arm_swap", "path_translation", "g0", "body",
                                  "K_registered", "wrapper_sha256", "multiseed_sha256",
                                  "no_statistics_here", "real_logfc_stats")}
    blob["seeds"] = list(SEEDS)
    blob["assembled_from"] = {str(s): {"file": f"raw_seed_{s}.json",
                                       "sha256": MS.sha256_file(WORK / f"raw_seed_{s}.json")}
                              for s in SEEDS}
    blob["run_shape"] = "one process per seed (this box OOMs if five block_b runs share one)"
    for field in ("per_seed_block_a", "per_seed_block_b", "per_seed_topk"):
        blob[field] = {str(s): parts[s][field][str(s)] for s in SEEDS}
    blob["top_overlap_interception"] = [p["top_overlap_interception"][0] for p in parts.values()]
    blob["walltime_s"] = sum(p["walltime_s"] for p in parts.values())
    blob["sha256_after_run"] = first["sha256_after_run"]
    blob["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p = WORK / "raw_deployed_arm.json"
    p.write_text(json.dumps(blob, ensure_ascii=False, indent=1, default=float) + "\n",
                 encoding="utf-8")
    print(f"[out] {p}\n[out] sha256 {MS.sha256_file(p)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("g0", "full", "one", "aggregate"), default="full")
    ap.add_argument("--ours-seed", type=int, default=DEPLOYED_SEED,
                    help="which deployed sampling seed the OURS arm reads (sensitivity "
                         "check: Table 1 has three, 20260987/88/89). Files are tagged with "
                         "it so a sensitivity run can never overwrite the main run.")
    ap.add_argument("--seed", type=int, default=None,
                    help="--mode one: run exactly this registered seed into its own file "
                         "(one process per seed; this box OOMs if five seeds share one)")
    a = ap.parse_args(argv)
    WORK.mkdir(parents=True, exist_ok=True)

    if a.mode == "aggregate":
        return aggregate()

    m, prov = MS.load_body(WORK / "body_out")
    by_local = manifest_local(m)

    t0 = time.time()
    g0 = gate_g0(m, by_local)
    print(f"[G0] worst |delta| = {g0['worst_abs_diff']:.3e}  pass={g0['pass']}", flush=True)
    if not g0["pass"]:
        (WORK / "g0_failed.json").write_text(
            json.dumps(g0, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        raise SystemExit("G0 FAILED: this machine does not reproduce the recorded numbers. "
                         "Comparing new against old would confound the arm swap with an "
                         "environment drift. STOP.")
    if a.mode == "g0":
        (WORK / "g0.json").write_text(json.dumps(g0, ensure_ascii=False, indent=1) + "\n",
                                      encoding="utf-8")
        print(f"[out] {WORK / 'g0.json'}")
        return 0

    arrays = {t: Path(str(q).replace(f"seed{DEPLOYED_SEED}", f"seed{a.ours_seed}"))
              for t, q in DEPLOYED.items()}
    by_new, swap = swap_ours(by_local, arrays)
    for t, r in swap.items():
        print(f"[swap] {t:12s} {r['was']}\n            -> {r['now']}  ({r['n_rows']} rows)",
              flush=True)

    if a.mode == "one":
        if a.seed not in SEEDS:
            raise SystemExit(f"--seed must be one of the registered {SEEDS}")
        seeds = [a.seed]
    else:
        seeds = list(SEEDS)

    per_a, per_b, per_topk, fc_stats, checks = {}, {}, {}, {}, []
    for s in seeds:
        print(f"\n=== seed {s} : block_a ===", flush=True)
        per_a[s] = m.block_a(by_new, np.random.default_rng(s))
        print(f"=== seed {s} : block_b ===", flush=True)
        rb, tk, fcs, chk = MS.run_block_b_with_k(m, by_new, s)
        per_b[s], per_topk[s] = rb, tk
        checks.append({"seed": s, **chk})
        if not fc_stats:
            fc_stats = fcs

    blob = {"ticket": 197,
            "produced_by": "analysis/e33_deployed_arm.py",
            "what_changed": "the OURS arm of block_a / block_b now reads the DEPLOYED "
                            "arrays (the same ones Table 1 and every 4.2 figure use); "
                            "scVI / scANVI / CFGen are untouched",
            "deployed_seed": a.ours_seed,
            "is_main_run": a.ours_seed == DEPLOYED_SEED,
            "arm_swap": swap,
            "path_translation": {"win_prefix": WIN_PREFIX, "local_root": str(REPO),
                                 "applied_to": "every arm, content unchanged"},
            "g0": g0,
            "body": prov, "seeds": SEEDS, "K_registered": KS,
            "wrapper_sha256": MS.sha256_file(Path(__file__)),
            "multiseed_sha256": MS.sha256_file(Path(MS.__file__)),
            "no_statistics_here": "block_a / block_b / logfc / top_overlap / load_manifest "
                                  "come from the untouched body; the seed loop, the K sweep "
                                  "and the aggregation are analysis/e33_multiseed.py's own "
                                  "functions, called not copied",
            "top_overlap_interception": checks,
            "per_seed_block_a": {str(k): v for k, v in per_a.items()},
            "per_seed_block_b": {str(k): v for k, v in per_b.items()},
            "per_seed_topk": {str(k): v for k, v in per_topk.items()},
            "real_logfc_stats": fc_stats,
            "walltime_s": time.time() - t0,
            "sha256_after_run": MS.sha256_file(MS.SRC),
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if a.mode == "one":
        tag = "" if a.ours_seed == DEPLOYED_SEED else f"_ours{a.ours_seed}"
        q = WORK / f"raw_seed_{a.seed}{tag}.json"
        q.write_text(json.dumps(blob, ensure_ascii=False, indent=1, default=float) + "\n",
                     encoding="utf-8")
        print(f"\n[out] {q}")
        print(f"[body] e33_downstream.py sha256 after the run: {blob['sha256_after_run']}")
        return 0
    p = WORK / "raw_deployed_arm.json"
    p.write_text(json.dumps(blob, ensure_ascii=False, indent=1, default=float) + "\n",
                 encoding="utf-8")
    print(f"\n[out] {p}")
    print(f"[out] sha256 {MS.sha256_file(p)}")
    print(f"[body] e33_downstream.py sha256 after the run: {blob['sha256_after_run']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
