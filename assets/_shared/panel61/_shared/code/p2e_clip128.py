"""P2(e) — re-score every arm after a COMMON clip to [0, NMAX=128].

Motivation from P2(d): OURS has a categorical head over {0..128} and therefore CANNOT
emit a count above 128 (gen_max = 128 in 32/32 files), while Countsdiff -- the nearest
competitor and also count-native -- predicts dropped counts with a softplus regression
head and has no such ceiling (gen_max = 218, exceeding in 32/32 files). Real data reaches
481 in train and 346 in val, and 21 of 61 genes have dmax >= 128.

So the two leading arms are NOT scored on the same support. This clips everything to the
same [0,128] box and re-scores, which isolates how much of the gap is the support ceiling
rather than model quality.

NEW FILE. Reads existing gens; overwrites nothing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SCRNA = Path(__file__).resolve().parents[1]
if str(SCRNA) not in sys.path:
    sys.path.insert(0, str(SCRNA))

from _shared.evaluate_common import evaluate_array   # noqa: E402
from _shared.seeds import METRICS4, NMAX             # noqa: E402

# arm -> (gens glob at that arm's published NFE*)   NFE* from baseline/scrna/README.md §2.1
NFE_STAR = {
    "OURS":       ("wake-sigma_T8.0_K1024_s*_samp*.npy", 1024),
    "Countsdiff": ("cd_Tnone_K1024_s*_samp*.npy", 1024),
    "D3PM":       ("d3pm_nfe250_s*_samp*.npy", 250),
    "DDPM":       ("ddpm_nfe100_s*_samp*.npy", 100),
    "NB-VAE":     ("nbvae_nfe1_s*_samp*.npy", 1),
    "scVI":       ("scvi_nfe1_s*_samp*.npy", 1),
    "scLDM":      ("scldm_nfe1000_s*_samp*.npy", 1000),
    "CFGen":      ("cfgen_nfe10_s*_samp*.npy", 10),
}
OUT = SCRNA / "_shared" / "p2e_clip128.json"


def s_train(vals: dict[int, list[float]]) -> tuple[float, float]:
    per = np.array([np.mean(v) for v in vals.values()], float)
    return float(np.mean(per)), (float(np.std(per, ddof=1)) if len(per) > 1 else 0.0)


def main() -> int:
    res = {"NMAX": int(NMAX), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "what": "all arms re-scored after a common clip to [0,128]", "arms": {}}
    for arm, (pat, nfe) in NFE_STAR.items():
        files = sorted((SCRNA / arm / "gens").glob(pat))
        if not files:
            alt = sorted((SCRNA / arm / "gens").glob("*.npy"))
            print(f"[SKIP] {arm}: no files match {pat} (dir has {len(alt)})")
            res["arms"][arm] = {"skipped": f"no files match {pat}"}
            continue
        raw_by_ck, clip_by_ck, changed = {}, {}, 0
        for f in files:
            ck = int(f.stem.split("_s")[1].split("_")[0])
            a = np.load(f)
            a = np.nan_to_num(a, nan=0.0, posinf=float(NMAX), neginf=0.0)
            clipped = np.clip(a, 0, NMAX)
            n_changed = int((clipped != a).sum())
            changed += n_changed
            mr = evaluate_array(a)
            mc = evaluate_array(clipped) if n_changed else mr
            raw_by_ck.setdefault(ck, []).append(mr)
            clip_by_ck.setdefault(ck, []).append(mc)
        entry = {"nfe_star": nfe, "n_files": len(files),
                 "entries_changed_by_clip": changed,
                 "clip_is_noop": changed == 0, "raw": {}, "clipped": {}, "delta": {}}
        for k in METRICS4:
            rm, rs = s_train({c: [m[k] for m in v] for c, v in raw_by_ck.items()})
            cm, cs = s_train({c: [m[k] for m in v] for c, v in clip_by_ck.items()})
            entry["raw"][k] = {"mean": rm, "s_train": rs}
            entry["clipped"][k] = {"mean": cm, "s_train": cs}
            entry["delta"][k] = cm - rm
        res["arms"][arm] = entry
        print(f"{arm:<11s} NFE*={nfe:>5d} files={len(files):>2d} "
              f"clip_changed={changed:>8d}  "
              f"FD {entry['raw']['FD_frechet']['mean']:.4f} -> "
              f"{entry['clipped']['FD_frechet']['mean']:.4f}   "
              f"resid {entry['raw']['resid_corr_sqrt2']['mean']:.4f} -> "
              f"{entry['clipped']['resid_corr_sqrt2']['mean']:.4f}")
        Path(OUT).write_text(json.dumps(res, indent=1), encoding="utf-8")

    # headline: OURS vs its nearest competitor, before and after
    if "OURS" in res["arms"] and "Countsdiff" in res["arms"]:
        o, c = res["arms"]["OURS"], res["arms"]["Countsdiff"]
        res["headline_gap_vs_countsdiff"] = {}
        for k in METRICS4:
            gr = (o["raw"][k]["mean"] - c["raw"][k]["mean"]) / abs(c["raw"][k]["mean"])
            gc = (o["clipped"][k]["mean"] - c["clipped"][k]["mean"]) / abs(c["clipped"][k]["mean"])
            res["headline_gap_vs_countsdiff"][k] = {
                "gap_raw_pct": 100 * gr, "gap_clipped_pct": 100 * gc,
                "narrowed": abs(gc) < abs(gr),
                "narrowed_by_pct_points": 100 * (abs(gr) - abs(gc))}
            print(f"\n[gap OURS vs Countsdiff] {k}: raw {100*gr:+.2f}%  "
                  f"clipped {100*gc:+.2f}%  "
                  f"{'NARROWED' if abs(gc) < abs(gr) else 'not narrowed'}")
    Path(OUT).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
