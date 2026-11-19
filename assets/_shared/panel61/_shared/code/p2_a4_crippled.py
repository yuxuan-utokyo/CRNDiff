"""P2 — quantify how much of A4 is actually operative on the 61-gene panel.

`pf_trunc` only truncates when `c + 1 <= p.shape[2] - 1`, i.e. when the gene's cap is
below NMAX. Caps come from `dmax = concatenate([train, val]).max(0)` and NMAX = 128,
while per-gene data max runs to 481 -- so for every gene with dmax >= 128 the truncation
is a no-op and the real ceiling is NMAX itself.

Reports (a) gene split, (b) mass/rows above 128, (c) clip-vs-discard in training,
(d) per-arm gen_max, (e) all arms re-scored after a common clip to [0,128].

NEW FILE. Reads existing artefacts only; overwrites nothing.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

SCRNA = Path(__file__).resolve().parents[1]
if str(SCRNA) not in sys.path:
    sys.path.insert(0, str(SCRNA))

from _shared.data import load_panel                 # noqa: E402
from _shared.evaluate_common import evaluate_array  # noqa: E402
from _shared.seeds import METRICS4, NMAX           # noqa: E402

ARMS = ["OURS", "Countsdiff", "D3PM", "DDPM", "NB-VAE", "scVI", "scLDM", "CFGen"]
OUT = SCRNA / "_shared" / "p2_a4_crippled.json"


def main() -> int:
    p = load_panel()
    res: dict = {"NMAX": int(NMAX), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---- (a) gene split -------------------------------------------------- #
    dmax = np.asarray(p.dmax)
    lo = int((dmax < NMAX).sum())
    hi = int((dmax >= NMAX).sum())
    res["a_gene_split"] = {
        "caps_source": "dmax = concatenate([train, val]).max(0)  (_shared/data.py:39)",
        "n_genes": int(len(dmax)), "dmax_lt_128": lo, "dmax_ge_128": hi,
        "dmax_min": int(dmax.min()), "dmax_max": int(dmax.max()),
        "dmax_per_gene": dmax.tolist(),
        "note": f"pf_trunc is a NO-OP on the {hi} genes with dmax >= NMAX; for those the "
                f"effective ceiling is NMAX={NMAX}, not the data max",
    }
    print(f"(a) genes with dmax<128: {lo}/61 ; dmax>=128: {hi}/61 "
          f"(dmax range {dmax.min()}..{dmax.max()})")

    # ---- (b) mass above 128 ---------------------------------------------- #
    def above(x):
        return {"mass_frac": float((x > NMAX).mean()),
                "row_frac": float((x > NMAX).any(1).mean()),
                "n_entries": int((x > NMAX).sum()), "n_rows": int((x > NMAX).any(1).sum()),
                "total_entries": int(x.size), "total_rows": int(x.shape[0]),
                "max": int(x.max())}
    res["b_above_128"] = {"train": above(p.train), "val": above(p.val)}
    for k, v in res["b_above_128"].items():
        print(f"(b) {k}: mass above 128 = {v['mass_frac']:.3e} "
              f"({v['n_entries']} entries), rows = {v['row_frac']:.3e} "
              f"({v['n_rows']}), max = {v['max']}")

    # ---- (c) clip or discard in training? -------------------------------- #
    src = (SCRNA / "_shared" / "vendor_ours.py").read_text(encoding="utf-8")
    clamps = re.findall(r".*clamp\(0,\s*nmax\).*", src)
    res["c_training_clip_or_discard"] = {
        "verdict": "CLIP (not discard)" if clamps else "UNDETERMINED from vendor_ours.py",
        "evidence_lines": [c.strip() for c in clamps],
        "meaning": "counts above NMAX are folded onto the NMAX class in the x0 target, so "
                   "the model is trained to put that mass AT 128 rather than never seeing "
                   "it; rows are not dropped",
    }
    print(f"(c) training treatment of >NMAX: {res['c_training_clip_or_discard']['verdict']}")
    for c in clamps[:3]:
        print(f"      {c.strip()}")

    # ---- (d) per-arm gen_max --------------------------------------------- #
    res["d_arm_gen_max"] = {}
    for arm in ARMS:
        gdir = SCRNA / arm / "gens"
        files = sorted(gdir.glob("*.npy")) if gdir.exists() else []
        if not files:
            res["d_arm_gen_max"][arm] = {"n_files": 0}
            continue
        gmax_global, gmax_per_gene, exceed = 0, np.zeros(p.dim, dtype=np.int64), 0
        for f in files:
            a = np.load(f)
            if a.ndim != 2:
                continue
            m = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            gmax_global = max(gmax_global, int(np.max(m)))
            gmax_per_gene = np.maximum(gmax_per_gene, np.max(m, axis=0).astype(np.int64))
            exceed += int((m > NMAX).any())
        res["d_arm_gen_max"][arm] = {
            "n_files": len(files), "gen_max_global": gmax_global,
            "gen_max_per_gene": gmax_per_gene.tolist(),
            "n_files_exceeding_128": exceed,
            "can_exceed_128": bool(gmax_global > NMAX),
        }
        print(f"(d) {arm:<11s} files={len(files):>3d} gen_max={gmax_global:>6d} "
              f"exceeds128={'YES' if gmax_global > NMAX else 'no':>3s} "
              f"({exceed}/{len(files)} files)")

    Path(OUT).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
