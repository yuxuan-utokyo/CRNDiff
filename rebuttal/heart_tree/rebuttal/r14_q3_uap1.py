# -*- coding: utf-8 -*-
"""jvM6-Q3: UAP1+ fraction with R14's own rule, no sampling, no training, CPU 1 thread.
Rule imported from analysis.score_intersect: cell is UAP1+ iff X[:, GENE] > 0, GENE = 128 (threshold: count > 0).
(1) A'-v2 Endothelial (paper's fine-tuned row = aprime_v2, tools/t215_paper_numbers.py), seeds 20260931-33, N 10,057 each.
(2) real val Endothelial (same val as R14's reference); train rate from the tilt table provenance for reference.
(3) R14 tilted FK (intersect_ladder/a1): recompute with the same expression and compare with intersect_intersect_ladder.json.
Writes rebuttal/out/summary/rebuttal_jvM6_Q3_uap1.json. Test is never read.
"""
import json, os, sys, hashlib
from pathlib import Path
import numpy as np
HVG = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HVG))
from analysis import score_intersect as SI          # R14's module: GENE, REQUEST, data paths
from hvg import config as C
GENE = SI.GENE
def pos(a): return (np.asarray(a[:, GENE]) > 0)      # the expression R14 uses: (arr[:, GENE] > 0)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = {"what": "jvM6-Q3 UAP1+ fraction, R14 rule", "rule": f"X[:, {GENE}] > 0 (UAP1 count > 0), imported GENE from analysis.score_intersect",
       "request_R14": SI.REQUEST, "caliber": "HVG2K main tables and all selection on val; test not used"}
# (1) A'
rows = []
for s in (20260931, 20260932, 20260933):
    p = HVG / "out/runs/aprime_v2/Endothelial_aprime_v2" / f"Endothelial_aprime_v2_N10057_seed{s}.npy"
    a = np.load(p, mmap_mode="r"); m = pos(a)
    rows.append({"seed": s, "n": int(a.shape[0]), "n_uap1_pos": int(m.sum()), "frac": float(m.mean()), "npy": str(p), "sha256": sha(p)})
fr = [r["frac"] for r in rows]
out["aprime_v2_Endothelial"] = {"per_seed": rows, "mean": float(np.mean(fr)), "sd_ddof1": float(np.std(fr, ddof=1))}
# (2) real
D = C.LEGACY_DATA / "all"
Xv = np.load(D / "val.npy", mmap_mode="r"); yv = np.load(D / "val_celltype.npy", allow_pickle=True).astype(str)
e = np.flatnonzero(yv == "Endothelial"); m = pos(Xv[e])
tab = json.loads((C.TABLES / f"logc_{SI.REQUEST}.json").read_text(encoding="utf-8"))["provenance"]
out["real"] = {"val_Endothelial": {"n": int(len(e)), "n_uap1_pos": int(m.sum()), "frac": float(m.mean()), "data": str(D / "val.npy")},
               "train_Endothelial_from_tilt_table": {"n": tab["n_cells_type"], "n_uap1_pos": tab["n_ref_cells"], "frac": tab["detection_rate_in_type"]}}
# (3) R14 recheck
S = json.loads((HVG / "out/summary/intersect_intersect_ladder.json").read_text(encoding="utf-8"))
srows = S["rows"] if isinstance(S, dict) and "rows" in S else S
arch = {r["stem"]: r.get("uap1_pos_frac_delivered") for r in srows if isinstance(r, dict) and "stem" in r}
chk = []
for arm in ("a1", "fk_only", "tilt_only"):
    for p in sorted((HVG / "out/runs/intersect_ladder" / arm).glob("*.npy")):
        a = np.load(p, mmap_mode="r"); f = float(pos(a).mean())
        chk.append({"arm": arm, "stem": p.stem, "n": int(a.shape[0]), "frac_recomputed": f, "frac_archived": arch.get(p.stem),
                    "identical": (arch.get(p.stem) == f)})
out["R14_recheck"] = chk
for arm in ("a1", "fk_only", "tilt_only"):
    v = [c["frac_recomputed"] for c in chk if c["arm"] == arm]
    if v: out.setdefault("R14_recheck_mean", {})[arm] = {"n_seeds": len(v), "mean": float(np.mean(v)), "sd_ddof1": float(np.std(v, ddof=1)) if len(v) > 1 else None}
o = HVG / "rebuttal/out/summary/rebuttal_jvM6_Q3_uap1.json"; o.parent.mkdir(parents=True, exist_ok=True)
if o.exists(): o.rename(o.with_name(o.stem + "_bak.json"))
o.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("aprime_v2_Endothelial", "real", "R14_recheck_mean")}, ensure_ascii=False, indent=1)[:3000])
print("all R14 identical:", all(c["identical"] for c in chk), "n runs", len(chk))
