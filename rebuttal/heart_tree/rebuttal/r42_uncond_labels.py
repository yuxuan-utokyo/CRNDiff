# -*- coding: utf-8 -*-
"""R42: per-cell CellTypist labels + v8 coarse class for the 3 Table 9 CRNDiff unconditional pools
(out/uncond_pools/ours_uncond_seed2026093{1,2,3}_n160000.npy), FIRST 20,000 rows each (the R14b slice, rebuttal/r14_fano_prep.py).
No test data. Ruler = the heart main-table purity ruler: analysis.score_purity.ruler_check() (gate G5) + annotate() (CellTypist 1.7.1
Healthy_Adult_Heart, normalize_total 1e4 + log1p, majority_voting False); v8 label sets from tools.t215_rescore_v8.ceilings()
(labels_v8 of Endothelial / Myeloid / Neuronal); coarse class = the type whose v8 set holds the label, else "other".
Gates (else exit 2): pool sha = the sha in R14b SOURCES.tsv; per-cell labels identical to R14b's <OURS_s<seed>_celltypist_labels.csv>.
CPU, 4 threads.
    python -m rebuttal.r42_uncond_labels <out_dir>
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"
import csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from hvg import config as C                                          # noqa: E402
from analysis.score_purity import annotate, ruler_check              # noqa: E402
from tools.t215_rescore_v8 import ceilings                           # noqa: E402
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
R14B = Path(r"work\exchange\r14_fano")
src = {(r["method"], r["seed"]): r for r in csv.DictReader(open(R14B / "SOURCES.tsv", encoding="utf-8"), delimiter="\t")}
rl = ruler_check(); cal = ceilings(); T = ["Endothelial", "Myeloid", "Neuronal"]
v8 = {t: set(cal[t]["labels_v8"]) for t in T}
assert not (v8["Endothelial"] & v8["Myeloid"] or v8["Endothelial"] & v8["Neuronal"] or v8["Myeloid"] & v8["Neuronal"]), "v8 sets overlap"
genes = json.loads((C.LEGACY_DATA / "all" / "meta.json").read_text(encoding="utf-8"))["genes"]
N = 20000; summary = {"ruler": rl, "v8_label_sets": {t: sorted(v8[t]) for t in T}, "n_rows": N, "pools": {}, "script_sha256": sha(Path(__file__))}
fail = []
for s in (20260931, 20260932, 20260933):
    p = ROOT / "out" / "uncond_pools" / f"ours_uncond_seed{s}_n160000.npy"; h = sha(p)
    if h != src[("OURS", str(s))]["pool_sha256"]: fail.append(f"pool sha {s}")
    x = np.load(p, mmap_mode="r")
    lab = annotate(np.asarray(x[:N], dtype=np.float64), genes)
    ref = [r["label"] for r in csv.DictReader(open(R14B / f"OURS_s{s}_celltypist_labels.csv", encoding="utf-8"))]
    same = len(ref) == N and all(a == b for a, b in zip(lab, ref)); fail += [] if same else [f"labels differ from R14b, seed {s}"]
    coarse = np.array([next((t for t in T if l in v8[t]), "other") for l in lab])
    np.savez(out / f"ours_uncond_seed{s}_first20000_labels.npz", row=np.arange(N, dtype=np.int64), label=lab.astype("U32"), coarse_v8=coarse.astype("U16"))
    with open(out / f"ours_uncond_seed{s}_first20000_labels.tsv", "w", encoding="utf-8") as f:
        f.write("row\tcelltypist_label\tcoarse_v8\n"); f.writelines(f"{i}\t{a}\t{b}\n" for i, (a, b) in enumerate(zip(lab, coarse)))
    u, c = np.unique(coarse, return_counts=True); u2, c2 = np.unique(lab, return_counts=True)
    summary["pools"][str(s)] = {"pool": str(p.relative_to(ROOT).as_posix()), "pool_sha256": h, "rows": f"0..{N - 1} (first {N})",
        "equals_r14b_labels": same, "coarse_v8_counts": {str(k): int(v) for k, v in zip(u, c)},
        "label_dist": {str(k): int(v) for k, v in zip(u2, c2)}}
    print(s, "R14b-identical" if same else "DIFFERS", {str(k): int(v) for k, v in zip(u, c)}, flush=True)
summary["gates"] = {"failures": fail, "passed": not fail}
(out / "r42_uncond_labels_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
sys.exit(0 if not fail else 2)
