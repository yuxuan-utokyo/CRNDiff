# -*- coding: utf-8 -*-
"""R19: v8 purity ceiling on HVG2K **test**, the 3 target classes.
Reads test ONCE: test_celltype.npy for the class rows, then test.npy rows of those classes only (Endothelial 10,059, Myeloid 2,304,
Neuronal 397). Same ruler as every v8 number: analysis/score_purity.py annotate() (CellTypist Healthy_Adult_Heart, normalize_total 1e4 +
log1p, majority_voting False; ruler_check() run first), label sets v7/v8 from tools/t215_rescore_v8.ceilings(), hvg.metrics.purity_from_hist.
Reports, per class: purity on ALL test cells of the class (v7, v8) and, as score_run does, the mean over matched blocks of 2,500 (classes
with >= 2,500 cells); plus the val-derived v8 ceiling from ceilings() for comparison. Writes a read log. 1 thread.
    python -m rebuttal.r19_src.r19_test_ceiling <out_dir>
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import hashlib, json, statistics, sys, time
from pathlib import Path
import numpy as np
HVG = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(HVG))
from hvg import config as C                                     # noqa: E402
from hvg.metrics import purity_from_hist                        # noqa: E402
from analysis.score_purity import BLOCK, annotate, ruler_check  # noqa: E402
from tools.t215_rescore_v8 import ceilings                      # noqa: E402
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rl = ruler_check(); cal = ceilings(); D = C.LEGACY_DATA / "all"
genes = json.loads((D / "meta.json").read_text(encoding="utf-8"))["genes"]
T = ["Endothelial", "Myeloid", "Neuronal"]
log = {"what": "HVG2K test read for R19 purity ceilings (v8 ruler)", "started": time.strftime("%F %T"),
       "files": {f: {"path": str(D / f), "sha256": sha(D / f)} for f in ("test_celltype.npy", "test.npy")}, "rows_read": {}}
assert log["files"]["test.npy"]["sha256"].startswith("ad800326") and log["files"]["test_celltype.npy"]["sha256"].startswith("851a3d3d")
y = np.load(D / "test_celltype.npy", allow_pickle=True).astype(str); x = np.load(D / "test.npy", mmap_mode="r")
res = {}
for t in T:
    idx = np.flatnonzero(y == t); log["rows_read"][t] = int(idx.size)
    lab = annotate(np.asarray(x[idx], dtype=np.float64), genes)
    u, c = np.unique(lab, return_counts=True); whole = {str(k): int(v) for k, v in zip(u, c)}
    e = cal[t]; p7, n = purity_from_hist(whole, e["labels_v7"]); p8, _ = purity_from_hist(whole, e["labels_v8"])
    nb = idx.size // BLOCK; blocks = []
    for b in range(nb):
        uu, cc = np.unique(lab[b * BLOCK:(b + 1) * BLOCK], return_counts=True)
        blocks.append(purity_from_hist({str(k): int(v) for k, v in zip(uu, cc)}, e["labels_v8"])[0])
    res[t] = {"n_test_cells": int(idx.size), "purity_v7_all": p7, "purity_v8_all": p8,
              "purity_v8_matched_blocks_of_2500": (statistics.fmean(blocks) if blocks else None), "n_blocks": nb,
              "per_block_v8": blocks, "val_ceiling_v8_derived": e["ceiling_v8_derived"], "val_ceiling_v7_published": e["ceiling_v7_published"],
              "label_dist_whole": whole}
    print(f"{t:12s} n={idx.size:6d} purity_v8_all {p8:.4f} (v7 {p7:.4f}) blocks {nb} mean_v8 {res[t]['purity_v8_matched_blocks_of_2500']} | val v8 ceiling {e['ceiling_v8_derived']:.4f}", flush=True)
log["finished"] = time.strftime("%F %T")
(out / "r19_test_ceiling_v8.json").write_text(json.dumps({"ruler": rl, "block": BLOCK, "rows": res, "script_sha256": sha(Path(__file__))}, indent=1, ensure_ascii=False), encoding="utf-8")
(out / "test_read_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
