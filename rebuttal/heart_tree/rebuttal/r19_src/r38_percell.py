# -*- coding: utf-8 -*-
"""R38: per-cell CellTypist subtype labels of ALL real TEST Endothelial (10,059) and Myeloid (2,304) cells.
rebuttal/r19_src/r19_neuronal_percell.py (adca5827, the Neuronal labels of 11-11) with the class list Neuronal -> Endothelial, Myeloid;
nothing else. Same ruler as every generated-cell subtype label: analysis/score_purity.annotate() (CellTypist 1.7.1 Healthy_Adult_Heart,
normalize_total 1e4 + log1p, majority_voting False; ruler_check() = gate G5 first). Same file shas asserted as R19 (ad800326 / 851a3d3d).
Gate (else exit 2): each class's label histogram equals R19's label_dist_whole (r19_test_ceiling_v8.json) exactly. Test read logged.
1 thread, as R19.
    python -m rebuttal.r19_src.r38_percell <out_dir>
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
HVG = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(HVG))
from hvg import config as C                                     # noqa: E402
from analysis.score_purity import annotate, ruler_check         # noqa: E402
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
R19 = json.loads((HVG / "rebuttal" / "out" / "R19" / "test_ceiling" / "r19_test_ceiling_v8.json").read_text(encoding="utf-8"))
rl = ruler_check(); D = C.LEGACY_DATA / "all"
genes = json.loads((D / "meta.json").read_text(encoding="utf-8"))["genes"]
log = {"what": "HVG2K test read: per-cell CellTypist subtype labels of the test Endothelial and Myeloid cells",
       "started": time.strftime("%F %T"), "files": {f: {"path": str(D / f), "sha256": sha(D / f)} for f in ("test_celltype.npy", "test.npy")},
       "rows_read": {}, "script_sha256": sha(Path(__file__))}
assert log["files"]["test.npy"]["sha256"].startswith("ad800326") and log["files"]["test_celltype.npy"]["sha256"].startswith("851a3d3d")
y = np.load(D / "test_celltype.npy", allow_pickle=True).astype(str); x = np.load(D / "test.npy", mmap_mode="r")
gates = {}
for t in ("Endothelial", "Myeloid"):
    idx = np.flatnonzero(y == t); log["rows_read"][t] = int(idx.size)
    lab = annotate(np.asarray(x[idx], dtype=np.float64), genes)
    u, c = np.unique(lab, return_counts=True); whole = {str(k): int(v) for k, v in zip(u, c)}
    ok = whole == R19["rows"][t]["label_dist_whole"]; gates[t] = ok
    s = t.lower()
    np.savez(out / f"test_{s}_celltypist_labels.npz", test_row_index=idx.astype(np.int64), label=lab.astype("U32"))
    with open(out / f"test_{s}_celltypist_labels.tsv", "w", encoding="utf-8") as f:
        f.write("test_row_index\tcelltypist_label\n"); f.writelines(f"{int(i)}\t{l}\n" for i, l in zip(idx, lab))
    (out / f"test_{s}_celltypist_labels.json").write_text(json.dumps({"ruler": rl, "class": t, "n": int(idx.size), "label_dist_whole": whole,
        "gate_equals_r19_hist": ok, "r19_hist": R19["rows"][t]["label_dist_whole"], "script_sha256": log["script_sha256"],
        "settings": "analysis.score_purity.annotate: Healthy_Adult_Heart.pkl, normalize_total 1e4, log1p, majority_voting False"},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{t} n {idx.size} gate {'PASS' if ok else 'FAIL'}", flush=True)
log["finished"] = time.strftime("%F %T"); log["gates"] = gates
(out / "test_read_log_r38_percell.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
sys.exit(0 if all(gates.values()) else 2)
