# -*- coding: utf-8 -*-
"""R12 side-by-side. Read-only.
1) [dropped, board 11:38] Copy the Table 1 OURS runs (train/samp 20260625/20260989, 20260627/20260987, 20260630/20260989; paths from rebuttal/w2_runs.py::run_list,
   as in R1) for Myeloid and Neuronal to _data_exchange/runs/table1_ours/<T>_a1/train<tr>_samp<s>.{npy,json}; SHA256SUMS + NAMES.tsv
   (source path). Endothelial is not copied.
2) out/summary/rebuttal_R12_side.json: frozen-ruler val numbers. OURS = existing records (out/summary/217_ours_grid_3x3.json cells, which
   point to their 214/214g scored files; not rescored); CountsDiff = today's final scoring (rebuttal_CD_final_scored.json + _v8.json).
   Noise floors per target from the CountsDiff final rows (same frozen rulers, same matched n).
    python -m rebuttal.r12_side
"""
import hashlib, json, shutil, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from rebuttal import w2_runs as W
# board 11-08 12:13: window A has manual CPU steps (R4 20260625 calibration, report edits, Count-FM C SHA256SUMS) between the CountsDiff
# final scoring and this step; they run while only the sampling line is on the GPU. This step waits for logs/GO_R12_SIDE, written by hand.
import time as _t
while not ((ROOT / "rebuttal/logs/GO_R12_SIDE").exists() and (ROOT / "rebuttal/logs/GO_R4_POOLS").exists()): _t.sleep(30)   # 17:36: also the R4 pools
S = ROOT / "out/summary"
X = Path(r"work\exchange\runs\table1_ours")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
X.mkdir(parents=True, exist_ok=True)
#
grid = json.loads((S / "217_ours_grid_3x3.json").read_text(encoding="utf-8"))["cells"]
cd = {r["request"]: r for r in json.loads((S / "rebuttal_CD_final_scored.json").read_text(encoding="utf-8"))["rows"]}
cd8 = json.loads((S / "rebuttal_CD_final_v8.json").read_text(encoding="utf-8"))["rows"]
out = {"split": "val (frozen rulers; test not read)", "sources": {"OURS": "out/summary/217_ours_grid_3x3.json (existing records, not rescored)",
       "CountsDiff": "out/summary/rebuttal_CD_final_scored.json + rebuttal_CD_final_v8.json"}, "targets": {}}
for t in ("Endothelial", "Myeloid", "Neuronal"):
    c = cd[t]
    out["targets"][t] = {
        "floor": {k: c.get(k) for k in ("floor_W1", "floor_sliced_W1", "floor_mmd2", "floor_pcc", "floor_scc")},
        "matched_n": {"W1": c.get("W1_matched_n"), "e21": c.get("e21_matched_n")},
        "OURS": [{"train": tr, "samp": s, **{k: grid[f"{t}|{tr}|{s}"].get(k) for k in ("purity_abs", "purity_abs_v7", "W1", "mmd2", "pcc", "stem", "source_file")}}
                 for tr, s in W.TABLE1["OURS"]],
        "CountsDiff": {"arm": c["arm"], "n": c["n_delivered"], "purity_v8": cd8[c["arm"]]["purity_v8"], "purity_v7": c["purity_whole_set"],
                       **{k: c.get(k) for k in ("W1", "W1_over_floor", "sliced_W1", "mmd2_rbf_biased", "pcc", "scc")}}}
p = S / "rebuttal_R12_side.json"; p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
shutil.copyfile(p, X / p.name)
with open(X / "SHA256SUMS", "a", encoding="utf-8", newline="\n") as f: f.write(f"{sha(X / p.name)}  {p.name}\n")
print("[out]", p)
