"""R12 side table with CountsDiff at 3 training seeds. Val only; HVG2K test is never touched.
CountsDiff: train seeds 20260625 / 20260626 / 20260627, the selected cells (Endothelial K256 n=10,057; Myeloid K32 n=2,500; Neuronal K256
n=2,500; guidance 1.0; sampling seed 20261061, Endothelial 20261061+20261062). Per cell: the three values, mean and range (min-max).
  W1, MMD^2, PCC   runs/cd_hvg2k_3seed/frozen_w1_mmd_pcc.json (Mac, r13_score.py --mode hvg2k, frozen ruler; s25 Myeloid/Neuronal
                   bit-identical to runs/countsdiff_hvg2k_scores_my_neu.json; s25 all three targets: W1 and sliced W1 bit-identical to
                   Windows' rebuttal_R12_side.json, MMD^2 within 9.5e-10, PCC within 2.3e-12)
  recall, coverage runs/cd_hvg2k_3seed/new_metrics_<target>_s<seed>.json (r9_new_metrics.py, frozen R3 PCA basis, k=5; s25 bit-identical
                   to runs/countsdiff_hvg2k_new_metrics_<target>.json)
  v8, v7           Windows CellTypist: s25 from rebuttal_R12_side.json; s26/s27 from --v8 JSON {"<target>": {"<train seed>": {"v8":, "v7":}}}
                   when posted; until then those cells are "pending" and not counted.
OURS: Table 1 runs (20260625/20260989, 20260627/20260987, 20260630/20260989); v8, v7, W1, MMD^2, PCC from rebuttal_R12_side.json,
recall and coverage from r3_numbers.json (sha256 fd7295c0...); mean +- sd (ddof=1).
Count: a cell is "CountsDiff better" when its 3-seed mean is better than OURS's mean (higher v8, v7, PCC, recall, coverage; lower W1,
MMD^2). "overlap" = CountsDiff's [min, max] intersects OURS's [mean - sd, mean + sd].
    python cd_3seed_side_table.py OUT.json OUT.md [--v8 V8.json]
"""
import argparse, hashlib, json
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("out_json"); ap.add_argument("out_md"); ap.add_argument("--v8")
a = ap.parse_args()
T = ["Endothelial", "Myeloid", "Neuronal"]; SEEDS = [20260625, 20260626, 20260627]
COLS = [("v8", "v8 ↑", True, "{:.3f}"), ("v7", "v7 ↑", True, "{:.3f}"), ("W1", "W1 ↓", False, "{:.4f}"), ("mmd2", "MMD² ↓", False, "{:.5f}"),
        ("pcc", "PCC ↑", True, "{:.4f}"), ("recall", "recall ↑", True, "{:.3f}"), ("coverage", "coverage ↑", True, "{:.3f}")]
TABLE1 = [(20260625, 20260989), (20260627, 20260987), (20260630, 20260989)]
side = json.load(open("from_windows/runs/table1_ours/rebuttal_R12_side.json"))["targets"]
r3p = "from_windows/rebuttal_code/results_for_check/r3_numbers.json"
assert hashlib.sha256(open(r3p, "rb").read()).hexdigest().startswith("fd7295c00b6822c2")
r3 = json.load(open(r3p))
fz = {(r["target"], r["seed"]): r for r in json.load(open("runs/cd_hvg2k_3seed/frozen_w1_mmd_pcc.json"))["runs"]}
v8in = json.load(open(a.v8)) if a.v8 else {}
res = {"split": "val", "targets": {}}
for t in T:
    cd = {}
    for s in SEEDS:
        nm = next(iter(json.load(open(f"runs/cd_hvg2k_3seed/new_metrics_{t}_s{s}.json"))["runs"].values()))[0]
        f = fz[(t, s)]
        row = {"W1": f["W1"], "mmd2": f["mmd2"], "pcc": f["pcc"], "recall": nm["recall"], "coverage": nm["coverage"], "n_generated": f["n_generated"]}
        if s == 20260625:
            row["v8"], row["v7"] = side[t]["CountsDiff"]["purity_v8"], side[t]["CountsDiff"]["purity_v7"]
        elif str(s) in v8in.get(t, {}):
            row["v8"], row["v7"] = v8in[t][str(s)]["v8"], v8in[t][str(s)]["v7"]
        cd[s] = row
    ours_side = {(r["train"], r["samp"]): r for r in side[t]["OURS"]}
    ours_r3 = {(r["train"], r["samp"]): r for r in r3[t]["runs"] if r["arm"] == "OURS"}
    ours = {k: [] for k, *_ in COLS}
    for key in TABLE1:
        o = ours_side[key]; q = ours_r3[key]
        for k, v in (("v8", o["purity_abs"]), ("v7", o["purity_abs_v7"]), ("W1", o["W1"]), ("mmd2", o["mmd2"]), ("pcc", o["pcc"]), ("recall", q["recall"]), ("coverage", q["coverage"])):
            ours[k].append(v)
    T_ = {"countsdiff": {str(s): cd[s] for s in SEEDS}, "cells": {}}
    for k, name, hi, fmt in COLS:
        om, osd = float(np.mean(ours[k])), float(np.std(ours[k], ddof=1))
        vals = [cd[s].get(k) for s in SEEDS]
        if any(v is None for v in vals):
            T_["cells"][k] = {"ours_mean": om, "ours_sd": osd, "cd_values": vals, "status": "pending"}; continue
        cm, lo, up = float(np.mean(vals)), float(min(vals)), float(max(vals))
        better = cm > om if hi else cm < om
        overlap = lo <= om + osd and up >= om - osd
        T_["cells"][k] = {"ours_mean": om, "ours_sd": osd, "ours_values": ours[k], "cd_values": vals, "cd_mean": cm, "cd_min": lo, "cd_max": up,
                          "countsdiff_better": bool(better), "overlap": bool(overlap)}
    res["targets"][t] = T_
cells = [(t, k, c) for t in T for k, c in res["targets"][t]["cells"].items()]
done = [(t, k, c) for t, k, c in cells if c.get("status") != "pending"]
res["count"] = {"cells_total": len(cells), "cells_complete": len(done), "countsdiff_better": [f"{t} {k}" for t, k, c in done if c["countsdiff_better"]],
                "countsdiff_better_with_overlap": [f"{t} {k}" for t, k, c in done if c["countsdiff_better"] and c["overlap"]],
                "ours_better_with_overlap": [f"{t} {k}" for t, k, c in done if not c["countsdiff_better"] and c["overlap"]],
                "pending": [f"{t} {k}" for t, k, c in cells if c.get("status") == "pending"]}
json.dump(res, open(a.out_json, "w"), indent=1)
L = ["| target | method | " + " | ".join(n for _, n, _, _ in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
for t in T:
    C = res["targets"][t]["cells"]
    L.append(f"| {t} | OURS(Table 1, 3 runs) | " + " | ".join((fmt.format(C[k]["ours_mean"]) + " ± " + fmt.format(C[k]["ours_sd"])) for k, _, _, fmt in COLS) + " |")
    cellf = lambda k, fmt: ("pending Windows" if C[k].get("status") == "pending" else fmt.format(C[k]["cd_mean"]) + "(" + fmt.format(C[k]["cd_min"]) + "–" + fmt.format(C[k]["cd_max"]) + ")")
    L.append(f"| | CountsDiff(3 seeds, mean (range)) | " + " | ".join(cellf(k, fmt) for k, _, _, fmt in COLS) + " |")
    L.append(f"| | CountsDiff per seed (25 / 26 / 27) | " + " | ".join(" / ".join("—" if v is None else fmt.format(v) for v in C[k]["cd_values"]) for k, _, _, fmt in COLS) + " |")
open(a.out_md, "w").write("\n".join(L) + "\n")
c = res["count"]
print(f"complete {c['cells_complete']} of {c['cells_total']}; CountsDiff better in {len(c['countsdiff_better'])}: {c['countsdiff_better']}")
print("  of which overlapping:", c["countsdiff_better_with_overlap"]); print("OURS better but overlapping:", c["ours_better_with_overlap"]); print("pending:", c["pending"])
