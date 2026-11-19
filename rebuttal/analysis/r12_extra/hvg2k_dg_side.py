"""HVG2K vs DG side by side. Existing results only; nothing is rescored and test is not read.
Rows: HVG2K Endothelial, Myeloid, Neuronal (val, Table 1 runs, mean of 3) and DG GlialProg, RadialGlia2, OPC, Granule
(the single DG test read of 11-08 03:22; CRNDiff 1 seed, baselines mean of 3).
Methods: ours, scVI, scANVI. Columns: train cells and share of the target, val/test cells, purity, W1, W1/floor, MMD^2,
PCC, recall, precision; plus ours - scANVI purity and ours/scANVI W1.
Sources:
  HVG2K train/val counts       data/hvg2k/{train,val}_celltype.npy
  HVG2K recall, precision      from_windows/rebuttal_code/results_for_check/r3_numbers.json (TABLE1 runs)
  HVG2K ours v8, W1, MMD2, PCC from_windows/runs/table1_ours/rebuttal_R12_side.json (+ its floors)
  HVG2K scVI/scANVI v8 etc.    optional BASELINES json (Windows, same format); missing -> None
  DG train counts              data/r13_dg/train_celltype.npy
  DG everything else           runs/r13_scores_test.json
Usage: hvg2k_dg_side.py OUT.json [BASELINES.json]
"""
import json, os, sys
import numpy as np

W = os.path.expanduser("~/research/crndiff-rebuttal")
TABLE1 = {"OURS": [(20260625, 20260989), (20260627, 20260987), (20260630, 20260989)],
          "scVI": [(20261050, 20261051), (20261053, 20261055), (20261054, 20261055)],
          "scANVI": [(20261060, 20261065), (20261063, 20261061), (20261064, 20261061)]}
MET = ["purity", "W1", "W1_over_floor", "mmd2", "pcc", "recall", "precision"]


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


out = {"rows": []}
# HVG2K
yt = np.load(f"{W}/data/hvg2k/train_celltype.npy", allow_pickle=True).astype(str)
yv = np.load(f"{W}/data/hvg2k/val_celltype.npy", allow_pickle=True).astype(str)
r3 = json.load(open(f"{W}/from_windows/rebuttal_code/results_for_check/r3_numbers.json"))
side = json.load(open(f"{W}/from_windows/runs/table1_ours/rebuttal_R12_side.json"))
base = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and os.path.exists(sys.argv[2]) else None
for t in ["Endothelial", "Myeloid", "Neuronal"]:
    fl = side["targets"][t]["floor"]["floor_W1"]
    row = {"dataset": "HVG2K", "split": "val", "target": t, "n_train": int((yt == t).sum()), "share_train": float((yt == t).mean()),
           "n_eval": int((yv == t).sum()), "purity_ruler": "CellTypist v8", "floor_W1": fl, "methods": {}}
    for a in ["OURS", "scVI", "scANVI"]:
        rr = [r for r in r3[t]["runs"] if r["arm"] == a and (r["train"], r["samp"]) in TABLE1[a]]; assert len(rr) == 3, (t, a)
        m = {"recall": mean([r["recall"] for r in rr]), "precision": mean([r["precision"] for r in rr])}
        if a == "OURS":
            o = side["targets"][t]["OURS"]; assert {(r["train"], r["samp"]) for r in o} == set(TABLE1["OURS"])
            m.update(purity=mean([r["purity_abs"] for r in o]), W1=mean([r["W1"] for r in o]), mmd2=mean([r["mmd2"] for r in o]), pcc=mean([r["pcc"] for r in o]))
        elif base:
            b = base["targets"][t][a]; assert {(r["train"], r["samp"]) for r in b} == set(TABLE1[a]), (t, a)
            assert all(abs(r["floor_W1"] - fl) < 1e-12 for r in b), (t, a, "floor differs from rebuttal_R12_side.json")
            m.update(purity=mean([r["purity_v8"] for r in b]), W1=mean([r["W1"] for r in b]), mmd2=mean([r["mmd2_rbf_biased"] for r in b]), pcc=mean([r["pcc"] for r in b]))
        else:
            m.update(purity=None, W1=None, mmd2=None, pcc=None)
        m["W1_over_floor"] = m["W1"] / fl if m["W1"] is not None else None
        row["methods"][a] = m
    out["rows"].append(row)
# DG (test, single read)
yd = np.load(f"{W}/data/r13_dg/train_celltype.npy", allow_pickle=True).astype(str)
dt = json.load(open(f"{W}/runs/r13_scores_test.json"))
SEL = {"OURS": "CRNDiff", "scVI": "scvi", "scANVI": "scanvi"}
for t in ["GlialProg", "RadialGlia2", "OPC", "Granule"]:
    fl = dt["floors"][t]["W1"]["W1"]
    row = {"dataset": "DG", "split": "test (single read 11-08 03:22)", "target": t, "n_train": int((yd == t).sum()), "share_train": float((yd == t).mean()),
           "n_eval": dt["floors"][t]["n_val"], "purity_ruler": "logistic regression (train)", "floor_W1": fl, "methods": {}}
    for a, k in SEL.items():
        rr = [r for r in dt["runs"] if r["target"] == t and r["method"] == k]; assert rr, (t, a)
        m = {"purity": mean([r["purity_logreg"] for r in rr]), "W1": mean([r["W1"] for r in rr]), "mmd2": mean([r["mmd2"] for r in rr]),
             "pcc": mean([r["pcc"] for r in rr]), "recall": mean([r["recall"] for r in rr]), "precision": mean([r["precision"] for r in rr]), "n_seeds": len(rr)}
        m["W1_over_floor"] = m["W1"] / fl
        row["methods"][a] = m
    out["rows"].append(row)
for row in out["rows"]:
    o, s = row["methods"]["OURS"], row["methods"]["scANVI"]
    row["ours_minus_scanvi_purity"] = o["purity"] - s["purity"] if None not in (o["purity"], s["purity"]) else None
    row["ours_over_scanvi_W1"] = o["W1"] / s["W1"] if None not in (o["W1"], s["W1"]) else None
    print(row["dataset"], row["target"], row["n_train"], round(row["share_train"], 4), row["n_eval"],
          {a: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in MET} for a, m in row["methods"].items()},
          "dPur", row["ours_minus_scanvi_purity"] and round(row["ours_minus_scanvi_purity"], 3), "W1ratio", row["ours_over_scanvi_W1"] and round(row["ours_over_scanvi_W1"], 3))
json.dump(out, open(sys.argv[1], "w"), indent=1)
