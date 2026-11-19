#!/usr/bin/env python
r"""Third rare HLCA target, mechanical rule: train share 0.15-0.9 %, test >= 200, lineage different from plasma cell and
ionocyte, multi-ciliated excluded; pick the share nearest 0.5 %, ties -> more test cells. Labels only (train/val/test_celltype.npy).
Lineage = coarse compartment assigned here from the label name (the rule does not define it): epithelial, endothelial, stromal,
immune-lymphoid, immune-myeloid. Plasma cell = immune-lymphoid, ionocyte = epithelial. Variant: immune merged into one lineage.
    python r16_third_target_pick.py OUT.json
"""
import hashlib, json, sys
import numpy as np
D = "./work/data/r16_hlca"; out = sys.argv[1]
y = {s: np.load(f"{D}/{s}_celltype.npy", allow_pickle=True).astype(str) for s in ("train", "val", "test")}
LIN = {"epithelial": ["respiratory basal cell", "type II pneumocyte", "club cell", "nasal mucosa goblet cell", "ciliated columnar cell of tracheobronchial tree",
                      "type I pneumocyte", "multi-ciliated epithelial cell", "respiratory hillock cell", "epithelial cell of lower respiratory tract", "bronchial goblet cell",
                      "serous secreting cell", "tracheobronchial serous cell", "epithelial cell of alveolus of lung", "acinar cell", "tracheobronchial goblet cell", "ionocyte",
                      "mucus secreting cell", "lung neuroendocrine cell", "brush cell of trachebronchial tree"],
       "endothelial": ["capillary endothelial cell", "vein endothelial cell", "pulmonary artery endothelial cell", "endothelial cell of lymphatic vessel"],
       "stromal": ["alveolar type 2 fibroblast cell", "alveolar type 1 fibroblast cell", "lung pericyte", "tracheobronchial smooth muscle cell", "bronchus fibroblast of lung",
                   "myofibroblast cell", "smooth muscle cell", "stromal cell", "fibroblast", "mesothelial cell"],
       "immune-lymphoid": ["CD8-positive, alpha-beta T cell", "CD4-positive, alpha-beta T cell", "natural killer cell", "B cell", "plasma cell", "T cell"],
       "immune-myeloid": ["alveolar macrophage", "elicited macrophage", "classical monocyte", "CD1c-positive myeloid dendritic cell", "non-classical monocyte", "mast cell",
                          "lung macrophage", "plasmacytoid dendritic cell", "conventional dendritic cell", "dendritic cell", "hematopoietic stem cell"]}
lin = {c: l for l, cs in LIN.items() for c in cs}; classes = sorted(set(y["train"])); assert set(classes) == set(lin), set(classes) ^ set(lin)
rows = []
for c in classes:
    rows.append({"class": c, "lineage": lin[c], "n_train": int((y["train"] == c).sum()), "share_pct": float((y["train"] == c).mean() * 100),
                 "n_val": int((y["val"] == c).sum()), "n_test": int((y["test"] == c).sum())})
def pick(excl_lineages):
    cand = [r for r in rows if 0.15 <= r["share_pct"] <= 0.9 and r["n_test"] >= 200 and r["lineage"] not in excl_lineages
            and r["class"] not in ("multi-ciliated epithelial cell", "plasma cell", "ionocyte")]
    cand.sort(key=lambda r: (abs(r["share_pct"] - 0.5), -r["n_test"])); return cand
A = pick({"immune-lymphoid", "epithelial"}); Bm = pick({"immune-lymphoid", "immune-myeloid", "epithelial"})
res = {"rule": "share 0.15-0.9 %, test >= 200, lineage != plasma (immune-lymphoid) / ionocyte (epithelial), no multi-ciliated, nearest 0.5 %, ties -> more test cells",
       "pick_5_lineages": A[0]["class"], "pick_immune_merged": Bm[0]["class"], "candidates_5_lineages": A, "candidates_immune_merged": Bm,
       "all_in_share_window": [r for r in rows if 0.15 <= r["share_pct"] <= 0.9], "labels_sha256": {s: hashlib.sha256(open(f"{D}/{s}_celltype.npy", "rb").read()).hexdigest()[:8] for s in y},
       "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
json.dump(res, open(out, "w"), indent=1)
print("pick (5 lineages):", A[0]["class"], "| pick (immune merged):", Bm[0]["class"])
for r in A: print(f"  {r['class']:42s} {r['lineage']:16s} {r['share_pct']:.3f}%  train {r['n_train']}  val {r['n_val']}  test {r['n_test']}  |d| {abs(r['share_pct']-0.5):.3f}")
