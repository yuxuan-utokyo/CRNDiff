#!/usr/bin/env python
"""R16, descriptive: median library size and zero fraction of the generated cells for every CRNDiff tau run
(seed 1, alpha 1), next to the real VAL cells of the class (all val cells of the class; for type II all 4,997, not the 2,500 subset).
Val only; test is not read. Counts are on the 2,000 HVGs, the space the generator works in. Inputs: the sha-checked local copies
of the tau runs (from_windows/rebuttal_R16/) and data/r16_hlca_score. Writes a json and the markdown table.
    python r16_libsize_table.py OUT_PREFIX
"""
import glob, hashlib, json, sys
import numpy as np
W = "./work"; out = sys.argv[1]
CL = [("classical monocyte", "classicalmonocyte"), ("multi-ciliated epithelial cell", "multiciliated"), ("plasma cell", "plasmacell"),
      ("ionocyte", "ionocyte"), ("type II pneumocyte", "typeIIpneumocyte")]
TAUS = ("0p2", "0p4", "0p6", "0p8", "1p0")
y = np.load(f"{W}/data/r16_hlca_score/val_celltype.npy", allow_pickle=True); X = np.load(f"{W}/data/r16_hlca_score/val.npy", mmap_mode="r")
picks = {}
for t, k in CL:
    try: picks[k] = json.load(open(f"{W}/runs/r16/tau_val/select_{k}.json"))["pick"]
    except FileNotFoundError: picks[k] = None
res = {"what": "median library size (sum over the 2,000 HVGs) and zero fraction; generated = CRNDiff seed 1, alpha 1; real = all val cells of the class",
       "split": "val", "classes": {}}
for t, k in CL:
    v = np.asarray(X[np.flatnonzero(y == t)]); rec = {"real_val": {"n": int(len(v)), "lib_median": float(np.median(v.sum(1))), "zero_frac": float((v == 0).mean())}, "tau": {}}
    for tau in TAUS:
        f = glob.glob(f"{W}/from_windows/rebuttal_R16/{k}_tau{tau}/*.npy"); assert len(f) == 1, (k, tau, f)
        g = np.asarray(np.load(f[0], mmap_mode="r"))
        rec["tau"][tau] = {"lib_median": float(np.median(g.sum(1))), "zero_frac": float((g == 0).mean()), "lib_ratio_to_real": float(np.median(g.sum(1)) / rec["real_val"]["lib_median"]),
                           "file_sha256": hashlib.sha256(open(f[0], "rb").read()).hexdigest()}
    rec["pick"] = picks[k]; res["classes"][t] = rec
json.dump(res, open(out + ".json", "w"), indent=1)
L = ["| class | real val (n) | " + " | ".join(f"τ {x.replace('p', '.')}" for x in TAUS) + " |", "|---|---|" + "---|" * len(TAUS)]
for t, r in res["classes"].items():
    rv = r["real_val"]
    cells = [f"{r['tau'][x]['lib_median']:,.0f} / {r['tau'][x]['zero_frac']:.3f}" + (" ★" if r["pick"] == f"tau{x}" else "") for x in TAUS]
    L.append(f"| {t} | {rv['lib_median']:,.0f} / {rv['zero_frac']:.3f}({rv['n']:,}) | " + " | ".join(cells) + " |")
L.append("")
L.append("each cell: median library size / zero fraction; ★ = τ selected on val. Ratio (generated median ÷ real val median):")
for t, r in res["classes"].items():
    L.append(f"- {t}:" + ",".join(f"τ {x.replace('p', '.')} ×{r['tau'][x]['lib_ratio_to_real']:.2f}" for x in TAUS))
open(out + ".md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
