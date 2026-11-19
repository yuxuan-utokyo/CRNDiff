#!/usr/bin/env python
r"""CountsDiff K128 rows for tab:hvg_de and tab:hvg_tstr, from Windows' single-arm runs
_data_exchange/runs/cd_hvg2k_de_tstr_k128/ (the run log 18:43; r12_cd_de_k128.py / r12_cd_tstr_k128.py, 3 training seeds x 5 eval seeds
20260833-37). The paper tables (CRNDiff repo scripts/reproduce/table_04 / table_05) print mean ± sd over the 5 EVALUATION seeds of one fixed
delivery (DE: top-100 overlap 3 dp, Spearman 4 dp; TSTR: macro-F1 and per-class F1, 3 dp). Two aggregations are given and labelled:
  A  per training seed, mean ± sd (ddof 1) over the 5 eval seeds -- the table's own convention;
  B  across training seeds: mean of the 3 per-seed means ± sd (ddof 1) of those 3 means.
Checks: the TSTR REAL-ceiling row is identical across the 3 runs and to the 11-08 CountsDiff run (r12_countsdiff_tstr11.json 63f175e9).
    python cd_k128_de_tstr_rows.py OUT.md OUT.json
"""
import hashlib, json, os, sys
import numpy as np
out_md, out_js = sys.argv[1:3]
X = os.path.expanduser("./work/exchange/runs/cd_hvg2k_de_tstr_k128")
W = "./work"
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
TG = ["Endothelial", "Myeloid", "Neuronal"]; SEEDS = [20260625, 20260626, 20260627]; EV = [20260833, 20260834, 20260835, 20260836, 20260837]
src = {}; de, ts = {}, {}
for s in SEEDS:
    f = f"{X}/de_e33_k128_s{s}/r12_countsdiff_de.json"; j = json.load(open(f)); src[f.replace(X, "cd_hvg2k_de_tstr_k128")] = sha(f)[:8]
    assert j["eval_seeds"] == EV, j["eval_seeds"]
    for t in TG:
        assert str(s) in j["cfgen_repoint"][t]["now"] and "K128" in j["cfgen_repoint"][t]["now"], (s, t, j["cfgen_repoint"][t]["now"])
    de[s] = {t: {"top100": [j["per_seed_topk"][str(e)][t]["CFGen"]["top100_overlap_fraction"] for e in EV],
                 "rho": [j["per_seed_block_b"][str(e)]["types"][t]["arms"]["CFGen"]["spearman_rho_all_genes"] for e in EV]} for t in TG}
    f = f"{X}/tstr11_k128_s{s}/r12_countsdiff_tstr11.json"; k = json.load(open(f)); src[f.replace(X, "cd_hvg2k_de_tstr_k128")] = sha(f)[:8]
    assert k["eval_seeds"] == EV
    paths = [v["path"] for kk, v in k["arm_arrays"].items() if kk.startswith("CFGen|")]; assert len(paths) == 3 and all(f"train{s}" in p and "K128" in p for p in paths), paths
    a = k["arms"]["CountsDiff"]
    ts[s] = {"macro": a["macro_F1_11class"]["per_seed"], **{t: a[f"F1_{t}"]["per_seed"] for t in TG}, "_real": k["arms"]["REAL (ceiling)"]}
old = json.load(open(f"{W}/from_windows/windows_full_r123/r12_countsdiff_tstr11.json"))["arms"]["REAL (ceiling)"]
real_same = all(ts[s]["_real"] == old for s in SEEDS)
st = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1)))
A_de = {s: {t: {k2: st(de[s][t][k2]) for k2 in ("top100", "rho")} for t in TG} for s in SEEDS}
A_ts = {s: {c: st(ts[s][c]) for c in ["macro"] + TG} for s in SEEDS}
B_de = {t: {k2: st([A_de[s][t][k2][0] for s in SEEDS]) for k2 in ("top100", "rho")} for t in TG}
B_ts = {c: st([A_ts[s][c][0] for s in SEEDS]) for c in ["macro"] + TG}
f = lambda m, d: f"{m[0]:.{d}f} ± {m[1]:.{d}f}"
L = ["### DE(`tab:hvg_de`):CountsDiff K128", "",
     "the ± in the paper table is the sd over **5 evaluation seeds on the same delivery** (`table_04_differential_expression.py`: EVALUATION seeds; the delivered arrays are fixed); top-100 with 3 decimals, Spearman with 4.", "",
     "Table D1: A: as in the table, one row per training seed (mean ± sd, 5 evaluation seeds, n − 1)", "",
     "| training seed | " + " | ".join(f"{t} top-100 | {t} ρ" for t in TG) + " |", "|---|" + "---|" * 6]
for s in SEEDS:
    L.append(f"| {s} | " + " | ".join(f"{f(A_de[s][t]['top100'], 3)} | {f(A_de[s][t]['rho'], 4)}" for t in TG) + " |")
L += ["", "Table D2: B: aggregated over 3 training seeds (mean of the 3 per-seed means ± sd of these 3 means, n − 1; **the ± is between training seeds**, a different definition from the other rows of the table)", "",
      "| target | top-100 marker overlap ↑ | Spearman ρ (all genes)↑ |", "|---|---|---|"]
for t in TG: L.append(f"| {t} | {f(B_de[t]['top100'], 3)} | {f(B_de[t]['rho'], 4)} |")
L += ["", "### downstream classification (`tab:hvg_tstr`, 11 classes, replacement):CountsDiff K128", "",
      "the ± in the paper table is likewise the sd over 5 evaluation seeds (`table_05_tstr.py`), all 4 columns with 3 decimals. The evaluation set is all 45,145 val cells.", "",
      "Table T1: A: as in the table, one row per training seed (mean ± sd, 5 evaluation seeds, n − 1)", "",
      "| training seed | macro-F1 (11 classes)↑ | Endothelial F1 | Myeloid F1 | Neuronal F1 |", "|---|---|---|---|---|"]
for s in SEEDS: L.append(f"| {s} | " + " | ".join(f(A_ts[s][c], 3) for c in ["macro"] + TG) + " |")
L += ["", "Table T2: B: aggregated over 3 training seeds (**the ± is between training seeds**)", "", "| macro-F1(11 classes) ↑ | Endothelial F1 | Neuronal F1 top | ", ]
L[-1] = "| macro-F1(11 classes)↑ | Endothelial F1 | Myeloid F1 | Neuronal F1 |"
L += ["|---|---|---|---|", "| " + " | ".join(f(B_ts[c], 3) for c in ["macro"] + TG) + " |", "",
      f"- check: the real-cell ceiling row is the same across the 3 TSTR runs and bit-identical to the ceiling row of the earlier CountsDiff run:{'is' if real_same else 'no'}(macro-F1 {old['macro_F1_11class']['mean']:.4f} ± {old['macro_F1_11class']['sd']:.4f}).",
      "- every run input was checked to be the K128 delivery of that training seed (train seed and K128 in the file name)."]
res = {"A_de": {str(s): v for s, v in A_de.items()}, "A_tstr": {str(s): v for s, v in A_ts.items()}, "B_de": B_de, "B_tstr": B_ts,
       "per_eval_seed": {"de": {str(s): v for s, v in de.items()}, "tstr": {str(s): {c: ts[s][c] for c in ["macro"] + TG} for s in SEEDS}},
       "real_ceiling_identical_to_1108": real_same, "sources": src, "script_sha256": sha(__file__)}
json.dump(res, open(out_js, "w"), indent=1); open(out_md, "w").write("\n".join(L) + "\n"); print("\n".join(L))
