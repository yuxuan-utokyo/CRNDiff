#!/usr/bin/env python
r"""R38: fine CellTypist labels for heart Endothelial and Myeloid on TEST. Real test cells = Windows' per-cell labels
(one test read 11-12 12:19, histogram gated equal to R19); generated = run histograms copied verbatim from the scored jsons (no relabelling).
Columns: the 5 most frequent labels among the real test cells of the class, then "other in-target" (remaining labels of the class's v8 label
set) and "off-target" (labels outside it). v8 label sets read from rebuttal_R10_cond_3gen_v8.json["labels"]["v8"].
Runs: CRNDiff = the paper's main-table runs (ours_<type>_train20260630_samp..., gen 20260630 x 3 sampling seeds; the *_fk duplicate rows
are not used); CountsDiff = K128, 3 training seeds; CFGen = 219_cfgen_cond_3runs (9); MDLM = 208 + 216 s34 + 216 s35 (9); scVI, scANVI = 214g (9).
Neuronal is unchanged (paper numbers_1111b section 2).
    python r38_subtypes.py OUT.md
"""
import hashlib, json, sys
import numpy as np
W = "./work"; S = f"{W}/from_windows/subtype_heart_EM"; sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:8]
H = json.load(open(f"{S}/heart_EM_label_hists_runs.json"))["rows"]; V8 = json.load(open(f"{W}/from_windows/r10_more/raw/rebuttal_R10_cond_3gen_v8.json"))["labels"]["v8"]
SEL = {"CRNDiff(main table: generator 20260630 × 3 sampling seeds)": lambda r, t: "214g_ours_s20260630" in r["source"] and r["arm"] == t and r["stem"].startswith(f"ours_{t}_train20260630"),
       "CountsDiff K128 [3 seeds]": lambda r, t: "rebuttal_CD_K128_" in r["source"] and r["arm"].startswith(f"{t}_K128"),
       "CFGen(9 runs)": lambda r, t: r["source"].endswith("219_cfgen_cond_3runs_scored.json") and r["arm"] == f"{t}_cfgen_cond",
       "MDLM(9 runs)": lambda r, t: r["source"].endswith(("208_mdlm_cond_scored.json", "216_mdlm_cond_s20260934_scored.json", "216_mdlm_cond_s20260935_scored.json")) and r["arm"].startswith(f"{t}_mdlm_cond"),
       "scVI(9 runs)": lambda r, t: "214g_scvi_" in r["source"] and r["arm"] == t, "scANVI(9 runs)": lambda r, t: "214g_scanvi_" in r["source"] and r["arm"] == t}
NEED = dict(zip(SEL, (3, 3, 9, 9, 9, 9)))
L = ["# R38 heart Endothelial and Myeloid subtype labels (test)", "",
     "CellTypist 1.7.1(Healthy_Adult_Heart, model b62e3930), fine-grained label for every cell. Real test cells: labelled once cell by cell with the same ruler (test read once); the histogram matches the R19 test ceiling item by item."
     "Generated cells: label histograms already stored at scoring time, copied as is, not relabelled. Columns: the 5 most frequent labels among real test cells, then other within-target labels (rest of the v8 label set of the class) and out-of-target labels (outside the v8 label set). Method rows are mean ± sd over runs. Neuronal unchanged.", ""]
summary = {}
for t, fn in (("Endothelial", "endothelial"), ("Myeloid", "myeloid")):
    T = json.load(open(f"{S}/test_{fn}_celltypist_labels.json")); assert T["gate_equals_r19_hist"] and T["class"] == t
    h = T["label_dist_whole"]; n = sum(h.values()); assert n == T["n"]; inset = set(V8[t])
    top = sorted(h, key=lambda k: -h[k])[:5]
    def frac(d):
        m = sum(d.values()); f = [d.get(k, 0) / m for k in top]
        return f + [sum(v for k, v in d.items() if k in inset and k not in top) / m, sum(v for k, v in d.items() if k not in inset and k not in top) / m]
    cols = top + ["other within-target", "off-target"]
    L += [f"**{t}**(real test {n:,} cells; v8 within-target labels:{', '.join(sorted(inset))}; off-target labels among the top 5:{', '.join(k for k in top if k not in inset) or 'none'})", "",
          "| cells | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols), f"| real test {t} ({n:,}) | " + " | ".join(f"{x:.3f}" for x in frac(h)) + " |"]
    summary[t] = {"real": frac(h), "cols": cols}
    for lab, f in SEL.items():
        rs = [r for r in H if f(r, t)]; assert len(rs) == NEED[lab], (t, lab, len(rs)); fr = np.array([frac(r["label_dist_whole"]) for r in rs])
        L.append(f"| {lab} | " + " | ".join(f"{np.mean(fr[:, i]):.3f} ± {np.std(fr[:, i], ddof=1):.3f}" for i in range(len(cols))) + " |")
        summary[t][lab] = fr.mean(0).tolist()
    L.append("")
# closeness to the real composition (L1 distance over the 7 columns), computed
L += ["**distance from the real composition** (L1 distance between the 7 column proportions and real test; smaller is closer; computed in code)", "", "| method | Endothelial | Myeloid |", "|---|---|---|"]
for lab in SEL: L.append(f"| {lab} | " + " | ".join(f"{np.abs(np.array(summary[t][lab]) - np.array(summary[t]['real'])).sum():.3f}" for t in ("Endothelial", "Myeloid")) + " |")
best = {t: min(SEL, key=lambda lab: np.abs(np.array(summary[t][lab]) - np.array(summary[t]["real"])).sum()) for t in ("Endothelial", "Myeloid")}
L += ["", "- closest to the real composition:" + ";".join(f"{t} {best[t].split('(')[0].split(' ')[0]}" for t in best) + ".", "",
      "> [!info]- reproducibility information", f"> - real labels: test_endothelial_celltypist_labels.json ({sha(S + '/test_endothelial_celltypist_labels.json')}), test_myeloid_celltypist_labels.json ({sha(S + '/test_myeloid_celltypist_labels.json')}), script r38_percell.py, test read recorded in test_read_log_r38_percell.json; subtype/ SUMS 17/17.",
      f"> - generated histograms: heart_EM_label_hists_runs.json ({sha(S + '/heart_EM_label_hists_runs.json')}, 106 rows, copied as is from each scored json); v8 label set: rebuttal_R10_cond_3gen_v8.json.",
      f"> - this document:`scripts/r38/r38_subtypes.py`({sha(__file__)})."]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L[3:]))
