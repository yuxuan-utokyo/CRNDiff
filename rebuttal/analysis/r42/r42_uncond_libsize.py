#!/usr/bin/env python
r"""R42: heart unconditional pools by target class. Pools = the Table 9 CRNDiff unconditional pools
ours_uncond_seed2026093{1,2,3}, first 20,000 cells (the R14b blocks, from_windows/r14_fano/, SHA256SUMS checked). Each cell's class = the
main-table purity rule: its CellTypist label (Windows, per cell, R14b) is in the class's v8 label set (rebuttal_R10_cond_3gen_v8.json).
Per class and seed: n cells called the class, mean library size, zero fraction, median Fano (B.8 definition: median over genes with mean > 0
of var(ddof=1) / mean, raw counts; r14b_common.dispersion). Reference: real val cells of the class and the deployed-tau row of R32
(results/R32_tau_count_stats.json). Val only, no test.
    python r42_uncond_libsize.py OUT.md
"""
import hashlib, json, sys
import numpy as np
W = "./work"; sys.path.insert(0, f"{W}/scripts/r14b"); import r14b_common as C
B = W.replace("research/crndiff-rebuttal", "work/shared")
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:8]
C.check_sums(); V8 = json.load(open(f"{W}/from_windows/r10_more/raw/rebuttal_R10_cond_3gen_v8.json"))["labels"]["v8"]
R32 = json.load(open(f"{B}/results/R32_tau_count_stats.json")); T = ["Endothelial", "Myeloid", "Neuronal"]
res = {t: [] for t in T}; CNT = {}
WIN = {20260931: [4568, 851, 520, 14061], 20260932: [4550, 799, 467, 14184], 20260933: [4607, 848, 484, 14061]}   # Windows re-labelling counts
for s in C.SEEDS:
    x = C.load_pool("OURS", s); lab = np.array([r["label"] for r in C.labels(f"OURS_s{s}")]); assert len(lab) == len(x)
    lib = x.sum(1); CNT[s] = [int(np.isin(lab, V8[t]).sum()) for t in T]; CNT[s].append(len(lab) - sum(CNT[s]))
    for t in T:
        m = np.isin(lab, V8[t]); xs = x[m]
        res[t].append({"seed": s, "n": int(m.sum()), "frac": float(m.mean()), "lib_mean": float(lib[m].mean()) if m.any() else None,
                       "zero": float((xs == 0).mean()) if m.any() else None, "fano": C.dispersion(xs)[0]["fano_median"] if m.sum() > 1 else None})
ms = lambda v, d: f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}"
L = ["# R42 library size of target-class cells in the heart unconditional pool (val)", "",
     "**Question to answer.** Before conditional sampling, is the library size of target-class cells produced by the generator already low, and in which direction does steering (tilt + FK) pull it?", "",
     "**Method.** The 3 CRNDiff unconditional pools of Table 9 (seed 20260931 / 32 / 33), first 20,000 cells each. The class of each cell follows the main-table purity rule: its CellTypist label (per-cell labels) falls in the v8 label set of that class."
     "for Endothelial, Myeloid and Neuronal separately: number of cells assigned to the class, mean library size, zero fraction, median Fano (definition of B.8: median of var / mean over genes with mean > 0, raw counts). Mean ± sd over 3 seeds."
     "reference: real val cells of that class and the R32 deployed τ rows (Endothelial τ 0.4, Myeloid 0.2, Neuronal 0.2, 3 sampling seeds). Reads val only, not test.", "", "**results.**", "",
     "| target | cell set | cells (per 20,000) | mean library size | zero fraction | median Fano |", "|---|---|---|---|---|---|"]
out = {}
for t in T:
    r = res[t]; rv = R32["targets"][t]["real_val"]; dep = [v for k, v in R32["targets"][t]["tau"].items() if v["deployed"]]; assert len(dep) == 1; dep = dep[0]; dtau = [k for k, v in R32["targets"][t]["tau"].items() if v["deployed"]][0]
    L.append(f"| {t} | cells assigned to {t} in the unconditional pool | {ms([x['n'] for x in r], 0)}({np.mean([x['frac'] for x in r]):.3f}) | {ms([x['lib_mean'] for x in r], 1)} | {ms([x['zero'] for x in r], 4)} | {ms([x['fano'] for x in r], 3)} |")
    L.append(f"| {t} | deployed τ {dtau} (tilt + FK, R32) | {R32['targets'][t]['tau'][dtau]['n_generated_per_seed'][0]:,} / seed | {dep['mean_library_size']['mean']:.1f} ± {dep['mean_library_size']['sd']:.1f} | {dep['zero_fraction']['mean']:.4f} ± {dep['zero_fraction']['sd']:.4f} | {dep['conditional_fano']['mean']:.3f} ± {dep['conditional_fano']['sd']:.3f} |")
    L.append(f"| {t} | real val | {R32['targets'][t]['n_real_val']:,} | {rv['mean_library_size']:.1f} | {rv['zero_fraction']:.4f} | {rv['conditional_fano']:.3f} |")
    out[t] = {"uncond_lib": float(np.mean([x["lib_mean"] for x in r])), "deployed_lib": dep["mean_library_size"]["mean"], "real_lib": rv["mean_library_size"]}
L += ["", "**how to read (computed in code).**"]
for t in T:
    o = out[t]; L.append(f"- {t}: mean library size of cells of the class in the unconditional pool {o['uncond_lib']:.0f}, the real val {o['uncond_lib'] / o['real_lib'] * 100:.0f}%; deployed after steering {o['deployed_lib']:.0f}({o['deployed_lib'] / o['real_lib'] * 100:.0f}%),"
                     + ("pull toward the real direction" if abs(o["deployed_lib"] - o["real_lib"]) < abs(o["uncond_lib"] - o["real_lib"]) else "farther from real") + ".")
L += ["", "> [!info]- reproducibility information", "> - pools and per-cell labels: from_windows/r14_fano/ (OURS_s2026093{1,2,3}_rows*.npz, OURS_s*_celltypist_labels.csv, checked against SHA256SUMS, same batch as R14b); v8 label set: rebuttal_R10_cond_3gen_v8.json.",
      f"> - deployed τ row and real val row: results/R32_tau_count_stats.json ({sha(B + '/results/R32_tau_count_stats.json')}). Fano: r14b_common.dispersion (definition of make_F8_dispersion_baselines).",
      "> - Label recheck: the 3 pools were relabelled cell by cell with the same CellTypist ruler; the labels are identical to the R14b per-cell labels and the pool sha are the same; v8 coarse-class counts (Endothelial / Myeloid / Neuronal / rest, per 20,000) match this document seed by seed:"
      + ";".join(f"{s} {' / '.join(str(c) for c in CNT[s])}" + ("(consistent)" if CNT[s] == WIN[s] else "(inconsistent)") for s in C.SEEDS) + ".",
      f"> - this document:`scripts/r42/r42_uncond_libsize.py`({sha(__file__)})."]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L[7:]))
