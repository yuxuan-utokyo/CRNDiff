#!/usr/bin/env python
r"""R32 aggregate: per (target, tau) the mean and sd (n - 1) over the 3 sampling seeds of the 4 count-structure statistics,
plus the real val values (denominators). Reads only runs/r32/{r3_<T>,r1r2,fano,manifest}.json written by scripts/r32/r32_run.sh (val only).
    python r32_aggregate.py OUT.json OUT.md
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
W = Path("./work"); O = W / "runs/r32"; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
OJ, OM = Path(sys.argv[1]), Path(sys.argv[2])
T = ["Endothelial", "Myeloid", "Neuronal"]; TAUS = ["0p2", "0p4", "0p6", "0p8", "1p0"]
MAN = json.load(open(O / "manifest.json")); R12 = json.load(open(O / "r1r2.json")); FA = json.load(open(O / "fano.json"))
assert R12["split"] == "val"; DEP = {e["target"]: e["arm"] for e in MAN if e["deployed"]}
res = {"split": "val", "generator": 20260625, "seeds": [20260987, 20260988, 20260989], "deployed_tau": {t: DEP[t][3:].replace("p", ".") for t in T},
       "definitions": {"var_ratio_top50": "total variance in the top-50 PCs of the target's train-fit basis (target_pca_bases.npz 24d2cc8f), generated / real val; r9_new_metrics.py, resample n = min(n_val, n_gen, 2500), k as R19",
                       "zero_fraction": "fraction of zero entries over all cells x 2,000 genes (r1r2_metrics R2 zero_overall)",
                       "mean_library_size": "mean over cells of the count sum over the 2,000 HVGs (r1r2_metrics R2 lib_mean)",
                       "conditional_fano": "median over genes with mean > 0 of var(ddof=1)/mean, raw counts (r2_fano.py)"},
       "targets": {}}
st = lambda v: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "per_seed": [float(x) for x in v]}
for t in T:
    R3 = json.load(open(O / f"r3_{t}.json")); assert R3["split"] == "val"
    real = R12["r2"][t]["real_val"]; fl = R3["floor_heldout"]["var_ratio_top50"]
    res["targets"][t] = {"n_real_val": real["n"], "real_val": {"var_ratio_top50": 1.0, "var_ratio_floor_train_vs_val_heldout_basis": fl, "zero_fraction": real["zero_overall"],
                                                              "mean_library_size": real["lib_mean"], "conditional_fano": FA["targets"][t]["real_val_fano"]}, "tau": {}}
    for tau in TAUS:
        arm = f"tau{tau}"; r3 = sorted(R3["runs"][arm], key=lambda x: x["file"]); r2 = sorted([x for x in R12["runs"] if x["target"] == t and x["arm"] == arm], key=lambda x: x["seed"])
        fa = sorted([x for x in FA["targets"][t]["runs"] if x["arm"] == arm], key=lambda x: x["seed"]); assert len(r3) == len(r2) == len(fa) == 3, (t, tau)
        assert [x["seed"] for x in r2] == [x["seed"] for x in fa] == [20260987, 20260988, 20260989] and all(f"seed{s}" in x["file"] for s, x in zip([20260987, 20260988, 20260989], r3))
        res["targets"][t]["tau"][tau.replace("p", ".")] = {"deployed": arm == DEP[t], "n_generated_per_seed": [x["n"] for x in r2],
            "var_ratio_top50": st([x["var_ratio_top50"] for x in r3]), "zero_fraction": st([x["r2"]["zero_overall"] for x in r2]),
            "mean_library_size": st([x["r2"]["lib_mean"] for x in r2]), "conditional_fano": st([x["fano"] for x in fa])}
res["inputs"] = {p.name: sha(p)[:8] for p in sorted(O.glob("*.json"))}; res["scripts"] = {"r32_run.sh": sha(W / "scripts/r32/r32_run.sh")[:8], "r32_aggregate.py": sha(__file__)[:8],
    "r9_new_metrics.py": "13ef56af", "r1r2_metrics.py": "da155341", "r2_fano.py": "95bc766e"}
json.dump(res, open(OJ, "w"), indent=1)
K = [("var_ratio_top50", "variance ratio (first 50 PCs, generated/real)", 3), ("zero_fraction", "zero fraction", 4), ("mean_library_size", "mean library size", 1), ("conditional_fano", "condition Fano", 3)]
L = ["# R32 τ count-structure statistics on the grid (val)", "",
     "HVG2K, generator 20260625, 3 sampling seeds (20260987–89), mean ± sd per cell (n − 1). Reference = all val cells of the class; val only, test not read. Definitions and scripts as in R19 / B.8 (see definitions in the json). Bold τ is the deployed value.", ""]
for t in T:
    g = res["targets"][t]; rv = g["real_val"]
    L += [f"**{t}**(real val {g['n_real_val']:,} cells)", "", "| τ | " + " | ".join(h for _, h, _ in K) + " |", "|---|" + "---|" * len(K)]
    for tau, v in g["tau"].items():
        L.append(f"| {'**' + tau + '**' if v['deployed'] else tau} | " + " | ".join(f"{v[k]['mean']:.{d}f} ± {v[k]['sd']:.{d}f}" for k, _, d in K) + " |")
    L.append(f"| real val | 1 (by definition; train-vs-val floor {rv['var_ratio_floor_train_vs_val_heldout_basis']['mean']:.3f}) | {rv['zero_fraction']:.4f} | {rv['mean_library_size']:.1f} | {rv['conditional_fano']:.3f} |")
    L.append("")
L += [f"generated cells per seed:Endothelial {res['targets']['Endothelial']['tau']['0.2']['n_generated_per_seed'][0]:,}, Myeloid / Neuronal 2,500.", "",
      "> [!info]- reproducibility information", f"> - arrays: τ grid from_windows/r17_hvg2k_tau/tau_sweep; Neuronal τ 0.4 / 1.0 from from_windows/r19_crndiff/variants_Neuronal; deployed τ from from_windows/r19_arrays/crndiff_table1_ours/<T>_a1; all 45 arrays checked against their delivered SHA256SUMS.",
      "> - script:" + ";".join(f"{k} {v}" for k, v in res["scripts"].items()) + "; basis target_pca_bases.npz 24d2cc8f, R3 floor runs/r3_floor_heldout.json 6f2edf45.",
      "> - inputs:" + ";".join(f"runs/r32/{k} {v}" for k, v in res["inputs"].items()) + "."]
OM.write_text("\n".join(L) + "\n"); print("\n".join(L[:40]))
