#!/usr/bin/env python
r"""Table 9 (tab:hvg_uncond_fidelity) row for CountsDiff K128, the Mac part: the columns that need no CellTypist.
Columns of the paper table (CRNDiff repo scripts/reproduce/table_09_unconditional_fidelity.py): library-size source, median Fano, library-size
CV, detected-genes CV, sliced-W1 (each the mean over the 3 sampling seeds), and the 3 "fraction of pool" columns (CellTypist; Windows).
Same computation as the table's other rows (CRNDiff repo scripts/score_unconditional_fidelity.py = Windows t200_uncond_fidelity.py):
  reference  = 20,000 val cells, np.sort(default_rng(20261200).choice(N_val, 20000, replace=False)), shared by every generator;
  pool       = its first 20,000 rows, unfiltered;
  sliced-W1  = evaluation/score_distances_logcp10k.sliced_w1(cp10k_log1p(pool), cp10k_log1p(ref), 512, 20260906) -- the table's own calibre,
               labelled "hvg_score_distribution_caliber, NOT a frozen ruler" in 202_tableB1_full.json;
  median Fano = make_F8_dispersion_baselines.dispersion (from_windows/r14_fano, Windows' copy);
  CVs        = score_unconditional_fidelity.heterogeneity (imported from the repo script, not rewritten).
Gate first: the same code on CRNDiff's (OURS) seed-20260931 pool (Windows' first-20,000-row blocks in from_windows/r14_fano) must reproduce
202_tableB1_full.json's row for the four columns exactly; else stop.
    python uncond_k128_score.py OUT.json POOL_DIR
"""
import hashlib, importlib.util, json, os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "8"
import numpy as np
W = "./work"; out, PD = sys.argv[1:3]
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
def mod(path, name):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sys.path.insert(0, f"{W}/CRNDiff")
SD = mod(f"{W}/CRNDiff/evaluation/score_distances_logcp10k.py", "_sd")
F8 = mod(f"{W}/from_windows/r14_fano/make_F8_dispersion_baselines.py", "_f8")
SU = mod(f"{W}/CRNDiff/scripts/score_unconditional_fidelity.py", "_su")
assert (SU.VAL_SUB_SEED, SU.N_SUB, SU.N_PROJ, SU.PROJ_SEED) == (20261200, 20000, 512, 20260906)
xv = np.load(f"{W}/data/hvg2k/val.npy", mmap_mode="r")
idx = np.sort(np.random.default_rng(SU.VAL_SUB_SEED).choice(xv.shape[0], SU.N_SUB, replace=False))
ref_log = SD.cp10k_log1p(np.asarray(xv[idx], dtype=np.float64))
def score(x):
    g = np.asarray(x[:SU.N_SUB], dtype=np.float64); assert len(g) == SU.N_SUB
    h = SU.heterogeneity(g); f = F8.dispersion(g)[0]; sw = SD.sliced_w1(SD.cp10k_log1p(g), ref_log, SU.N_PROJ, SU.PROJ_SEED)
    return {"sliced_W1": sw["mean"], "median_fano": f["fano_median"], "libsize_CV": h["libsize_CV"], "detected_genes_CV": h["detected_genes_CV"],
            "libsize_mean": h["libsize_mean"], "detected_genes_mean": h["detected_genes_mean"]}
# gate: reproduce the OURS seed-20260931 row of 202_tableB1_full.json
sys.path.insert(0, f"{W}/scripts/r14b"); import r14b_common as RC
B1 = json.load(open(f"{W}/from_windows/r14_fano/202_tableB1_full.json"))
want = [r for r in B1["rows"] if r["generator"] == "OURS" and r["seed"] == 20260931][0]
got = score(RC.load_pool("OURS", 20260931))
gate = {k: {"mac": got[k], "table": (want[k] if k in want else want["heterogeneity"][k]), } for k in ("sliced_W1", "median_fano", "libsize_CV", "detected_genes_CV")}
for v in gate.values(): v["identical"] = v["mac"] == v["table"]; v["abs_diff"] = abs(v["mac"] - v["table"])
print("gate (OURS s20260931 vs 202_tableB1_full.json):", json.dumps({k: (v["identical"], f"{v['abs_diff']:.1e}") for k, v in gate.items()}))
assert all(v["abs_diff"] <= 1e-12 for v in gate.values()), gate
rows = []
for s in (20260931, 20260932, 20260933):
    p = f"{PD}/uncond_K128_seed{s}.npy"; sc = json.load(open(p[:-4] + ".json")); assert sc["K"] == 128 and sc["type"] is None and sc["seed"] == s, sc
    x = np.load(p, mmap_mode="r"); r = {"seed": s, "pool": p, "pool_sha256": sha(p), "n_pool": int(x.shape[0]), **score(x)}
    rows.append(r); print(s, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
mean = {k: float(np.mean([r[k] for r in rows])) for k in ("sliced_W1", "median_fano", "libsize_CV", "detected_genes_CV", "libsize_mean")}
res = {"generator": "CountsDiff K128 g1.0 uncond (train 20260625)", "library_size_source": "generated", "rows": rows, "mean_over_3_seeds": mean,
       "gate_vs_202": gate, "reference": {"n": SU.N_SUB, "seed": SU.VAL_SUB_SEED, "val_sha256": sha(f"{W}/data/hvg2k/val.npy")[:16]},
       "pool_fraction_columns": "NOT computed here (CellTypist, Windows)", "script_sha256": sha(__file__)}
json.dump(res, open(out, "w"), indent=1); print("mean", {k: round(v, 4) for k, v in mean.items()})
