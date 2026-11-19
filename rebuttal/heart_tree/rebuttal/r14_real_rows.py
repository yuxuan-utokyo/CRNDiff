# -*- coding: utf-8 -*-
"""R14 table, two real-cell rows. r14_score.py (ac7bc4c6) lineage; ONE test read, logged.
Frozen scorers imported unedited and sha-asserted: analysis/score_intersect.py 86da6ce1, analysis/score_e21_intersect.py e3693a2a
(its _frozen e21_scc_mmd / e21c_pcc via E21.guard + E21.load_frozen), analysis/score_purity.py 9b1cb6a0 (ruler_check / annotate).
The test reference is r14_score.build_reference_test, copied verbatim (same file pair, rule, SPLIT_SEED halves, SUB_SEED, n).
  row 1  purity of ALL 858 real test Endothelial & UAP1+ cells: score_purity.annotate (CellTypist, ruler gate G5) and
         purity_from_hist on the master_purity_v7 Endothelial fixed_labels (the R14 alias INTERSECT_Endothelial_g128 -> Endothelial);
         whole-set purity (858 < the 2500 block, so no blocks). The label histogram is saved (v8 can be read off it).
  row 2  pool = ALL real test Endothelial cells (any UAP1) MINUS the reference half A (idx_A, 429 cells; = the reference itself);
         ONE draw of n = 429 with default_rng(DRAW_SEED = SI.SUB_SEED 20261071), sorted; scored exactly like a generated row:
         SI.score (marginal W1 mean/median, sliced W1) vs the same ref, ratios to the same test floor (half B);
         E21: same sigma (exact_bandwidth of ref), same k_rr, E.score + PCC.corr_pair (MMD2 / PCC / SCC).
Gates (else exit 2): the in-script floor W1 / sliced W1 equal intersect_R14_test.json, and sigma + floor MMD2/PCC equal
e21_intersect_rebuttal_R14g_test.json (the generated rows' numbers), so the rows sit on the same ruler.
    python -u rebuttal/r14_real_rows.py
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HVG = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HVG))
from hvg import config as C
SHA = {"analysis/score_intersect.py": "86da6ce1b56ad967cdd0e858c3fa11a3b899f58267a678ac34e3b634928892f8",
       "analysis/score_e21_intersect.py": "e3693a2a2f883ad5a7a96a34d4fa029c496734e2c8f689c2eb97c75958483cc4",
       "analysis/score_purity.py": "9b1cb6a0c5514f9c"}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for k, v in SHA.items():
    assert sha(HVG / k).startswith(v), f"{k} changed"
import analysis.score_intersect as SI
import analysis.score_e21_intersect as E21
import analysis.score_purity as SP
from hvg.metrics import load_fixed_label_sets, purity_from_hist
OUT = HVG / "rebuttal" / "out" / "R14"
OLD_I = json.loads((OUT / "intersect_R14_test.json").read_text(encoding="utf-8"))
OLD_E = json.loads((OUT / "e21_intersect_rebuttal_R14g_test.json").read_text(encoding="utf-8"))
DRAW_SEED = SI.SUB_SEED
t0 = time.time()
D = C.LEGACY_DATA / "all"; files = {f: D / f for f in ("test.npy", "test_celltype.npy")}
log = {"what": "R14 real-cell rows: purity of the 858 test Endothelial & UAP1+ cells; all test Endothelial "
               "minus reference half A, one draw n=429, vs the reference half (W1/jSWD/MMD2/PCC/SCC)", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
       "files": {k: {"path": str(v), "sha256": sha(v)} for k, v in files.items()}, "script_sha256": sha(Path(__file__))}


def build_reference_test(arm_sizes):
    """score_intersect.build_reference with ONLY the file pair changed to the test split."""
    Xv = np.load(files["test.npy"], mmap_mode="r")
    yv = np.load(files["test_celltype.npy"], allow_pickle=True).astype(str)
    sel = np.flatnonzero((yv == "Endothelial") & (np.asarray(Xv)[:, SI.GENE] > 0))
    rs = np.random.default_rng(SI.SPLIT_SEED); perm = rs.permutation(len(sel)); half = len(sel) // 2
    idx_A, idx_B = np.sort(sel[perm[:half]]), np.sort(sel[perm[half:2 * half]])
    n = int(min([len(idx_A), len(idx_B)] + list(arm_sizes)))
    rng = np.random.default_rng(SI.SUB_SEED)
    ref = np.asarray(Xv[idx_A[np.sort(rng.choice(len(idx_A), n, replace=False))]], dtype=np.float64)
    flo = np.asarray(Xv[idx_B[np.sort(rng.choice(len(idx_B), n, replace=False))]], dtype=np.float64)
    return {"ref": ref, "flo": flo, "n": n, "n_total": int(len(sel)), "half_sizes": [int(len(idx_A)), int(len(idx_B))],
            "data": str(files["test.npy"].resolve()), "_sel": sel, "_idx_A": idx_A, "_idx_B": idx_B, "_X": Xv, "_y": yv}


R = build_reference_test([429]); ref, flo, n, X, y = R["ref"], R["flo"], R["n"], R["_X"], R["_y"]
assert n == 429 and R["n_total"] == 858 and len(R["_idx_A"]) == 429, (n, R["n_total"])
fail = []
# ---- row 1: purity of all 858 -------------------------------------------------------------------------------------------
ruler = SP.ruler_check()
genes = json.loads((D / "meta.json").read_text(encoding="utf-8"))["genes"]
labels_v7 = load_fixed_label_sets(C.MASTER_PURITY_V7)["Endothelial"]
lab = SP.annotate(np.asarray(X[R["_sel"]], dtype=np.float64), genes)
u, c = np.unique(lab, return_counts=True); hist = {str(k): int(v) for k, v in zip(u, c)}
p7, n7 = purity_from_hist(hist, labels_v7)
row1 = {"row": "real test Endothelial & UAP1+ (all)", "n": int(n7), "purity_v7_whole_set": p7, "label_set": sorted(labels_v7),
        "label_set_source": str(C.MASTER_PURITY_V7), "label_dist_whole": hist, "ruler": ruler,
        "note": "whole-set purity on all 858 cells (fewer than the 2500 block of score_purity, so no blocks)"}
print(f"[row1] purity v7 = {p7:.4f} (n {n7})", flush=True)
# ---- row 2: pool and one draw -------------------------------------------------------------------------------------------
endo_all = np.flatnonzero(y == "Endothelial")
pool = np.setdiff1d(endo_all, R["_idx_A"])
rng = np.random.default_rng(DRAW_SEED); pick = np.sort(pool[np.sort(rng.choice(len(pool), n, replace=False))])
g = np.asarray(X[pick], dtype=np.float64)
M, sw1 = SI.load_frozen(); proj = M.make_projections(ref.shape[1], n_proj=SI.N_PROJ, seed=SI.PROJ_SEED)
floor = SI.score(flo, ref, proj, M, sw1)
for k in ("marginal_w1_mean", "sliced_w1"):
    if abs(floor[k] - OLD_I["floor"][k]) > 1e-12: fail.append(f"floor {k} {floor[k]} != {OLD_I['floor'][k]}")
s = SI.score(g, ref, proj, M, sw1)
E21.guard(); E, PCC = E21.load_frozen()
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
ref_np, _ = E.cp10k_log1p(ref); ref_gpu = torch.from_numpy(ref_np).to(dev); sigma = E.exact_bandwidth(ref_gpu)
k_rr = E.kernel_mean(ref_gpu, ref_gpu, sigma)
ef = E.score(ref_np, ref_gpu, flo, sigma, k_rr); flo_np, _ = E.cp10k_log1p(flo)
_, f_pcc = PCC.corr_pair(ref_np.mean(0, dtype=np.float64), flo_np.mean(0, dtype=np.float64))
of = OLD_E["gate_floor_sanity"]["floor"]
if abs(sigma - OLD_E["bandwidth"]["sigma_used"]) > 1e-6: fail.append(f"sigma {sigma} != {OLD_E['bandwidth']['sigma_used']}")
if abs(ef["mmd2_rbf_biased"] - of["mmd2_rbf_biased"]) > 1e-9 or abs(f_pcc - of["pcc"]) > 1e-9:
    fail.append(f"e21 floor {ef['mmd2_rbf_biased']}/{f_pcc} != {of['mmd2_rbf_biased']}/{of['pcc']}")
es = E.score(ref_np, ref_gpu, E.select_rows(g, n, np.random.default_rng(E.SEED)), sigma, k_rr)
g_np, _ = E.cp10k_log1p(g); scc2, pcc = PCC.corr_pair(ref_np.mean(0, dtype=np.float64), g_np.mean(0, dtype=np.float64))
if abs(scc2 - es["scc"]) > 1e-12: fail.append("SCC paths disagree")
row2 = {"row": "real test Endothelial (all UAP1 states) minus reference half A, one draw", "pool_n": int(len(pool)),
        "pool_rule": "test Endothelial (any gene128) minus idx_A (the 429 reference-half cells)", "draw_seed": int(DRAW_SEED), "n": int(n),
        "drawn_test_rows": pick.tolist(), "n_drawn_in_half_B": int(np.isin(pick, R["_idx_B"]).sum()),
        "uap1_pos_frac_in_draw": float((g[:, SI.GENE] > 0).mean()), "uap1_pos_frac_in_pool": float((np.asarray(X[pool])[:, SI.GENE] > 0).mean()),
        **{k: s[k] for k in s}, "mW1_over_floor": s["marginal_w1_mean"] / floor["marginal_w1_mean"],
        "jSWD_over_floor": s["sliced_w1"] / floor["sliced_w1"], "mmd2_rbf_biased": es["mmd2_rbf_biased"], "pcc": pcc, "scc": es["scc"],
        "e21_device": dev}
print(f"[row2] pool {len(pool)} mW1/floor {row2['mW1_over_floor']:.3f} jSWD/floor {row2['jSWD_over_floor']:.3f} "
      f"MMD2 {es['mmd2_rbf_biased']:.6f} PCC {pcc:.6f} SCC {es['scc']:.6f}", flush=True)
res = {"split": "test", "reference": {k: R[k] for k in ("n", "n_total", "half_sizes", "data")}, "floor_w1": floor,
       "floor_e21": {**{k: ef[k] for k in ("mmd2_rbf_biased", "scc")}, "pcc": f_pcc}, "sigma_used": sigma,
       "gates": {"failures": fail, "passed": not fail, "against": ["intersect_R14_test.json floor", "e21_intersect_rebuttal_R14g_test.json sigma/floor"]},
       "row1_purity": row1, "row2_all_endothelial": row2, "script_sha256": log["script_sha256"], "seconds": round(time.time() - t0, 1)}
(OUT / "R14_real_rows_test.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); log["rows_read"] = {"test Endothelial": int(len(endo_all)), "UAP1+": R["n_total"], "pool": int(len(pool))}
(OUT / "test_read_log_real_rows.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
print("[gates]", "PASS" if not fail else fail, flush=True)
sys.exit(0 if not fail else 2)
