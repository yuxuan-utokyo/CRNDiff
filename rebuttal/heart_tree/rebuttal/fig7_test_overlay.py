# -*- coding: utf-8 -*-
"""Fig. 7 overlay on TEST (vice 10:3x, for the manuscript; CPU, <= 4 threads). Projects through the frozen heart UMAP reducer that
Fig 7d used (figures/hvgfig.py FrozenUMAP: assets/legacy_root/Processing_data/data/umap_reducer.pkl, PCA50 -> UMAP, transform only,
never refit), with the same preprocessing (cpm_log1p -> frozen pca50_val.npz -> reducer.transform), exactly as rebuttal/r1_fig7d.py:
  (a) all HVG2K test cells + their cell-type labels (background)        -> xy_test, y_test
  (b) real test Neuronal cells = the Neuronal subset of (a)             -> mask_test_neuronal
  (c) the Fig 7d generated Neuronal run (gen 20260625 / samp 20260987)  -> xy_gen   (gate: equals the paper panel's cached coords)
Only the real cells change from val to test. Reducer gate (embedding_ == umap_xy_val_real.npy) as FrozenUMAP.gate_against_archive.
Writes rebuttal/out/fig7_test/{fig7_test_overlay.npz, fig7_test_overlay.json}; the test files read are logged with sha256.
    python -u rebuttal/fig7_test_overlay.py
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "4"
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from figures import hvgfig as H
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUT = ROOT / "rebuttal" / "out" / "fig7_test"; OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
um = H.FrozenUMAP(); gate = um.gate_against_archive()
if not gate["passed"]: raise SystemExit(f"reducer gate failed: {gate['max_abs_delta']}")
tx, ty = H.DATA / "test.npy", H.DATA / "test_celltype.npy"
xt = np.asarray(np.load(tx, mmap_mode="r"), dtype=np.float64); yt = np.load(ty, allow_pickle=True).astype(str)
print(f"[test] {xt.shape} neuronal {int((yt == 'Neuronal').sum())}", flush=True)
xy_test = um.project(xt); print(f"[umap] test projected ({time.time() - t0:.0f}s)", flush=True)
gp = ROOT / "out" / "runs" / "table1_ours" / "Neuronal_a1" / \
    "Neuronal_residual_hvgtwist_fk_tiltevery_trigany_K32_M5000_N2500_alpha1p0_J16_int8_th0p5_tau0p2_cc256_odds_seed20260987.npy"
xg = np.asarray(np.load(gp), dtype=np.float64); xy_gen = um.project(xg)
cached = np.load(H.FIG / "fig_umap_panels.coords.npz")["Neuronal_umap"]
delta = float(np.abs(xy_gen - cached).max()); print(f"[gate] generated s987 vs paper panel coords max |delta| = {delta:.3e}", flush=True)
np.savez_compressed(OUT / "fig7_test_overlay.npz", xy_test=xy_test, y_test=yt, mask_test_neuronal=(yt == "Neuronal"), xy_gen=xy_gen,
                    xy_val_real_reference=um.real_xy)
json.dump({"what": "Fig 7 overlay on TEST through the frozen heart UMAP reducer (transform only)", "reducer_gate": gate,
           "test_files": {str(tx): sha(tx), str(ty): sha(ty)}, "n_test": int(len(yt)), "n_test_neuronal": int((yt == "Neuronal").sum()),
           "generated_run": {"path": str(gp), "sha256": sha(gp), "n": int(len(xg)), "gen": 20260625, "samp": 20260987},
           "gate_generated_vs_paper_panel_max_abs_delta": delta, "script_sha256": sha(Path(__file__)),
           "preprocessing": "hvgfig.cpm_log1p -> (x - pca50.mean) @ pca50.components.T -> reducer.transform (FrozenUMAP.project)",
           "threads": 4, "seconds": round(time.time() - t0, 1)}, open(OUT / "fig7_test_overlay.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"[done] {time.time() - t0:.0f}s", flush=True)
