# -*- coding: utf-8 -*-
"""R14 gen27/30 prep (vice 07:0x/07:1x: prep only, a1 FK held for the morning). For generator 20260627 or 20260630, reproduce the s25
R14 (INTERSECT_Endothelial_g128) recipe step by step, changing only the generator checkpoint and the generator-derived inputs; every
frozen script is imported/run UNEDITED and only its output paths are redirected under rebuttal/out/R14g/ (nothing is written under
assets/ or legacy_root/).

  table: tools/build_ref_table.py (260ceb2d) --ref INTERSECT:Endothelial:128 --x-root assets/legacy_root
         --pilot <legacy_root>/Baseline/OURS/results/linspace_T_O_K32_spilot_run214_s<GEN>_samp20260902_batched1_chunk256_noa4.npy
         --ratio-cap 1000 --eps 0.5 -o rebuttal/out/R14g/s<GEN>/tables_ref            (s25: same, --pilot = its own calibration pool)
  pool:  sample_hvg2k_noa4.py (6cc17088) --ckpt <legacy_root>/Baseline/OURS/models/pilot_run214_s<GEN>.pt --ks 32 --samps 20261044
         --n-gen 10000 --cell-chunk 256 --batched --tau 0.4 --logpi <the table above> --no-a4 --tag _UNI_INTERSECT_Endo_g128_n10000_g0p4
         with S.ROOT/S.LOG -> rebuttal/out/R14g/s<GEN>/pool_root (as rebuttal/r10_make_proposal.py)   (s25: ckpt pilot_run1c_s20260625.pt)
  disc:  CGE/HVG2K/universality_fk/code/fit_ref_disc.py (8d0d4d77) --ref INTERSECT:Endothelial:128 --pool <the pool above>
         --fit-n 4590 --val-pos-n 2295 --val-neg-n 2000 -o rebuttal/out/R14g/s<GEN>/disc_INTERSECT_g128
         with its module constant D -> assets/legacy_root/_shared/data/all (as the 184C bit-exact reproduction did), seed default 20260817;
         then tools/export_hgbt_portable.py (4a99b3cd) <out>/residual_classifier.joblib <out>/residual_classifier_portable.npz
    python -u rebuttal/r14g_prep.py <table|pool|disc> --gen 20260627
"""
import argparse, importlib.util, os, runpy, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
HVG = Path(__file__).resolve().parents[1]
LR = HVG / "assets" / "legacy_root"
CGE = HVG.parents[1] / "Conditional generation experiments" / "HVG2K"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "4"
ap = argparse.ArgumentParser(); ap.add_argument("step", choices=("table", "pool", "disc")); ap.add_argument("--gen", required=True, choices=("20260627", "20260630"))
a = ap.parse_args()
O = HVG / "rebuttal" / "out" / "R14g" / f"s{a.gen}"; O.mkdir(parents=True, exist_ok=True)
TAB = O / "tables_ref" / "logc_INTERSECT_Endothelial_g128.npy"
TAG = "_UNI_INTERSECT_Endo_g128_n10000_g0p4"
POOL = O / "pool_root" / "Baseline" / "OURS" / "results" / f"linspace_T_O_K32_spilot_run214_s{a.gen}_samp20261044_batched1_chunk256{TAG}_noa4.npy"
if a.step == "table":
    pilot = LR / "Baseline" / "OURS" / "results" / f"linspace_T_O_K32_spilot_run214_s{a.gen}_samp20260902_batched1_chunk256_noa4.npy"
    assert pilot.exists(), pilot
    sys.argv = ["build_ref_table.py", "--ref", "INTERSECT:Endothelial:128", "--x-root", str(LR), "--pilot", str(pilot),
                "--ratio-cap", "1000", "--eps", "0.5", "-o", str(O / "tables_ref")]
    runpy.run_path(str(HVG / "tools" / "build_ref_table.py"), run_name="__main__")
elif a.step == "pool":
    assert TAB.exists(), TAB
    sys.path.insert(0, str(LR / "Baseline" / "OURS" / "code"))
    import torch; torch.set_num_threads(4)
    import sample_hvg2k_noa4 as S
    root = O / "pool_root"; (root / "Baseline" / "OURS" / "results").mkdir(parents=True, exist_ok=True)
    S.ROOT, S.LOG = root, root / "RUN_LOG.md"
    sys.argv = ["sample_hvg2k_noa4.py", "--ckpt", str(LR / "Baseline" / "OURS" / "models" / f"pilot_run214_s{a.gen}.pt"),
                "--ks", "32", "--samps", "20261044", "--n-gen", "10000", "--cell-chunk", "256", "--batched", "--tau", "0.4",
                "--logpi", str(TAB), "--no-a4", "--tag", TAG]
    S.main()
else:
    assert POOL.exists(), POOL
    spec = importlib.util.spec_from_file_location("fit_ref_disc", CGE / "universality_fk" / "code" / "fit_ref_disc.py")
    F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
    F.D = LR / "_shared" / "data" / "all"
    out = O / "disc_INTERSECT_g128"
    sys.argv = ["fit_ref_disc.py", "--ref", "INTERSECT:Endothelial:128", "--pool", str(POOL), "--fit-n", "4590", "--val-pos-n", "2295",
                "--val-neg-n", "2000", "-o", str(out)]
    F.main()
    rc = subprocess.run([sys.executable, "-u", str(HVG / "tools" / "export_hgbt_portable.py"), str(out / "residual_classifier.joblib"),
                         str(out / "residual_classifier_portable.npz")], cwd=HVG).returncode
    sys.exit(rc)
