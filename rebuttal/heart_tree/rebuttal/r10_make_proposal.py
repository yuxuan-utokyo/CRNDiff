# -*- coding: utf-8 -*-
"""R10: regenerate the 20,000-cell CONDFULL Neuronal proposal of train 20260625.

The original file lived under the retired Experiments/X tree and is not on disk any more. The
frozen sampler is imported unedited with the arguments in tools/t214_ours_downstream.py stage 3 (--logpi, --tag, --root
select another pool, e.g. the R9-B top-k tables);
only its two module-level output paths are redirected, so nothing is written under assets/.
Original run for comparison (RUN_LOG.md): mean=0.1694 max=512 zeros=0.9487.

    cd Experiments/HVG && python -m rebuttal.r10_make_proposal
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HVG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HVG / "assets" / "legacy_root" / "Baseline" / "OURS" / "code"))
import os, time
# Board 11-08 01:03: 2 CPU threads; board 11-07 19:37: wait while logs/PAUSE_FOR_Q exists, before touching the GPU.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "2"
while (HVG / "rebuttal/logs/PAUSE_FOR_Q").exists(): time.sleep(30)
import torch
torch.set_num_threads(2)
if os.environ.get("GPU_CAP_GB"):
    torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_CAP_GB"]) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
import sample_hvg2k_noa4 as S
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--logpi", default=str(HVG / "assets/tables/tables_kz/logc_Neuronal.npy")); _ap.add_argument("--tag", default="_CONDFULL_Neuronal_tables_kz_n20000_g0p4"); _ap.add_argument("--root", default=str(HVG / "rebuttal" / "out" / "R10" / "pool_root"))
_a = _ap.parse_args()
ROOT = Path(_a.root); (ROOT / "Baseline" / "OURS" / "results").mkdir(parents=True, exist_ok=True)
S.ROOT, S.LOG = ROOT, ROOT / "RUN_LOG.md"
sys.argv = ["sample_hvg2k_noa4.py", "--ckpt", str(HVG / "assets/models/generator/pilot_run1c_s20260625.pt"),
            "--ks", "32", "--samps", "20260902", "--n-gen", "20000", "--batched", "--cell-chunk", "256",
            "--tau", "0.4", "--logpi", _a.logpi,
            "--tag", _a.tag, "--no-a4"]
S.main()
