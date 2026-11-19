# -*- coding: utf-8 -*-
"""experiments.run_one under a per-process GPU memory cap. PyTorch empties its cache before failing at the cap;
sampling code and numbers are untouched. Used so that two sampling runs fit on one GPU.
    python -m rebuttal.run_capped --mem-gb 9.5 -- <run_one arguments>
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import os
# Board 11-08 01:03: at most 2 CPU threads per GPU line; pinned here, before numpy/torch load, over whatever the queue set.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "2"
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
# Board 11-07 19:37: while logs/PAUSE_FOR_Q exists, a new run waits here (before torch, so it holds no GPU memory).
import time as _t
while (Path(__file__).resolve().parents[1] / "rebuttal/logs/PAUSE_FOR_Q").exists(): _t.sleep(30)
# Board 11-07 16:59: R3c waits for a verdict on R3b. Held here, before torch, so a waiting process holds no GPU memory.
import time
if "rebuttal_R3c" in sys.argv:
    while any((ROOT / "rebuttal/logs").glob("HOLD_R3c*")): time.sleep(15)   # HOLD_R3c (R13) and HOLD_R3c_mdlm
import torch
torch.set_num_threads(2)
i = sys.argv.index("--"); gb = float(sys.argv[sys.argv.index("--mem-gb") + 1]); argv = sys.argv[i + 1:]; print("[capped] argv", argv[:8], flush=True)
torch.cuda.set_per_process_memory_fraction(gb * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
from experiments import run_one
raise SystemExit(run_one.main(argv))
