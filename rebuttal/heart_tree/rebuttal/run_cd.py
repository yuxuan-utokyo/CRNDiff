# -*- coding: utf-8 -*-
"""Run the Mac's countsdiff2k.py (rebuttal/r12_src, sha256-checked copy) under a per-process GPU memory cap.
The script itself is not edited. Honours the pause gate (logs/PAUSE_FOR_Q) before touching the GPU.
    python -m rebuttal.run_cd --mem-gb 8 -- train --data ... --out ... --seed ... --nmax 512
"""
import os, runpy, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
# Board 11-08 01:03: at most 2 CPU threads per GPU line; pinned here, before numpy/torch load, over whatever the queue set.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "2"
ROOT = Path(__file__).resolve().parents[1]
while (ROOT / "rebuttal/logs/PAUSE_FOR_Q").exists(): time.sleep(30)
# Board 11-09 13:25: DG 42 moved to the GPU server. A DG sample started by the 4090 line-1 queue waits here, before torch
# (no GPU memory, no work), so that queue can be stopped cleanly; it is then stopped by hand.
while (ROOT / "rebuttal/logs/HOLD_LINE1_DG").exists() and any("r13" in a and "OUT" in a for a in sys.argv): time.sleep(10)
i = sys.argv.index("--"); gb = float(sys.argv[sys.argv.index("--mem-gb") + 1]); argv = sys.argv[i + 1:]
import torch
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(gb * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
src = ROOT / "rebuttal/r12_src/countsdiff2k.py"
sys.argv = [str(src)] + argv; print("[run_cd] cap", gb, "GB; argv", argv, flush=True)
runpy.run_path(str(src), run_name="__main__")
