# -*- coding: utf-8 -*-
"""R16 tilted FK on HLCA (vice 11-10 03:1x): rebuttal/r13_run_fk.py (sha 450f4890...) with only these changes:
  1. RG.use_hlca() instead of RG.use_dg() (HLCA data, gene hash, NMAX 4096, T_O from R16_HLCA_DIR), NMAX assert 4096;
  2. rebuttal/bank_place.install() before run_one builds the bank (BASE.DedupBank -> frozen DedupBankV2 host_resident, R16_BANK_MMAP_DIR cache).
  3. (v2, vice 11-10 03:32) optional per-step CUDA memory print (R16_MEMLOG_STEPS), logging only.
Original R13 docstring: experiments.run_one unedited, with the DG config set on the shared config object
in this process (resumable_gen.use_dg: DATA_DIR, GENE_HASH, NMAX=256, T_O=3.7404), as r13_make_pool does.
The DG generator, the Mac tilt table and the Mac classifier are passed explicitly (--ckpt, --logpi, --residual-dir).
    python -m rebuttal.r13_run_fk -- <run_one arguments>
"""
import os, sys
from pathlib import Path
sys.dont_write_bytecode = True
# Board 11-08 01:03: at most 2 CPU threads per GPU line; pinned here, before numpy/torch load, over whatever the queue set.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "2"
HVG = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HVG))
# Board 11-07 19:37: while logs/PAUSE_FOR_Q exists, a new run waits here (before torch, so it holds no GPU memory).
import time as _t
while (Path(__file__).resolve().parents[1] / "rebuttal/logs/PAUSE_FOR_Q").exists(): _t.sleep(30)
import torch
torch.set_num_threads(2)
if os.environ.get("GPU_CAP_GB"):
    torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_CAP_GB"]) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
from rebuttal import resumable_gen as RG
RG.use_hlca()
from hvg import config as C
C.add_base_code_to_path()
import sample_hvg2k_noa4 as S
assert S.CFG is RG.CFG and S.M is RG.M and S.CFG.NMAX == 4096, "sampler must read the HLCA config object"
from rebuttal import bank_place      # change 2: placement only, frozen shas asserted inside
bank_place.install()
# v2 (vice 11-10 03:32): logging only. For the first $R16_MEMLOG_STEPS calls of HVGEngine.pass_all, print the step and
# torch.cuda.max_memory_allocated / max_memory_reserved after the call. No argument, return value or RNG call is touched.
_memlog_n = int(os.environ.get("R16_MEMLOG_STEPS", "0"))
if _memlog_n > 0:
    from hvg import sampler as _HS
    _orig_pass_all, _memlog_count = _HS.HVGEngine.pass_all, [0]
    def _pass_all_memlog(self, state, step, *args, **kw):
        out = _orig_pass_all(self, state, step, *args, **kw)
        if _memlog_count[0] < _memlog_n:
            _memlog_count[0] += 1
            print(f"[r16_mem] pass_all #{_memlog_count[0]} step {step} max_alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
                  f"max_reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB", flush=True)
        return out
    _HS.HVGEngine.pass_all = _pass_all_memlog
argv = sys.argv[sys.argv.index("--") + 1:]
_tilt_only = "--sampler" in argv and argv[argv.index("--sampler") + 1] == "tilt_only"   # board 11-09 06:11: GATE_FAILED classes run tilt-only
for k in ("--ckpt", "--logpi") + (() if _tilt_only else ("--residual-dir",)):            # tilt_only loads no discriminator (run_one.py:280)
    assert k in argv, f"{k} must be explicit for HLCA"
if _tilt_only:
    assert "--residual-dir" not in argv and "--alpha" not in argv, "tilt_only takes no discriminator and no alpha"
from experiments import run_one
raise SystemExit(run_one.main(argv))
