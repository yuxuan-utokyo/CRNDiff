# -*- coding: utf-8 -*-
"""R16 pools on HLCA: rebuttal/r13_make_pool.py (sha 9a70a6ae...) with three changes only, for the vice 11-09 20:28 plan:
  1. RG.use_hlca() instead of RG.use_dg()  (HLCA data, gene hash, NMAX 4096 and T_O 5.7487 from ~/QIU/hlca/data_hvg2000 via R16_HLCA_DIR);
  2. the sample seed is an argument (--samps; checkpoint 1: 20260987 for s20260625, 20260989 for s20260630) instead of fixed 20260902;
  3. output root from R16_POOL_ROOT (default ~/QIU/hlca/pools), not rebuttal/out/R13/pool_root;
  4. rebuttal/bank_place.install(): the bank is the frozen DedupBankV2(host_resident=True) instead of DedupBank (placement only);
  5. (v2, vice 11-09 22:5x) cell chunk 128 for all HLCA sampling: chunk 256 OOMs at NMAX 4096 (smoke 22:53, row tensor 15.6 GB).
Everything else as R13 / HVG2K: the frozen sampler imported unedited, K=32, --batched, no A4, tau as given.
    python -m rebuttal.r16_make_pool --ckpt <hlca_generator.pt> --tau 0 --samps 20260987 --n-gen 25000 --tag _R16_uncond_s20260625
"""
import argparse, os, sys
from pathlib import Path
sys.dont_write_bytecode = True
HVG = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HVG))
import time as _t
while (HVG / "rebuttal/logs/PAUSE_FOR_Q").exists(): _t.sleep(30)
import torch
if os.environ.get("GPU_CAP_GB"):
    torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_CAP_GB"]) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
from rebuttal import resumable_gen as RG     # imports the model and config used by the sampler
RG.use_hlca()
import sample_hvg2k_noa4 as S
assert S.CFG is RG.CFG and S.M is RG.M, "sampler must read the HLCA config object"
assert RG.CFG.NMAX == 4096, RG.CFG.NMAX
from rebuttal import bank_place          # change 4 (vice 21:0x, option a): bank placement only, frozen files sha-checked
bank_place.install()
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tau", required=True); ap.add_argument("--logpi"); ap.add_argument("--tag", required=True)
ap.add_argument("--n-gen", default="25000"); ap.add_argument("--samps", required=True)
a = ap.parse_args()
ROOT = Path(os.environ.get("R16_POOL_ROOT", str(Path.home() / "QIU/hlca/pools"))); (ROOT / "Baseline" / "OURS" / "results").mkdir(parents=True, exist_ok=True)
S.ROOT, S.LOG = ROOT, ROOT / "RUN_LOG.md"
sys.argv = ["sample_hvg2k_noa4.py", "--ckpt", a.ckpt, "--ks", "32", "--samps", a.samps, "--n-gen", a.n_gen, "--batched", "--cell-chunk", "128", "--tau", a.tau, "--tag", a.tag, "--no-a4"] + (["--logpi", a.logpi] if a.logpi else [])
S.main()
