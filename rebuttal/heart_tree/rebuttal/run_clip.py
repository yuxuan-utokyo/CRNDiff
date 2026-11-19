# -*- coding: utf-8 -*-
"""Run the frozen sampler with an upper clip on the odds, for R4 clip. Frozen files are not edited: the reward
rule is wrapped in this process only. Everything else is experiments.run_one. Writes how many reward evaluations
were clipped and the share of raw odds weight above the bound to rebuttal/out/R4clip/clip_stats_s<seed>.json.
    python -m rebuttal.run_clip --U <odds> -- <run_one arguments>
"""
import atexit, json, math, sys
from pathlib import Path
sys.dont_write_bytecode = True
import os
# Board 11-08 01:03: at most 2 CPU threads per GPU line; pinned here, before numpy/torch load, over whatever the queue set.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "2"
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
# Board 11-07 23:26: clip extra seeds postponed. While logs/SKIP_R4CLIP exists a launch exits at once with code 3
# (recorded as skipped in gpu_line.log by the queue's own exit note), so the queue goes on to scoring.
if (Path(__file__).resolve().parents[1] / "rebuttal/logs/SKIP_R4CLIP").exists():
    print("[skip] R4 clip extra seeds postponed; logs/SKIP_R4CLIP present", flush=True); raise SystemExit(3)
# Board 11-07 19:37: while logs/PAUSE_FOR_Q exists, a new run waits here (before torch, so it holds no GPU memory).
import time as _t
while (Path(__file__).resolve().parents[1] / "rebuttal/logs/PAUSE_FOR_Q").exists(): _t.sleep(30)
import numpy as np
i = sys.argv.index("--"); U = float(sys.argv[sys.argv.index("--U") + 1]); argv = sys.argv[i + 1:]
import os, torch
if os.environ.get("GPU_CAP_GB"):
    torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_CAP_GB"]) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
torch.set_num_threads(2)
import hvg.sampler as S
_orig = S.apply_reward_mode; LU = math.log(U)
st = {"U": U, "n_eval": 0, "n_clipped": 0, "raw_odds_sum": 0.0, "clipped_odds_sum": 0.0, "max_log_rho": -math.inf, "calls": 0}


def clipped(p, floor, mode):
    r = _orig(p, floor, mode); c = np.minimum(r, LU)
    st["calls"] += 1; st["n_eval"] += int(r.size); st["n_clipped"] += int((r > LU).sum())
    st["raw_odds_sum"] += float(np.exp(r).sum()); st["clipped_odds_sum"] += float(np.exp(c).sum()); st["max_log_rho"] = max(st["max_log_rho"], float(r.max()))
    return c


S.apply_reward_mode = clipped


def _write():
    st["share_evals_clipped"] = st["n_clipped"] / max(st["n_eval"], 1)
    st["share_raw_weight_removed"] = 1 - st["clipped_odds_sum"] / max(st["raw_odds_sum"], 1e-300)
    out = ROOT / "rebuttal/out/R4clip"; out.mkdir(parents=True, exist_ok=True); seed = argv[argv.index("--seed") + 1]; exp = argv[argv.index("--experiment") + 1]; (out / (f"clip_stats_s{seed}.json" if exp == "rebuttal_R4clip" else f"clip_stats_{exp}_s{seed}.json")).write_text(json.dumps(st, indent=1))
    print("[clip stats]", st, flush=True)


atexit.register(_write)
from experiments import run_one
print(f"[clip] log odds capped at log({U})", flush=True)
raise SystemExit(run_one.main(argv))
