# -*- coding: utf-8 -*-
"""[generated verbatim from train_pilot.py by gen_ls_train.py; only --label-smooth / --ls-kernel added.]

⚠ ****must be trained in pairs**:`pilot_run1c` was produced by three resumed segments (step_offset=7800; in each segment the optimizer and
EMA cold start); comparing an LS model trained from scratch against it would mix in"different training schedule". Train one each with the same script, steps and seed:

    python train_pilot_ls.py --steps 12000 --label-smooth 0     --name cectrl
    python train_pilot_ls.py --steps 12000 --label-smooth 3e-4  --name ls3e4

s_per_step≈0.44(batch 384); 12000 steps take about 88 minutes per arm.

hvg2k · OURS pilot training.

    python OURS/code/train_pilot.py --smoke          # 200 step smoke test (first check that the loss decreases and memory does not blow up)
    python OURS/code/train_pilot.py                  # formal pilot (1 seed)
    python OURS/code/train_pilot.py --batch 128      # reduce when out of memory batch

criterion (main line B5): stop this configuration after 3 consecutive val evaluations without improvement, record, and try another.
outputs:`OURS/models/pilot_s<seed>.pt` + `OURS/loss/pilot_s<seed>_{train,val,step}.npy`
      + `OURS/models/pilot_s<seed>.json`(including the trigger values of all criteria).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent                  # .../hvg2k/OURS/code
ROOT = Path(__file__).resolve().parents[3]                                  # .../toy/hvg2k
sys.path.insert(0, str(HERE))
import config_hvg2k as CFG                              # noqa: E402
import model_hvg2k_ls as M                           # noqa: E402  ← the only import change

LOG = ROOT / "RUN_LOG.md"


def rl(m):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"- `{time.strftime('%H:%M:%S')}` {m}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-smooth", type=float, default=0.0,
                    help="★ training-side back-off. 0 = bit-identical to train_pilot.py."
                         "hvg2k eps budget: under uniform_sup, 2%% extra mass corresponds to 2.4e-4,"
                         "suggest trying 1e-4 / 3e-4 (toy optimum is 3e-4)")
    ap.add_argument("--ls-kernel", default="uniform_sup",
                    choices=["uniform_sup", "uniform"],
                    help="uniform leaves a horizontal line above d_g, and is 8 times more expensive on hvg2k")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=CFG.SEED_PILOT)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", default=None, help="resume training from a .pt/.best/.last (optimizer cold start)")
    ap.add_argument("--t-max", type=float, default=None,
                    help="upper bound of t sampling during training. Default = CFG.T_HORIZON (T_O=5.2021)."
                         "the T_m arm of the horizon ablation passes 8.0: **each horizon gets its own training interval**:"
                         "run1c during training t~log-uniform[1e-2, 5.2021]; the network has never seen t>5.2,"
                         "sampling it at T_m=8 would be extrapolation, and the ablation would be invalid.")
    ap.add_argument("--name", default=None,
                    help="output name (overrides the name built from tag), e.g. run2_T8")
    ap.add_argument("--step-offset", type=int, default=0,
                    help="starting global step. Experiment X was cut into three segments by early stopping; run1c passes 7800,"
                         "so the curves can be joined by global step (main line 2026-08-10 loss record section 1)")
    a = ap.parse_args()
    steps = a.steps or (CFG.STEPS_SMOKE if a.smoke else CFG.STEPS_PILOT)
    batch = a.batch or CFG.BATCH

    d, meta = M.load_data(("train", "val"))
    tr, va = d["train"], d["val"]
    der = M.derive(tr)
    T = M.horizon()
    tag = a.name or (("smoke" if a.smoke else "pilot") + a.tag)
    T_MAX = a.t_max if a.t_max is not None else T
    rl(f"### hvg2k OURS {tag} start · {CFG.DATASET}/ train{list(tr.shape)} val{list(va.shape)}")
    rl(f"  gene_hash check passed {CFG.GENE_HASH[:16]}…")
    rl(f"  V: log2 2^{CFG.V_GRID_LO}..2^{CFG.V_GRID_HI} -> #V={der['n_distinct_V']} "
       f"sum_V={der['sum_V']:.1f} **sum_V/lib_mean={der['sum_V_over_lib']:.4f}** "
       f"(B2 criterion [0.95,1.10] {'pass' if 0.95 <= der['sum_V_over_lib'] <= 1.10 else 'fails'})")
    rl(f"  NMAX={CFG.NMAX} clip_x0={CFG.CLIP_X0_TO_NMAX};"
       f"**training T_MAX={T_MAX:.4f}** (default horizon {CFG.T_HORIZON}={T:.4f});"
       f"e^-T_MAX·NMAX={np.exp(-T_MAX)*CFG.NMAX:.3f} (published upper bound 6.52 = 10D D1 main-text row)")
    rl(f"  network AttnX0Posterior d{CFG.D_MODEL}/h{CFG.NHEADS}/L{CFG.NLAYERS},"
       f"batch={batch} steps={steps} lr={CFG.LR} ema={CFG.EMA_DECAY}")

    t0 = time.time()
    try:
        net, curve, info = M.train_ours_hvg2k(
            tr, va, der["V"], der["iscale"], a.seed, steps,
            batch=batch, log_fn=lambda x: print(x, flush=True), t_max=T_MAX,
            ckpt_path=ROOT / "Baseline" / "OURS" / "models" / f"{tag}_s{a.seed}.pt",
            resume_from=a.resume,
            label_smooth=a.label_smooth, ls_kernel=a.ls_kernel)
    except torch.cuda.OutOfMemoryError as e:
        rl(f"  [SCALE-CUT] batch={batch} out of memory ({repr(e)[:100]}): suggest halving and retrying")
        return 2
    dt = time.time() - t0
    (ROOT / "Baseline" / "OURS" / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "Baseline" / "OURS" / "loss").mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), ROOT / "Baseline" / "OURS" / "models" / f"{tag}_s{a.seed}.pt")
    # step always record the **global** step (step within segment + offset), so the three segments join into one curve
    curve["step"] = [int(s) + a.step_offset for s in curve["step"]]
    for k in ("train", "val", "step"):
        np.save(ROOT / "Baseline" / "OURS" / "loss" / f"{tag}_s{a.seed}_{k}.npy", np.asarray(curve[k]))
    blob = {"tag": tag, "seed": a.seed, "walltime_s": dt, "dataset": CFG.DATASET,
            "gene_hash": CFG.GENE_HASH, "T_horizon": CFG.T_HORIZON, "T": T,
            "grid": CFG.GRID, "F_BASE": CFG.F_BASE,
            "V_grid": [CFG.V_GRID_LO, CFG.V_GRID_HI],
            **{k: der[k] for k in ("n_distinct_V", "sum_V", "lib_mean", "sum_V_over_lib")},
            "steps_requested": steps, "step_offset": a.step_offset,
            "resume_from": (Path(a.resume).name if a.resume else None),
            "s_per_step": dt / max(len(curve["step"]) and steps, 1), **info,
            # ★ json used to store only the last 5 points of curve_tail, making the single entry point incomplete.
            #   now the **full** record is stored, together with the relative paths of the three .npy files (main-line loss record section 3).
            "curve": {k: [float(x) for x in v] for k, v in curve.items()},
            "curve_npy": {k: f"OURS/loss/{tag}_s{a.seed}_{k}.npy"
                          for k in ("train", "val", "step")}}
    (ROOT / "Baseline" / "OURS" / "models" / f"{tag}_s{a.seed}.json").write_text(
        json.dumps(blob, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    rl(f"  {tag} done {dt/60:.1f} min,n_params={info['n_params']:,},"
       f"stopped_at={info['stopped_at']},best_val={info['best_val']:.4f},"
       f"loss {curve['train'][0]:.4f} -> {curve['train'][-1]:.4f}"
       + (f", peak memory {torch.cuda.max_memory_allocated()/2**30:.1f} GB"
          if torch.cuda.is_available() else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
