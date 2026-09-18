# -*- coding: utf-8 -*-
"""hvg2k · OURS pilot 训练。

    python OURS/code/train_pilot.py --smoke          # 200 步冒烟（先确认 loss 在降、显存不炸）
    python OURS/code/train_pilot.py                  # 正式 pilot（1 个种子）
    python OURS/code/train_pilot.py --batch 128      # 显存不够时降 batch

判据（主线 B5）：连续 3 次 val 评估没有改善就停这个配置、记录、换一个再试。
产出：`OURS/models/pilot_s<seed>.pt` + `OURS/loss/pilot_s<seed>_{train,val,step}.npy`
      + `OURS/models/pilot_s<seed>.json`（含全部判据的触发值）。
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
import model_hvg2k as M                                 # noqa: E402

LOG = ROOT / "RUN_LOG.md"


def rl(m):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"- `{time.strftime('%H:%M:%S')}` {m}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=CFG.SEED_PILOT)
    # 工单 214-A 稳定化工具箱。三个都不传 = 与改动前逐字等价的默认配方。
    ap.add_argument("--clip-grad", type=float, default=None,
                    help="clip_grad_norm_ 的 max_norm；本仓库唯一真正缺的那一项"
                         "（warmup=500 与 cosine 调度在 model_hvg2k 里本来就有）")
    ap.add_argument("--lr", type=float, default=None, help=f"覆盖 CFG.LR（{CFG.LR}）")
    ap.add_argument("--warmup", type=int, default=None,
                    help=f"覆盖 CFG.WARMUP（{CFG.WARMUP}）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", default=None, help="从某个 .pt/.best/.last 续训（冷启优化器）")
    ap.add_argument("--t-max", type=float, default=None,
                    help="训练时 t 的采样上界。默认 = CFG.T_HORIZON(T_O=5.2021)。"
                         "视界消融的 T_m 臂传 8.0 —— **每个视界配它自己的训练区间**："
                         "run1c 训练时 t~log-uniform[1e-2, 5.2021]，网络从没见过 t>5.2 的样本，"
                         "拿它按 T_m=8 采样是外推，消融会废。")
    ap.add_argument("--name", default=None,
                    help="产物名（覆盖 tag 拼出来的名字），如 run2_T8")
    ap.add_argument("--step-offset", type=int, default=0,
                    help="全局步数起点。实验 X 被早停砍成三段，run1c 传 7800，"
                         "曲线才能按全局步数接起来（主线 2026-08-10 loss 记录 §1）")
    a = ap.parse_args()
    steps = a.steps or (CFG.STEPS_SMOKE if a.smoke else CFG.STEPS_PILOT)
    batch = a.batch or CFG.BATCH

    d, meta = M.load_data(("train", "val"))
    tr, va = d["train"], d["val"]
    der = M.derive(tr)
    T = M.horizon()
    tag = a.name or (("smoke" if a.smoke else "pilot") + a.tag)
    T_MAX = a.t_max if a.t_max is not None else T
    rl(f"### hvg2k OURS {tag} 开始 · {CFG.DATASET}/ train{list(tr.shape)} val{list(va.shape)}")
    rl(f"  gene_hash 校验通过 {CFG.GENE_HASH[:16]}…")
    rl(f"  V: log2 2^{CFG.V_GRID_LO}..2^{CFG.V_GRID_HI} -> #V={der['n_distinct_V']} "
       f"sum_V={der['sum_V']:.1f} **sum_V/lib_mean={der['sum_V_over_lib']:.4f}** "
       f"（B2 判据 [0.95,1.10] {'通过' if 0.95 <= der['sum_V_over_lib'] <= 1.10 else '不通过'}）")
    rl(f"  NMAX={CFG.NMAX} clip_x0={CFG.CLIP_X0_TO_NMAX}；"
       f"**训练 T_MAX={T_MAX:.4f}**（默认视界 {CFG.T_HORIZON}={T:.4f}）；"
       f"e^-T_MAX·NMAX={np.exp(-T_MAX)*CFG.NMAX:.3f}（已发表上界 6.52 = 10D D1 正文行）")
    rl(f"  网络 AttnX0Posterior d{CFG.D_MODEL}/h{CFG.NHEADS}/L{CFG.NLAYERS}，"
       f"batch={batch} steps={steps} lr={CFG.LR} ema={CFG.EMA_DECAY}")

    t0 = time.time()
    try:
        net, curve, info = M.train_ours_hvg2k(
            tr, va, der["V"], der["iscale"], a.seed, steps,
            batch=batch, log_fn=lambda x: print(x, flush=True), t_max=T_MAX,
            ckpt_path=ROOT / "Baseline" / "OURS" / "models" / f"{tag}_s{a.seed}.pt",
            resume_from=a.resume,
            clip_grad=a.clip_grad, lr=a.lr, warmup=a.warmup)
    except torch.cuda.OutOfMemoryError as e:
        rl(f"  [SCALE-CUT] batch={batch} 显存不够（{repr(e)[:100]}）—— 建议减半重试")
        return 2
    except M.DivergedError as e:
        # 工单 214-A：发散即非零退出，欠训模型不得进入下游。
        rl(f"  [DIVERGED] {e} —— 本次训练作废，不产出 {tag}_s{a.seed}.pt，非零退出")
        return 3
    dt = time.time() - t0
    (ROOT / "Baseline" / "OURS" / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "Baseline" / "OURS" / "loss").mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), ROOT / "Baseline" / "OURS" / "models" / f"{tag}_s{a.seed}.pt")
    # step 一律落**全局**步数（段内 step + offset），三段才能直接接成一条曲线
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
            # ★ json 曾经只存 curve_tail 的最后 5 个点，成了「唯一入口却是残缺的」。
            #   现在存**全量**，并同时写出三个 .npy 的相对路径（主线 loss 记录 §3）。
            "curve": {k: [float(x) for x in v] for k, v in curve.items()},
            "curve_npy": {k: f"OURS/loss/{tag}_s{a.seed}_{k}.npy"
                          for k in ("train", "val", "step")}}
    (ROOT / "Baseline" / "OURS" / "models" / f"{tag}_s{a.seed}.json").write_text(
        json.dumps(blob, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    rl(f"  {tag} 结束 {dt/60:.1f} min，n_params={info['n_params']:,}，"
       f"stopped_at={info['stopped_at']}，best_val={info['best_val']:.4f}，"
       f"loss {curve['train'][0]:.4f} -> {curve['train'][-1]:.4f}"
       + (f"，峰值显存 {torch.cuda.max_memory_allocated()/2**30:.1f} GB"
          if torch.cuda.is_available() else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
