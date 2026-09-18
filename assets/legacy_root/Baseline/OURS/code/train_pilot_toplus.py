# -*- coding: utf-8 -*-
"""EXP179-F · $T_{O+}$ 的**匹配训练**：t 的采样分布 = 0.9 对数均匀[F_BASE, T_O] + 0.1 点质量 t=8.0。

    python Baseline/OURS/code/train_pilot_toplus.py --seed 20260625 --steps 20000 --batch 384

## 为什么要这个配置（主线 EXP179 更正 §2）

$T_{O+}$ 的反向链**只在 $t=8$ 与 $t\\in[0,T_O]$ 上查询网络，中间那段 $(T_O, 8)$ 一次都不查**。
而 `pilot_Tm8` 是在 $[0.01, 8]$ 上对数均匀抽 t 训练的 —— 有相当一部分容量花在了
**永远不会被问到的区间**上。拿它跑 D 臂，是「一个分心的模型」对「一个专注的模型」，
输了也说明不了问题。本文件给 $T_{O+}$ 一个自己的训练配置。

比例 **0.9 / 0.1 事前定死，不调**：257 步里只有 1 步在 t=8，但那一步决定起点，
给它 10% 的训练预算是保守的过配。实际实现的比例由 `_STAT` 计数后写进 RUN_LOG。

## 怎么做到「不改 model_hvg2k.py 一个字节」

`train_ours_hvg2k` 里 t 的采样在一个闭包 `noise` 里，没法从外面覆盖。
所以本文件**不手抄那 150 行**，而是：

    src = inspect.getsource(M.train_ours_hvg2k)      # 取原函数源码
    src = src.replace(OLD_NOISE, NEW_NOISE)          # **断言只命中一次**的一处替换
    exec(src, dict(M.__dict__) | {注入的常量})        # 在 model_hvg2k 的同一份全局里编译

这样这份拷贝**可证明忠实** —— 除了被断言过的那一处，逐字与原函数相同，
而 `model_hvg2k.py` 从头到尾只被读取。运行时会把替换前后的那几行打印出来备查。
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
import config_hvg2k as CFG                              # noqa: E402
import model_hvg2k as M                                 # noqa: E402

LOG = ROOT / "RUN_LOG.md"
TOPLUS_T, TOPLUS_P = 8.0, 0.1
_STAT = {"n_hi": 0, "n": 0}

OLD_NOISE = """    def noise(x0, gg):
        u = torch.rand(x0.shape[0], generator=gg).to(DEV)
        t = torch.exp(lf + u * (lt - lf))
        r = torch.exp(-t).unsqueeze(1)"""
NEW_NOISE = """    def noise(x0, gg):
        u = torch.rand(x0.shape[0], generator=gg).to(DEV)
        t = torch.exp(lf + u * (lt - lf))
        # ===== EXP179-F 唯一的替换：0.9 对数均匀[F_BASE, T_O] + 0.1 点质量 t=8.0 =====
        pick = torch.rand(x0.shape[0], generator=gg).to(DEV)
        hi = pick < TOPLUS_P
        t = torch.where(hi, torch.full_like(t, TOPLUS_T), t)
        _STAT["n_hi"] += int(hi.sum()); _STAT["n"] += int(x0.shape[0])
        # ==========================================================================
        r = torch.exp(-t).unsqueeze(1)"""


def rl(m):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"- `{time.strftime('%H:%M:%S')}` {m}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _build():
    src = inspect.getsource(M.train_ours_hvg2k)
    n = src.count(OLD_NOISE)
    assert n == 1, f"[ABORT] `noise` 的锚点命中 {n} 次，拒绝盲改"
    src2 = src.replace(OLD_NOISE, NEW_NOISE, 1)
    ns = dict(M.__dict__)
    ns.update(TOPLUS_T=TOPLUS_T, TOPLUS_P=TOPLUS_P, _STAT=_STAT)
    exec(compile(src2, "<train_ours_hvg2k::EXP179F>", "exec"), ns)
    return ns["train_ours_hvg2k"], src, src2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=CFG.SEED_PILOT)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--name", default="pilot_TOplus")
    a = ap.parse_args()

    ck = ROOT / "Baseline" / "OURS" / "models" / f"{a.name}_s{a.seed}.pt"
    for p in (ck, Path(str(ck) + ".best"), Path(str(ck) + ".last"),
              ROOT / "Baseline" / "OURS" / "models" / f"{a.name}_s{a.seed}.json"):
        if p.exists():
            raise SystemExit(f"[ABORT] 重名，拒绝覆盖：{p.name}")

    train_fn, src, src2 = _build()
    d, meta = M.load_data(("train", "val"))
    tr, va = d["train"], d["val"]
    der = M.derive(tr)
    T = M.horizon()

    rl(f"### EXP179-F 训练开始 · {a.name} train{list(tr.shape)} val{list(va.shape)} "
       f"batch={a.batch} steps={a.steps} seed={a.seed}")
    rl(f"  **t 的采样分布 = {1-TOPLUS_P:.1f} 对数均匀[{CFG.F_BASE}, {T:.4f}] + "
       f"{TOPLUS_P:.1f} 点质量 t={TOPLUS_T}**（事前定死，不调）；"
       f"e^-{TOPLUS_T}·NMAX={np.exp(-TOPLUS_T)*CFG.NMAX:.4f}，"
       f"e^-T_O·NMAX={np.exp(-T)*CFG.NMAX:.4f}")
    rl(f"  拷贝方式：`inspect.getsource(train_ours_hvg2k)` + **一处断言过的替换**，"
       f"`model_hvg2k.py` 只读未改（原 {len(src)} B -> 新 {len(src2)} B）")

    t0 = time.time()
    net, curve, info = train_fn(tr, va, der["V"], der["iscale"], a.seed, a.steps,
                                batch=a.batch, log_fn=lambda x: print(x, flush=True),
                                t_max=T, ckpt_path=ck)
    dt = time.time() - t0
    if ck.exists():
        raise SystemExit(f"[ABORT] 重名，拒绝覆盖：{ck.name}")
    torch.save(net.state_dict(), ck)
    for k in ("train", "val", "step"):
        np.save(ROOT / "Baseline" / "OURS" / "loss" / f"{a.name}_s{a.seed}_{k}.npy",
                np.asarray(curve[k]))
    frac = _STAT["n_hi"] / max(_STAT["n"], 1)
    (ROOT / "Baseline" / "OURS" / "models" / f"{a.name}_s{a.seed}.json").write_text(
        json.dumps({"tag": a.name, "seed": a.seed, "walltime_s": dt,
                    "dataset": CFG.DATASET, "gene_hash": CFG.GENE_HASH,
                    "t_dist": {"log_uniform_range": [CFG.F_BASE, T],
                               "point_mass_t": TOPLUS_T, "point_mass_p": TOPLUS_P,
                               "realised_fraction": frac,
                               "n_draws": _STAT["n"]},
                    "s_per_step": dt / max(a.steps, 1), **info,
                    "curve": {k: [float(x) for x in v] for k, v in curve.items()}},
                   ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    rl(f"  EXP179-F 训练结束 {dt/60:.1f} min（{dt/a.steps:.3f} s/步），"
       f"best_val={info['best_val']:.5f}，n_params={info['n_params']:,}；"
       f"**t=8.0 的实际比例 {frac:.5f}**（配置 {TOPLUS_P}，"
       f"{_STAT['n_hi']:,}/{_STAT['n']:,} 次抽样）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
