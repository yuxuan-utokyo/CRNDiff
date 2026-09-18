# -*- coding: utf-8 -*-
"""hvg2k · OURS pilot 训练 + **logit adjustment（LA）**。T3 层。

    python Baseline/OURS/code/train_pilot_la.py --tau 0 --steps 2500 --name t3base
    python Baseline/OURS/code/train_pilot_la.py --tau 1 --steps 2500 --name t3la1

## 这是什么

    L = -log softmax( f(n_t)_{d,:} + tau * log pi_d )_{x0_d}

`pi_d` = **第 d 个基因自己的计数直方图**（从 train 数出来，`[dim, NMAX+1]`），
不需要任何联合分布、不需要重采样、不需要去偏。1 个基因、10 个、2000 个写法完全一样
（任务书 M4：迁移摩擦为零）。`tau=0` 就是现状，用来做**同预算的对照臂**。

推断时用**同一个表达式**：`p_hat = softmax(f + tau*log pi)`。所以这是
一个精确的重参数化，不是近似 —— 见 `sample_hvg2k_la.py`。

## 为什么在 2000 维上它该比 1D 更有用（T2 已验，见 exp08）

主线的 head 是**逐 token 共享**的 `Linear(d_model, NMAX+1)`，逐基因的先验
只能通过 `gene_emb` 挤进去。2000 个基因的先验横跨十个数量级
（`gene_mean` 从 5.5e-6 到 12.15，zerofrac 0.941），让一个共享头去表示它是纯浪费。
LA 把先验**解析地**加进去，网络只需要学「相对先验的偏离」。

## 怎么做到「不改 `model_hvg2k.py` 一个字节」

照 `train_pilot_toplus.py` 已经用过的模式：

    src = inspect.getsource(M.train_ours_hvg2k)   # 取原函数源码
    src = src.replace(OLD, NEW)                   # **断言只命中一次**的两处替换
    exec(src, dict(M.__dict__) | {注入的常量})     # 在 model_hvg2k 的同一份全局里编译

两处替换：训练那一步的 CE、以及 val 评估那一步的 CE（val 也要加 LA，否则 val 曲线
量的不是这个臂在优化的目标）。除这两处外逐字与原函数相同，`model_hvg2k.py`
从头到尾只被读取。运行时把替换前后的行打印出来备查。

## pi 的平滑

`pi_d(n) = (count_d(n) + EPS) / (N + EPS*(NMAX+1))`，`EPS=0.5`。
不平滑的话没出现过的计数 log pi = -inf，该类别**永远**生成不出来 ——
那比 A4 截断更硬（A4 只砍 n > dmax_d，dmax 以下没出现过的计数仍然留着）。
EPS=0.5 时地板是 0.5/361205 = 1.4e-6，配合采样时的 A4 截断（`n > dmax_d` 置零）
不会漏出假质量。

## 红线

`model_hvg2k.py` / `sample_hvg2k.py` / `vendor_ours*.py` / `bank_dedup.py`
一个字节没动。产物名带 `--name`，与已有 pilot_* 不重名。
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
PI_EPS = 0.5

OLD_TRAIN = """        with amp:
            logits = model(n_t, t)
            loss = F.cross_entropy(logits.reshape(-1, CFG.NMAX + 1),
                                   x0.long().clamp(0, CFG.NMAX).reshape(-1))"""
NEW_TRAIN = """        with amp:
            logits = model(n_t, t) + _LA_TAU * _LA_LOGPI.unsqueeze(0)
            loss = F.cross_entropy(logits.reshape(-1, CFG.NMAX + 1),
                                   x0.long().clamp(0, CFG.NMAX).reshape(-1))"""

OLD_VAL = """                    s_ += float(F.cross_entropy(
                        model(nb, tb_).reshape(-1, CFG.NMAX + 1),
                        xb.long().clamp(0, CFG.NMAX).reshape(-1), reduction="sum"))"""
NEW_VAL = """                    s_ += float(F.cross_entropy(
                        (model(nb, tb_)
                         + _LA_TAU * _LA_LOGPI.unsqueeze(0)).reshape(-1, CFG.NMAX + 1),
                        xb.long().clamp(0, CFG.NMAX).reshape(-1), reduction="sum"))"""


def rl(m):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"- `{time.strftime('%H:%M:%S')}` {m}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_logpi(train, nmax, eps=PI_EPS):
    """pi_d(n)，[dim, nmax+1] float32（平滑过、逐行归一）。**只从 train 数**。"""
    X = np.clip(train, 0, nmax).astype(np.int64)
    n, dim = X.shape
    cnt = np.zeros((dim, nmax + 1), np.float64)
    for d in range(dim):
        cnt[d] = np.bincount(X[:, d], minlength=nmax + 1)
    pi = (cnt + eps) / (n + eps * (nmax + 1))
    pi /= pi.sum(1, keepdims=True)
    return np.log(pi), pi, cnt


def make_trainer(tau, logpi_t):
    """把 `train_ours_hvg2k` 的源码取出来，做两处**被断言过**的替换，再编译。"""
    src = inspect.getsource(M.train_ours_hvg2k)
    assert src.count(OLD_TRAIN) == 1, "训练 CE 那一处没有恰好命中一次 —— 上游改过，停"
    assert src.count(OLD_VAL) == 1, "val CE 那一处没有恰好命中一次 —— 上游改过，停"
    src = src.replace(OLD_TRAIN, NEW_TRAIN).replace(OLD_VAL, NEW_VAL)
    print("  [PATCH] 训练 CE:\n" + "\n".join("    " + l for l in NEW_TRAIN.splitlines()))
    print("  [PATCH] val  CE:\n" + "\n".join("    " + l for l in NEW_VAL.splitlines()))
    g = dict(M.__dict__)
    g["_LA_TAU"] = float(tau)
    g["_LA_LOGPI"] = logpi_t
    ns = {}
    exec(compile(src, "<train_ours_hvg2k+LA>", "exec"), g, ns)
    return ns["train_ours_hvg2k"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True,
                    help="LA 强度。0 = 现状对照臂（但走同一条代码路径）")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=CFG.SEED_PILOT)
    ap.add_argument("--name", required=True, help="产物名，如 t3base / t3la1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    steps = 60 if a.smoke else a.steps
    batch = a.batch or CFG.BATCH

    d, meta = M.load_data(("train", "val"))
    tr, va = d["train"], d["val"]
    der = M.derive(tr)
    T = M.horizon()
    tag = a.name

    t0 = time.time()
    lp_dir = ROOT / "Baseline" / "OURS" / "models"
    lp_dir.mkdir(parents=True, exist_ok=True)
    cache = lp_dir / f"logpi_nmax{CFG.NMAX}_eps{PI_EPS}.npy"
    if cache.exists():
        logpi = np.load(cache)
        pi = np.exp(logpi)
    else:
        logpi, pi, cnt = build_logpi(tr, CFG.NMAX)
        np.save(cache, logpi)
    logpi_t = torch.as_tensor(logpi, dtype=torch.float32, device=M.DEV)
    rl(f"### hvg2k OURS **T3/{tag}** tau={a.tau} steps={steps} batch={batch} "
       f"seed={a.seed}  ({time.time()-t0:.1f}s 建 log_pi)")
    rl(f"  log_pi [{logpi.shape[0]},{logpi.shape[1]}]  "
       f"min={logpi.min():.3f} max={logpi.max():.3f}  "
       f"pi(0) 中位={np.median(pi[:, 0]):.5f}  "
       f"逐基因 pi 的有效类别数中位="
       f"{np.median(np.exp(-(pi*np.log(np.maximum(pi,1e-300))).sum(1))):.3f}")
    rl(f"  train{list(tr.shape)} val{list(va.shape)}  T_MAX={T:.4f}  NMAX={CFG.NMAX}  "
       f"sum_V/lib={der['sum_V_over_lib']:.4f}")

    trainer = make_trainer(a.tau, logpi_t)
    t0 = time.time()
    try:
        net, curve, info = trainer(
            tr, va, der["V"], der["iscale"], a.seed, steps,
            batch=batch, log_fn=lambda x: print(x, flush=True), t_max=T,
            ckpt_path=lp_dir / f"{tag}_s{a.seed}.pt")
    except torch.cuda.OutOfMemoryError as e:
        rl(f"  [SCALE-CUT] batch={batch} 显存不够（{repr(e)[:100]}）")
        return 2
    dt = time.time() - t0
    torch.save(net.state_dict(), lp_dir / f"{tag}_s{a.seed}.pt")
    (ROOT / "Baseline" / "OURS" / "loss").mkdir(parents=True, exist_ok=True)
    for k in ("train", "val", "step"):
        np.save(ROOT / "Baseline" / "OURS" / "loss" / f"{tag}_s{a.seed}_{k}.npy",
                np.asarray(curve[k]))
    blob = {"tag": tag, "LA_tau": a.tau, "PI_EPS": PI_EPS, "seed": a.seed,
            "walltime_s": dt, "steps_requested": steps, "batch": batch,
            "dataset": CFG.DATASET, "gene_hash": CFG.GENE_HASH, "T": T,
            "logpi_cache": cache.name,
            "note": ("LA 臂的 val 与 tau=0 臂的 val **不可直接比大小** —— "
                     "两者优化的是不同的目标（加了 tau*log pi 的 CE）。"
                     "臂间比较一律用生成产物上的读数，不用 val。"),
            "s_per_step": dt / max(steps, 1),
            **{k: der[k] for k in ("n_distinct_V", "sum_V", "lib_mean", "sum_V_over_lib")},
            **info, "curve": {k: [float(x) for x in v] for k, v in curve.items()}}
    (lp_dir / f"{tag}_s{a.seed}.json").write_text(
        json.dumps(blob, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    rl(f"  T3/{tag} 结束 {dt/60:.1f} min ({dt/max(steps,1):.3f} s/步)，"
       f"best_val={info['best_val']:.5f}，train {curve['train'][0]:.4f} -> "
       f"{curve['train'][-1]:.4f}"
       + (f"，峰值显存 {torch.cuda.max_memory_allocated()/2**30:.1f} GB"
          if torch.cuda.is_available() else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
