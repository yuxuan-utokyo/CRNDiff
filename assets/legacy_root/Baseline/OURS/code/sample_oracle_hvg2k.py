# -*- coding: utf-8 -*-
"""EXP174 · **oracle 后验**采样。从 `sample_hvg2k.py` 拷贝，只替换 `p = net.post(...)` 一处。

    python Baseline/OURS/code/sample_oracle_hvg2k.py --stage gate     # 先过两个闸门
    python Baseline/OURS/code/sample_oracle_hvg2k.py --stage sample --ks 256

**`sample_hvg2k.py` / `model_hvg2k.py` / `bank_dedup.py` 一个字节没动**（红线）。
本文件是新建的拷贝，网络**完全不加载**。

## 要回答的问题（EXP174 §0）

低表达基因的方差不足，是 (A) 学出来的 x0 后验不够散，还是 (B) bridge 反向链本身
在低计数上有离散化损失？把学出来的后验换成**可解析的逐基因贝叶斯后验**，其余不动：
低表达档回到 1.0 -> (A)；仍是 0.85 -> (B)。

## oracle 后验（§1）

    p_oracle[b,d,c]  ∝  K_{t_k}^{(d)}[c, n_t[b,d]]  ·  π_d[c]

* `K` 复用 bank 里已有的核，**不新算**。
* `π_d` = 基因 d 在 **train** 上的计数经验分布，clip 到 NMAX，加 `1e-8` 再归一。

### ★ 索引约定：我自己核实过，取**列**（与主线的读法一致）

`vendor_ours.kmat` 的 docstring 与实现都写死了

    K[n0, n] = q_tau(n | n0) = Bin(n0, e^-tau) (*) Pois(V(1-e^-tau))

（`K[n0] = np.convolve(binom(n0,·), pois)`，**行下标是 n0**）。
`DedupBank` 里 `Kgrid[k]` 存的是 `layer(tau)` **未转置**，所以
`Kgrid[k][d][c, n_t]` 就是 `P(n_t | n_0=c)` —— oracle 要的是**第 n_t 列**。

旁证：同一个 `DedupBank` 里 `Kdelta_T` 存的是 `layer(t).transpose(1, 2)`，
而采样器对它用的是 `index_select(0, n_t)` —— 取转置后的行 = 取原矩阵的**列**，
正是桥的第二个因子 `K_Δ(n_s -> n_t)` 需要的。两处约定自洽。

实现上用 `U.transpose(1,2)` 之后按 `[vi, n_t]` 高级索引，一次拿到 `[B, dim, C]`，
与 batched 分支里 `Knext.U[vi_next, n0]` 是同一种索引模式。

## dtype

bank 是 float64，所以 oracle 的 `p` 是 **float64**（网络那版是 float32）。
这是**升精度不是降精度**，不违反「float64 不得降为 float32」。
两次 multinomial 的次数与形状不变，RNG 消费顺序逐字保留。
"""
from __future__ import annotations

import argparse
import gc
import importlib
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
from bank_dedup import DedupBank                        # noqa: E402

VO = importlib.import_module("panel61_shared.vendor_ours")
LOG = ROOT / "RUN_LOG.md"
DEV = M.DEV
TAG = "_ORACLE"


def rl(m):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"- `{time.strftime('%H:%M:%S')}` {m}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- 以下三个函数逐字拷自 sample_hvg2k.py ----
def _cat(p, gen):
    """`vendor_ours._categorical_torch` 逐字语义。"""
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    tot = p.sum(dim=1)
    bad = ~torch.isfinite(tot) | (tot <= 0)
    if bad.any():
        p = p.clone(); p[bad] = 0.0; p[bad, 0] = 1.0
    return torch.multinomial(p, 1, generator=gen).squeeze(1)


def _slot_v_idx(slot, dev, _cache={}):
    key = id(slot)
    t = _cache.get(key)
    if t is None or t.device != dev:
        t = torch.as_tensor(slot.g2v, dtype=torch.long, device=dev)
        _cache[key] = t
    return t


# ---- 新增：oracle 后验 ----
def emp_prior(train, nmax, dev):
    """π_d[c] = 基因 d 在 **train** 上的计数经验分布，clip 到 nmax，+1e-8 再归一。"""
    X = np.clip(np.asarray(train), 0, nmax).astype(np.int64)
    dim = X.shape[1]
    P = np.zeros((dim, nmax + 1), np.float64)
    for d in range(dim):
        P[d] = np.bincount(X[:, d], minlength=nmax + 1)[:nmax + 1]
    P /= P.sum(1, keepdims=True)
    P += 1e-8                                    # 拉普拉斯平滑，避免零概率永久锁死
    P /= P.sum(1, keepdims=True)
    return torch.as_tensor(P, dtype=torch.float64, device=dev)


def _slot_UT(slot, _cache={}):
    """`U[v][n0, n]` -> `UT[v][n, n0]`，供 `UT[vi, n_t]` 一次取出 `P(n_t | n_0=·)`。"""
    key = id(slot)
    t = _cache.get(key)
    if t is None:
        t = slot.U.transpose(1, 2).contiguous()
        _cache.clear()                           # 只留当前这层，31.6 MB 不值得攒
        _cache[key] = t
    return t


def oracle_post(bank, k, n_t, pi, nmax):
    """p[b,d,c] ∝ K_{t_k}^{(d)}[c, n_t[b,d]] · π_d[c]。**未归一**（归一化留给下游逐字那段）。"""
    slot = bank.Kgrid[k]
    UT = _slot_UT(slot)                                        # [n_V, C, C] float64
    n, dim = n_t.shape
    vi = _slot_v_idx(slot, n_t.device)[None, :].expand(n, dim)
    p = UT[vi, n_t.clamp(0, nmax)]                             # [B, dim, C] = P(n_t|n_0=c)
    p = p * pi[None, :, :]                                     # × 经验先验
    return p


@torch.no_grad()
def sample_chunk_oracle(bank, pi, dmax_t, n, seed, nmax, batched: bool = True):
    """与 `sample_hvg2k.sample_chunk` 逐字相同，**只把 `p = net.post(nt, tt)` 换成 oracle**。"""
    dev = bank.device
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    dim, tgrid = bank.dim, bank.tgrid
    prior = bank.prior.cpu().numpy()
    cdf = np.cumsum(prior, axis=1)
    u = rng.random((n, dim))
    n_t = torch.as_tensor(
        np.stack([np.searchsorted(cdf[d], u[:, d]) for d in range(dim)], 1).clip(0, nmax),
        dtype=torch.long, device=dev)
    for k in range(len(tgrid) - 1):
        # ---- 唯一的替换点：网络后验 -> oracle 后验（其余一个字不动）----
        p = oracle_post(bank, k, n_t, pi, nmax)                  # [B, dim, C] float64
        p = p.masked_fill(torch.arange(p.shape[2], device=dev)[None, None, :]
                          > dmax_t[None, :, None], 0.0)          # A4 posterior-support 截断
        s = p.sum(-1, keepdim=True)
        p = torch.where(s > 0, p / torch.clamp_min(s, 1e-30), p)
        Knext, KdelT = bank.Kgrid[k + 1], bank.Kdelta_T[k]
        C = p.shape[2]
        n0 = _cat(p.reshape(n * dim, C), gen).view(n, dim)       # 第 1 次大 multinomial
        del p
        vi_next = _slot_v_idx(Knext, dev)[None, :].expand(n, dim)
        vi_delt = _slot_v_idx(KdelT, dev)[None, :].expand(n, dim)
        row = Knext.U[vi_next, n0]                               # [B, dim, C] float64
        row.mul_(KdelT.U[vi_delt, n_t.clamp(0, nmax)])           # 原地乘
        n_t = _cat(row.reshape(n * dim, C), gen).view(n, dim)    # 第 2 次
        del row, n0
    return n_t.cpu().numpy().astype(np.int64)


# =============================== 闸门（§1）=============================== #
@torch.no_grad()
def gates(bank, pi, dmax_t, train, nmax, n_probe=4000, seed=0):
    """闸 1：最后一个网格点（t->0）上 argmax_c p_oracle == n_t，命中率须 > 0.99。
       闸 2：p 每行和为 1，无 NaN/Inf。"""
    dev = bank.device
    k_last = len(bank.tgrid) - 2                 # 循环里用到的最小 tau
    tau = float(bank.tgrid[k_last])
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(train), 64, replace=False)
    n_t = torch.as_tensor(np.clip(np.asarray(train)[np.sort(rows)], 0, nmax),
                          dtype=torch.long, device=dev)
    p = oracle_post(bank, k_last, n_t, pi, nmax)
    p = p.masked_fill(torch.arange(p.shape[2], device=dev)[None, None, :]
                      > dmax_t[None, :, None], 0.0)
    s = p.sum(-1, keepdim=True)
    p = torch.where(s > 0, p / torch.clamp_min(s, 1e-30), p)

    # 闸 2
    rs = p.sum(-1)
    g2 = {"tau": tau, "n_rows": int(rs.numel()),
          "max_abs_sum_minus_1": float((rs - 1.0).abs().max()),
          "n_nan": int(torch.isnan(p).sum()), "n_inf": int(torch.isinf(p).sum())}
    # 闸 1：只在 A4 截断没有把真值切掉的位置上判（c=n_t 必须还在支撑里）
    am = p.argmax(-1)
    keep = (n_t <= dmax_t[None, :])
    hit = ((am == n_t) & keep).sum().item()
    tot = int(keep.sum().item())
    g1 = {"tau": tau, "n_checked": tot, "n_hit": int(hit),
          "hit_rate": hit / max(tot, 1),
          "n_excluded_by_A4": int((~keep).sum().item())}
    del p
    return g1, g2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gate", "sample"], default="gate")
    ap.add_argument("--ks", type=int, nargs="+", default=[256])
    ap.add_argument("--samps", type=int, nargs="+", default=[20260901])
    ap.add_argument("--n-gen", type=int, default=5000)
    ap.add_argument("--cell-chunk", type=int, default=64)
    a = ap.parse_args()

    d, meta = M.load_data(("train",))
    tr = d["train"]
    der = M.derive(tr)
    dim, nmax = tr.shape[1], CFG.NMAX
    T = M.horizon()
    logfact = VO._logfact_arr(nmax)
    dmax_t = torch.as_tensor(der["dmax"], dtype=torch.long, device=DEV)
    (ROOT / "Baseline" / "OURS" / "results").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pi = emp_prior(tr, nmax, DEV)
    rl(f"### EXP174 oracle {a.stage} · K={a.ks} n_gen={a.n_gen} chunk={a.cell_chunk} "
       f"T={T:.4f} NMAX={nmax} dim={dim}；经验先验 π 从 **train** 算，"
       f"clip {nmax} + 1e-8 归一，{time.time()-t0:.1f}s；**网络完全不加载**")

    out = {"T": T, "NMAX": nmax, "dim": dim, "n_gen": a.n_gen,
           "cell_chunk": a.cell_chunk, "batched": True, "oracle": True, "cells": {}}
    for K in a.ks:
        gf, nfe = M.build_grid(K, T)
        t0 = time.time()
        bank = DedupBank(der["V"], gf, nmax,
                         lambda tau, v, nm: VO.kmat(tau, v, nm, logfact),
                         lambda v, nm: VO.pois_prior(v, nm, logfact), DEV)
        g = bank.verify(lambda tau, v, nm: VO.kmat(tau, v, nm, logfact), der["V"])
        rl(f"  K={K} bank {time.time()-t0:.1f}s  去重闸门 bit_exact={g['bit_exact']} "
           f"({g['n_checked']} 个抽查)  n_V={g['n_V']}")
        if not g["bit_exact"]:
            rl(f"  [STOP] K={K} 去重闸门未过")
            return 2

        g1, g2 = gates(bank, pi, dmax_t, tr, nmax)
        rl(f"  **闸 1**（t->0，tau={g1['tau']:.4f}）argmax 命中率 "
           f"**{g1['hit_rate']:.5f}**（{g1['n_hit']}/{g1['n_checked']}，触发线 >0.99；"
           f"A4 截断排除 {g1['n_excluded_by_A4']} 个）")
        rl(f"  **闸 2** 行和偏离 1 的最大值 **{g2['max_abs_sum_minus_1']:.3e}**，"
           f"NaN {g2['n_nan']}，Inf {g2['n_inf']}")
        ok = (g1["hit_rate"] > 0.99 and g2["max_abs_sum_minus_1"] < 1e-9
              and g2["n_nan"] == 0 and g2["n_inf"] == 0)
        out.setdefault("gates", {})[f"K{K}"] = {"gate1": g1, "gate2": g2, "pass": bool(ok)}
        if not ok:
            rl("  **[STOP] 闸门未过，按 §7 停止条件停下来报主线**")
            (HERE.parents[1] / "results" / f"oracle_gate_K{K}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=1, default=float),
                encoding="utf-8")
            return 3
        rl("  闸门全过")
        if a.stage == "gate":
            del bank; gc.collect(); torch.cuda.empty_cache(); continue

        for sp in a.samps:
            t1 = time.time()
            parts = []
            for ci, lo in enumerate(range(0, a.n_gen, a.cell_chunk)):
                b = min(a.cell_chunk, a.n_gen - lo)
                parts.append(sample_chunk_oracle(bank, pi, dmax_t, b,
                                                 int(sp) * 1000 + ci, nmax))
            gen = np.concatenate(parts, 0)
            wt = time.time() - t1
            f = (ROOT / "Baseline" / "OURS" / "results" /
                 f"linspace_T_O_K{K}_sORACLE_samp{sp}"
                 f"_batched1_chunk{a.cell_chunk}_n{a.n_gen}{TAG}.npy")
            if f.exists():
                raise SystemExit(f"[ABORT] 重名，拒绝覆盖：{f.name}")
            np.save(f, gen.astype(np.int16))
            out["cells"][f"K{K}|samp{sp}"] = {
                "K": K, "nfe": nfe, "samp_seed": int(sp), "walltime_s": wt,
                "s_per_step": wt / max(nfe, 1), "gen_file": f.name,
                "cell_chunk": a.cell_chunk,
                "peak_alloc_gb": float(torch.cuda.max_memory_allocated() / 2 ** 30),
                "mean": float(gen.mean()), "max": int(gen.max()),
                "zerofrac": float((gen == 0).mean())}
            rl(f"  K={K} samp{sp} **{wt/60:.1f}min（{wt/nfe:.3f} s/步）** -> {f.name}  "
               f"mean={gen.mean():.4f} max={gen.max()} zeros={(gen==0).mean():.4f}"
               f"  （真实 train mean={tr.mean():.4f} max={tr.max()} "
               f"zeros={(tr==0).mean():.4f}）")
        del bank; gc.collect(); torch.cuda.empty_cache()

    lg = (ROOT / "Baseline" / "OURS" / "results" /
          f"oracle_log_K{'-'.join(str(k) for k in a.ks)}_{a.stage}.json")
    if lg.exists():
        raise SystemExit(f"[ABORT] 重名，拒绝覆盖：{lg.name}")
    lg.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                  encoding="utf-8")
    rl(f"### EXP174 oracle {a.stage} 结束 -> {lg.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
