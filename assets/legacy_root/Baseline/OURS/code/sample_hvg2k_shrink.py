# -*- coding: utf-8 -*-
"""EXP178 · 大 t 处把后验向经验边缘收缩（**零训练**）。

    python Baseline/OURS/code/sample_hvg2k_shrink.py --ckpt <run1c.pt> --ks 32 --s 0.1

从 `sample_hvg2k.py` **逐字拷贝**，只在 `p = net.post(nt, tt)` 之后、
**A4 截断之前**插入一行混合，其余一个字不动。
**`sample_hvg2k.py` / `model_hvg2k.py` / `bank_dedup.py` 一个字节没动**（红线）。

## 修法

    p_used(n0 | n_t) = (1 - lam_d(t)) * p_net + lam_d(t) * pi_d

`pi_d` = EXP174/177 用的那个经验边缘（train 直方图，clip NMAX + 1e-8 归一），
**不用学，直接数出来的**。

    lam(t, d) = (1 - beta_t) * exp(-beta_t * nbar_d / s),   beta_t = e^{-t}

（主线 2026-08-13 更正后的形式。）两个因子各管一件事：

* `(1 - beta_t)` 保证 **t->0 时 lam->0**。t=0.01 时它只有 0.00995，所以**无论哪个 s，
  lam <= 0.01**，小 t 端网络的跨基因条件完全保留 —— EXP177 实测那里是真优势，一点不能动。
* `exp(-beta_t * nbar_d / s)` 管**大 t 处的逐基因差异**。t=T_O 时 `1-beta_t = 0.9945`，
  第二项对稀有基因约等于 1 -> lam≈0.99；常见基因在中等 t 仍被第二项压住。

> 旧形式 `exp(-beta_t*nbar_d/s)` 单独用是错的：稀有基因 `beta_t*nbar_d≈0.06`，
> t->0 时 lam **不收敛到 0**（s>=0.1 在 t=0.01 处还有 0.62~0.95 的收缩），
> 正好把该保留的那一端换掉了。根因是拿 `beta_t*nbar_d` 当信息量，而 t->0 时观测**就是**
> n_0，不管基因多稀有信息都是满的。已按主线更正换掉。

`--s 0` 表示 **lam 恒等于 0**，即现状对照。

## dtype：混合在 **float32** 上做

现有采样路径里 `p` 本来就是 float32（`sample_hvg2k.py` 的注释写明），float64 的是
`row = Knext.U[...]` 那一支。红线是「**不得把现在是 float64 的东西降成 float32**」，
不是「所有东西都必须 float64」。所以把 `pi` 从 float64 **cast 到 float32** 再与 `p` 混合，
**精度状态与现状逐位一致**（pi 最小值 1e-8，float32 表示得下），临时张量也减半。

`--s 0` 因此与主表那个 K=32 产物走的是同一条数值路径。
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
from bank_dedup_v2 import DedupBankV2                   # noqa: E402  （--bank v2 才用）

VO = importlib.import_module("panel61_shared.vendor_ours")
# ★ EXP179：E1/E2/F1/F2 需要「收缩 + 非默认网格」同时生效。网格逻辑**不重写一遍**，
#   直接 import EXP176 那个文件里的 `grid_variant` —— 这样 EXP176 的 A/B/D 与
#   EXP179 的 E1/E2/F1/F2 用的是**同一个函数**，网格逐位同源，不可能因为抄错而错位。
from sample_hvg2k_toplus import T_PLUS_START, grid_variant   # noqa: E402,F401
LOG = ROOT / "RUN_LOG.md"
DEV = M.DEV


def emp_prior(train, nmax):
    """pi_d[c]：train 上的计数经验分布，clip 到 nmax，+1e-8 再归一（与 EXP174/177 逐字相同）。"""
    Xc = np.clip(np.asarray(train), 0, nmax).astype(np.int64)
    P = np.stack([np.bincount(Xc[:, d], minlength=nmax + 1)[:nmax + 1]
                  for d in range(Xc.shape[1])]).astype(np.float64)
    P /= P.sum(1, keepdims=True)
    P += 1e-8
    P /= P.sum(1, keepdims=True)
    return P


def lam_of(t, nbar, s):
    """lam(t,d) = (1 - beta_t) * exp(-beta_t * nbar_d / s)，beta_t = e^{-t}。

    s<=0 -> 恒 0（现状对照）。主线 2026-08-13 更正后的形式，见文件头。
    """
    nbar = np.asarray(nbar, np.float64)
    if s is None or s <= 0:
        return np.zeros_like(nbar)
    beta = float(np.exp(-float(t)))
    return (1.0 - beta) * np.exp(-beta * nbar / float(s))


_GATE = {"max_rowsum_dev": 0.0}


def rl(m):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"- `{time.strftime('%H:%M:%S')}` {m}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _cat(p, gen):
    """`vendor_ours._categorical_torch` 逐字语义。"""
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    tot = p.sum(dim=1)
    bad = ~torch.isfinite(tot) | (tot <= 0)
    if bad.any():
        p = p.clone(); p[bad] = 0.0; p[bad, 0] = 1.0
    return torch.multinomial(p, 1, generator=gen).squeeze(1)


def _slot_v_idx(slot, dev, _cache={}):
    """`_Slot.g2v`（CPU int 列表）-> GPU long 张量，供 batched 路径做高级索引。

    只读 `slot.g2v`，**不改 `bank_dedup.py` 一个字节**（红线）。按 id 缓存，
    整条梯子上每个 slot 只建一次。
    """
    key = id(slot)
    t = _cache.get(key)
    if t is None or t.device != dev:
        t = torch.as_tensor(slot.g2v, dtype=torch.long, device=dev)
        _cache[key] = t
    return t


@torch.no_grad()
def sample_chunk(net, bank, dmax_t, n, seed, nmax, batched: bool = False,
                 pi=None, nbar=None, s=0.0):
    """一块细胞的完整反向链。

    `batched=False`（默认）：结构与 `ours_fast_v2.sample_bridge_v2(batched=False)` 一致 ——
        逐基因 2·dim 次 multinomial。dim=2000 时每反向步 4000 次 kernel 启动，
        `n_gen=20000` + `cell_chunk=500` 是 40 块 -> **16 万次 / 反向步**，
        实测 69 s/步，而按维度从 61gene 的 0.215 s/步线性外推只该是 7.05 s/步。
        多出来的十倍全是 kernel 启动延迟。

    `batched=True`（EXP160，**B 类**）：逐字照 `ours_fast_v2.sample_bridge_v2` 的
        batched 分支 —— 把 2·dim 次压成 **2 次** `[B*dim, C]` 的大 multinomial。
        改变 torch 生成器的消费顺序，所以**不可能**与 batched=False 逐位相同，
        必须过分布等价闸（`_diag/t1_batched_gate.py`）。
        主线 2026-08-10 撤销了 EXP157 的禁令：实验 X 是全新实验，没有已发表的数要对齐。

    ★ 数值精度不变：bank 仍是 float64，`row*col` 在 float64 上做，
      `p` 仍是网络输出的 float32。batched 只改**批处理方式**，不改任何 dtype。
    """
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
        t = float(tgrid[k])
        nt = n_t.to(torch.float32)
        tt = torch.full((n,), t, device=dev)
        p = net.post(nt, tt)                                     # [B, dim, C] float32
        # ===== EXP178 唯一的插入点：向经验边缘收缩（在 A4 截断之前）=====
        # 混合走 **float32**：p 本来就是 float32，pi cast 下来，精度状态与现状逐位一致。
        if pi is not None and s and s > 0:
            lam = torch.as_tensor(lam_of(t, nbar, s), dtype=torch.float32,
                                  device=dev)[None, :, None]     # [1, dim, 1]
            p = (1.0 - lam) * p + lam * pi[None, :, :]
        # ================================================================
        p = p.masked_fill(torch.arange(p.shape[2], device=dev)[None, None, :]
                          > dmax_t[None, :, None], 0.0)          # A4 posterior-support 截断
        ssum = p.sum(-1, keepdim=True)
        p = torch.where(ssum > 0, p / torch.clamp_min(ssum, 1e-30), p)
        # 闸门（§3）：混合 + 重归一之后，行和必须为 1、无 NaN/Inf
        # 阈值 1e-5：float32 下 513 项求和的舍入本底约 1e-6，1e-9 是 float64 的标准，
        # 拿到 float32 路径上会误报。实测最大值照样写进 RUN_LOG，不只写"通过"。
        _rs = float((p.sum(-1) - 1.0).abs().max())
        if not (_rs < 1e-5) or bool(torch.isnan(p).any()) or bool(torch.isinf(p).any()):
            raise SystemExit(f"[STOP] EXP178 归一化闸门未过：行和偏离 {_rs:.3e}，"
                             f"NaN={int(torch.isnan(p).sum())} Inf={int(torch.isinf(p).sum())}")
        _GATE["max_rowsum_dev"] = max(_GATE["max_rowsum_dev"], _rs)
        Knext, KdelT = bank.Kgrid[k + 1], bank.Kdelta_T[k]
        if not batched:
            for d in range(dim):
                n0 = _cat(p[:, d, :], gen)
                row = Knext[d].index_select(0, n0)
                col = KdelT[d].index_select(0, n_t[:, d].clamp(0, nmax))
                n_t[:, d] = _cat(row * col, gen)
        else:
            C = p.shape[2]
            n0 = _cat(p.reshape(n * dim, C), gen).view(n, dim)   # 第 1 次大 multinomial
            del p
            vi_next = _slot_v_idx(Knext, dev)[None, :].expand(n, dim)
            vi_delt = _slot_v_idx(KdelT, dev)[None, :].expand(n, dim)
            row = Knext.U[vi_next, n0]                           # [B, dim, C] float64
            row.mul_(KdelT.U[vi_delt, n_t.clamp(0, nmax)])       # 原地乘，省一份 4 GB
            n_t = _cat(row.reshape(n * dim, C), gen).view(n, dim)  # 第 2 次
            del row, n0
            continue
        del p
    return n_t.cpu().numpy().astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=CFG.K_LADDER_PILOT)
    ap.add_argument("--samps", type=int, nargs="+", default=CFG.SAMP_SEEDS[:1])
    ap.add_argument("--n-gen", type=int, default=CFG.N_GEN)
    ap.add_argument("--cell-chunk", type=int, default=500)
    ap.add_argument("--batched", action="store_true",
                    help="EXP160：2 次大 multinomial 代替 2·dim 次（B 类，须过分布等价闸）")
    ap.add_argument("--tag", default="", help="附加到文件名，避免与已有产物重名")
    ap.add_argument("--bank", choices=["v1", "v2"], default="v1",
                    help="EXP180：v2 在 V 去重之外再按 **tau 精确去重**"
                         "（Kdelta_T K-1 层 -> 5 层），K=512 才装得下。默认 v1 = 旧行为。")
    ap.add_argument("--mmap-dir", default=None,
                    help="EXP180：host_resident 的 Kgrid 改放磁盘 memmap。"
                         "K=1024 的 30 GB 在 41 GB 空闲的机器上 pin 会 CUDA OOM、"
                         "可分页会撑爆 CPU 堆，用 memmap 兜住。")
    ap.add_argument("--host-resident", action="store_true",
                    help="EXP180：Kgrid 常驻 CPU、逐层上传。K=1024 需要（GPU 要 30.3 GB）。")
    ap.add_argument("--horizon", choices=["T8", "T_O", "T_Oplus"], default="T_O",
                    help="EXP179：与 EXP176 的 A/B/D 同一套网格。T_Oplus 的 NFE = K+1。")
    ap.add_argument("--s", type=float, default=0.0,
                    help="lam = exp(-e^{-t}·nbar_d/s)。**--s 0 表示 lam 恒 0（现状对照）**。"
                         "§2 写死的扫描范围：0, 0.01, 0.03, 0.1, 0.3, 1.0，不要自己加别的形式。")
    a = ap.parse_args()

    d, meta = M.load_data(("train",))
    tr = d["train"]
    der = M.derive(tr)
    dim, nmax = tr.shape[1], CFG.NMAX
    T = M.horizon()
    logfact = VO._logfact_arr(nmax)
    net = M.build_net(dim, der["iscale"]).to(DEV)
    net.load_state_dict(torch.load(a.ckpt, map_location=DEV, weights_only=False))
    net.eval()
    dmax_t = torch.as_tensor(der["dmax"], dtype=torch.long, device=DEV)
    (ROOT / "Baseline" / "OURS" / "results").mkdir(parents=True, exist_ok=True)

    nbar = np.asarray(tr, np.float64).mean(0)
    pi_t = torch.as_tensor(emp_prior(tr, nmax), dtype=torch.float32, device=DEV)
    det = 1 - (np.asarray(tr) == 0).mean(0)
    rare = det < 0.05
    t_hi = T_PLUS_START if a.horizon in ('T8', 'T_Oplus') else T
    l_hi, l_lo = lam_of(t_hi, nbar, a.s), lam_of(CFG.F_BASE, nbar, a.s)
    rl(f"### 收缩采样 **s={a.s} horizon={a.horizon}** ckpt={Path(a.ckpt).name} K={a.ks} "
       f"samp={a.samps} n_gen={a.n_gen} cell_chunk={a.cell_chunk} T={T:.4f} "
       f"NMAX={nmax} dim={dim}")
    rl(f"  **lam 中位数**：t={t_hi:.4f}(起点) 全体 {np.median(l_hi):.4f} / "
       f"稀有(det<5%,n={int(rare.sum())}) {np.median(l_hi[rare]):.4f}；"
       f"t=F_BASE({CFG.F_BASE}) 全体 {np.median(l_lo):.4f} / "
       f"稀有 {np.median(l_lo[rare]):.4f}")
    out = {"ckpt": Path(a.ckpt).name, "T": T, "NMAX": nmax, "dim": dim,
           "n_gen": a.n_gen, "cell_chunk": a.cell_chunk, "batched": bool(a.batched),
           "cells": {}}
    for K in a.ks:
        gf, nfe = grid_variant(K, a.horizon)      # EXP179：与 EXP176 同一个函数
        t0 = time.time()
        if a.bank == "v2":
            bank = DedupBankV2(der["V"], gf, nmax,
                               lambda tau, v, nm: VO.kmat(tau, v, nm, logfact),
                               lambda v, nm: VO.pois_prior(v, nm, logfact), DEV,
                               host_resident=a.host_resident,
                               mmap_dir=a.mmap_dir)
        else:
            bank = DedupBank(der["V"], gf, nmax,
                             lambda tau, v, nm: VO.kmat(tau, v, nm, logfact),
                             lambda v, nm: VO.pois_prior(v, nm, logfact), DEV)
        g = bank.verify(lambda tau, v, nm: VO.kmat(tau, v, nm, logfact), der["V"])
        t_bank = time.time() - t0
        rl(f"  K={K} bank {t_bank:.1f}s  去重闸门 bit_exact={g['bit_exact']} "
           f"({g['n_checked']} 个抽查)  n_V={g['n_V']}")
        if not g["bit_exact"]:
            rl(f"  [SKIP] K={K} 去重闸门未过，跳过")
            del bank; gc.collect(); torch.cuda.empty_cache(); continue
        for sp in a.samps:
            t1 = time.time()
            parts = []
            for ci, lo in enumerate(range(0, a.n_gen, a.cell_chunk)):
                b = min(a.cell_chunk, a.n_gen - lo)
                parts.append(sample_chunk(net, bank, dmax_t, b, int(sp) * 1000 + ci, nmax,
                                          batched=a.batched, pi=pi_t, nbar=nbar, s=a.s))
            gen = np.concatenate(parts, 0)
            wt = time.time() - t1
            # ★ `batched` 与 `cell_chunk` 都进文件名：两者都改 RNG 消费顺序，是**口径的一部分**，
            #   不同取值的产物必须并存，不能互相覆盖（红线：不覆盖任何已有 .npy）。
            suf = ((f"_batched{int(a.batched)}_chunk{a.cell_chunk}" if a.batched else "")
                   + f"_{a.horizon}_nfe{nfe}_SHRINK_s{a.s:g}"
                   + ("" if a.bank == "v1" else
                      f"_bank{a.bank}{'hr' if a.host_resident else ''}") + a.tag)
            f = (ROOT / "Baseline" / "OURS" / "results" /
                 f"linspace_T_O_K{K}_s{Path(a.ckpt).stem}_samp{sp}{suf}.npy")
            if f.exists():
                raise SystemExit(f"[ABORT] 重名，拒绝覆盖：{f.name}")
            np.save(f, gen.astype(np.int16))
            out["cells"][f"K{K}|samp{sp}"] = {
                "K": K, "nfe": nfe, "samp_seed": int(sp), "walltime_s": wt,
                "s_per_step": wt / max(nfe, 1), "gen_file": f.name,
                "batched": bool(a.batched), "cell_chunk": a.cell_chunk,
                "peak_alloc_gb": float(torch.cuda.max_memory_allocated() / 2 ** 30)
                if torch.cuda.is_available() else 0.0,
                "mean": float(gen.mean()), "max": int(gen.max()),
                "zerofrac": float((gen == 0).mean())}
            rl(f"  K={K} samp{sp} {wt/60:.1f}min ({wt/nfe:.3f} s/步) -> {f.name}  "
               f"mean={gen.mean():.4f} max={gen.max()} zeros={(gen==0).mean():.4f}"
               f"  （真实 train mean={tr.mean():.4f} max={tr.max()} zeros={(tr==0).mean():.4f}）")
        del bank; gc.collect(); torch.cuda.empty_cache()
    # ★ 2026-08-10 修：日志名**必须带 K**。原来只带 batched/chunk，于是同一个 chunk 下
    #   第二档起就撞名，在 main() 末尾被「不许覆盖」挡下、rc=1 ——
    #   **样本其实都存好了**，但链条日志把这些档标成「失败」，汇总时会被当成缺档跳过。
    ktag = "K" + "-".join(f"{k:03d}" for k in a.ks)
    # ★ EXP172 修：日志名里**没有 ckpt**，于是同一个 (K, chunk, tag) 换个训练种子再采
    #   就会撞名 —— 样本 .npy 已经落好了，却在最后一步 SystemExit，链条把这一档记成 rc=1。
    #   种子复现要对同一个 K 跑三个 ckpt，这个坑必踩，故把 ckpt stem 加进日志名。
    #   **只是日志文件名，不碰任何数值路径。**
    lg = ROOT / "Baseline" / "OURS" / "results" / (
        f"sample_log_{ktag}_s{Path(a.ckpt).stem}_{a.horizon}_SHRINK_s{a.s:g}"
        f"{'' if a.bank == 'v1' else '_bank' + a.bank + ('hr' if a.host_resident else '')}"
        f"{'_chunk%d_batched1' % a.cell_chunk if a.batched else ''}{a.tag}.json")
    if lg.exists():
        raise SystemExit(f"[ABORT] 重名，拒绝覆盖：{lg.name}")
    lg.write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    rl(f"### EXP178 s={a.s} 采样结束，落盘 {len(out['cells'])} 格；**归一化闸门** 行和偏离 1 的最大值 {_GATE['max_rowsum_dev']:.3e}（触发线 <1e-9）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
