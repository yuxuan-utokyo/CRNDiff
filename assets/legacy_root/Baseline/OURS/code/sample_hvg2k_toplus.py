# -*- coding: utf-8 -*-
"""EXP176 · 三条网格臂的采样（A=T8 / B=T_O / D=T_Oplus 跳跃）。

    python Baseline/OURS/code/sample_hvg2k_toplus.py --ckpt <Tm8.pt> --horizon T8
    python Baseline/OURS/code/sample_hvg2k_toplus.py --ckpt <Tm8.pt> --horizon T_O
    python Baseline/OURS/code/sample_hvg2k_toplus.py --ckpt <Tm8.pt> --horizon T_Oplus

从 `sample_hvg2k.py` **逐字拷贝**，只替换「建网格」那一处（`M.build_grid` -> `grid_variant`），
外加一个 `--horizon` 开关与文件名后缀。
**`sample_hvg2k.py` / `model_hvg2k.py` / `bank_dedup.py` 一个字节没动**（红线）。

## 三条网格（EXP176 3)

    A  T8       linspace(8.0 -> F_BASE, K) + [0]            NFE = K
    B  T_O      linspace(T_O -> F_BASE, K) + [0]            NFE = K       <- 控制变量
    D  T_Oplus  [8.0] + linspace(T_O -> F_BASE, K) + [0]    NFE = K + 1   <- 跳跃

**B 的网格与旧 run1c（C 臂）逐位同源**：两者都调用 `M.build_grid(K, M.horizon())`,
同一个函数同一个入参，不是照着重写一遍。
**D 的中段与 B 逐位相同**：D 就是在 B 的网格前面 concatenate 一个 8.0,
两条臂只差最前面多一步 —— 这才是干净的单变量对照。

约定逐字照 toy 的 `exp139_step3.py::build_grid(horizon="T_Oplus")`：
起点一个 `T_PLUS_START = 8.0`，中段不变，末尾补 0。

NFE 一律用**实测值**（D 是 257 不是 256），进文件名、进 json、出表横轴用它。
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


T_PLUS_START = 8.0          # EXP176 跳跃臂的起点（与 toy exp139_step3 同名常量）


def grid_variant(K, horizon):
    """唯一被替换的那一处。"""
    if horizon == "T8":
        return M.build_grid(K, T_PLUS_START)
    gf, nfe = M.build_grid(K, M.horizon())              # B：与 C 臂逐位同源
    if horizon == "T_Oplus":
        gf = np.concatenate([[T_PLUS_START], gf])       # D：在 B 前面多一步
        nfe = int(len(gf) - 1)
    return gf, nfe


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
def sample_chunk(net, bank, dmax_t, n, seed, nmax, batched: bool = False):
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
        p = p.masked_fill(torch.arange(p.shape[2], device=dev)[None, None, :]
                          > dmax_t[None, :, None], 0.0)          # A4 posterior-support 截断
        s = p.sum(-1, keepdim=True)
        p = torch.where(s > 0, p / torch.clamp_min(s, 1e-30), p)
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
    ap.add_argument("--horizon", choices=["T8", "T_O", "T_Oplus"], default="T_O",
                    help="A=T8 全程 T_m / B=T_O 控制变量 / D=T_Oplus 跳跃（NFE=K+1）")
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

    rl(f"### EXP176 采样开始 **horizon={a.horizon}** ckpt={Path(a.ckpt).name} "
       f"K={a.ks} samp={a.samps} n_gen={a.n_gen} cell_chunk={a.cell_chunk} "
       f"T_O={T:.4f} NMAX={nmax} dim={dim}")
    out = {"ckpt": Path(a.ckpt).name, "T": T, "NMAX": nmax, "dim": dim,
           "n_gen": a.n_gen, "cell_chunk": a.cell_chunk, "batched": bool(a.batched),
           "cells": {}}
    for K in a.ks:
        gf, nfe = grid_variant(K, a.horizon)      # <- 唯一的替换点
        t0 = time.time()
        bank = DedupBank(der["V"], gf, nmax,
                         lambda tau, v, nm: VO.kmat(tau, v, nm, logfact),
                         lambda v, nm: VO.pois_prior(v, nm, logfact), DEV)
        g = bank.verify(lambda tau, v, nm: VO.kmat(tau, v, nm, logfact), der["V"])
        t_bank = time.time() - t0
        rl(f"  K={K} horizon={a.horizon} **实测 NFE={nfe}** "
           f"网格[{gf[0]:.4f}, {gf[1]:.4f}, ..., {gf[-2]:.4f}, {gf[-1]:.4f}] "
           f"len={len(gf)}；bank {t_bank:.1f}s 去重闸门 bit_exact={g['bit_exact']} "
           f"({g['n_checked']} 个抽查) n_V={g['n_V']}")
        if not g["bit_exact"]:
            rl(f"  [SKIP] K={K} 去重闸门未过，跳过")
            del bank; gc.collect(); torch.cuda.empty_cache(); continue
        for sp in a.samps:
            t1 = time.time()
            parts = []
            for ci, lo in enumerate(range(0, a.n_gen, a.cell_chunk)):
                b = min(a.cell_chunk, a.n_gen - lo)
                parts.append(sample_chunk(net, bank, dmax_t, b, int(sp) * 1000 + ci, nmax,
                                          batched=a.batched))
            gen = np.concatenate(parts, 0)
            wt = time.time() - t1
            # ★ `batched` 与 `cell_chunk` 都进文件名：两者都改 RNG 消费顺序，是**口径的一部分**，
            #   不同取值的产物必须并存，不能互相覆盖（红线：不覆盖任何已有 .npy）。
            suf = ((f"_batched{int(a.batched)}_chunk{a.cell_chunk}" if a.batched else "")
                   + f"_{a.horizon}_nfe{nfe}" + a.tag)
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
        f"sample_log_{ktag}_s{Path(a.ckpt).stem}_{a.horizon}"
        f"{'_chunk%d_batched1' % a.cell_chunk if a.batched else ''}{a.tag}.json")
    if lg.exists():
        raise SystemExit(f"[ABORT] 重名，拒绝覆盖：{lg.name}")
    lg.write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    rl(f"### hvg2k 采样结束，落盘 {len(out['cells'])} 格")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
