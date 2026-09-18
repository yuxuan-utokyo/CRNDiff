# -*- coding: utf-8 -*-
"""EXP180 §4 · `DedupBankV2`：在 V 去重之外再按 **tau** 去重，外加可选的 CPU 常驻。

新文件；`bank_dedup.py` / `sample_hvg2k.py` / `model_hvg2k.py` / `vendor_ours*.py`
一个字节没动（红线）。

## 两处改动

**1. tau 去重。** `DedupBank` 对 `Kgrid` 的每个 `tgrid[k]`、`Kdelta_T` 的每个
`|diff(tgrid)[k]|` 各建一份 `[n_V, C, C]`。但同一个 `(tau, V)` 本来就是同一份浮点数组
（`kmat` 是纯函数），所以多个 k 可以指向同一块 `U` —— **不是近似**，与按 V 去重同一个论证。

    v1:  2·(K+1) 层            K=512 -> 30.2 GB     K=1024 -> 60.3 GB
    v2:  (K+1) + n_uniq 层     K=512 -> 15.2 GB     K=1024 -> 30.3 GB

> [!warning] 去重按 **float64 精确相等**，不许 round（§2 红线）
> `np.linspace` 的相邻差**不是逐位相同**的：等间距那一族里差着 1~2 个 ulp。
> 实测 `n_uniq_delta` 在 K=128/256/512/1024 上**都是 5**（不是任务书 §2 猜的 2）：
> 四个「等间距」的 ulp 变体 + 末段的 `F_BASE-0 = 0.010000000000000000208`。
> 照实用。多出来的 3 层只多 0.09 GB。
>
> 我先前写过一个用 `round(Δ,12)` 合并成 2 层的版本，**被自己的逐位闸门抓到**
> （`n_bad_Kdelta = 31/32`），已作废进 `_quarantine`。

**2. `host_resident`（可选，默认关）。** 为真时 `Kgrid` 的层留在 CPU 内存里，
`Kgrid[k]` 第一次被取时上传该层、并释放上一层。采样循环里 k 单调递增
（第 k 步只用 `Kgrid[k+1]` 与 `Kdelta_T[k]`），滚动窗口 1 层就够。
`Kdelta_T` 去重后只有 5 层（0.15 GB），始终常驻 GPU。

**dtype 全程 float64，一个字节都不许降。**

## 接口

与 `DedupBank` 完全一致：`.dim .nmax .tgrid .K .uV .n_V .prior .Kgrid[k] .Kdelta_T[k]`，
层对象同时支持 `[d]`（逐基因，非 batched 路径）与 `.U` / `.g2v`（batched 路径）。
直接复用 `bank_dedup._Slot` / `_SlotList`，所以那两个类的语义**逐字相同**。
"""
from __future__ import annotations

import time

import numpy as np
import torch

from pathlib import Path

from bank_dedup import _Slot, _SlotList          # 只读 import，不修改


class _HostSlotList:
    """`Kgrid` 常驻 CPU 版：按 k 取层 -> 上传 GPU，只保留最近 `keep` 层。"""

    __slots__ = ("host", "idx", "g2v", "device", "dtype", "_cache", "keep",
                 "n_upload", "upload_s")

    def __init__(self, host, idx, g2v, device, dtype, keep=2):
        self.host, self.idx, self.g2v = host, idx, g2v
        self.device, self.dtype, self.keep = device, dtype, keep
        self._cache, self.n_upload, self.upload_s = {}, 0, 0.0

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        s = self._cache.get(k)
        if s is None:
            t0 = time.time()
            h = self.host[self.idx[k]]
            if not torch.is_tensor(h):                    # memmap 后端
                h = torch.from_numpy(np.ascontiguousarray(h))
            U = h.to(self.device, dtype=self.dtype, non_blocking=True)
            self.upload_s += time.time() - t0
            self.n_upload += 1
            s = _Slot(U, self.g2v)
            self._cache[k] = s
            if len(self._cache) > self.keep:
                for old in sorted(self._cache)[:-self.keep]:
                    self._cache.pop(old, None)
        return s


class DedupBankV2:
    """接口与 `DedupBank` 一致；多一层 tau 去重与可选的 CPU 常驻。"""

    def __init__(self, V, tgrid, nmax, kmat_fn, prior_fn, device,
                 dtype=torch.float64, verbose=True, host_resident=False,
                 pin_memory=True, mmap_dir=None):
        V = np.asarray(V, np.float64)
        self.device, self.dtype = device, dtype
        self.dim, self.nmax = len(V), nmax
        self.tgrid = np.asarray(tgrid, np.float64)
        self.K = len(self.tgrid)
        self.host_resident = bool(host_resident)
        uV, inv = np.unique(V, return_inverse=True)
        self.uV, self.n_V = uV, len(uV)
        g2v = [int(i) for i in np.asarray(inv).ravel()]
        self.g2v = g2v
        C = nmax + 1

        taus_grid = self.tgrid
        taus_delta = np.abs(np.diff(self.tgrid))
        # ★ float64 精确相等去重，**不做任何容差合并**
        ug, ig = np.unique(taus_grid, return_inverse=True)
        ud, id_ = np.unique(taus_delta, return_inverse=True)
        self.n_uniq_grid, self.n_uniq_delta = int(len(ug)), int(len(ud))

        def layer(tau):
            return torch.as_tensor(
                np.stack([kmat_fn(float(tau), float(v), nmax) for v in uV]),
                dtype=dtype)

        if verbose:
            n_gpu = (self.n_uniq_delta if host_resident
                     else self.n_uniq_grid + self.n_uniq_delta)
            per = self.n_V * C * C * 8 / 2 ** 30
            print(f"  [DedupBankV2] dim={self.dim} n_V={self.n_V} K={self.K}；"
                  f"tau 去重 Kgrid {self.K}->{self.n_uniq_grid} 层、"
                  f"Kdelta_T {self.K-1}->**{self.n_uniq_delta}** 层"
                  f"（精确相等，未 round）；"
                  f"GPU {n_gpu*per:.2f} GB"
                  + (f" + CPU 常驻 {self.n_uniq_grid*per:.1f} GB" if host_resident else "")
                  + f"（v1 要 {2*self.K*per:.1f} GB）", flush=True)

        t0 = time.time()
        # ---- Kdelta_T：始终常驻 GPU（去重后只有几层）----
        dl = [layer(t).to(device).transpose(1, 2).contiguous() for t in ud]
        self.Kdelta_T = _SlotList([_Slot(dl[int(j)], g2v) for j in id_])

        # ---- Kgrid ----
        if host_resident:
            # ★ 30 GB（K=1024）在 41 GB 空闲的机器上：pin_memory 会 CUDA OOM，
            #   退回可分页内存又把 CPU 堆撑爆（实测 STATUS_HEAP_CORRUPTION）。
            #   给一个 `mmap_dir` 就改用磁盘 memmap：常驻内存降到 ~0，
            #   热层由 OS 页缓存兜住，**数值逐位不变**。
            shape = (self.n_uniq_grid, self.n_V, C, C)
            if mmap_dir is not None:
                import hashlib
                md = Path(mmap_dir); md.mkdir(parents=True, exist_ok=True)
                key = hashlib.sha256(
                    np.concatenate([ug, uV, [nmax]]).tobytes()).hexdigest()[:16]
                fp = md / f"Kgrid_{key}_n{self.n_uniq_grid}_V{self.n_V}_C{C}.f64"
                ok = fp.exists() and fp.stat().st_size == int(np.prod(shape)) * 8
                host = np.memmap(fp, dtype=np.float64,
                                 mode="r" if ok else "w+", shape=shape)
                self.mmap_path = fp
                if verbose:
                    print(f"    [DedupBankV2] Kgrid 走磁盘 memmap "
                          f"{np.prod(shape, dtype=np.int64)*8/2**30:.1f} GB -> {fp.name}"
                          f"{'（命中缓存）' if ok else ''}", flush=True)
                if ok:
                    self.Kgrid = _HostSlotList(host, [int(j) for j in ig], g2v,
                                              device, dtype)
                    self._host = host
                    self.build_s = time.time() - t0
                    self.prior = torch.as_tensor(
                        np.stack([prior_fn(float(v), nmax) for v in V]),
                        dtype=dtype, device=device)
                    return
            else:
                host = torch.empty(shape, dtype=dtype)
                if pin_memory:
                    try:
                        host = host.pin_memory()
                    except Exception as e:
                        print(f"    [DedupBankV2] pin_memory 失败，退回可分页："
                              f"{type(e).__name__}", flush=True)
            for j, t in enumerate(ug):
                lj = layer(t)
                host[j] = lj.numpy() if not torch.is_tensor(host) else lj
                if verbose and (j + 1) % 128 == 0:
                    print(f"    Kgrid {j+1}/{self.n_uniq_grid} 层  "
                          f"{time.time()-t0:.0f}s", flush=True)
            if not torch.is_tensor(host):
                host.flush()
                host = np.memmap(self.mmap_path, dtype=np.float64, mode="r",
                                 shape=shape)
            self.Kgrid = _HostSlotList(host, [int(j) for j in ig], g2v, device, dtype)
            self._host = host
        else:
            gl = [layer(t).to(device) for t in ug]
            self.Kgrid = _SlotList([_Slot(gl[int(j)], g2v) for j in ig])
        self.build_s = time.time() - t0
        if verbose:
            print(f"  [DedupBankV2] 建完 {self.build_s:.1f}s", flush=True)

        self.prior = torch.as_tensor(
            np.stack([prior_fn(float(v), nmax) for v in V]), dtype=dtype, device=device)

    def verify(self, kmat_fn, V, n_probe=8, seed=0):
        """与 `DedupBank.verify` 逐字同义。"""
        rng = np.random.default_rng(seed)
        ds = rng.choice(self.dim, min(n_probe, self.dim), replace=False)
        bad = 0
        for k in (0, len(self.tgrid) // 2, len(self.tgrid) - 1):
            for d in ds:
                a = self.Kgrid[k][int(d)].cpu().numpy()
                b = kmat_fn(float(self.tgrid[k]), float(V[int(d)]), self.nmax)
                if not np.array_equal(a, b):
                    bad += 1
        return {"bit_exact": bad == 0, "n_bad": int(bad),
                "n_checked": 3 * len(ds), "n_V": int(self.n_V), "dim": int(self.dim)}


def gate_vs_v1(V, tgrid, nmax, kmat_fn, prior_fn, device, host_resident=False):
    """闸门：小 K 上把 v2 与 v1 **逐层逐位**比对。不过就别用 v2。"""
    from bank_dedup import DedupBank
    a = DedupBank(V, tgrid, nmax, kmat_fn, prior_fn, device, verbose=False)
    b = DedupBankV2(V, tgrid, nmax, kmat_fn, prior_fn, device, verbose=False,
                    host_resident=host_resident)
    bad_g = sum(int(not torch.equal(a.Kgrid[k].U, b.Kgrid[k].U))
                for k in range(len(tgrid)))
    bad_d = sum(int(not torch.equal(a.Kdelta_T[k].U, b.Kdelta_T[k].U))
                for k in range(len(tgrid) - 1))
    # 逐基因路径也查（非 batched 分支用的是 `slot[d]`）
    bad_gene = sum(int(not torch.equal(a.Kgrid[k][d], b.Kgrid[k][d]))
                   for k in (0, len(tgrid) // 2, len(tgrid) - 1)
                   for d in (0, 1, len(V) // 2, len(V) - 1))
    return {"bit_exact": bad_g == 0 and bad_d == 0 and bad_gene == 0
            and bool(torch.equal(a.prior, b.prior)),
            "n_bad_Kgrid": bad_g, "n_bad_Kdelta": bad_d, "n_bad_per_gene": bad_gene,
            "prior_equal": bool(torch.equal(a.prior, b.prior)),
            "n_layers": len(tgrid), "n_uniq_delta": b.n_uniq_delta,
            "n_uniq_grid": b.n_uniq_grid, "host_resident": host_resident}
