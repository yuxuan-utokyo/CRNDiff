# -*- coding: utf-8 -*-
"""hvg2k 专用：**按 V 去重的 GPU bank**。新文件；不改 `vendor_ours*.py` / `ours_fast_v2.py`。

## 为什么必须有这一层

前向核 `K_tau[V]` 只依赖 `(tau, V)`，与基因身份无关。`vendor_ours.KernelBank._k`
本来就是按 `(tau, V)` 缓存的，**CPU 端已经去重**。但 `FastBank` / `FastBankV2` 上传时
做的是 `np.stack(bank.Kgrid[k])`，把逐基因的列表摊成 `[dim, C, C]` —— 又展开回去了。

61gene 上无所谓（dim=61、C=129，单 K 才 0.136 GB）；hvg2k 上是灾难：

```
2·(K+1)·dim·C²·8 B,  dim=2000, C=513:   K=8 -> 70.3 GB,  K=32 -> 258 GB
```

hvg2k 的 V 量化到 log2 网格之后**只有 15 个不同值**，所以去重之后：

```
2·(K+1)·n_V·C²·8 B,  n_V=15, C=513:     K=8 -> 0.57 GB,  K=32 -> 2.0 GB,  K=128 -> 8.1 GB
```

**放得下。** 这一层就是把 `[dim, C, C]` 换成 `[n_V, C, C]` + 一张 `gene -> V槽` 的索引表。

## 数值等价

不做任何近似：同一个 `(tau, V)` 的核本来就是同一个浮点数组（`KernelBank._k` 的缓存
保证了这一点），去重只是不再复制它。`_Slot.__getitem__(d)` 返回 `U[g2v[d]]`，
与 `FastBank.Kgrid[k][d]` 是同一块数据，所以 `sample_bridge_v2` 的循环逐位不变。
`verify()` 里对随机若干个基因做逐位比对当作现场闸门。
"""
from __future__ import annotations

import numpy as np
import torch


class _Slot:
    """让 `bank.Kgrid[k][d]` 仍然按基因索引，底下取的是去重后的 V 槽。

    ★ `g2v` **必须留在 CPU**（Python int 列表）。第一版把它放在 GPU 上，
    于是 `self.g2v[d]`（d 是 Python int）返回的是一个 **GPU 标量张量**，
    再拿它去索引 `self.U` 会触发一次 **device→host 同步**。
    dim=2000 时每步 2×2000 次、K=32 时共 130 万次同步，采样直接卡死
    （2026-08-10 的 T4a 跑了 27 分钟没出结果就是这个）。
    换成 Python int 之后 `self.U[i]` 是纯 GPU 侧的视图，零同步。
    """

    __slots__ = ("U", "g2v")

    def __init__(self, U: torch.Tensor, g2v):
        self.U = U
        self.g2v = g2v if isinstance(g2v, list) else list(map(int, g2v))

    def __getitem__(self, d):
        return self.U[self.g2v[d]]


class _SlotList:
    __slots__ = ("slots",)

    def __init__(self, slots):
        self.slots = slots

    def __getitem__(self, k):
        return self.slots[k]

    def __len__(self):
        return len(self.slots)


class DedupBank:
    """与 `FastBankV2` 接口一致（device/dtype/K/dim/nmax/tgrid/prior/Kgrid/Kdelta_T），
    但 Kgrid/Kdelta_T 只存 `n_V` 份而不是 `dim` 份。"""

    def __init__(self, V, tgrid, nmax, kmat_fn, prior_fn, device,
                 dtype=torch.float64, verbose=True):
        V = np.asarray(V, np.float64)
        self.device, self.dtype = device, dtype
        self.dim, self.nmax = len(V), nmax
        self.tgrid = np.asarray(tgrid, np.float64)
        self.K = len(self.tgrid)
        uV, inv = np.unique(V, return_inverse=True)
        self.uV, self.n_V = uV, len(uV)
        g2v = [int(i) for i in np.asarray(inv).ravel()]        # CPU 侧，见 _Slot 的说明
        C = nmax + 1
        taus_grid = self.tgrid
        taus_delta = np.abs(np.diff(self.tgrid))

        def layer(tau):
            return torch.as_tensor(
                np.stack([kmat_fn(float(tau), float(v), nmax) for v in uV]),
                dtype=dtype, device=device)

        if verbose:
            gbytes = 2 * self.K * self.n_V * C * C * 8 / 2 ** 30
            print(f"  [DedupBank] dim={self.dim} -> n_V={self.n_V} 个不同 V；"
                  f"nmax={nmax} K={self.K}；显存 {gbytes:.2f} GB "
                  f"（不去重是 {gbytes*self.dim/self.n_V:.1f} GB）", flush=True)
        self.Kgrid = _SlotList([_Slot(layer(t), g2v) for t in taus_grid])
        self.Kdelta_T = _SlotList(
            [_Slot(layer(t).transpose(1, 2).contiguous(), g2v) for t in taus_delta])
        self.prior = torch.as_tensor(
            np.stack([prior_fn(float(v), nmax) for v in V]), dtype=dtype, device=device)

    def verify(self, kmat_fn, V, n_probe=8, seed=0):
        """现场闸门：随机抽若干基因，去重取出来的核必须与直接算的逐位相同。"""
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
