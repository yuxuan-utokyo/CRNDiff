# -*- coding: utf-8 -*-
"""hvg2k dedicated: **GPU bank deduplicated by V**. New file; does not modify `vendor_ours*.py` / `ours_fast_v2.py`.

## why this layer is needed

forward kernel `K_tau[V]` depends only on `(tau, V)`, independent of gene identity.`vendor_ours.KernelBank._k`
was already by `(tau, V)` cached, **already deduplicated on the CPU side**. But `FastBank` / `FastBankV2` when uploading
does `np.stack(bank.Kgrid[k])`, spreading the per-gene list into `[dim, C, C]` — and expanded it back.

61gene irrelevant there (dim=61, C=129, a single K is only 0.136 GB); on hvg2k it is a disaster:

```
2·(K+1)·dim·C²·8 B,  dim=2000, C=513:   K=8 -> 70.3 GB,  K=32 -> 258 GB
```

hvg2k V takes **only 15 distinct values** after quantization to the log2 grid, so after deduplication:

```
2·(K+1)·n_V·C²·8 B,  n_V=15, C=513:     K=8 -> 0.57 GB,  K=32 -> 2.0 GB,  K=128 -> 8.1 GB
```

**fits.** This layer is exactly what turns `[dim, C, C]` replaced by `[n_V, C, C]` + one `gene -> Vslot` index table.

## numerically equivalent

no approximation at all: the same `(tau, V)` kernels are the same float array anyway (`KernelBank._k` cache
guarantees this); deduplication just stops copying it.`_Slot.__getitem__(d)` returns `U[g2v[d]]`,
and `FastBank.Kgrid[k][d]` are the same block of data, so `sample_bridge_v2` loop is bit-identical.
`verify()` a bit-exact comparison of some random genes serves as the on-the-fly gate.
"""
from __future__ import annotations

import numpy as np
import torch


class _Slot:
    """makes `bank.Kgrid[k][d]` still indexed by gene; underneath it reads the deduplicated V slot.

    ★ `g2v` ****must stay on CPU** (Python int list). The first version kept it on GPU,
    so `self.g2v[d]`(d is a Python int) returns a **GPU scalar tensor**,
    then use it to index `self.U` triggers a **device→host sync**.
    dim=2000 2×2000 syncs per step, 1.3 million in total at K=32; sampling grinds to a halt
    (this is why T4a ran 27 minutes without a result on 2026-08-10).
    after switching to Python int `self.U[i]` is a pure GPU-side view, zero syncs.
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
    """and `FastBankV2` same interface (device/dtype/K/dim/nmax/tgrid/prior/Kgrid/Kdelta_T),
    but Kgrid/Kdelta_T only store `n_V` copies instead of `dim` copies."""

    def __init__(self, V, tgrid, nmax, kmat_fn, prior_fn, device,
                 dtype=torch.float64, verbose=True):
        V = np.asarray(V, np.float64)
        self.device, self.dtype = device, dtype
        self.dim, self.nmax = len(V), nmax
        self.tgrid = np.asarray(tgrid, np.float64)
        self.K = len(self.tgrid)
        uV, inv = np.unique(V, return_inverse=True)
        self.uV, self.n_V = uV, len(uV)
        g2v = [int(i) for i in np.asarray(inv).ravel()]        # CPU side; see the _Slot notes
        C = nmax + 1
        taus_grid = self.tgrid
        taus_delta = np.abs(np.diff(self.tgrid))

        def layer(tau):
            return torch.as_tensor(
                np.stack([kmat_fn(float(tau), float(v), nmax) for v in uV]),
                dtype=dtype, device=device)

        if verbose:
            gbytes = 2 * self.K * self.n_V * C * C * 8 / 2 ** 30
            print(f"  [DedupBank] dim={self.dim} -> n_V={self.n_V} distinct V;"
                  f"nmax={nmax} K={self.K}; memory {gbytes:.2f} GB "
                  f"(without deduplication {gbytes*self.dim/self.n_V:.1f} GB)", flush=True)
        self.Kgrid = _SlotList([_Slot(layer(t), g2v) for t in taus_grid])
        self.Kdelta_T = _SlotList(
            [_Slot(layer(t).transpose(1, 2).contiguous(), g2v) for t in taus_delta])
        self.prior = torch.as_tensor(
            np.stack([prior_fn(float(v), nmax) for v in V]), dtype=dtype, device=device)

    def verify(self, kmat_fn, V, n_probe=8, seed=0):
        """on-the-fly gate: draw some random genes; the deduplicated kernels must be bit-identical to directly computed ones."""
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
