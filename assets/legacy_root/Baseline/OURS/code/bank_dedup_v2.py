# -*- coding: utf-8 -*-
"""EXP180 §4 · `DedupBankV2`: deduplicates by **tau** in addition to V, plus optional CPU residency.

new file;`bank_dedup.py` / `sample_hvg2k.py` / `model_hvg2k.py` / `vendor_ours*.py`
not a single byte changed (red line).

## two changes

**1. tau deduplicated.** `DedupBank` for `Kgrid` each `tgrid[k]`,`Kdelta_T` each
`|diff(tgrid)[k]|` one each `[n_V, C, C]`. But the same `(tau, V)` was already the same float array
(`kmat` is a pure function), so several k can point to the same block `U` — **not an approximation**; same argument as deduplication by V.

    v1:  2·(K+1) layers            K=512 -> 30.2 GB     K=1024 -> 60.3 GB
    v2:  (K+1) + n_uniq layers     K=512 -> 15.2 GB     K=1024 -> 30.3 GB

> [!warning] deduplication uses **exact float64 equality**, no rounding (section 2 red line)
> `np.linspace` adjacent differences are **not bit-identical**: the equally spaced family differs by 1~2 ulp.
> measured `n_uniq_delta` is **5 at K=128/256/512/1024** (not 2 as guessed in spec section 2):
> four equally spaced ulp variants + the final `F_BASE-0 = 0.010000000000000000208`.
> used as measured. The 3 extra layers add only 0.09 GB.
>
> I previously wrote one using `round(Δ,12)` the version merged into 2 layers **was caught by its own bit-exact gate**
> (`n_bad_Kdelta = 31/32`), voided into `_quarantine`.

**2. `host_resident`(optional, off by default).** When true `Kgrid` layers stay in CPU memory,
`Kgrid[k]` upload the layer on first access and free the previous one. k increases monotonically in the sampling loop
(step k only uses `Kgrid[k+1]` and `Kdelta_T[k]`); a rolling window of 1 layer is enough.
`Kdelta_T` after deduplication only 5 layers (0.15 GB), always resident on GPU.

**dtype float64 throughout; not a single byte may be downcast.**

## interface

and `DedupBank` exactly consistent:`.dim .nmax .tgrid .K .uV .n_V .prior .Kgrid[k] .Kdelta_T[k]`,
the layer object supports both `[d]`(per gene, non-batched path) and `.U` / `.g2v`(batched path).
reuse directly `bank_dedup._Slot` / `_SlotList`, so the semantics of those two classes are **identical**.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from pathlib import Path

from bank_dedup import _Slot, _SlotList          # read-only import, no modification


class _HostSlotList:
    """`Kgrid` CPU-resident version: fetch the layer for k -> upload to GPU, keep only the latest `keep` layers."""

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
            if not torch.is_tensor(h):                    # memmap backend
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
    """interface identical to `DedupBank` consistent; adds a tau deduplication layer and an optional CPU-resident mode."""

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
        # ★ float64 deduplication by exact equality, **no tolerance merging**
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
            print(f"  [DedupBankV2] dim={self.dim} n_V={self.n_V} K={self.K};"
                  f"tau deduplicated Kgrid {self.K}->{self.n_uniq_grid} layers,"
                  f"Kdelta_T {self.K-1}->**{self.n_uniq_delta}** layers"
                  f"(exact equality, no rounding);"
                  f"GPU {n_gpu*per:.2f} GB"
                  + (f" + CPU resident {self.n_uniq_grid*per:.1f} GB" if host_resident else "")
                  + f"(v1 needs {2*self.K*per:.1f} GB)", flush=True)

        t0 = time.time()
        # ---- Kdelta_T: always resident on GPU (only a few layers after deduplication)----
        dl = [layer(t).to(device).transpose(1, 2).contiguous() for t in ud]
        self.Kdelta_T = _SlotList([_Slot(dl[int(j)], g2v) for j in id_])

        # ---- Kgrid ----
        if host_resident:
            # ★ 30 GB(K=1024) on a machine with 41 GB free: pin_memory gives CUDA OOM,
            #   falling back to pageable memory blew up the CPU heap (measured STATUS_HEAP_CORRUPTION).
            #   give a `mmap_dir` switch to an on-disk memmap: resident memory drops to ~0,
            #   the hot layers are backed by the OS page cache; **values are bit-identical**.
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
                    print(f"    [DedupBankV2] Kgrid via disk memmap "
                          f"{np.prod(shape, dtype=np.int64)*8/2**30:.1f} GB -> {fp.name}"
                          f"{'(cache hit)' if ok else ''}", flush=True)
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
                        print(f"    [DedupBankV2] pin_memory failed, falling back to pageable memory:"
                              f"{type(e).__name__}", flush=True)
            for j, t in enumerate(ug):
                lj = layer(t)
                host[j] = lj.numpy() if not torch.is_tensor(host) else lj
                if verbose and (j + 1) % 128 == 0:
                    print(f"    Kgrid {j+1}/{self.n_uniq_grid} layers  "
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
            print(f"  [DedupBankV2] after building {self.build_s:.1f}s", flush=True)

        self.prior = torch.as_tensor(
            np.stack([prior_fn(float(v), nmax) for v in V]), dtype=dtype, device=device)

    def verify(self, kmat_fn, V, n_probe=8, seed=0):
        """and `DedupBank.verify` same meaning."""
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
    """gate: on small K compare v2 against v1 **layer by layer, bit for bit**. If it fails, do not use v2."""
    from bank_dedup import DedupBank
    a = DedupBank(V, tgrid, nmax, kmat_fn, prior_fn, device, verbose=False)
    b = DedupBankV2(V, tgrid, nmax, kmat_fn, prior_fn, device, verbose=False,
                    host_resident=host_resident)
    bad_g = sum(int(not torch.equal(a.Kgrid[k].U, b.Kgrid[k].U))
                for k in range(len(tgrid)))
    bad_d = sum(int(not torch.equal(a.Kdelta_T[k].U, b.Kdelta_T[k].U))
                for k in range(len(tgrid) - 1))
    # the per-gene path is also checked (the non-batched branch uses `slot[d]`)
    bad_gene = sum(int(not torch.equal(a.Kgrid[k][d], b.Kgrid[k][d]))
                   for k in (0, len(tgrid) // 2, len(tgrid) - 1)
                   for d in (0, 1, len(V) // 2, len(V) - 1))
    return {"bit_exact": bad_g == 0 and bad_d == 0 and bad_gene == 0
            and bool(torch.equal(a.prior, b.prior)),
            "n_bad_Kgrid": bad_g, "n_bad_Kdelta": bad_d, "n_bad_per_gene": bad_gene,
            "prior_equal": bool(torch.equal(a.prior, b.prior)),
            "n_layers": len(tgrid), "n_uniq_delta": b.n_uniq_delta,
            "n_uniq_grid": b.n_uniq_grid, "host_resident": host_resident}
