# 搬自 _shared_discussion/ours_fast.py（整文件），改动（逐条）：
#   1. 文件顶部加入本行注释；
#   2. `import_module("ours")` → `import_module("_shared.vendor_ours")`（共 2 处：sample_bridge_fast
#      与 verify_bitexact）。指向的是同一份逐字复制的代码（见 vendor_ours.py 文件头），
#      发布目录不能依赖 sys.path 上恰好有个叫 `ours` 的模块。
# 其余逐字未动（FastBank / _cat / sample_bridge_fast 的算法、dtype、生成器消费顺序全部原样，
# 因此仍与 ours.sample_bridge 逐位一致）。
"""
Drop-in vectorised replacement for `ours.sample_bridge`.

WHY: the reference implementation loops over genes in Python and, for every gene of
every step, does two host<->device round trips plus two float64 numpy fancy-index
gathers of shape [B, nmax+1]. At B=20000, dim=61, nmax=128 that is 244 transfers and
122 gathers of 20.6 MB each, per step. Measured cost is ~1.7 s/step, i.e. 1.4 us per
coordinate for what is one gather, one multiply and one length-129 categorical draw.

WHAT CHANGES: nothing mathematically. The kernel tables live on the device as
[dim, nmax+1, nmax+1] tensors; the gathers become `index_select`; the running state
`n` never leaves the device. The per-gene `torch.multinomial` calls are kept **in the
same order** so the generator is consumed identically -- therefore the output is
bit-identical to the reference, which `verify_bitexact` below asserts.

DTYPE: the reference builds the bridge weights in float64 (Kgrid/Kdelta are float64
numpy). Bit-exactness therefore requires float64 here too. `dtype=torch.float32`
is offered as a faster non-exact mode; it must be validated statistically, not by
the bit-exactness gate.

FURTHER SPEEDUP (not implemented here, deliberately): batching all `dim` genes into a
single multinomial over [B*dim, nmax+1] removes the remaining Python loop, but changes
the order in which the generator is consumed and is therefore NOT bit-exact. Add it
only behind a separate flag with its own statistical equivalence test.
"""
from __future__ import annotations
import numpy as np
import torch


class FastBank:
    """Device-resident view of a ours.KernelBank. Built once, reused for every seed."""

    def __init__(self, bank, device, dtype=torch.float64):
        self.device, self.dtype, self.K = device, dtype, bank.K
        self.dim, self.nmax = bank.dim, bank.nmax
        self.tgrid = bank.tgrid
        # Kgrid[k] : [dim, nmax+1, nmax+1], rows indexed by n0
        self.Kgrid = [torch.as_tensor(np.stack(bank.Kgrid[k]), dtype=dtype, device=device)
                      for k in range(bank.K)]
        # Kdelta[k] stored ALREADY TRANSPOSED to [dim, nmax+1(n), nmax+1(n_s)] so the
        # gather is a row gather (contiguous) instead of the reference's column gather.
        self.Kdelta_T = [torch.as_tensor(np.stack(bank.Kdelta[k]), dtype=dtype, device=device)
                         .transpose(1, 2).contiguous() for k in range(bank.K - 1)]
        self.prior = torch.as_tensor(bank.prior, dtype=dtype, device=device)  # [dim, nmax+1]


def _cat(p, gen):
    """Same semantics as ours._categorical_torch: nan/inf -> 0, clamp>=0, all-zero row -> 0."""
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    tot = p.sum(dim=1)
    bad = ~torch.isfinite(tot) | (tot <= 0)
    if bad.any():
        p = p.clone()
        p[bad] = 0.0
        p[bad, 0] = 1.0
    return torch.multinomial(p, 1, generator=gen).squeeze(1)


def sample_bridge_fast(post_fn, fbank: FastBank, *, n_samples: int, seed: int):
    """Bit-identical to ours.sample_bridge(post_fn, bank, n_samples=..., seed=...)."""
    dev, dt = fbank.device, fbank.dtype
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    dim, nmax, tgrid = fbank.dim, fbank.nmax, fbank.tgrid

    # initial draw: reference uses numpy _categorical per gene on the tiled prior
    from importlib import import_module
    O = import_module("_shared.vendor_ours")
    n = np.stack([O._categorical(np.tile(fbank.prior[d].cpu().numpy(), (n_samples, 1)), rng)
                  for d in range(dim)], axis=1)
    n_t = torch.as_tensor(n, dtype=torch.long, device=dev)                      # [B, dim]

    for k in range(len(tgrid) - 1):
        t = float(tgrid[k])
        post = post_fn(n_t.detach().cpu().numpy().astype(np.float32), t)        # [B,dim,nmax+1]
        post_t = torch.as_tensor(post, device=dev)                              # float32, as reference
        Knext, KdelT = fbank.Kgrid[k + 1], fbank.Kdelta_T[k]
        for d in range(dim):
            n0 = _cat(post_t[:, d, :], gen)                                     # [B]
            row = Knext[d].index_select(0, n0).to(dt)                           # [B, nmax+1] over n_s
            col = KdelT[d].index_select(0, n_t[:, d].clamp(0, nmax))            # [B, nmax+1] over n_s
            n_t[:, d] = _cat(row * col, gen)
    return n_t.detach().cpu().numpy().astype(np.int64)


# --------------------------------------------------------------------------- #
def verify_bitexact(post_fn, bank, *, n_samples=256, seed=12345, device=None):
    """Hard gate: fast path must reproduce the reference sampler element for element."""
    from importlib import import_module
    O = import_module("_shared.vendor_ours")
    dev = device or O.DEVICE
    a = O.sample_bridge(post_fn, bank, n_samples=n_samples, seed=seed)
    b = sample_bridge_fast(post_fn, FastBank(bank, dev), n_samples=n_samples, seed=seed)
    ok = bool(np.array_equal(a, b))
    return ok, int((a != b).sum()), a.shape
