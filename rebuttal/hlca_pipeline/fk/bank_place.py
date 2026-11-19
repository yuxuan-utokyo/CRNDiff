# -*- coding: utf-8 -*-
"""R16 (vice 11-09 21:0x, option (a)): bank PLACEMENT wrapper for the frozen sampler. No formula, RNG call order or dtype changes.

What it does: replace the name `DedupBank` inside the already-imported frozen module `sample_hvg2k_noa4` with a factory that
builds the frozen `bank_dedup_v2.DedupBankV2(..., host_resident=True, pin_memory=True)` instead of `bank_dedup.DedupBank`.
Both the pool sampler (`sample_hvg2k_noa4.main`, line 291 `bank = DedupBank(...)`) and the FK runner
(`experiments/run_one.py` line 321 and `hvg/_frozen/fk_hvg_v2.py` line 590, both `BASE.DedupBank(...)`) resolve the name
through that module, so one attribute assignment covers both. No frozen file is edited; their sha256 are asserted on install.

DedupBankV2 (frozen, EXP180 section 4; already used by sample_hvg2k_shrink.py) differs from DedupBank only in placement:
  * Kgrid: the (tau, V) layers are deduplicated by EXACT float64 equality of tau (no rounding) and kept in pinned CPU memory;
    `Kgrid[k]` uploads that layer to the GPU on first use and keeps the last 2 (the sampler and FK loops use Kgrid[k+1] with k
    increasing). Upload is a dtype-preserving copy (float64 -> float64).
  * Kdelta_T: deduplicated the same way (about 5 distinct layers on a linspace grid) and resident on the GPU.
  * prior, dim, nmax, tgrid, K, uV, n_V, verify(): the same as DedupBank. Layer objects are bank_dedup._Slot (imported, not copied).
The frozen `bank_dedup_v2.gate_vs_v1` checks layer-by-layer bit equality against DedupBank; `gate()` below runs it.

    import bank_place; bank_place.install()        # after sys.path holds the frozen code dir, before the bank is built
v1 (vice 11-09 22:5x): optional R16_BANK_MMAP_DIR -> the frozen DedupBankV2(mmap_dir=...) disk cache for Kgrid, guarded by a
"<file>.complete" marker (see below); unset = v0 behaviour (pinned host memory).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import importlib
import sys
from pathlib import Path

FROZEN_SHA = {   # the copies under assets/legacy_root/Baseline/OURS/code (byte-identical to hvg/_frozen/)
    "sample_hvg2k_noa4.py": "6cc170883d793c120229719bd3c7aa087de0382231351b849e7309459930263a",
    "bank_dedup.py": "e1e6db1c4d797ca05bf479516b7602b8953b94ee5550e1698484cd91c2a5be2e",
    "bank_dedup_v2.py": "2ac081ddd10b7951b092e02fddb6ad40aacc6e9231ffb9f2012558e0003b435e",
}
STATE = {"installed": False, "built": []}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def install(verbose=True):
    S = importlib.import_module("sample_hvg2k_noa4")
    code = Path(S.__file__).resolve().parent
    for f, want in FROZEN_SHA.items():
        got = _sha(code / f)
        if got != want:
            raise SystemExit(f"[bank_place] frozen file changed: {code / f} sha {got} != {want}")
    V2 = importlib.import_module("bank_dedup_v2")
    assert Path(V2.__file__).resolve().parent == code, (V2.__file__, code)
    assert S.DedupBank.__module__ == "bank_dedup", f"sample_hvg2k_noa4.DedupBank already replaced ({S.DedupBank})"
    orig = S.DedupBank

    def DedupBankPlaced(V, tgrid, nmax, kmat_fn, prior_fn, device, dtype=None, verbose=True):
        kw = {} if dtype is None else {"dtype": dtype}
        md = mmap_dir_from_env()
        fp = cache_path(md, V, tgrid, nmax) if md else None
        if fp is not None:
            check_cache_or_claim(fp)
        b = V2.DedupBankV2(V, tgrid, nmax, kmat_fn, prior_fn, device, verbose=verbose,
                           host_resident=True, pin_memory=True, mmap_dir=md, **kw)
        if fp is not None:
            assert Path(b.mmap_path) == fp, (b.mmap_path, fp)
            mark_complete(fp, b)
        STATE["built"].append({"nmax": int(nmax), "K": int(b.K), "n_V": int(b.n_V), "n_uniq_grid": b.n_uniq_grid,
                               "n_uniq_delta": b.n_uniq_delta, "build_s": round(b.build_s, 1),
                               "mmap": str(fp) if fp is not None else None})
        print(f"[bank_place] DedupBankV2(host_resident=True) {STATE['built'][-1]}", flush=True)
        return b

    DedupBankPlaced.original = orig
    S.DedupBank = DedupBankPlaced
    STATE["installed"] = True
    if verbose:
        print(f"[bank_place] installed: sample_hvg2k_noa4.DedupBank -> DedupBankV2(host_resident=True, pin_memory=True); "
              f"frozen shas OK ({', '.join(f'{k} {v[:8]}' for k, v in FROZEN_SHA.items())})", flush=True)
    return S


def gate(V, tgrid, nmax, kmat_fn, prior_fn, device):
    """The frozen layer-by-layer bit gate (bank_dedup_v2.gate_vs_v1) with host_resident=True.
    v1: if R16_BANK_MMAP_DIR is set, the frozen gate runs unchanged but its DedupBankV2 gets mmap_dir (module name swapped for
    the duration of the call, restored after), so the memmap write path and, on a second run, the cache-hit path are compared."""
    V2 = importlib.import_module("bank_dedup_v2")
    md = mmap_dir_from_env()
    if not md:
        return V2.gate_vs_v1(V, tgrid, nmax, kmat_fn, prior_fn, device, host_resident=True)
    fp = cache_path(md, V, tgrid, nmax); check_cache_or_claim(fp); hit = fp.exists() and (fp.parent / (fp.name + ".complete")).exists()
    orig = V2.DedupBankV2
    V2.DedupBankV2 = functools.partial(orig, mmap_dir=md)
    try:
        r = V2.gate_vs_v1(V, tgrid, nmax, kmat_fn, prior_fn, device, host_resident=True)
    finally:
        V2.DedupBankV2 = orig
    if r["bit_exact"] and not hit:
        (fp.parent / (fp.name + ".complete")).write_text(json.dumps({"bytes": fp.stat().st_size, "by": "gate"}))
    r.update({"mmap_file": str(fp), "cache_hit": bool(hit)})
    return r


# ---- v1 (vice 11-09 22:5x): optional disk cache for the host-resident Kgrid, via the frozen DedupBankV2(mmap_dir=...) ----------
# The frozen code reuses a cache file whenever its SIZE matches, and np.memmap(mode="w+") creates the full-size file at once,
# so a crashed or concurrent build would look like a valid cache. Guard: a "<file>.complete" marker is written only after the
# frozen constructor returns (all layers written and flushed); a cache file without its marker stops the run.
def mmap_dir_from_env():
    v = os.environ.get("R16_BANK_MMAP_DIR", "").strip()
    return v or None


def cache_path(md, V, tgrid, nmax):
    """The file name the frozen DedupBankV2 uses (bank_dedup_v2.py, mmap branch), recomputed with the same expression."""
    import numpy as np
    ug = np.unique(np.asarray(tgrid, np.float64)); uV = np.unique(np.asarray(V, np.float64)); C = nmax + 1
    key = hashlib.sha256(np.concatenate([ug, uV, [nmax]]).tobytes()).hexdigest()[:16]
    return Path(md) / f"Kgrid_{key}_n{len(ug)}_V{len(uV)}_C{C}.f64"


def check_cache_or_claim(fp):
    done = fp.parent / (fp.name + ".complete")
    if fp.exists() and not done.exists():
        raise SystemExit(f"[bank_place] cache file without completion marker (crashed or concurrent build?): {fp} -- move it to "
                         f"~/QIU/_to_delete and rerun")
    print(f"[bank_place] mmap cache {'HIT' if done.exists() else 'WRITE (new)'}: {fp}", flush=True)


def mark_complete(fp, b):
    done = fp.parent / (fp.name + ".complete")
    if not done.exists():
        done.write_text(json.dumps({"bytes": fp.stat().st_size, "build_s": round(b.build_s, 1), "by": "pool/fk"}))
