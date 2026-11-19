# moved from the data loading and derived statistics lines of cc/experiments/EXP134_more_ckpts/run_exp134.py::main()
#   (astype(np.int64) of train/val, Vg, iscale, dmax, spectrum()).
# changes (item by item):
#   1. path changed from EXP35_generalization/data to _shared/data (a byte-identical copy; see DATA.md and the P0 gate assertion).
#   2. collected into a single function load_panel() returning a dataclass; in EXP134 they were local variables in main(); expressions unchanged.
#   3. spectrum() moved verbatim from EXP134::spectrum (needed for the wake-sigma grid of OURS); only OURS uses it in this round.
# expressions and constants unchanged.
"""lock data + derived statistics (all estimated from train only, zero leakage)."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .seeds import DATA_DIR, GENE_HASH, NMAX


@dataclass
class Panel:
    train: np.ndarray
    val: np.ndarray
    meta: dict
    dim: int
    Vg: np.ndarray
    iscale: np.ndarray
    dmax: np.ndarray
    nmax: int


def load_panel():
    meta = json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    assert meta["gene_hash"] == GENE_HASH, f"GENE HASH {meta['gene_hash']} != {GENE_HASH}"
    train = np.load(DATA_DIR / "train.npy").astype(np.int64)
    val = np.load(DATA_DIR / "val.npy").astype(np.int64)
    dim = train.shape[1]
    Vg = np.asarray(meta["Vg"], np.float64)
    iscale = 1.0 / (1.0 + train.mean(0))
    dmax = np.concatenate([train, val]).max(0).astype(int)
    return Panel(train=train, val=val, meta=meta, dim=dim, Vg=Vg, iscale=iscale, dmax=dmax, nmax=NMAX)


def spectrum(train):
    """EXP134::spectrum verbatim. Estimated from train only."""
    tr = train.astype(np.float64); mu0 = tr.mean(0)
    Psi0 = np.cov(tr, rowvar=False); np.fill_diagonal(Psi0, np.diag(Psi0) - mu0)
    Dm = 1 / np.sqrt(np.maximum(mu0, 1e-9)); C0 = Dm[:, None] * Psi0 * Dm[None, :]
    w, U = np.linalg.eigh(C0); o = np.argsort(w)[::-1]; w = w[o]; U = U[:, o]
    keep = w > 1.0
    return w[keep]
