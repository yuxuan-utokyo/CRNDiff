# 搬自 cc/experiments/EXP134_more_ckpts/run_exp134.py::main() 的数据加载与派生统计那几行
#   （train/val 的 astype(np.int64)、Vg、iscale、dmax、spectrum()）。
# 改动（逐条）：
#   1. 路径从 EXP35_generalization/data 改为 _shared/data（逐字节相同的一份，见 DATA.md 与 P0 闸门断言）。
#   2. 收拢成 load_panel() 一个函数返回 dataclass，EXP134 里是散在 main() 里的局部变量；表达式未改。
#   3. spectrum() 逐字搬自 EXP134::spectrum（OURS 的 wake-sigma 网格需要），本轮只有 OURS 用。
# 表达式与常量逐字未动。
"""锁定数据 + 派生统计（全部只从 train 估，零泄漏）。"""
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
    """EXP134::spectrum 逐字。只从 train 估。"""
    tr = train.astype(np.float64); mu0 = tr.mean(0)
    Psi0 = np.cov(tr, rowvar=False); np.fill_diagonal(Psi0, np.diag(Psi0) - mu0)
    Dm = 1 / np.sqrt(np.maximum(mu0, 1e-9)); C0 = Dm[:, None] * Psi0 * Dm[None, :]
    w, U = np.linalg.eigh(C0); o = np.argsort(w)[::-1]; w = w[o]; U = U[:, o]
    keep = w > 1.0
    return w[keep]
