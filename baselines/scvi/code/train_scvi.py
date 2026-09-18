# -*- coding: utf-8 -*-
"""STEP A * train scVI on the shared 2000-HVG split, official defaults.

Implements `PREREG_scvi.md`. Nothing here is tuned: the model is constructed
with library defaults and `train()` is called with no `max_epochs`, so
scvi-tools picks its own schedule. The resolved value is recorded.

`cell_type` is the `batch_key`, which is the slot scVI conditions its decoder
on and therefore what makes "generate a cell of type T" well defined.

The gene_hash of the split is ASSERTED, not assumed -- the spec stops the run if
it differs, because the whole point of this arm is that it sits on the same
data as ours and cfDiffusion.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
X = Path(__file__).resolve().parents[3] / "assets" / "legacy_root"
DATA = X / "_shared" / "data" / "all"
GENE_HASH = "6f7963523c7a4c5bf1df7264c4c11e0a42fb0a9c4d52ca91d5724ae5e1eadb12"
SEED = 20261050


def main(max_epochs: int | None = None) -> None:
    import anndata as ad
    import pandas as pd
    import scvi
    import torch
    from scipy import sparse

    scvi.settings.seed = SEED

    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    if meta["gene_hash"] != GENE_HASH:
        raise SystemExit(f"gene_hash mismatch: {meta['gene_hash']} != {GENE_HASH}")
    genes = eval(meta["genes"]) if isinstance(meta["genes"], str) else meta["genes"]
    print(f"[data] gene_hash asserted OK  ({GENE_HASH[:16]}...)  n_genes={len(genes)}")

    counts = np.load(DATA / "train.npy")
    labels = np.load(DATA / "train_celltype.npy", allow_pickle=True).astype(str)
    assert counts.shape[1] == 2000 == len(genes)
    assert float(np.mean(counts == np.round(counts))) == 1.0, "train.npy not integral"
    print(f"[data] train {counts.shape[0]:,} cells x {counts.shape[1]} genes, "
          f"{len(set(labels))} types")

    adata = ad.AnnData(
        X=sparse.csr_matrix(counts.astype(np.float32)),
        obs=pd.DataFrame({"cell_type": pd.Categorical(labels)},
                         index=[f"c{i}" for i in range(len(counts))]),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")))

    scvi.model.SCVI.setup_anndata(adata, batch_key="cell_type")
    model = scvi.model.SCVI(adata)          # library defaults, nothing passed
    print(model)

    t0 = time.time()
    model.train(**({} if max_epochs is None else {"max_epochs": max_epochs}))                            # unchanged library schedule unless an explicit test budget is supplied
    secs = time.time() - t0

    mdir = HERE / "out" / "model"
    model.save(str(mdir), overwrite=False, save_anndata=False)

    hist = {k: [float(x) for x in v.iloc[:, 0].tolist()]
            for k, v in model.history.items()}
    rec = {
        "scvi_version": scvi.__version__,
        "seed": SEED,
        "gene_hash": GENE_HASH,
        "n_train_cells": int(counts.shape[0]),
        "n_genes": int(counts.shape[1]),
        "batch_key": "cell_type",
        "n_batches": int(adata.obs["cell_type"].nunique()),
        "batch_order": [str(c) for c in adata.obs["cell_type"].cat.categories],
        "max_epochs_resolved": int(model.trainer.max_epochs),
        "actual_epochs": int(model.trainer.current_epoch),
        "walltime_s": secs,
        "model_kwargs": "library defaults (n_latent=10, n_hidden=128, "
                        "n_layers=1, dropout_rate=0.1, dispersion='gene', "
                        "gene_likelihood='zinb')",
        "history_keys": list(hist.keys()),
        "history": hist,
        "cuda": bool(torch.cuda.is_available()),
    }
    p = HERE / "logs" / "train_scvi.json"
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")

    print(f"\n[train] resolved max_epochs = {rec['max_epochs_resolved']} "
          f"(library default for {counts.shape[0]:,} cells)")
    print(f"[train] {secs/60:.1f} min")
    for k in ("elbo_train", "reconstruction_loss_train"):
        if k in hist and hist[k]:
            print(f"[hist] {k}: {hist[k][0]:.1f} -> {hist[k][-1]:.1f}")
    print(f"[out] {mdir}")
    print(f"[out] {p}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="optional runtime-test budget; omitted means the original library schedule")
    args = ap.parse_args()
    if args.max_epochs is not None and args.max_epochs <= 0:
        ap.error("--max-epochs must be positive")
    main(max_epochs=args.max_epochs)
