# -*- coding: utf-8 -*-
"""scANVI baseline: train. Official defaults, `cell_type` as labels_key.

Why this arm exists (SPEC_queue_scanvi_cfgen.md): CFGen's own paper conditions
on cell type with scANVI/scPoli, not vanilla scVI. Our existing scVI arm uses
`batch_key=cell_type` plus prior sampling, which is the weaker mode. scANVI is
the fair label-conditioned VAE baseline, and running it closes the "you
compared against a crippled scVI" objection.

Nothing is tuned: `SCANVI.from_scvi_model` / `SCANVI` defaults, `train()` with
no `max_epochs` so the library picks its own schedule. The resolved value is
recorded.
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
SEED = 20261060


def main(max_epochs: int | None = None) -> None:
    import anndata as ad
    import pandas as pd
    import scvi
    import torch
    from scipy import sparse

    scvi.settings.seed = SEED
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    if meta["gene_hash"] != GENE_HASH:
        raise SystemExit("gene_hash mismatch")
    genes = eval(meta["genes"]) if isinstance(meta["genes"], str) else meta["genes"]

    counts = np.load(DATA / "train.npy")
    labels = np.load(DATA / "train_celltype.npy", allow_pickle=True).astype(str)
    print(f"[data] {counts.shape[0]:,} cells x {counts.shape[1]} genes, "
          f"{len(set(labels))} types; gene_hash OK", flush=True)

    adata = ad.AnnData(
        X=sparse.csr_matrix(counts.astype(np.float32)),
        obs=pd.DataFrame({"cell_type": pd.Categorical(labels)},
                         index=[f"c{i}" for i in range(len(counts))]),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")))

    # labels_key, NOT batch_key: this is the label-conditioned slot.
    # `unlabeled_category` is required by the API; every cell here is labelled,
    # so the category is present but empty, which is the intended usage.
    scvi.model.SCANVI.setup_anndata(adata, labels_key="cell_type",
                                    unlabeled_category="Unknown")
    model = scvi.model.SCANVI(adata)
    print(model, flush=True)

    t0 = time.time()
    model.train(**({} if max_epochs is None else {"max_epochs": max_epochs}))
    secs = time.time() - t0

    model.save(str(HERE / "out" / "model"), overwrite=True, save_anndata=False)
    hist = {k: [float(x) for x in v.iloc[:, 0].tolist()]
            for k, v in model.history.items()}
    rec = {"scvi_version": scvi.__version__, "seed": SEED,
           "gene_hash": GENE_HASH, "labels_key": "cell_type",
           "unlabeled_category": "Unknown",
           "n_train_cells": int(counts.shape[0]),
           "label_order": [str(c) for c in adata.obs["cell_type"].cat.categories],
           "max_epochs_resolved": int(model.trainer.max_epochs),
           "actual_epochs": int(model.trainer.current_epoch),
           "walltime_s": secs,
           "model_kwargs": "library defaults (SCANVI)",
           "history": hist, "cuda": bool(torch.cuda.is_available())}
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / "train_scanvi.json").write_text(json.dumps(rec, indent=1),
                                                     encoding="utf-8")
    print(f"[train] max_epochs={rec['max_epochs_resolved']} in {secs/60:.1f} min",
          flush=True)
    for k in ("elbo_train", "reconstruction_loss_train"):
        if k in hist and hist[k]:
            print(f"[hist] {k}: {hist[k][0]:.1f} -> {hist[k][-1]:.1f}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="optional runtime-test budget; omitted means the original library schedule")
    args = ap.parse_args()
    if args.max_epochs is not None and args.max_epochs <= 0:
        ap.error("--max-epochs must be positive")
    main(max_epochs=args.max_epochs)
