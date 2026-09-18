# -*- coding: utf-8 -*-
"""CFGen: build the h5ad and hydra dataset configs it expects.

CFGen's dataset config wants counts in a NAMED LAYER (`layer_key: X_counts`)
and a `covariate_keys` list, so the shared h5ad is re-emitted with
`layers['X_counts']` populated. The counts themselves are untouched -- the same
frozen 2000-HVG split, gene_hash asserted.

Two config schemas, NOT one. `configs_encoder/dataset/*.yaml` and
`configs_sccfm/dataset/*.yaml` are different files in upstream: the flow-matching
one additionally carries `one_hot_encode_features`, `cov_embedding_dimensions`,
`size_factor_covariate` and `guidance_weights`, all of which
`CfgenEstimator.__init__` dereferences. Writing the encoder schema into both (as
an earlier version of this script did) blows up at flow-matching start. Each
schema below is copied key-for-key from upstream's own `pbmc3k.yaml` in the
corresponding directory, so nothing about their shape is invented and no default
value is altered.

A second, stratified SMOKE dataset is emitted alongside the real one: same 2000
genes, same 11 cell types (so `n_cat` and therefore the architecture are
identical), just few cells. It exists so the sampling and evaluation code can be
validated against a real checkpoint in minutes instead of hours.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HVG = HERE.parents[2]
CFGEN = HVG / "baseline_cfgen"
UP = CFGEN / "upstream"
X = Path(__file__).resolve().parents[3] / "assets" / "legacy_root"
DATA = X / "_shared" / "data" / "all"
GENE_HASH = "6f7963523c7a4c5bf1df7264c4c11e0a42fb0a9c4d52ca91d5724ae5e1eadb12"

SMOKE_PER_TYPE = 300
SMOKE_SEED = 20261072

# Upstream schema, key for key, from configs_*/dataset/pbmc3k.yaml.
ENCODER_KEYS = """layer_key: X_counts
covariate_keys: [cell_type]
subsample_frac: 1
normalization_type: log_gexp
is_binarized: False
theta_covariate: cell_type
split_rates: [0.90, 0.10]
"""

SCCFM_KEYS = """layer_key: X_counts
covariate_keys: [cell_type]
subsample_frac: 1
normalization_type: log_gexp
one_hot_encode_features: False
split_rates: [0.90, 0.10]
cov_embedding_dimensions: 100
is_binarized: False
theta_covariate: cell_type
size_factor_covariate: cell_type
guidance_weights:
  cell_type: 1
"""


def write_h5ad(path: Path, counts, labels, genes) -> None:
    import anndata as ad
    import pandas as pd
    from scipy import sparse

    Xs = sparse.csr_matrix(np.asarray(counts, dtype=np.float32))
    a = ad.AnnData(
        X=Xs,
        obs=pd.DataFrame({"cell_type": pd.Categorical(labels)},
                         index=[f"c{i}" for i in range(Xs.shape[0])]),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    a.layers["X_counts"] = Xs.copy()      # the layer CFGen reads
    a.uns["gene_hash"] = GENE_HASH
    a.write_h5ad(path, compression="gzip")
    print(f"[out] {path.name} {Xs.shape} ({path.stat().st_size/1e6:.0f} MB)",
          flush=True)


def main() -> None:
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    if meta["gene_hash"] != GENE_HASH:
        raise SystemExit("gene_hash mismatch")
    genes = eval(meta["genes"]) if isinstance(meta["genes"], str) else meta["genes"]

    out_dir = CFGEN / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    full = out_dir / "hvg2k_train_cfgen.h5ad"
    smoke = out_dir / "hvg2k_smoke_cfgen.h5ad"

    labels = np.load(DATA / "train_celltype.npy", allow_pickle=True).astype(str)
    order = sorted(set(labels.tolist()))
    print(f"[info] {len(order)} cell types, np.unique order = index order used "
          f"by RNAseqLoader.id2cov:", flush=True)
    for i, t in enumerate(order):
        print(f"       {i:>2}  {t}", flush=True)

    if not full.exists():
        counts = np.load(DATA / "train.npy")
        write_h5ad(full, counts, labels, genes)
        del counts
    else:
        print(f"[have] {full.name}", flush=True)

    if not smoke.exists():
        rng = np.random.default_rng(SMOKE_SEED)
        keep = []
        for t in order:                       # stratified: every type survives,
            idx = np.flatnonzero(labels == t)  # so n_cat is 11 here too
            k = min(SMOKE_PER_TYPE, len(idx))
            keep.append(rng.choice(idx, k, replace=False))
        keep = np.sort(np.concatenate(keep))
        counts = np.load(DATA / "train.npy", mmap_mode="r")
        write_h5ad(smoke, np.asarray(counts[keep]), labels[keep], genes)
        del counts
    else:
        print(f"[have] {smoke.name}", flush=True)

    n_written = 0
    for sub, keys in (("configs_encoder", ENCODER_KEYS),
                      ("configs_sccfm", SCCFM_KEYS)):
        d = UP / "configs" / sub / "dataset"
        if not d.exists():
            print(f"[warn] {d} missing", flush=True)
            continue
        for name, h5 in (("hvg2k", full), ("hvg2k_smoke", smoke)):
            f = d / f"{name}.yaml"
            f.write_text(f"dataset_path: {h5.as_posix()}\n" + keys,
                         encoding="utf-8")
            print(f"[out] {sub}/dataset/{name}.yaml -> {h5.name}", flush=True)
            n_written += 1
    if n_written != 4:
        raise SystemExit(f"expected 4 dataset configs, wrote {n_written}")
    print("CFGEN DATA PREP OK", flush=True)


if __name__ == "__main__":
    main()
