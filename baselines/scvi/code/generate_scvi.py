# -*- coding: utf-8 -*-
"""STEP A * conditional generation from the trained scVI.

Implements `PREREG_scvi.md` section 3, using scVI's OWN generative path:

    z ~ N(0, I)                      prior, 10-D
    px = module.generative(z, library, batch_index = target type)
    counts ~ px  (the fitted ZINB)

so the counts come from the model's fitted likelihood, not from a hand-rolled
transform. They are integer and non-negative by construction -- a structural
property of scVI, recorded rather than credited as a pass.

Library sizes are drawn from the REAL val cells of the target type
(seed 20261052), the same choice the cfDiffusion arm makes, so the two
baselines stay comparable. That is generous to both.

Also produces the UNCONDITIONAL row required by the three-row purity protocol:
identical procedure, but the batch covariate is drawn from the empirical
training mix instead of being fixed to the target. Same model, so the
conditional/unconditional gap isolates conditioning.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
X = Path(__file__).resolve().parents[3] / "assets" / "legacy_root"
DATA = X / "_shared" / "data" / "all"
GENE_HASH = "6f7963523c7a4c5bf1df7264c4c11e0a42fb0a9c4d52ca91d5724ae5e1eadb12"

TYPES = ["Endothelial", "Myeloid", "Neuronal"]
N_GEN = 20000
SAMPLE_SEED = 20261051
LIB_SEED = 20261052
BATCH = 4096


def real_library_pool(ctype: str) -> np.ndarray:
    Xv = np.load(DATA / "val.npy", mmap_mode="r")
    yv = np.load(DATA / "val_celltype.npy", allow_pickle=True).astype(str)
    idx = np.flatnonzero(yv == ctype)
    if len(idx) == 0:
        raise SystemExit(f"no real val cells of type {ctype}")
    return np.asarray(Xv[np.sort(idx)], dtype=np.float64).sum(1)


def main() -> None:
    import torch
    import scvi
    from scvi.distributions import ZeroInflatedNegativeBinomial

    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    if meta["gene_hash"] != GENE_HASH:
        raise SystemExit("gene_hash mismatch")

    rec = json.loads((HERE / "logs" / "train_scvi.json").read_text(encoding="utf-8"))
    order = rec["batch_order"]
    print(f"[model] batch order: {order}")

    import anndata as ad
    import pandas as pd
    from scipy import sparse
    genes = eval(meta["genes"]) if isinstance(meta["genes"], str) else meta["genes"]
    counts = np.load(DATA / "train.npy", mmap_mode="r")
    labels = np.load(DATA / "train_celltype.npy", allow_pickle=True).astype(str)
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.asarray(counts[:64], dtype=np.float32)),
        obs=pd.DataFrame({"cell_type": pd.Categorical(labels[:64],
                                                      categories=order)},
                         index=[f"c{i}" for i in range(64)]),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    scvi.model.SCVI.setup_anndata(adata, batch_key="cell_type")
    model = scvi.model.SCVI.load(str(HERE / "out" / "model"), adata=adata)
    mod = model.module.eval()
    dev = next(mod.parameters()).device
    n_latent = mod.n_latent
    print(f"[model] loaded, n_latent={n_latent}, device={dev}")

    outdir = HERE / "out" / "counts"
    outdir.mkdir(parents=True, exist_ok=True)
    train_mix = pd.Series(labels).value_counts(normalize=True).reindex(order).values

    meta_out = {"scvi_version": rec["scvi_version"], "n": N_GEN,
                "sample_seed": SAMPLE_SEED, "library_seed": LIB_SEED,
                "batch_order": order, "arms": {}}

    jobs = [(t, "conditional") for t in TYPES] + [(t, "unconditional") for t in TYPES]
    for ctype, mode in jobs:
        rng = np.random.default_rng(LIB_SEED)
        lib_pool = real_library_pool(ctype)
        lib = rng.choice(lib_pool, size=N_GEN, replace=True)
        g = torch.Generator(device="cpu").manual_seed(
            SAMPLE_SEED + (0 if mode == "conditional" else 1))
        brng = np.random.default_rng(SAMPLE_SEED + 7)
        if mode == "conditional":
            bidx_all = np.full(N_GEN, order.index(ctype), dtype=np.int64)
        else:
            bidx_all = brng.choice(len(order), size=N_GEN, p=train_mix)

        out = np.empty((N_GEN, len(genes)), dtype=np.int64)
        with torch.no_grad():
            for i in range(0, N_GEN, BATCH):
                j = min(i + BATCH, N_GEN)
                z = torch.randn(j - i, n_latent, generator=g).to(dev)
                lb = torch.tensor(np.log(lib[i:j]), dtype=torch.float32,
                                  device=dev).unsqueeze(1)
                bi = torch.tensor(bidx_all[i:j], device=dev).unsqueeze(1)
                px = mod.generative(z=z, library=lb, batch_index=bi)["px"]
                out[i:j] = px.sample().cpu().numpy().astype(np.int64)

        tag = "cond" if mode == "conditional" else "uncond"
        f = outdir / f"scvi_{tag}_{ctype}_n{N_GEN}_seed{SAMPLE_SEED}.npy"
        if f.exists():
            raise SystemExit(f"refusing to overwrite {f}")
        np.save(f, out)
        info = {"file": f.name, "mode": mode, "type": ctype,
                "n": int(N_GEN),
                "min": int(out.min()), "max": int(out.max()),
                "frac_zero": float((out == 0).mean()),
                "library_median": float(np.median(out.sum(1))),
                "library_median_assigned": float(np.median(lib)),
                "library_median_real": float(np.median(lib_pool)),
                "all_integer": True, "any_negative": bool((out < 0).any())}
        meta_out["arms"][f.name] = info
        print(f"  [{mode:<13}] {ctype:<12} max {info['max']:>6}  "
              f"zero {info['frac_zero']:.4f}  lib med {info['library_median']:.0f} "
              f"(assigned {info['library_median_assigned']:.0f})")

    p = HERE / "out" / "generate_meta.json"
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.write_text(json.dumps(meta_out, indent=1), encoding="utf-8")
    print(f"\n[out] {p}")


if __name__ == "__main__":
    main()
