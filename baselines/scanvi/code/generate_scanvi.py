# -*- coding: utf-8 -*-
"""scANVI baseline: conditional generation through its label-conditioned path.

    z ~ N(0, I)
    px = module.generative(z, library, batch_index, y = <target label>)
    counts ~ px

Counts come from the model's own fitted likelihood, so they are integer and
non-negative BY CONSTRUCTION. That is a structural property of a count-decoder
VAE and is reported as such -- it is not a legality test the model passed.

Library sizes are drawn from the REAL val cells of the target type
(seed 20261052), identical to the scVI and cfDiffusion arms so the baselines
stay comparable; this is generous to all of them.

The unconditional row uses the SAME model with labels drawn from the empirical
training mix, so the conditional/unconditional gap isolates conditioning.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
X = Path(__file__).resolve().parents[3] / "assets" / "legacy_root"
DATA = X / "_shared" / "data" / "all"
TYPES = ["Endothelial", "Myeloid", "Neuronal"]
N_GEN = 20000
SAMPLE_SEED = 20261061
LIB_SEED = 20261052
BATCH = 4096
SHELL_PER_TYPE = 6      # registry-carrier only; must cover every trained type
# CPU by default: this is a small MLP decoder, and the GPU is held by the CFGen
# run. Sampling is deterministic in the CPU generator either way, so the device
# does not change which numbers come out.
ACCELERATOR = "cpu"


def lib_pool(ctype):
    Xv = np.load(DATA / "val.npy", mmap_mode="r")
    yv = np.load(DATA / "val_celltype.npy", allow_pickle=True).astype(str)
    return np.asarray(Xv[np.sort(np.flatnonzero(yv == ctype))],
                      dtype=np.float64).sum(1)


def main() -> None:
    import anndata as ad
    import pandas as pd
    import scvi
    import torch
    from scipy import sparse

    rec = json.loads((HERE / "logs" / "train_scanvi.json").read_text(encoding="utf-8"))
    order = rec["label_order"]
    print(f"[model] label order: {order}", flush=True)

    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    genes = eval(meta["genes"]) if isinstance(meta["genes"], str) else meta["genes"]
    counts = np.load(DATA / "train.npy", mmap_mode="r")
    labels = np.load(DATA / "train_celltype.npy", allow_pickle=True).astype(str)

    # The shell adata only carries the registry -- generation below calls
    # `mod.generative` directly and never touches X. But the registry has to
    # come out with the SAME label set the checkpoint was trained on, and that
    # is fragile in two ways:
    #
    #   * a plain head slice (`labels[:64]`) contains only 6 of the 11 types, so
    #     re-registration ends up counting the `unlabeled_category` as a real
    #     class and the module is built with n_labels=12 against an 11-class
    #     checkpoint -- every layer touching y is then off by one and
    #     `load_state_dict` dies on a size mismatch. Measured, not guessed:
    #     forcing `categories=order` on the pandas side does NOT prevent it.
    #   * calling `setup_anndata` here would re-register from scratch instead of
    #     letting `load` migrate the archived registry.
    #
    # So: stratified shell containing every type, built exactly the way
    # `train_scanvi.py` builds it (plain `pd.Categorical`, no `categories=`
    # kwarg), and no `setup_anndata` -- `load` transfers the saved registry.
    keep = np.sort(np.concatenate(
        [np.flatnonzero(labels == t)[:SHELL_PER_TYPE] for t in order]))
    shell_lab = pd.Categorical(labels[keep])
    if list(shell_lab.categories) != list(order):
        raise SystemExit(f"shell label set drift: {list(shell_lab.categories)} "
                         f"vs trained {list(order)}")
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.asarray(counts[keep], dtype=np.float32)),
        obs=pd.DataFrame({"cell_type": shell_lab},
                         index=[f"c{i}" for i in range(len(keep))]),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    model = scvi.model.SCANVI.load(str(HERE / "out" / "model"), adata=adata,
                                   accelerator=ACCELERATOR)
    mod = model.module.eval()
    dev = next(mod.parameters()).device
    n_latent = mod.n_latent
    # Fail here rather than silently generating from a mis-indexed label space.
    if int(mod.y_prior.shape[1]) != len(order) or int(mod.n_labels) != len(order):
        raise SystemExit(
            f"label-space mismatch: module has n_labels={mod.n_labels}, "
            f"y_prior={tuple(mod.y_prior.shape)}, but training registered "
            f"{len(order)} types {order}")
    print(f"[model] loaded, n_latent={n_latent}, device={dev}, "
          f"n_labels={mod.n_labels} (== {len(order)} trained types)", flush=True)

    # Guard: the label must actually move the z1 prior. If `decoder_z1_z2`
    # returned the same mean for two different labels, every "conditional" cell
    # below would be an unconditional cell and the arm would look falsely weak.
    with torch.no_grad():
        _z2 = torch.randn(256, n_latent, generator=torch.Generator("cpu")
                          .manual_seed(SAMPLE_SEED)).to(dev)
        _m = [mod.decoder_z1_z2(_z2, torch.full((256, 1), k, dtype=torch.long,
                                                device=dev))[0]
              for k in (order.index(TYPES[0]), order.index(TYPES[1]))]
        _sep = float((_m[0] - _m[1]).abs().mean())
    if not _sep > 1e-6:
        raise SystemExit("label has no effect on the z1 prior "
                         f"(mean |delta| = {_sep:.3e}); conditioning is dead")
    print(f"[check] label moves the z1 prior: mean |delta| = {_sep:.4f}",
          flush=True)

    out = HERE / "out" / "counts"
    out.mkdir(parents=True, exist_ok=True)
    mix = pd.Series(labels).value_counts(normalize=True).reindex(order).values
    meta_out = {"n": N_GEN, "sample_seed": SAMPLE_SEED,
                "library_seed": LIB_SEED, "label_order": order, "arms": {}}

    for ctype in TYPES:
        for mode in ("conditional", "unconditional"):
            rng = np.random.default_rng(LIB_SEED)
            pool = lib_pool(ctype)
            lib = rng.choice(pool, size=N_GEN, replace=True)
            g = torch.Generator(device="cpu").manual_seed(
                SAMPLE_SEED + (0 if mode == "conditional" else 1))
            lrng = np.random.default_rng(SAMPLE_SEED + 7)
            if mode == "conditional":
                yidx = np.full(N_GEN, order.index(ctype), dtype=np.int64)
            else:
                yidx = lrng.choice(len(order), size=N_GEN, p=mix)

            arr = np.empty((N_GEN, len(genes)), dtype=np.int64)
            with torch.no_grad():
                for i in range(0, N_GEN, BATCH):
                    j = min(i + BATCH, N_GEN)
                    yy = torch.tensor(yidx[i:j], device=dev).unsqueeze(1)
                    # scANVI's label enters ONLY through the z2 -> z1 prior.
                    # `SCANVAE` inherits `generative` from `VAE`, whose
                    # observation decoder declares `n_cat_list=[n_batch]`, so the
                    # trailing `y` it is handed is never consumed: p(x | z1) does
                    # not see the label. Sampling z1 ~ N(0, I) and passing y to
                    # `generative` therefore produces UNCONDITIONAL cells with a
                    # label argument that does nothing -- measured, an earlier
                    # version of this script did exactly that and the conditional
                    # and unconditional rows came out identical to three decimals.
                    # The real ancestral path, as written in `SCANVAE.loss`, is
                    #   z2 ~ N(0, I);  pz1_m, pz1_v = decoder_z1_z2(z2, y);
                    #   z1 ~ N(pz1_m, sqrt(pz1_v));  x ~ p(x | z1, library)
                    z2 = torch.randn(j - i, n_latent, generator=g).to(dev)
                    pz1_m, pz1_v = mod.decoder_z1_z2(z2, yy)
                    eps = torch.randn(pz1_m.shape, generator=g).to(dev)
                    z1 = pz1_m + eps * torch.sqrt(pz1_v)
                    lb = torch.tensor(np.log(lib[i:j]), dtype=torch.float32,
                                      device=dev).unsqueeze(1)
                    bi = torch.zeros(j - i, 1, dtype=torch.long, device=dev)
                    px = mod.generative(z=z1, library=lb, batch_index=bi,
                                        y=yy)["px"]
                    arr[i:j] = px.sample().cpu().numpy().astype(np.int64)

            tag = "cond" if mode == "conditional" else "uncond"
            f = out / f"scanvi_{tag}_{ctype}_n{N_GEN}_seed{SAMPLE_SEED}.npy"
            np.save(f, arr)
            meta_out["arms"][f.name] = {
                "mode": mode, "type": ctype, "n": int(N_GEN),
                "min": int(arr.min()), "max": int(arr.max()),
                "frac_zero": float((arr == 0).mean()),
                "library_median": float(np.median(arr.sum(1))),
                "library_median_assigned": float(np.median(lib)),
                "any_negative": bool((arr < 0).any())}
            print(f"  [{mode:<13}] {ctype:<12} max {arr.max():>6} "
                  f"zero {(arr == 0).mean():.4f} "
                  f"lib med {np.median(arr.sum(1)):.0f}", flush=True)

    (HERE / "out" / "generate_meta.json").write_text(
        json.dumps(meta_out, indent=1), encoding="utf-8")
    print("[out] generate_meta.json", flush=True)


if __name__ == "__main__":
    main()
