# -*- coding: utf-8 -*-
"""CFGen: conditional sampling.

Written against a real trained checkpoint (a short smoke run first, then the
full run), not against an assumption about the API.

Model reconstruction follows upstream's OWN generation notebook
(`notebooks/generate_with_cfgen/dentategyrus.ipynb`) -- build `EncoderModel`,
build the denoising net, hand both to `FM`, then
`generative_model.load_state_dict(ckpt["state_dict"])`. One deviation is forced:
the notebook recovers `denoising_model` and `feature_embeddings` from
`ckpt["hyper_parameters"]`, but pytorch-lightning 1.9.5 on this machine prints

    attribute 'denoising_model' removed from hparams because it cannot be pickled

so those entries do not exist in checkpoints written here. `train_cfgen.py`
therefore snapshots the architecture arguments and the trained covariate
embeddings itself; both are read back below. Where the checkpoint DOES happen to
carry `hyper_parameters["feature_embeddings"]`, the two copies are compared and
any mismatch is fatal -- silently sampling with untrained embeddings is exactly
the failure this guards against.

Sampling caliber is copied from the scANVI arm so the arms stay comparable:
  * N = 20000 per (type, arm)
  * library sizes drawn from the SAME pool -- real val cells of that type,
    LIB_SEED 20261052. This is also upstream's own recipe: their notebook passes
    `log_size_factor` taken from real cells rather than the log-normal prior.
  * "unconditional" = the label is drawn from the training mixture instead of
    being fixed, so label assignment is the ONLY difference between the two arms
  * n_sample_steps = 2, upstream's default in every notebook and in
    `launch_evaluation_metrics_unimodal.py`
Nothing in the model, the guidance weights, or the solver is altered.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HVG = HERE.parents[2]
CFGEN = HERE.parent
UP = Path(os.environ.get("CRNDIFF_CFGEN_UPSTREAM", str(CFGEN / "upstream"))).resolve()
X = Path(__file__).resolve().parents[3] / "assets" / "legacy_root"
DATA = X / "_shared" / "data" / "all"

TYPES = ["Endothelial", "Myeloid", "Neuronal"]
N_GEN = 20000
SAMPLE_SEED = 20261061
LIB_SEED = 20261052
N_SAMPLE_STEPS = 2          # upstream default
BATCH = 2000


def lib_pool(ctype):
    Xv = np.load(DATA / "val.npy", mmap_mode="r")
    yv = np.load(DATA / "val_celltype.npy", allow_pickle=True).astype(str)
    return np.asarray(Xv[np.sort(np.flatnonzero(yv == ctype))],
                      dtype=np.float64).sum(1)


def build_model(state, device):
    """Rebuild FM exactly as CfgenEstimator.init_model did, then load weights."""
    import torch
    from omegaconf import OmegaConf
    from cfgen.models.base.encoder_model import EncoderModel
    from cfgen.models.featurizers.category_featurizer import CategoricalFeaturizer
    from cfgen.models.fm.denoising_model import MLPTimeStep
    from cfgen.models.fm.fm import FM

    cfg = OmegaConf.create(state["cfg_yaml"])
    in_dim = state["in_dim"]

    # feature embeddings -- trained, but absent from FM.state_dict()
    feature_embeddings = {}
    for cov, sd in state["feature_embeddings"].items():
        fe = CategoricalFeaturizer(state["feature_embeddings_ncat"][cov],
                                   cfg.dataset.one_hot_encode_features,
                                   device,
                                   embedding_dimensions=cfg.denoising_module.embedding_dim)
        fe.load_state_dict(sd)
        feature_embeddings[cov] = fe.to(device)

    denoising_model = MLPTimeStep(
        in_dim=sum(in_dim.values()) if isinstance(in_dim, dict) else in_dim,
        hidden_dim=cfg.denoising_module.hidden_dim,
        dropout_prob=cfg.denoising_module.dropout_prob,
        n_blocks=cfg.denoising_module.n_blocks,
        size_factor_min=state["min_size_factor"],
        size_factor_max=state["max_size_factor"],
        embed_size_factor=cfg.denoising_module.embed_size_factor,
        covariate_list=cfg.dataset.covariate_keys,
        embedding_dim=cfg.denoising_module.embedding_dim,
        normalization=cfg.denoising_module.normalization,
        conditional=cfg.denoising_module.conditional,
        is_binarized=cfg.encoder.is_binarized,
        modality_list=state["modality_list"],
        guided_conditioning=cfg.denoising_module.guided_conditioning).to(device)

    encoder_model = EncoderModel(
        in_dim=state["gene_dim"],
        n_cat=state["feature_embeddings_ncat"][cfg.dataset.theta_covariate],
        conditioning_covariate=cfg.dataset.theta_covariate,
        **OmegaConf.to_container(cfg.encoder, resolve=True))

    fm = FM(encoder_model=encoder_model,
            denoising_model=denoising_model,
            feature_embeddings=feature_embeddings,
            plotting_folder=None,
            in_dim=in_dim,
            size_factor_statistics={"mean": state["size_factor_mu"],
                                    "sd": state["size_factor_sd"]},
            covariate_list=cfg.dataset.covariate_keys,
            theta_covariate=cfg.dataset.theta_covariate,
            size_factor_covariate=cfg.dataset.size_factor_covariate,
            is_binarized=cfg.encoder.is_binarized,
            modality_list=state["modality_list"],
            guidance_weights=OmegaConf.to_container(cfg.dataset.guidance_weights,
                                                    resolve=True),
            **OmegaConf.to_container(cfg.generative_model, resolve=True))

    ckpt = torch.load(state["fm_ckpt"], weights_only=False, map_location="cpu")
    fm.load_state_dict(ckpt["state_dict"], strict=True)   # strict: shape drift is fatal
    fm = fm.to(device).eval()

    # cross-check the embeddings against the checkpoint when it kept a copy
    hp = ckpt.get("hyper_parameters", {})
    if isinstance(hp, dict) and "feature_embeddings" in hp:
        for cov, fe in hp["feature_embeddings"].items():
            a = fe.state_dict()["embeddings.weight"].detach().cpu()
            b = feature_embeddings[cov].state_dict()["embeddings.weight"].detach().cpu()
            if not torch.allclose(a, b):
                raise SystemExit(
                    f"feature embedding mismatch for '{cov}': the snapshot and "
                    f"the checkpoint disagree, refusing to sample")
        print("[check] covariate embeddings match the checkpoint copy", flush=True)
    else:
        print("[check] checkpoint carries no hparams copy of feature_embeddings; "
              "using the snapshot written by train_cfgen.py", flush=True)
    return fm, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=["smoke", "full", "e50"], default="full")
    ap.add_argument("--n", type=int, default=0, help="0 = caliber default 20000")
    args = ap.parse_args()

    import sys
    if not (UP / "cfgen").is_dir():
        raise SystemExit(f"missing CFGen upstream at {UP}; see baselines/README.md")
    sys.path.insert(0, str(UP))
    import torch
    import pandas as pd

    n_gen = args.n if args.n else N_GEN
    batch = min(BATCH, n_gen)
    # batched_sample slices its inputs per repetition, so every slice must be a
    # full batch; draw a padded number of cells and cut back afterwards. At the
    # caliber default (20000 / 2000) n_pad == n_gen, so the real run draws
    # exactly as many random numbers as the scANVI arm does.
    reps = -(-n_gen // batch)
    n_pad = reps * batch

    sp = CFGEN / "out" / f"cfgen_sampling_state_{args.tag}.pt"
    if not sp.exists():
        raise SystemExit(f"missing {sp}; run train_cfgen.py --tag {args.tag}")
    state = torch.load(sp, weights_only=False, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fm, cfg = build_model(state, device)
    cov = cfg.dataset.theta_covariate
    id2cov = state["id2cov"][cov]
    order = [t for t, _ in sorted(id2cov.items(), key=lambda kv: kv[1])]
    print(f"[model] {args.tag} checkpoint, label order = {order}", flush=True)
    for t in TYPES:
        if t not in id2cov:
            raise SystemExit(f"cell type {t} absent from the trained label set")

    labels = np.load(DATA / "train_celltype.npy", allow_pickle=True).astype(str)
    mix = pd.Series(labels).value_counts(normalize=True).reindex(order).values

    out = CFGEN / "out" / "counts"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"tag": args.tag, "n": n_gen, "sample_seed": SAMPLE_SEED,
            "library_seed": LIB_SEED, "n_sample_steps": N_SAMPLE_STEPS,
            "label_order": order, "fm_ckpt": state["fm_ckpt"],
            "encoder_ckpt": state["encoder_ckpt"], "arms": {}}

    for ctype in TYPES:
        pool = lib_pool(ctype)
        for mode in ("conditional", "unconditional"):
            rng = np.random.default_rng(LIB_SEED)
            lib = rng.choice(pool, size=n_pad, replace=True)
            lrng = np.random.default_rng(SAMPLE_SEED + 7)
            if mode == "conditional":
                yidx = np.full(n_pad, id2cov[ctype], dtype=np.int64)
            else:
                yidx = lrng.choice(len(order), size=n_pad, p=mix)

            torch.manual_seed(SAMPLE_SEED + (0 if mode == "conditional" else 1))
            gen = fm.batched_sample(
                batch_size=batch,
                repetitions=reps,
                n_sample_steps=N_SAMPLE_STEPS,
                theta_covariate=cov,
                size_factor_covariate=cov,
                conditioning_covariates=[cov],
                covariate_indices={cov: torch.tensor(yidx)},
                log_size_factor={"rna": torch.log(
                    torch.tensor(lib, dtype=torch.float32,
                                 device=device)).view(-1, 1)})
            g = gen["rna"].cpu().numpy()[:n_gen]
            if not np.isfinite(g).all():
                raise SystemExit(f"non-finite counts for {ctype}/{mode}")
            arr = g.astype(np.int64)          # int64: the upper tail is unbounded

            tag = "cond" if mode == "conditional" else "uncond"
            f = out / f"cfgen_{tag}_{ctype}_n{n_gen}_seed{SAMPLE_SEED}.npy"
            np.save(f, arr)
            libs = arr.sum(1)
            meta["arms"][f.name] = {
                "mode": mode, "type": ctype, "n": int(n_gen),
                "min": int(arr.min()), "max": int(arr.max()),
                "frac_zero": float((arr == 0).mean()),
                "library_median": float(np.median(libs)),
                "library_median_assigned": float(np.median(lib[:n_gen])),
                "n_empty_cells": int((libs == 0).sum()),
                "any_negative": bool((arr < 0).any())}
            print(f"  [{mode:<13}] {ctype:<12} max {arr.max():>6} "
                  f"zero {(arr == 0).mean():.4f} "
                  f"lib med {np.median(libs):.0f} "
                  f"(assigned {np.median(lib[:n_gen]):.0f}) "
                  f"empty {int((libs == 0).sum())}", flush=True)

    mp = CFGEN / "out" / f"cfgen_generate_meta_{args.tag}.json"
    mp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[out] {mp.name}", flush=True)
    print("CFGEN SAMPLE OK", flush=True)


if __name__ == "__main__":
    main()
