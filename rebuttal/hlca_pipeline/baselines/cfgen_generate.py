#!/usr/bin/env python
"""R16: generate CFGen HLCA cells from the public checkpoints, n per target class (and an unconditional pool). UNTESTED: written
from CFGen's notebooks/generate_with_cfgen/hlca.ipynb and cfgen/models/fm/fm.py (github.com/theislab/CFGen, branch
add_evaluation_example; read as web pages, never run). Run on the GPU server from notebooks/generate_with_cfgen/ of a CFGen checkout
(relative config and checkpoint paths as in the notebook) or pass the paths. Try --n 200 first.

What the notebook does, and what this script changes:
  - model: EncoderModel + FM built exactly as the notebook (encoder_large.yaml, configs_sccfm hlca_core.yaml and generative_model
    default.yaml; AE ckpt train_autoencoder_hlca_core_final/checkpoints/last.ckpt; FM ckpt train_fm_hlca_core_final/last.ckpt;
    guidance_weights {cell_type: 1}; batched_sample(batch_size=100, repetitions=…, n_sample_steps=2, theta/size_factor/conditioning
    covariate cell_type)). The class index mapping is rebuilt as the notebook does: RNAseqLoader on the TRAIN file, id2cov = sorted
    unique train labels. The loader also supplies the per-class size-factor statistics the FM needs.
  - labels: the notebook samples 1,000 TEST cells per batch and uses their labels and library sizes. Here the labels are fixed:
    n of each target class (cond) or M drawn from train-minus-val (uncond pool, labels follow train proportions).
  - library sizes: --size-factor empirical (default) = log library sizes of n randomly chosen TRAIN-minus-val cells of the same
    class (never test, never val; the notebook used test cells); --size-factor model = log_size_factor None, so the FM draws from
    its per-class normal fitted on the train file (fm.py fallback).
  - output: rna counts on all 27,997 genes; saved cut to the 2,000 HVGs as the npz table2_metrics.py reads (counts, labels, genes),
    plus full-gene library_size per cell.
  - seeds: torch / numpy / random seeded with --seed (one sampling seed per run).
    python cfgen_generate.py --train hlca_core_train.h5ad --val-ids val_ids.txt --mode cond --seed 20260987 --out cfgen_cond_s1.npz
    python cfgen_generate.py --train hlca_core_train.h5ad --val-ids val_ids.txt --mode uncond --uncond-n 25000 --seed 20260987 --out cfgen_uncond_s1.npz

v2:
  - data-arg fix (all devices): RNAseqLoader's first parameter is `data`, not `data_path`; the train file is passed positionally,
    as the notebook does. v1 would stop here with a TypeError.
  - env var R16_CFGEN_DEVICE (default "cuda"). The two CPU-only fixes from Windows' copy, map_location="cpu" in both torch.load
    calls and resetting the pickled featurizers' .device, are applied ONLY when the device is not "cuda". With the default, every
    line after the loader call does exactly what v1 did, so the GPU behaviour is v1's apart from the data-arg fix.
    (Windows' copy applied both unconditionally; whether map_location="cpu" changes anything on GPU depends on whether the
    pickled featurizers hold tensors outside a registered module, which was not checked, so it is kept off the GPU path.)
  - the npz also records device (one more key; table2_metrics.py ignores it).
  - memory: RNAseqLoader reads the whole train h5ad (22 GB file, about 86 GB host RAM peak on Windows): one CFGen job at a time.
"""
import argparse, hashlib, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r16_common import TARGETS, read_obs_var, read_rows, train_pool, pick_rows, pick_uncond
DEV = os.environ.get("R16_CFGEN_DEVICE", "cuda")   # v2; default cuda as v1


def label_plan(a):
    """Labels and log library sizes to generate (no cfgen needed; unit-testable)."""
    labels, obs_names, var_names, hv = read_obs_var(a.train)
    pool = train_pool(labels, obs_names, a.val_ids)
    if a.mode == "cond":
        rows, info = pick_rows(labels, pool, a.classes, a.n, a.seed)
    else:
        rows, info = pick_uncond(labels, pool, a.uncond_n, a.seed), {"uncond_n": a.uncond_n}
    lab = labels[rows]
    uniq = np.unique(rows); lib_u = np.asarray(read_rows(a.train, uniq).sum(1)).ravel()
    lib = lib_u[np.searchsorted(uniq, rows)]
    assert (lib > 0).all(), "zero-library train cell picked"
    return lab, np.log(lib.astype(np.float64)), var_names, hv, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True); ap.add_argument("--val-ids", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("cond", "uncond"), required=True); ap.add_argument("--classes", nargs="*", default=TARGETS)
    ap.add_argument("--n", type=int, default=5000); ap.add_argument("--uncond-n", type=int, default=25000); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--size-factor", choices=("empirical", "model"), default="empirical"); ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--n-sample-steps", type=int, default=2)
    ap.add_argument("--dataset-conf", default="../../configs/configs_sccfm/dataset/hlca_core.yaml")
    ap.add_argument("--encoder-conf", default="../../configs/configs_encoder/encoder/encoder_large.yaml")
    ap.add_argument("--gen-conf", default="../../configs/configs_sccfm/generative_model/default.yaml")
    ap.add_argument("--ae-ckpt", default="../../project_folder/experiments/autoencoder_ckpt/train_autoencoder_hlca_core_final/checkpoints/last.ckpt")
    ap.add_argument("--fm-ckpt", default="../../project_folder/experiments/cfgen_ckpt/train_fm_hlca_core_final/last.ckpt")
    ap.add_argument("--plan-only", action="store_true", help="build labels and library sizes, print them, stop (no cfgen import)")
    a = ap.parse_args()
    lab, loglib, var_names, hv, info = label_plan(a)
    N = len(lab); pad = (-N) % a.batch_size
    print("plan:", info, "cells", N, "(padded by", pad, "to a multiple of the batch size; padding rows are dropped)")
    if a.plan_only:
        return
    import torch
    from omegaconf import OmegaConf
    from cfgen.data.scrnaseq_loader import RNAseqLoader
    from cfgen.models.base.encoder_model import EncoderModel
    from cfgen.models.fm.fm import FM
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dc = OmegaConf.load(a.dataset_conf); ec = OmegaConf.load(a.encoder_conf); gc = OmegaConf.load(a.gen_conf)
    dataset = RNAseqLoader(a.train, layer_key=dc.layer_key, covariate_keys=dc.covariate_keys, subsample_frac=dc.subsample_frac,
                           normalization_type=dc.normalization_type, is_binarized=dc.is_binarized)
    id2 = dataset.id2cov["cell_type"]; miss = [c for c in set(lab) if c not in id2]; assert not miss, miss
    in_dim = dataset.X["rna"].shape[1]; assert in_dim == len(var_names), (in_dim, len(var_names))
    sfs = {"mean": dataset.log_size_factor_mu, "sd": dataset.log_size_factor_sd}
    enc = EncoderModel(in_dim={"rna": in_dim}, n_cat=None, conditioning_covariate="cell_type", **ec)
    mloc = {} if DEV == "cuda" else {"map_location": "cpu"}   # v2: GPU path loads exactly as v1
    enc.load_state_dict(torch.load(a.ae_ckpt, weights_only=False, **mloc)["state_dict"])
    ck = torch.load(a.fm_ckpt, weights_only=False, **mloc)
    if DEV != "cuda":   # v2, from Windows' copy: the featurizers' device was pickled as cuda
        for _f in ck["hyper_parameters"]["feature_embeddings"].values():
            if hasattr(_f, "device"): _f.device = DEV
    den =ck["hyper_parameters"]["denoising_model"]; den.multimodal = False
    fm = FM(encoder_model=enc, denoising_model=den, feature_embeddings=ck["hyper_parameters"]["feature_embeddings"], plotting_folder=None,
            in_dim={"rna": 100}, size_factor_statistics=sfs, covariate_list=["cell_type"], theta_covariate="cell_type",
            size_factor_covariate="cell_type", is_binarized=False, modality_list=["rna"], guidance_weights={"cell_type": 1}, **gc)
    fm.load_state_dict(ck["state_dict"]); fm = fm.to(DEV).eval()
    idx = np.array([id2[c] for c in lab] + [id2[lab[0]]] * pad)
    cov = {"cell_type": torch.tensor(idx, dtype=torch.long)}
    lsf = None if a.size_factor == "model" else {"rna": torch.tensor(np.r_[loglib, [loglib[0]] * pad], dtype=torch.float32).unsqueeze(1).to(DEV)}
    with torch.no_grad():
        out = fm.batched_sample(batch_size=a.batch_size, repetitions=(N + pad) // a.batch_size, n_sample_steps=a.n_sample_steps,
                                theta_covariate="cell_type", size_factor_covariate="cell_type", conditioning_covariates=["cell_type"],
                                covariate_indices=cov, log_size_factor=lsf, unconditional=(a.mode == "uncond"))
    X = out["rna"].cpu().numpy()[:N]
    assert X.shape == (N, len(var_names)), X.shape
    np.savez(a.out, counts=np.rint(X[:, hv]).astype(np.int32), labels=lab.astype(str), genes=var_names[hv].astype(str),
             library_size=X.sum(1), seed=a.seed, mode=a.mode, size_factor=a.size_factor, device=DEV,
             fm_ckpt_sha256=hashlib.sha256(open(a.fm_ckpt, "rb").read()).hexdigest(),
             ae_ckpt_sha256=hashlib.sha256(open(a.ae_ckpt, "rb").read()).hexdigest(),
             script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest())
    print("wrote", a.out, X.shape, "->", int(hv.sum()), "HVG columns")


if __name__ == "__main__":
    main()
