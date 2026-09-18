# -*- coding: utf-8 -*-
"""Draw the unconditional pool of each generator, at the pool size and seeds of this round.

The generation path follows each baseline's own archived sampling script word for word, with
three changes, each of them required:

1. the data root moves from the retired tree to assets/legacy_root (a path fix only);
2. the pool size and the seeds are this round's;
3. the LIBRARY SIZE SOURCE. The archived unconditional rows drew library sizes from the real
   validation cells of the target type, which is a conditional quantity; an unconditional pool
   must not receive it, so the source is stated explicitly in the output json.

    python scripts/sample_unconditional_pool.py --help
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.io import sha256_file                                       # noqa: E402

ARCHIVE = C.ARCHIVE
DATA = C.LEGACY_DATA / "all"
BATCH = 4096
N_GEN = 100000


def train_library_pools(order: list[str]) -> dict:
    """The library-size pool of each type, taken from the real TRAINING cells."""
    x = np.load(DATA / "train.npy", mmap_mode="r")
    y = np.load(DATA / "train_celltype.npy", allow_pickle=True).astype(str)
    out = {}
    for t in order:
        idx = np.sort(np.flatnonzero(y == t))
        out[t] = np.asarray(x[idx], dtype=np.float64).sum(1)
    return out


def draw_labels_and_library(order, mix, n, seed):
    """Labels are drawn from the training mixture; the library size is then drawn from the real training cells OF THE DRAWN LABEL."""
    lrng = np.random.default_rng(seed + 7)
    yidx = lrng.choice(len(order), size=n, p=mix)
    pools = train_library_pools(order)
    rng = np.random.default_rng(seed)
    lib = np.empty(n, dtype=np.float64)
    for k, t in enumerate(order):
        m = (yidx == k)
        if m.any():
            lib[m] = rng.choice(pools[t], size=int(m.sum()), replace=True)
    return yidx, lib


def genes_and_labels():
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    g = eval(meta["genes"]) if isinstance(meta["genes"], str) else meta["genes"]
    labels = np.load(DATA / "train_celltype.npy", allow_pickle=True).astype(str)
    return meta, g, labels


def run_scvi(seed: int, n: int, out_npy: Path) -> dict:
    import anndata as ad
    import pandas as pd
    import scvi
    import torch
    from scipy import sparse

    HERE = ARCHIVE / "baseline_scvi"
    rec = json.loads((HERE / "logs" / "train_scvi.json").read_text(encoding="utf-8"))
    order = rec["batch_order"]
    meta, genes, labels = genes_and_labels()
    counts = np.load(DATA / "train.npy", mmap_mode="r")
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.asarray(counts[:64], dtype=np.float32)),
        obs=pd.DataFrame({"cell_type": pd.Categorical(labels[:64], categories=order)},
                         index=[f"c{i}" for i in range(64)]),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    scvi.model.SCVI.setup_anndata(adata, batch_key="cell_type")
    model = scvi.model.SCVI.load(str(HERE / "out" / "model"), adata=adata)
    mod = model.module.eval()
    dev = next(mod.parameters()).device
    mix = pd.Series(labels).value_counts(normalize=True).reindex(order).values
    yidx, lib = draw_labels_and_library(order, mix, n, seed)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    arr = np.empty((n, len(genes)), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, n, BATCH):
            j = min(i + BATCH, n)
            z = torch.randn(j - i, mod.n_latent, generator=g).to(dev)
            lb = torch.tensor(np.log(lib[i:j]), dtype=torch.float32, device=dev).unsqueeze(1)
            bi = torch.tensor(yidx[i:j], device=dev).unsqueeze(1)
            arr[i:j] = mod.generative(z=z, library=lb,
                                      batch_index=bi)["px"].sample().cpu().numpy()
    np.save(out_npy, arr)
    return {"order": order, "scvi_version": rec.get("scvi_version"),
            "model_dir": str(HERE / "out" / "model"), "n_latent": int(mod.n_latent),
            "device": str(dev), "label_counts": np.bincount(yidx, minlength=len(order)).tolist()}


def run_scanvi(seed: int, n: int, out_npy: Path) -> dict:
    import anndata as ad
    import pandas as pd
    import scvi
    import torch
    from scipy import sparse

    HERE = ARCHIVE / "baseline_scanvi"
    rec = json.loads((HERE / "logs" / "train_scanvi.json").read_text(encoding="utf-8"))
    order = rec["label_order"]
    meta, genes, labels = genes_and_labels()
    counts = np.load(DATA / "train.npy", mmap_mode="r")
    SHELL_PER_TYPE = 6
    keep = np.sort(np.concatenate(
        [np.flatnonzero(labels == t)[:SHELL_PER_TYPE] for t in order]))
    shell_lab = pd.Categorical(labels[keep])
    if list(shell_lab.categories) != list(order):
        raise SystemExit(f"shell label set drift: {list(shell_lab.categories)} vs {order}")
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.asarray(counts[keep], dtype=np.float32)),
        obs=pd.DataFrame({"cell_type": shell_lab},
                         index=[f"c{i}" for i in range(len(keep))]),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    model = scvi.model.SCANVI.load(str(HERE / "out" / "model"), adata=adata,
                                   accelerator="cpu")
    mod = model.module.eval()
    dev = next(mod.parameters()).device
    if int(mod.y_prior.shape[1]) != len(order) or int(mod.n_labels) != len(order):
        raise SystemExit(f"label-space mismatch: n_labels={mod.n_labels} vs {len(order)}")
    mix = pd.Series(labels).value_counts(normalize=True).reindex(order).values
    yidx, lib = draw_labels_and_library(order, mix, n, seed)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    arr = np.empty((n, len(genes)), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, n, BATCH):
            j = min(i + BATCH, n)
            yy = torch.tensor(yidx[i:j], device=dev).unsqueeze(1)
            z2 = torch.randn(j - i, mod.n_latent, generator=g).to(dev)
            pz1_m, pz1_v = mod.decoder_z1_z2(z2, yy)
            eps = torch.randn(pz1_m.shape, generator=g).to(dev)
            z1 = pz1_m + eps * torch.sqrt(pz1_v)
            lb = torch.tensor(np.log(lib[i:j]), dtype=torch.float32, device=dev).unsqueeze(1)
            bi = torch.zeros(j - i, 1, dtype=torch.long, device=dev)
            arr[i:j] = mod.generative(z=z1, library=lb, batch_index=bi,
                                      y=yy)["px"].sample().cpu().numpy()
    np.save(out_npy, arr)
    return {"order": order, "model_dir": str(HERE / "out" / "model"),
            "n_latent": int(mod.n_latent), "device": str(dev),
            "label_path": "z2 -> decoder_z1_z2(y) -> z1 -> p(x|z1) (the ancestral path in "
                          "SCANVAE.loss; passing y to generative alone does nothing)",
            "label_counts": np.bincount(yidx, minlength=len(order)).tolist()}


def run_cfgen(seed: int, n: int, out_npy: Path, device: str = "cpu") -> dict:
    """CFGen. The generative path is the archived script's own model builder, imported by file

        The generation path uses the archived script's own model builder, imported by file path,
        path rather than reimplemented; only the label and library-size draws follow this
        rather than reimplementing it; only the label and library-size draws follow this round's
        convention. The number of sampling steps is the upstream default, unchanged.
        round's convention, and the number of sampling steps is the upstream default.
    """
    import importlib.util
    import torch
    import pandas as pd

    CFGEN = ARCHIVE / "baseline_cfgen"
    UP = CFGEN / "upstream"
    gen_py = C.BASELINES / "cfgen" / "code" / "generate_cfgen.py"
    for p in (CFGEN, UP, gen_py):
        if not p.exists():
            raise SystemExit(f"missing CFGen asset: {p}")
    sys.path.insert(0, str(UP))
    spec = importlib.util.spec_from_file_location("_archived_generate_cfgen", gen_py)
    G = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = G
    spec.loader.exec_module(G)                       # imported only for its model builder

    sp = CFGEN / "out" / "cfgen_sampling_state_full.pt"
    state = torch.load(C.require(sp, "CFGen sampling state"), weights_only=False,
                       map_location="cpu")
    fm, cfg = G.build_model(state, device)
    cov = cfg.dataset.theta_covariate
    id2cov = state["id2cov"][cov]
    order = [t for t, _ in sorted(id2cov.items(), key=lambda kv: kv[1])]

    meta, genes, labels = genes_and_labels()
    mix = pd.Series(labels).value_counts(normalize=True).reindex(order).values
    yidx, lib = draw_labels_and_library(order, mix, n, seed)

    batch = min(G.BATCH, n)
    reps = -(-n // batch)
    n_pad = reps * batch
    if n_pad != n:                     # the batched sampler slices by repetition, so the batch must be whole
        pad = n_pad - n
        yidx = np.concatenate([yidx, yidx[:pad]])
        lib = np.concatenate([lib, lib[:pad]])

    torch.manual_seed(seed + 1)
    gen = fm.batched_sample(
        batch_size=batch, repetitions=reps, n_sample_steps=G.N_SAMPLE_STEPS,
        theta_covariate=cov, size_factor_covariate=cov, conditioning_covariates=[cov],
        covariate_indices={cov: torch.tensor(yidx)},
        log_size_factor={"rna": torch.log(
            torch.tensor(lib, dtype=torch.float32, device=device)).view(-1, 1)})
    g = gen["rna"].cpu().numpy()[:n]
    if not np.isfinite(g).all():
        raise SystemExit("non-finite counts from CFGen")
    arr = g.astype(np.int64)           # int64: the upper tail is unbounded, as the archived script's comment says
    np.save(out_npy, arr)
    return {"order": order, "n_sample_steps": G.N_SAMPLE_STEPS, "batch": batch,
            "repetitions": reps, "device": device,
            "fm_ckpt": state["fm_ckpt"], "encoder_ckpt": state["encoder_ckpt"],
            "sampling_state": str(sp), "sampling_state_sha256": sha256_file(sp),
            "label_counts": np.bincount(yidx[:n], minlength=len(order)).tolist(),
            "model_rebuild": "verbatim via the archived generate_cfgen.build_model "
                             "(imported by path, not copied)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True, choices=("scvi", "scanvi", "cfgen"))
    ap.add_argument("--device", default="cpu",
                    help="for CFGen: stay on cpu while the generation queue has the GPU")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=N_GEN)
    ap.add_argument("--out-dir", default=str(C.OUT / "uncond_pools"))
    a = ap.parse_args()

    outd = Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    npy = outd / f"{a.generator}_uncond_seed{a.seed}_n{a.n}.npy"
    jsn = npy.with_suffix(".json")
    if npy.exists() or jsn.exists():
        raise SystemExit(f"refusing to overwrite {npy}")

    t0 = time.time()
    if a.generator == "cfgen":
        info = run_cfgen(a.seed, a.n, npy, device=a.device)
    else:
        info = {"scvi": run_scvi, "scanvi": run_scanvi}[a.generator](a.seed, a.n, npy)
    arr = np.load(npy, mmap_mode="r")
    blob = {"ticket": 200, "block": "M1-4", "generator": a.generator,
            "mode": "unconditional", "seed": a.seed, "n": int(a.n),
            "machine": "machine1", "device_class": "A",
            "library_size_source": "real-train-drawn (per drawn label)",
            "library_size_note":
                "the archived unconditional rows drew library sizes from the real validation "
                "cells of the TARGET type, which a mixed pool has no notion of, so the source "
                "here is the real training cells of the drawn label; this row does not rank them.",
            "sampling_code_source": "verbatim generative path of the archived "
                                    f"{a.generator} generate script (only data root, n, seed "
                                    "and the library-size rule above differ)",
            "data_root": str(C.LEGACY_ROOT),
            "data_root_note": "the data root hard-coded in the archived script was retired; "
                              "this is a path fix only and does not affect the generative path",
            "argv": sys.argv, "walltime_s": time.time() - t0,
            "dtype": str(arr.dtype), "shape": list(arr.shape),
            "npy_sha256": sha256_file(npy),
            "count_legality": {"any_negative": bool((np.asarray(arr[:20000]) < 0).any()),
                               "all_integer": True,
                               "note": "integer and non-negative BY CONSTRUCTION of the "
                                       "count decoder; recorded, not credited as a pass"},
            "library_size_mean": float(np.asarray(arr[:20000]).sum(1).mean()),
            "zero_fraction": float((np.asarray(arr[:20000]) == 0).mean()),
            "environment": {"python": sys.version, "numpy": np.__version__,
                            "platform": platform.platform()},
            **info,
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    jsn.write_text(json.dumps(blob, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[{a.generator}] {npy.name} {arr.shape} {(time.time()-t0)/60:.1f} min "
          f"sha={blob['npy_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
