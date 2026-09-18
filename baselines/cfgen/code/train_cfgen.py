# -*- coding: utf-8 -*-
"""CFGen: run both official training stages and snapshot what sampling needs.

Why a wrapper instead of two `python -m cfgen.train_encoder / train_sccfm`
calls:

1. `configs_sccfm/training_config/default.yaml` ships
   `encoder_ckpt: "path/to/encoder_ckpt"` with the comment "modified directly in
   the scripts". It is NOT None, so the flow-matching stage unconditionally does
   `torch.load("path/to/encoder_ckpt")` and dies. The encoder checkpoint lives
   under a freshly minted uuid directory, so only something that ran the first
   stage can name it. This wrapper does.
2. `FM.feature_embeddings` is a plain dict, not a ModuleDict. Its parameters are
   appended to the optimizer by hand in `configure_optimizers`, so they ARE
   trained, but they are NOT in `FM.state_dict()`. Sampling from the checkpoint
   alone would therefore silently use *untrained* covariate embeddings -- the
   conditioning would be quietly destroyed rather than error out. This wrapper
   saves them explicitly next to the checkpoint.
3. Sampling needs data-derived quantities that live on the dataset object
   (`id2cov`, per-type log-size-factor mean/sd, min/max size factor). Snapshotting
   them here means the sampler never has to re-load the 361k x 2000 dense tensor.

What is NOT changed: every hyperparameter, the forward pass, the loss, the
optimizer, the schedule, the architecture. Two platform-level deviations, both
logged into the snapshot and reported in FAIRNESS:

  * `num_workers: 4 -> 0`. Windows has no fork; each worker would re-pickle the
    fully-materialised 2.9 GB dense count tensor (4 x 2.9 GB on a 32 GB box).
    batch_size, shuffle and drop_last are untouched. This is a data-interface
    change, not a training change.
  * an optional wall-clock cap that sets `trainer.should_stop`, which is the
    pre-registered "budget-capped, evaluate the latest checkpoint" rule already
    applied to scDiffusion. Off unless `--max-minutes` is given.

Usage
  python train_cfgen.py --tag smoke --max-minutes 10 --smoke-steps 30
  python train_cfgen.py --tag full  --max-minutes 240
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
HVG = HERE.parents[2]
CFGEN = HERE.parent
UP = Path(os.environ.get("CRNDIFF_CFGEN_UPSTREAM", str(CFGEN / "upstream"))).resolve()

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fresh(cfg):
    """A deep, independent copy.

    `EncoderModel.__init__` mutates `encoder_kwargs['rna']['dims']` in place
    (it prepends in_dim, then reverses it for the decoder). Handing the same
    config object to a second construction inside one process would silently
    build a different network.
    """
    from omegaconf import OmegaConf
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def compose(config_dir: Path, overrides: list[str]):
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base=None,
                                     config_dir=str(config_dir)):
        return hydra.compose(config_name="train", overrides=overrides)


def zero_workers(est) -> None:
    """Rebuild the two dataloaders with num_workers=0, everything else equal."""
    import torch
    for name, ds, shuffle in (("train_dataloader", est.train_data, True),
                              ("valid_dataloader", est.valid_data, False)):
        old = getattr(est, name)
        setattr(est, name, torch.utils.data.DataLoader(
            ds, batch_size=old.batch_size, shuffle=shuffle,
            num_workers=0, drop_last=old.drop_last))
    log("dataloaders rebuilt with num_workers=0 (Windows; batch/shuffle/"
        "drop_last unchanged)")


def budget_callback(minutes: float):
    import pytorch_lightning as pl

    class Budget(pl.Callback):
        def __init__(self):
            self.t0 = time.time()
            self.fired = False

        def on_train_batch_end(self, trainer, *a, **k):
            if not self.fired and (time.time() - self.t0) > minutes * 60:
                self.fired = True
                trainer.should_stop = True
                log(f"wall-clock budget {minutes:.0f} min reached at step "
                    f"{trainer.global_step}; stopping cleanly so `last.ckpt` "
                    f"is written")
    return Budget()


def newest_ckpt(d: Path) -> Path:
    last = d / "last.ckpt"
    if last.exists():
        return last
    c = sorted(d.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not c:
        raise SystemExit(f"no checkpoint written under {d}")
    return c[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=["smoke", "full", "e22_d1", "e50"], required=True)
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="0 = no cap (official schedule runs to completion)")
    ap.add_argument("--smoke-steps", type=int, default=30)
    args = ap.parse_args()

    if not (UP / "cfgen").is_dir():
        raise SystemExit(f"missing CFGen upstream at {UP}; see baselines/README.md")
    sys.path.insert(0, str(UP))
    import torch
    from omegaconf import OmegaConf, open_dict

    ds = {"smoke": "hvg2k_smoke", "full": "hvg2k",
          "e22_d1": "hvg2k_e22_d1",
          "e50": "hvg2k_e50"}[args.tag]
    out = CFGEN / "out"
    out.mkdir(parents=True, exist_ok=True)

    # Only filesystem locations change; both numerical dataset configs stay intact.
    data_dir = Path(os.environ.get("CRNDIFF_CFGEN_DATA_DIR", str(CFGEN / "data"))).resolve()
    os.environ["CRNDIFF_CFGEN_DATA_DIR"] = data_dir.as_posix()
    filename = {"smoke": "hvg2k_smoke_cfgen.h5ad", "full": "hvg2k_train_cfgen.h5ad",
                "e22_d1": "hvg2k_e22_d1_cfgen.h5ad", "e50": "hvg2k_e50_cfgen.h5ad"}[args.tag]
    if not (data_dir / filename).is_file():
        raise SystemExit(f"missing CFGen input: {data_dir / filename}; see baselines/README.md")
    ov = [f"dataset={ds}", "~launcher", "logger.offline=True"]
    if args.tag == "smoke":
        # A few dozen optimiser steps, identical structure. detect_anomaly off
        # only here: it is upstream's default and stays on for the real run.
        ov += [f"+trainer.limit_train_batches={args.smoke_steps}",
               "+trainer.limit_val_batches=3",
               "trainer.max_epochs=1",
               "trainer.detect_anomaly=False",
               "training_config.batch_size=64"]

    t_all = time.time()
    stamp = {"tag": args.tag, "overrides": ov,
             "started": datetime.now(timezone.utc).isoformat(),
             "deviations": [
                 "num_workers 4->0 (Windows, data interface only)",
                 ("wall-clock cap via trainer.should_stop"
                  if args.max_minutes else "no wall-clock cap")]}

    # ---------------------------------------------------------- stage 1
    log(f"=== stage 1/2: encoder  (dataset={ds}) ===")
    cfg_e = compose(UP / "configs" / "configs_encoder", ov)
    log(OmegaConf.to_yaml(cfg_e.dataset).strip().replace("\n", " | "))
    from cfgen.estimator.encoder_estimator import EncoderEstimator
    est_e = EncoderEstimator(fresh(cfg_e))
    zero_workers(est_e)
    if args.max_minutes:
        est_e.trainer_generative.callbacks.append(
            budget_callback(args.max_minutes * 0.4))
    t0 = time.time()
    est_e.train()
    enc_ckpt = newest_ckpt(est_e.training_dir / "checkpoints")
    stamp["encoder"] = {"training_dir": str(est_e.training_dir),
                        "ckpt": str(enc_ckpt),
                        "seconds": time.time() - t0,
                        "n_cells": len(est_e.dataset),
                        "global_step": int(est_e.trainer_generative.global_step)}
    log(f"encoder done in {stamp['encoder']['seconds']/60:.1f} min, "
        f"{stamp['encoder']['global_step']} steps -> {enc_ckpt.name}")
    del est_e
    torch.cuda.empty_cache()

    # ---------------------------------------------------------- stage 2
    log("=== stage 2/2: flow matching ===")
    cfg_f = compose(UP / "configs" / "configs_sccfm", ov)
    with open_dict(cfg_f):
        cfg_f.training_config.encoder_ckpt = str(enc_ckpt)
    from cfgen.estimator.cfgen_estimator import CfgenEstimator
    est_f = CfgenEstimator(fresh(cfg_f))
    zero_workers(est_f)
    if args.max_minutes:
        est_f.trainer_generative.callbacks.append(
            budget_callback(args.max_minutes * 0.6))
    t0 = time.time()
    est_f.train()
    fm_ckpt = newest_ckpt(est_f.training_dir / "checkpoints")
    stamp["fm"] = {"training_dir": str(est_f.training_dir),
                   "ckpt": str(fm_ckpt),
                   "seconds": time.time() - t0,
                   "global_step": int(est_f.trainer_generative.global_step)}
    log(f"flow matching done in {stamp['fm']['seconds']/60:.1f} min, "
        f"{stamp['fm']['global_step']} steps -> {fm_ckpt.name}")

    # ------------------------------------------- snapshot for the sampler
    d = est_f.dataset
    state = {
        "cfg_yaml": OmegaConf.to_yaml(cfg_f, resolve=True),
        "id2cov": {c: {str(k): int(v) for k, v in m.items()}
                   for c, m in d.id2cov.items()},
        "size_factor_mu": d.log_size_factor_mu,
        "size_factor_sd": d.log_size_factor_sd,
        "min_size_factor": d.min_size_factor,
        "max_size_factor": d.max_size_factor,
        "gene_dim": est_f.gene_dim,
        "in_dim": est_f.in_dim,
        "modality_list": est_f.modality_list,
        "num_classes": est_f.num_classes,
        # the piece that FM.state_dict() does NOT carry:
        "feature_embeddings": {c: fe.state_dict()
                               for c, fe in est_f.feature_embeddings.items()},
        "feature_embeddings_ncat": {c: int(fe.n_cat)
                                    for c, fe in est_f.feature_embeddings.items()},
        "encoder_ckpt": str(enc_ckpt),
        "fm_ckpt": str(fm_ckpt),
        "tag": args.tag,
    }
    sp = out / f"cfgen_sampling_state_{args.tag}.pt"
    torch.save(state, sp)
    log(f"[out] {sp.name}  (covariate embeddings included: "
        f"{sorted(state['feature_embeddings'])})")

    stamp["sampling_state"] = str(sp)
    stamp["seconds_total"] = time.time() - t_all
    stamp["finished"] = datetime.now(timezone.utc).isoformat()
    jp = out / f"cfgen_train_{args.tag}.json"
    jp.write_text(json.dumps(stamp, indent=1), encoding="utf-8")
    log(f"[out] {jp.name}")
    log(f"CFGEN TRAIN OK ({args.tag}) in {stamp['seconds_total']/60:.1f} min")


if __name__ == "__main__":
    main()
