#!/usr/bin/env python
"""R12: class-conditional CountsDiff on 2,000 genes, with the same attention trunk as OURS.

Method (arXiv 2604.03779): forward pure death x_t ~ Binomial(x_0, p(t)), p(t) = cos^2(pi t / 2);
the network predicts y_t = x_0 - x_t with a softplus scalar head per gene; the loss is
w(t) * (yhat - y_t log yhat), w(t) = (pi/2) sin(pi t).

Training recipe: the authors' public scRNA configuration (configs/scrna_heart.yaml and trainer.py): loss averaged over
genes, Adam with weight decay 1e-4 (betas 0.9, 0.999), lr 2e-4 constant after a linear warmup of 1000 steps, gradient
clipping at 1.0, EMA 0.999, p_uncond 0.2; sampling with sigma = eta * sigma_max, eta 0.005. The 61-gene baseline used a summed loss and lr 2e-3
without clipping, which is not carried over: the Blackout arm, with this trunk and this loss form, diverged on HVG2K.

Changes from the 61-gene baseline (EXP22 CountsDiffAttn + vendor_countsdiff), each one required:
  1. a label embedding, added to every token like the time embedding, zeroed with probability
     p_uncond during training, for classifier-free guidance (the paper's section 3.4);
  2. NMAX 128 -> 512 with x_0 clipped to NMAX, as OURS does at 2,000 genes;
  3. bf16 autocast, gradient checkpointing and batch 384, as OURS does at 2,000 genes;
  4. chunked forward passes in validation and sampling, because a full pass does not fit in memory;
  5. a checkpoint that holds the optimiser, schedule, EMA and RNG states, so a stopped run resumes.
  6. a divergence guard: stop when val stays above 1.5 x best for 2,000 steps, the rule of the Blackout runs.

Guidance: log yhat^(g) = g log yhat(c) + (1 - g) log yhat(null). g = 1 is purely conditional.

Usage:
  python countsdiff2k.py train  --data DATA --out OUT --seed S [--steps N] [--batch B] [--no-label]
  python countsdiff2k.py sample --data DATA --ckpt CKPT --type T --n 2500 --steps-k 128 \
         --guidance 1.0 --seed S --out-npy F
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

NMAX = 512
D_MODEL, NHEADS, NLAYERS = 128, 4, 3
LR, WD, WARMUP, EMA, GRAD_CLIP = 2e-4, 1e-4, 1000, 0.999, 1.0
DIV_RATIO, DIV_STEPS = 1.5, 2000
P_UNCOND = 0.2  # configs/scrna_heart.yaml and scrna_fetus.yaml
VAL_CHUNK = 256
VAL_CELLS = 2048  # the value OURS uses
EVAL_EVERY = 200
EPS = 1e-6


def device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def p_schedule(t):
    return torch.cos(math.pi * t / 2.0) ** 2 if isinstance(t, torch.Tensor) else math.cos(math.pi * t / 2.0) ** 2


def w_weight(t: torch.Tensor) -> torch.Tensor:
    return (math.pi / 2.0) * torch.sin(math.pi * t)


def binomial(n: torch.Tensor, p: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """torch.binomial on CPU with an explicit generator: MPS has no binomial sampler."""
    return torch.binomial(n.cpu(), p.cpu(), generator=gen).to(n.device)


class CountsDiffAttnCond(nn.Module):
    """The CountsDiffAttn trunk (identical to OURS) plus an optional label embedding."""

    def __init__(self, dim: int, input_scale: np.ndarray, num_labels: int = 0, nmax: int = NMAX):
        super().__init__()
        self.register_buffer("input_scale", torch.tensor(input_scale, dtype=torch.float32))
        self.register_buffer("gene_ids", torch.arange(dim))
        self.count_proj = nn.Linear(2, D_MODEL)
        self.gene_emb = nn.Embedding(dim, D_MODEL)
        self.time_mlp = nn.Sequential(nn.Linear(1, D_MODEL), nn.SiLU(), nn.Linear(D_MODEL, D_MODEL))
        enc = nn.TransformerEncoderLayer(D_MODEL, NHEADS, dim_feedforward=4 * D_MODEL, batch_first=True,
                                         activation="gelu", dropout=0.0)
        self.transformer = nn.TransformerEncoder(enc, NLAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 1)
        self.label_emb = nn.Embedding(num_labels, D_MODEL) if num_labels else None
        self._lognmax = math.log1p(nmax)
        self.use_ckpt = False

    def forward(self, x_t, t, y=None, drop=None):
        """y: label indices or None. drop: boolean mask of cells that get the null label."""
        if t.dim() == 1:
            t = t.unsqueeze(1)
        sc = (x_t * self.input_scale).unsqueeze(-1)
        lg = (torch.log1p(x_t) / self._lognmax).unsqueeze(-1)
        tok = self.count_proj(torch.cat([sc, lg], -1)) + self.gene_emb(self.gene_ids)[None]
        tok = tok + self.time_mlp(t)[:, None, :]
        if self.label_emb is not None and y is not None:
            e = self.label_emb(y)
            if drop is not None:
                e = e * (~drop).float().unsqueeze(1)
            tok = tok + e[:, None, :]
        h = tok
        for layer in self.transformer.layers:
            h = cp.checkpoint(layer, h, use_reentrant=False) if (self.use_ckpt and self.training) else layer(h)
        if self.transformer.norm is not None:
            h = self.transformer.norm(h)
        return F.softplus(self.head(self.norm(h)).squeeze(-1))


def warmup_sched(opt, warmup: int):
    """Linear warmup, then the base lr, as trainer.py does."""
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / max(warmup, 1), 1.0))


def load_split(d: Path, split: str, order: list[str] | None, nmax: int):
    X = np.load(d / f"{split}.npy", mmap_mode="r")
    y = np.load(d / f"{split}_celltype.npy", allow_pickle=True).astype(str)
    idx = None if order is None else np.array([order.index(v) for v in y], dtype=np.int64)
    return np.clip(np.asarray(X, dtype=np.float32), 0, nmax), idx


def loss_on(model, x0, y, gen, dev, p_uncond: float):
    t = torch.rand(x0.shape[0], generator=gen)
    x_t = binomial(x0, p_schedule(t).unsqueeze(1).expand_as(x0).to(x0.device), gen)
    target = x0 - x_t
    t = t.to(dev)
    drop = None
    if y is not None and p_uncond > 0:
        drop = (torch.rand(x0.shape[0], generator=gen) < p_uncond).to(dev)
    amp = dev.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
        yhat = model(x_t, t, y, drop)
    yhat = yhat.float()
    per = yhat - target * torch.log(yhat.clamp_min(EPS))
    return (w_weight(t).unsqueeze(1) * per).mean(1).mean()


@torch.no_grad()
def val_loss(model, Xv, yv, dev, n_cells: int = VAL_CELLS) -> float:
    gen = torch.Generator(device="cpu").manual_seed(12345)
    idx = torch.randint(0, Xv.shape[0], (min(n_cells, Xv.shape[0]),), generator=gen).numpy()
    total, n = 0.0, 0
    for i in range(0, len(idx), VAL_CHUNK):
        j = idx[i:i + VAL_CHUNK]
        x0 = torch.from_numpy(Xv[j]).to(dev)
        y = None if yv is None else torch.from_numpy(yv[j]).to(dev)
        total += float(loss_on(model, x0, y, gen, dev, 0.0)) * len(j)
        n += len(j)
    return total / n


def cmd_train(a) -> int:
    dev = device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    meta = json.loads((a.data / "meta.json").read_text(encoding="utf-8"))
    order = None if a.no_label else sorted({str(v) for v in np.load(a.data / "train_celltype.npy", allow_pickle=True)})
    Xtr, ytr = load_split(a.data, "train", order, a.nmax)
    Xv, yv = load_split(a.data, "val", order, a.nmax)
    mu = np.load(a.data / "train.npy", mmap_mode="r").mean(0)
    iscale = 1.0 / (1.0 + np.asarray(mu, dtype=np.float64))
    model = CountsDiffAttnCond(Xtr.shape[1], iscale, len(order) if order else 0, a.nmax).to(dev)
    model.use_ckpt = dev.type == "cuda"
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=WD)
    sched = warmup_sched(opt, WARMUP)
    gen = torch.Generator(device="cpu").manual_seed(a.seed)

    a.out.mkdir(parents=True, exist_ok=True)
    last, best = a.out / f"s{a.seed}.last.pt", a.out / f"s{a.seed}.best.pt"
    start, best_val, curve, run0, above_since = 0, float("inf"), {"step": [], "train": [], "val": []}, None, None
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); gen.set_state(ck["gen"]); torch.set_rng_state(ck["rng"])
        start, best_val, curve, run0, above_since = ck["step"], ck["best_val"], ck["curve"], ck["run"], ck["above_since"]
        print(f"resumed at step {start}", flush=True)

    conf = {"seed": a.seed, "steps": a.steps, "batch": a.batch, "device": str(dev), "gene_hash": meta["gene_hash"],
            "labels": order, "p_uncond": P_UNCOND if order else 0.0, "nmax": a.nmax, "lr": a.lr, "grad_clip": GRAD_CLIP, "warmup": WARMUP, "optimizer": "Adam", "lr_after_warmup": "constant", "loss": "mean over genes",
            "n_params": sum(p.numel() for p in model.parameters()), "iscale": iscale.tolist()}
    print({k: v for k, v in conf.items() if k != "iscale"}, flush=True)
    t0, run = time.time(), run0
    for st in range(start, a.steps):
        model.train()
        i = torch.randint(0, Xtr.shape[0], (a.batch,), generator=gen).numpy()
        x0 = torch.from_numpy(Xtr[i]).to(dev)
        y = None if ytr is None else torch.from_numpy(ytr[i]).to(dev)
        loss = loss_on(model, x0, y, gen, dev, conf["p_uncond"])
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {st}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(EMA).add_(pm, alpha=1 - EMA)
        lv = float(loss.detach())
        run = lv if run is None else 0.98 * run + 0.02 * lv
        if (st + 1) % a.eval_every == 0 or st + 1 == a.steps:
            ema.eval()
            vl = val_loss(ema, Xv, yv, dev, a.val_cells)
            curve["step"].append(st + 1); curve["train"].append(run); curve["val"].append(vl)
            print(f"step {st+1}/{a.steps} train {run:.4f} val {vl:.4f} "
                  f"{(time.time()-t0)/max(st+1-start,1):.3f} s/step", flush=True)
            if vl < best_val:
                best_val, above_since = vl, None
                torch.save({"ema": ema.state_dict(), "step": st + 1, "val": vl, "conf": conf}, best)
            elif vl <= DIV_RATIO * best_val:
                above_since = None
            elif above_since is None:
                above_since = st + 1
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "gen": gen.get_state(), "rng": torch.get_rng_state(),
                        "step": st + 1, "best_val": best_val, "curve": curve, "conf": conf, "run": run, "above_since": above_since}, last)
            if above_since is not None and st + 1 - above_since >= DIV_STEPS:
                (a.out / f"s{a.seed}.json").write_text(json.dumps(
                    {**{k: v for k, v in conf.items() if k != "iscale"}, "stopped_reason": "divergence_guard", "stopped_at": st + 1,
                     "above_since": above_since, "best_val": best_val, "curve": curve}, indent=1), encoding="utf-8")
                print(f"DIVERGED: val above {DIV_RATIO} x best since step {above_since}; stopped at {st+1}. Report this seed.", flush=True)
                return 3
            if a.stop_after and st + 1 >= a.stop_after:
                print(f"stopped at step {st+1} by --stop-after", flush=True)
                return 0
    (a.out / f"s{a.seed}.json").write_text(json.dumps(
        {**{k: v for k, v in conf.items() if k != "iscale"}, "stopped_reason": "finished", "best_val": best_val, "curve": curve,
         "walltime_s_this_run": time.time() - t0, "finished_steps": a.steps}, indent=1), encoding="utf-8")
    print("finished", flush=True)
    return 0


@torch.no_grad()
def cmd_sample(a) -> int:
    dev = device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    conf = ck["conf"]
    order = conf["labels"]
    model = CountsDiffAttnCond(len(conf["iscale"]), np.array(conf["iscale"]), len(order) if order else 0, conf["nmax"])
    model.load_state_dict(ck["ema"])
    model.to(dev).eval()
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    torch.manual_seed(a.seed)
    if a.type is None:
        label = None
    else:
        if not order:
            raise SystemExit("--type needs a checkpoint trained with labels")
        label = order.index(a.type)
    edges = torch.linspace(1.0, 0.0, a.steps_k + 1)
    out, t0 = [], time.time()
    for start in range(0, a.n, a.batch):
        b = min(a.batch, a.n - start)
        x = torch.zeros(b, model.gene_ids.shape[0], device=dev)
        y = None if label is None else torch.full((b,), label, dtype=torch.long, device=dev)
        for k in range(a.steps_k):
            t, s = float(edges[k]), float(edges[k + 1])
            p_t, p_s = float(p_schedule(t)), float(p_schedule(s))
            sigma_max = 1.0 if p_t <= 0 else min(1.0, (1.0 - p_s) / p_t)
            sigma = a.eta * sigma_max
            beta = 0.0 if (1.0 - p_t) <= 0 else (p_s - (1.0 - sigma) * p_t) / (1.0 - p_t)
            beta = min(max(beta, 0.0), 1.0)
            tv = torch.full((b,), t, device=dev)
            amp = dev.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                if y is None or a.guidance == 1.0:
                    yhat = model(x, tv, y).float()
                else:
                    cond = model(x, tv, y).float()
                    unc = model(x, tv, y, torch.ones(b, dtype=torch.bool, device=dev)).float()
                    yhat = torch.exp(a.guidance * torch.log(cond.clamp_min(EPS))
                                     + (1 - a.guidance) * torch.log(unc.clamp_min(EPS)))
            floor = torch.floor(yhat)
            yc = (floor + torch.bernoulli((yhat - floor).cpu(), generator=gen).to(dev)).clamp_min(0.0)
            nb = binomial(yc, torch.full_like(yc, beta), gen)
            keep = binomial(x, torch.full_like(x, 1.0 - sigma), gen)
            x = nb + keep
        out.append(x.cpu().numpy().astype(np.int32))
        print(f"sampled {start+b}/{a.n} {(time.time()-t0):.0f}s", flush=True)
    arr = np.concatenate(out, axis=0)
    a.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out_npy, arr)
    rec = {"ckpt": str(a.ckpt), "ckpt_step": ck.get("step"), "type": a.type, "n": a.n, "K": a.steps_k,
           "nfe": a.steps_k * (1 if (label is None or a.guidance == 1.0) else 2), "guidance": a.guidance,
           "eta": a.eta, "seed": a.seed, "walltime_s": time.time() - t0, "device": str(dev),
           "zero_frac": float((arr == 0).mean()), "lib_median": float(np.median(arr.sum(1))), "max": int(arr.max())}
    a.out_npy.with_suffix(".json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(rec, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--data", type=Path, required=True)
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--seed", type=int, required=True)
    tr.add_argument("--steps", type=int, default=20000)
    tr.add_argument("--batch", type=int, default=384)
    tr.add_argument("--nmax", type=int, default=NMAX, help="count cap; set it per dataset by the HVG2K rule")
    tr.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    tr.add_argument("--no-label", action="store_true", help="unconditional model")
    tr.add_argument("--val-cells", type=int, default=VAL_CELLS)
    tr.add_argument("--lr", type=float, default=LR)
    tr.add_argument("--stop-after", type=int, help="stop after this step, at an evaluation point")
    tr.add_argument("--device", help="cuda, mps or cpu; default picks the best available")
    tr.set_defaults(fn=cmd_train)
    sa = sub.add_parser("sample")
    sa.add_argument("--data", type=Path)
    sa.add_argument("--ckpt", type=Path, required=True)
    sa.add_argument("--type", default=None, help="cell type; leave out for the unconditional mix")
    sa.add_argument("--n", type=int, default=2500)
    sa.add_argument("--steps-k", type=int, default=128)
    sa.add_argument("--guidance", type=float, default=1.0)
    sa.add_argument("--eta", type=float, default=0.005, help="sigma = eta * sigma_max, the authors' rescaled schedule")
    sa.add_argument("--seed", type=int, required=True)
    sa.add_argument("--batch", type=int, default=256)
    sa.add_argument("--out-npy", type=Path, required=True)
    sa.add_argument("--device", help="cuda, mps or cpu; default picks the best available")
    sa.set_defaults(fn=cmd_sample)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
