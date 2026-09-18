# -*- coding: utf-8 -*-
"""Conditional training: the label-embedding variant of the generator. No frozen file is touched.

The requirement is to import the frozen `model_hvg2k` (through `C.add_base_code_to_path()`) and
add an 11-class label embedding on top of its time embedding, with the label channel initialised

There is a single injection point, inside `panel61 backbone.AttnX0Posterior.forward`:

    tok = self.count_proj(torch.cat([sc, lg], -1))
    tok = tok + self.gene_emb(self.gene_ids)[None]
    tok = tok + self.time_mlp(t)[:, None, :]        # the label embedding is added here

so adding a label channel really is one local change. Zero initialisation gives
`label_emb(y) == 0`, so `tok + 0` is bitwise equal to the frozen forward pass, which is what

The forward (noising) process does not see the condition; only the reverse denoiser does, so the

Three objects:
    `ConditionalAttnX0Posterior`  the conditional network; its parent is the frozen backbone class
    `FixedLabel`                  wraps (n_t, t, y=const) into the `.post(n_t, t)` the sampler wants
    `train_conditional`           the training loop, matched line by line to the frozen trainer
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
try:
    import torch
except ModuleNotFoundError as _exc:                                      # noqa: E402
    raise SystemExit(
        "this entry point needs PyTorch, which the offline reproduction does not "
        "install. See README section 7; in short: pip install torch") from _exc
import torch.nn.functional as F
from torch import nn

from . import config as C

C.add_base_code_to_path()
import config_hvg2k as CFG                                           # noqa: E402
import model_hvg2k as M                                              # noqa: E402

BB = M.BB
DEV = M.DEV

# The label order is np.unique of the frozen label file, read out rather than transcribed
def label_vocabulary() -> list[str]:
    y = np.load(C.LEGACY_DATA / "all" / "train_celltype.npy", allow_pickle=True).astype(str)
    return sorted(set(y.tolist()))


class ConditionalAttnX0Posterior(BB.AttnX0Posterior):
    """A subclass of the frozen backbone plus a zero-initialised label embedding.

        `forward(n_t, t, y)` is bitwise equal to the parent's `forward(n_t, t)` when y is None or its
        embedding is zero. During training the three transformer layers are gradient-checkpointed
        layer by layer, exactly as the frozen checkpointed variant does.
    """

    def __init__(self, nmax, dim, n_labels, d_model=128, nheads=4, nlayers=3,
                 input_scale=None):
        super().__init__(nmax, dim, d_model=d_model, nheads=nheads, nlayers=nlayers,
                         input_scale=input_scale)
        self.n_labels = int(n_labels)
        self.label_emb = nn.Embedding(self.n_labels, d_model)
        nn.init.zeros_(self.label_emb.weight)          # the precondition of gate G-A1

    def _tokens(self, n_t, t, y):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        sc = (n_t * self.input_scale).unsqueeze(-1)
        lg = (torch.log1p(n_t) / self._lognmax).unsqueeze(-1)
        tok = self.count_proj(torch.cat([sc, lg], -1))
        tok = tok + self.gene_emb(self.gene_ids)[None]
        tok = tok + self.time_mlp(t)[:, None, :]
        if y is not None:
            tok = tok + self.label_emb(y)[:, None, :]
        return tok

    def forward(self, n_t, t, y=None):                              # type: ignore[override]
        h = self._tokens(n_t, t, y)
        if self.training and CFG.USE_GRAD_CKPT:
            import torch.utils.checkpoint as _cp
            for layer in self.transformer.layers:
                h = _cp.checkpoint(layer, h, use_reentrant=False)
            if self.transformer.norm is not None:
                h = self.transformer.norm(h)
            return self.head(self.norm(h))
        return self.head(self.norm(self.transformer(h)))

    @torch.no_grad()
    def post(self, n_t, t, y=None):                                 # type: ignore[override]
        return F.softmax(self.forward(n_t, t, y), dim=2)


class FixedLabel:
    """The frozen sampler only ever calls `net.post(n_t, t)`; this thin shell pins the label and

        The sampler's numerical path is untouched, which is what makes gate G-A2 meaningful.
    """

    def __init__(self, net: ConditionalAttnX0Posterior, label_index: int | None):
        self.net = net
        self.label_index = label_index

    @torch.no_grad()
    def post(self, n_t, t):
        y = (None if self.label_index is None else
             torch.full((n_t.shape[0],), int(self.label_index),
                        dtype=torch.long, device=n_t.device))
        return self.net.post(n_t, t, y)


def build_conditional(dim: int, iscale, n_labels: int) -> ConditionalAttnX0Posterior:
    return ConditionalAttnX0Posterior(CFG.NMAX, dim, n_labels, d_model=CFG.D_MODEL,
                                      nheads=CFG.NHEADS, nlayers=CFG.NLAYERS,
                                      input_scale=iscale)


def load_frozen_weights(net: ConditionalAttnX0Posterior, ckpt: Path) -> dict:
    """Load the frozen checkpoint into the conditional network; `label_emb` is not in the checkpoint"""
    sd = torch.load(str(ckpt), map_location=DEV, weights_only=False)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [k for k in missing]
    if unexpected:
        raise SystemExit(f"unexpected keys in frozen ckpt: {unexpected}")
    if missing != ["label_emb.weight"]:
        raise SystemExit(f"expected only label_emb.weight to be missing, got {missing}")
    if float(net.label_emb.weight.abs().max()) != 0.0:
        raise SystemExit("label_emb is not zero after loading -- G-A1 precondition broken")
    return {"missing": missing, "unexpected": list(unexpected)}


@torch.no_grad()
def gate_g_a1(dim: int, iscale, ckpt: Path, n_labels: int, n: int = 8, seed: int = 20260817):
    """Gate G-A1 (instant): with the label channel zeroed, the conditional forward pass is bitwise"""
    frozen = M.build_net(dim, iscale).to(DEV)
    frozen.load_state_dict(torch.load(str(ckpt), map_location=DEV, weights_only=False))
    frozen.eval()
    cond = build_conditional(dim, iscale, n_labels).to(DEV)
    load_frozen_weights(cond, ckpt)
    cond.eval()

    g = torch.Generator(device="cpu").manual_seed(seed)
    n_t = torch.randint(0, CFG.NMAX + 1, (n, dim), generator=g).to(DEV).to(torch.float32)
    t = torch.exp(torch.rand(n, generator=g) * 5.0).to(DEV)

    a = frozen(n_t, t)
    b0 = cond(n_t, t, None)
    y0 = torch.zeros(n, dtype=torch.long, device=DEV)
    b1 = cond(n_t, t, y0)
    r = {"bitwise_no_label": bool(torch.equal(a, b0)),
         "bitwise_zero_label": bool(torch.equal(a, b1)),
         "max_abs_diff_no_label": float((a - b0).abs().max()),
         "max_abs_diff_zero_label": float((a - b1).abs().max()),
         "n_rows": int(n), "dim": int(dim), "nmax": int(CFG.NMAX)}
    r["passed"] = bool(r["bitwise_no_label"] and r["bitwise_zero_label"])
    del frozen, cond
    torch.cuda.empty_cache()
    return r


def train_conditional(train: np.ndarray, labels: np.ndarray, val: np.ndarray,
                      val_labels: np.ndarray, V, iscale, seed: int, n_labels: int,
                      *, max_seconds: float, steps_cap: int = 10 ** 9,
                      resume_from: Path | None = None, ckpt_path: Path | None = None,
                      log_fn=print, batch: int | None = None,
                      divergence_guard: tuple[float, int] | None = None):
    """The conditional version of the frozen training loop. The optimiser, the learning rate, the
        batch size, the EMA and the objective are unchanged; only three differences are needed:

            1. the network is `ConditionalAttnX0Posterior` and its forward takes a label;
            2. each batch draws its labels alongside its counts;
            3. the budget is a five-hour wall-clock cap, the same pre-registered rule every external
                  baseline gets, rather than a fixed step count, so the cosine schedule needs a nominal
         step count, which `steps_cap` supplies; the actual step count goes into the json.
        Early stopping is recorded but not enforced, exactly as in the frozen trainer.
    """
    batch = batch or CFG.BATCH
    T_MAX = M.horizon()
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    cap = CFG.NMAX if CFG.CLIP_X0_TO_NMAX else None
    Xtr = torch.tensor(np.clip(train, 0, cap) if cap else train, dtype=torch.float32)
    Xva = torch.tensor(np.clip(val, 0, cap) if cap else val, dtype=torch.float32)
    Ytr = torch.tensor(np.asarray(labels), dtype=torch.long)
    Yva = torch.tensor(np.asarray(val_labels), dtype=torch.long)
    n, dim = Xtr.shape
    bf16 = CFG.USE_BF16
    Vt = torch.tensor(np.asarray(V, np.float64), dtype=torch.float32, device=DEV)
    model = build_conditional(dim, iscale, n_labels).to(DEV)
    if resume_from is not None:
        load_frozen_weights(model, Path(resume_from))
        log_fn(f"    [conditional] starting from the frozen checkpoint {Path(resume_from).name}; "
               f"label_emb is all zeros; optimiser state and EMA are NOT inherited")
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if (bf16 and torch.cuda.is_available()) else contextlib.nullcontext())
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
    sched = BB._cosine_sched(opt, steps_cap, CFG.WARMUP)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    lt, lf = math.log(T_MAX), math.log(CFG.F_BASE)
    curve = {"step": [], "train": [], "val": [], "walltime_s": []}

    def noise(x0, gg):
        u = torch.rand(x0.shape[0], generator=gg).to(DEV)
        t = torch.exp(lf + u * (lt - lf))
        r = torch.exp(-t).unsqueeze(1)
        n_t = (torch.binomial(x0, r.expand_as(x0))
               + torch.poisson(Vt.unsqueeze(0) * (1 - r))).clamp(0, CFG.NMAX)
        return n_t, t

    run_tr, best, st = None, float("inf"), 0
    best_step = None
    t_start = time.time()
    stop_reason = "steps_cap"
    # Mechanical early-stopping guard: val > ratio * best_val for patience_steps in a row stops
    # immediately and keeps best_val. Written before the run, purely to save time; no criterion moves.
    guard_ratio, guard_steps = divergence_guard or (None, None)
    diverged_since = None
    guard_fired_at = None
    while st < steps_cap:
        if time.time() - t_start >= max_seconds:
            stop_reason = "wallclock_cap"
            break
        model.train()
        idx = torch.randint(0, n, (batch,), generator=g)
        x0 = Xtr[idx].to(DEV)
        y = Ytr[idx].to(DEV)
        n_t, t = noise(x0, g)
        with amp:
            logits = model(n_t, t, y)
            loss = F.cross_entropy(logits.reshape(-1, CFG.NMAX + 1),
                                   x0.long().clamp(0, CFG.NMAX).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(CFG.EMA_DECAY).add_(pm, alpha=1 - CFG.EMA_DECAY)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        lv = float(loss.detach())
        run_tr = lv if run_tr is None else 0.98 * run_tr + 0.02 * lv
        st += 1
        if st % CFG.EVAL_EVERY == 0:
            model.eval()
            _cpu_rng = torch.get_rng_state()
            _cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(12345)
            with torch.no_grad():
                gv = torch.Generator(device="cpu").manual_seed(12345)
                vi = torch.randint(0, Xva.shape[0], (min(2048, Xva.shape[0]),), generator=gv)
                vx0, vy = Xva[vi].to(DEV), Yva[vi].to(DEV)
                vnt, vt = noise(vx0, gv)
                s_, n_ = 0.0, 0
                for lo in range(0, vx0.shape[0], CFG.VAL_CHUNK):
                    hi = lo + CFG.VAL_CHUNK
                    s_ += float(F.cross_entropy(
                        model(vnt[lo:hi], vt[lo:hi], vy[lo:hi]).reshape(-1, CFG.NMAX + 1),
                        vx0[lo:hi].long().clamp(0, CFG.NMAX).reshape(-1), reduction="sum"))
                    n_ += vx0[lo:hi].numel()
                vl = s_ / max(n_, 1)
                del vx0, vnt, vt, vy
            torch.set_rng_state(_cpu_rng)
            if _cuda_rng is not None:
                torch.cuda.set_rng_state_all(_cuda_rng)
            el = time.time() - t_start
            curve["step"].append(st); curve["train"].append(run_tr)
            curve["val"].append(vl); curve["walltime_s"].append(el)
            imp = vl < best
            if imp:
                best, best_step = vl, st
            if ckpt_path is not None:
                ema.eval()
                torch.save(ema.state_dict(), str(ckpt_path) + ".last")
                if imp:
                    torch.save(ema.state_dict(), str(ckpt_path) + ".best")
                ema.train()
            if guard_ratio is not None and best < float("inf"):
                if vl > guard_ratio * best:
                    diverged_since = st if diverged_since is None else diverged_since
                    if st - diverged_since >= guard_steps:
                        guard_fired_at = st
                        stop_reason = "divergence_guard"
                        log_fn(f"    [DIVERGENCE GUARD] val {vl:.4f} > {guard_ratio} x "
                               f"best {best:.4f} for {st - diverged_since} steps "
                               f"(since {diverged_since}) -> stop at {st}, take best_val")
                        break
                else:
                    diverged_since = None
            pk = (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else 0
            log_fn(f"    step {st:6d} train={run_tr:.4f} val={vl:.4f} best={best:.4f} "
                   f"[{el/3600:.2f} h / {max_seconds/3600:.2f} h] [GPU {pk:.1f} GB]")
    ema.eval()
    # whether the cap was reached on a plateau: val-loss slope over the last 20% of steps
    slope = None
    if len(curve["step"]) >= 4:
        k = max(2, int(0.2 * len(curve["step"])))
        xs = np.asarray(curve["step"][-k:], dtype=np.float64)
        ys = np.asarray(curve["val"][-k:], dtype=np.float64)
        slope = float(np.polyfit(xs, ys, 1)[0] * 1000.0)
    meta = {"n_params": int(sum(p.numel() for p in ema.parameters())),
            "steps_done": st, "stop_reason": stop_reason, "best_val": best,
            "best_val_step": best_step,
            "best_val_step_gt_5000": (None if best_step is None else bool(best_step > 5000)),
            "divergence_guard": ({"ratio": guard_ratio, "patience_steps": guard_steps,
                                  "fired_at_step": guard_fired_at}
                                 if guard_ratio is not None else None),
            "final_val": (curve["val"][-1] if curve["val"] else None),
            "final_train_ema": run_tr,
            "loss_plateau_slope_per_1000_steps_last20pct": slope,
            "walltime_s": time.time() - t_start, "max_seconds": max_seconds,
            "batch": batch, "lr": CFG.LR, "wd": CFG.WD, "warmup": CFG.WARMUP,
            "steps_cap_for_cosine": steps_cap, "T_MAX": T_MAX, "NMAX": CFG.NMAX,
            "ema_decay": CFG.EMA_DECAY, "bf16": bool(bf16),
            "grad_ckpt": bool(CFG.USE_GRAD_CKPT), "seed": seed, "n_labels": n_labels}
    return ema, curve, meta
