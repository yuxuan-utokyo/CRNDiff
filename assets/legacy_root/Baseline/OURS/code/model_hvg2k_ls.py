# -*- coding: utf-8 -*-
"""[generated verbatim from model_hvg2k.py by gen_ls_train.py; the only change = the training loss supports
label smoothing; with label_smooth=0 bit-identical to the original file. The val evaluation definition is **unchanged**, still plain CE,
so val curves of different eps can be compared side by side.]

hvg2k · OURS model

the network uses directly `_shared/code/backbone.py::AttnX0Posterior`(the very same class as in 61gene).
training loop follows `EXP21::train_ours_generic` moved; changes listed item by item in `train_ours_hvg2k` in the docstring.
"""
from __future__ import annotations

import contextlib
import copy
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent                  # .../hvg2k/OURS/code
ROOT = Path(__file__).resolve().parents[3]                                  # .../toy/hvg2k
G61 = ROOT.parent / "_shared" / "panel61"                            # read-only
sys.path.insert(0, str(HERE))
import config_hvg2k as CFG                              # noqa: E402

# reuse the frozen 61gene backbone (read-only import, nothing written there)
PKG = "panel61_shared"
SHARED61 = G61 / "_shared" / "code"
if PKG not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        PKG, SHARED61 / "__init__.py", submodule_search_locations=[str(SHARED61)])
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules[PKG] = _pkg
    _spec.loader.exec_module(_pkg)
    importlib.import_module(f"{PKG}.seeds").DATA_DIR = G61 / "_shared" / "data"
    sys.modules.setdefault("_shared", sys.modules[PKG])
BB = importlib.import_module(f"{PKG}.backbone")
DEV = BB.DEV


# =========================================================================== #
# data
# =========================================================================== #
def quantize_log2_V(mu, lo=CFG.V_GRID_LO, hi=CFG.V_GRID_HI):
    """quantize the per-gene mean to the log2 grid 2^lo … 2^hi (nearest level). See the B2 criterion in config."""
    lv = 2.0 ** np.arange(lo, hi + 1, dtype=np.float64)
    m = np.maximum(np.asarray(mu, np.float64), lv[0])
    return lv[np.abs(np.log2(m)[:, None] - np.log2(lv)[None, :]).argmin(1)], lv


def load_data(split=("train", "val")):
    meta = json.loads((CFG.DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    assert meta["gene_hash"] == CFG.GENE_HASH, \
        f"gene_hash mismatch:{meta['gene_hash']} != {CFG.GENE_HASH}"      # red line §5.3
    out = {s: np.load(CFG.DATA_DIR / f"{s}.npy") for s in split}
    return out, meta


def derive(train):
    """all derived quantities estimated from train. **Estimated from train only; val/test not involved.**"""
    X = train.astype(np.int64)
    mu = X.mean(0)
    V, levels = quantize_log2_V(mu)
    iscale = 1.0 / (1.0 + mu)
    dmax = X.max(0).astype(int)
    if CFG.CLIP_X0_TO_NMAX:
        dmax = np.minimum(dmax, CFG.NMAX)
    return {"mu": mu, "V": V, "levels": levels, "iscale": iscale, "dmax": dmax,
            "n_distinct_V": int(len(np.unique(V))),
            "sum_V": float(V.sum()), "lib_mean": float(X.sum(1).mean()),
            "sum_V_over_lib": float(V.sum() / X.sum(1).mean())}


def horizon():
    return float(CFG.T_O if CFG.T_HORIZON == "T_O" else CFG.T_HORIZON)


def build_grid(K, T=None):
    """linspace(T, F_BASE, K), 0 appended at the end (same convention as model.py::gfull_nfe)."""
    T = horizon() if T is None else T
    g = np.linspace(T, CFG.F_BASE, K)
    gf = np.concatenate([np.sort(np.unique(g[g > 1e-9]))[::-1], [0.0]])
    return gf, int(len(gf) - 1)


class AttnX0PosteriorCkpt(BB.AttnX0Posterior):
    """and `AttnX0Posterior` **subclass with **identical parameters**, only in forward on 3 layers transformer
    gradient checkpointing layer by layer. Parent class file `_shared/code/backbone.py` is read-only, not a single byte changed;
    parameter names and shapes are unchanged, so the two state_dicts can load each other (use the parent class when sampling).

    only on `self.training and CFG.USE_GRAD_CKPT` uses checkpointing; at inference/sampling time identical to the parent class.
    """

    def forward(self, n_t, t):
        if not (self.training and CFG.USE_GRAD_CKPT):
            return super().forward(n_t, t)
        import torch.utils.checkpoint as _cp
        if t.dim() == 1:
            t = t.unsqueeze(1)
        sc = (n_t * self.input_scale).unsqueeze(-1)
        lg = (torch.log1p(n_t) / self._lognmax).unsqueeze(-1)
        tok = self.count_proj(torch.cat([sc, lg], -1))
        tok = tok + self.gene_emb(self.gene_ids)[None]
        tok = tok + self.time_mlp(t)[:, None, :]
        h = tok
        for layer in self.transformer.layers:            # nn.TransformerEncoder.layers
            h = _cp.checkpoint(layer, h, use_reentrant=False)
        if self.transformer.norm is not None:
            h = self.transformer.norm(h)
        return self.head(self.norm(h))


def build_net(dim, iscale, ckpt=False):
    cls = AttnX0PosteriorCkpt if ckpt else BB.AttnX0Posterior
    return cls(CFG.NMAX, dim, d_model=CFG.D_MODEL, nheads=CFG.NHEADS,
               nlayers=CFG.NLAYERS, input_scale=iscale)


# =========================================================================== #
# training
# =========================================================================== #
def train_ours_hvg2k(train, val, V, iscale, seed, steps, *, batch=None, log_fn=print,
                     eval_every=CFG.EVAL_EVERY, patience=CFG.PATIENCE, t_max=None,
                     ckpt_path=None, bf16=None, grad_ckpt=None, resume_from=None,
                     label_smooth=0.0, ls_kernel="uniform_sup"):
    """`EXP21::train_ours_generic` hvg2k version. Changes item by item:

    1. `T_MAX` from `ours.T_MAX`(8.0) replaced by the horizon of this dataset (default T_O=5.2021);
       `T_FLOOR` still `F_BASE=0.01`. At training time the upper bound of t **must** match the sampling horizon.
    2. `x0` clip to NMAX before feeding into the binomial (`CFG.CLIP_X0_TO_NMAX`);
       61gene clamps only the **sum** of binomial + Poisson. Reason in the config `CLIP_X0_TO_NMAX`.
    3. batch reduced from 1024 to 256: logits are [B, 2000, 513]; with B=1024 the forward pass alone takes 4.2 GB.
    4. added early stopping (main line B5 criterion: consecutive `patience` consecutive val evaluations without improvement stop it),
       61gene that version ran a fixed number of steps to completion.
    5. val evaluation uses an **independent CPU Generator** (fixed 12345) and does not consume the one used for training `g` ——
       EXP143 fell into this pitfall (val evaluation touching the global RNG changes the training trajectory).

    everything else (AdamW 2e-3/1e-4, cosine with warmup 500, EMA 0.999, cross-entropy objective) unchanged.
    """
    batch = batch or CFG.BATCH
    T_MAX = horizon() if t_max is None else t_max
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    cap = CFG.NMAX if CFG.CLIP_X0_TO_NMAX else None
    Xtr = torch.tensor(np.clip(train, 0, cap) if cap else train, dtype=torch.float32)
    Xva = torch.tensor(np.clip(val, 0, cap) if cap else val, dtype=torch.float32)
    n, dim = Xtr.shape
    bf16 = CFG.USE_BF16 if bf16 is None else bf16
    grad_ckpt = CFG.USE_GRAD_CKPT if grad_ckpt is None else grad_ckpt
    Vt = torch.tensor(np.asarray(V, np.float64), dtype=torch.float32, device=DEV)
    model = build_net(dim, iscale, ckpt=grad_ckpt).to(DEV)
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if (bf16 and torch.cuda.is_available()) else contextlib.nullcontext())
    log_fn(f"    [training precision] bf16 autocast={bf16} (weights still fp32)  gradient checkpointing={grad_ckpt}")
    # ==== label smoothing back-off distribution u_g, [dim, NMAX+1], not back-propagated ====
    ls_w, ls_u = float(label_smooth), None
    if ls_w > 0.0:
        C_ = CFG.NMAX + 1
        dg = np.clip(np.asarray(train), 0, CFG.NMAX).max(0).astype(int)
        u_ = np.zeros((dim, C_), np.float32)
        if ls_kernel == "uniform_sup":
            # floor only within the observed support; above d_g the training signal is exactly the same as plain CE.
            for d_ in range(dim):
                u_[d_, :dg[d_] + 1] = 1.0 / (dg[d_] + 1)
        elif ls_kernel == "uniform":
            u_[:] = 1.0 / C_
        else:
            raise ValueError(f"unknown ls_kernel: {ls_kernel}")
        ls_u = torch.tensor(u_, device=DEV)
        extra = float(ls_w * (u_ * np.arange(C_)).sum()
                      / max(np.clip(np.asarray(train), 0, CFG.NMAX).mean() * dim, 1e-12))
        log_fn(f"    [label smoothing] eps={ls_w:g}  kernel={ls_kernel}  "
               f"d_g median={np.median(dg):.0f}  **expected extra relative mass {extra:+.3%}**")
        log_fn(f"      — this is a count-weighted mass cost; a little probability in the far tail is a large mass;"
               f"eps budget was set from it (2% corresponds to eps<=2.4e-4)")
    else:
        log_fn("    [label smoothing] off (eps=0, bit-identical to model_hvg2k.py)")
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
    sched = BB._cosine_sched(opt, steps, CFG.WARMUP)
    if resume_from is not None:
        sd_ = torch.load(resume_from, map_location=DEV, weights_only=False)
        model.load_state_dict(sd_)
        log_fn(f"    [RESUME] resuming training from {Path(resume_from).name}: "
               f"**optimizer state and EMA were not saved; this is a cold start from the weights**,"
               f"so the warmup/cosine schedule restarts. This is written to the log; do not treat it as a seamless resume.")
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    lt, lf = math.log(T_MAX), math.log(CFG.F_BASE)
    curve = {"step": [], "train": [], "val": []}

    def noise(x0, gg):
        u = torch.rand(x0.shape[0], generator=gg).to(DEV)
        t = torch.exp(lf + u * (lt - lf))
        r = torch.exp(-t).unsqueeze(1)
        n_t = (torch.binomial(x0, r.expand_as(x0))
               + torch.poisson(Vt.unsqueeze(0) * (1 - r))).clamp(0, CFG.NMAX)
        return n_t, t

    run_tr, best, bad, stopped = None, float("inf"), 0, steps
    es_ref, es_would_stop = float("inf"), None   # early-stop reference point / where it would have stopped if enabled
    for st in range(steps):
        model.train()
        idx = torch.randint(0, n, (batch,), generator=g)
        x0 = Xtr[idx].to(DEV)
        n_t, t = noise(x0, g)
        with amp:
            logits = model(n_t, t)
            if ls_u is None:
                # eps=0: **unchanged**, bit-identical to model_hvg2k.py
                loss = F.cross_entropy(logits.reshape(-1, CFG.NMAX + 1),
                                       x0.long().clamp(0, CFG.NMAX).reshape(-1))
            else:
                # q = (1-eps)*onehot + eps*u_g  =>  loss = -(q * log_softmax).sum(-1).mean()
                # .float() consistent with F.cross_entropy under autocast (which upcasts to fp32 anyway)
                lp_ = F.log_softmax(logits.reshape(-1, CFG.NMAX + 1).float(), -1)
                tg_ = x0.long().clamp(0, CFG.NMAX).reshape(-1)
                uf_ = ls_u.unsqueeze(0).expand(x0.shape[0], dim, CFG.NMAX + 1) \
                          .reshape(-1, CFG.NMAX + 1)
                loss = ((1.0 - ls_w) * F.nll_loss(lp_, tg_)
                        - ls_w * (uf_ * lp_).sum(-1).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(EMA := CFG.EMA_DECAY).add_(pm, alpha=1 - EMA)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        lv = float(loss.detach())
        run_tr = lv if run_tr is None else 0.98 * run_tr + 0.02 * lv
        if (st + 1) % eval_every == 0 or st == steps - 1:
            model.eval()
            # ===== 2026-08-10: val must be a **deterministic function of the weights** and must not touch the training RNG =====
            # originally only `noise` in `torch.rand`(drawing t) is handed to a fixed CPU `gv`,
            # but the corruption itself: `torch.binomial(x0, r)` / `torch.poisson(...)` — input
            # on DEV it uses the **global CUDA RNG**,`gv` cannot reach it. So two things happen at once:
            #   1. the corruption noise differs at every evaluation -> sampling noise leaks into the val curve (exactly the main-line question);
            #   2. val evaluation **consumes the global CUDA RNG** -> which in turn changes the training trajectory,
            #      exactly the EXP143 pitfall, only through the CUDA stream rather than the CPU one.
            # fix as in EXP138: fixed val subset + fixed corruption seed + **save and restore the global RNG state** around evaluation.
            _cpu_rng = torch.get_rng_state()
            _cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(12345)          # seeds both CPU and all CUDA devices
            with torch.no_grad():
                gv = torch.Generator(device="cpu").manual_seed(12345)
                vi = torch.randint(0, Xva.shape[0], (min(2048, Xva.shape[0]),), generator=gv)
                vx0 = Xva[vi].to(DEV)
                vnt, vt = noise(vx0, gv)
                # ★ ****must be chunked.** A forward pass of 2048 cells at dim=2000 is **21.6 GB**
                #   (T2 measured 10.8 MB/cell at inference), 2.5 times more than a training step at batch=128 (8.5 GB).
                #   2026-08-09 this, not the batch size, is the real cause of the overnight stalls:
                #   61gene in that recipe 2048×61 genes is negligible; copied to 2000 dimensions it becomes the peak.
                #   use reduction="sum" then dividing by the total number of elements is **numerically identical** to taking the mean at once,
                #   so the val curves remain comparable with earlier runs.
                # val always evaluated in fp32: the curves of the two precision configurations must be directly comparable
                s_, n_ = 0.0, 0
                for lo in range(0, vx0.shape[0], CFG.VAL_CHUNK):
                    hi = lo + CFG.VAL_CHUNK
                    xb, nb, tb_ = vx0[lo:hi], vnt[lo:hi], vt[lo:hi]
                    s_ += float(F.cross_entropy(
                        model(nb, tb_).reshape(-1, CFG.NMAX + 1),
                        xb.long().clamp(0, CFG.NMAX).reshape(-1), reduction="sum"))
                    n_ += xb.numel()
                vl = s_ / max(n_, 1)
                del vx0, vnt, vt
            torch.set_rng_state(_cpu_rng)                       # training trajectory unchanged
            if _cuda_rng is not None:
                torch.cuda.set_rng_state_all(_cuda_rng)
            curve["step"].append(st + 1); curve["train"].append(run_tr); curve["val"].append(vl)
            # ================= early stopping: logic fixed on 2026-08-10 bug =================
            # old code (which cut both training runs of experiment X):
            #     imp = vl < best * (1 - REL)
            #     best, bad = (vl, 0) if imp else (min(best, vl), bad + 1)
            # the error is in **`best` and [the reference point of the relative threshold] are the same variable**:
            #   `min(best, vl)` makes best ratchet down at every refresh, so
            #   the step that [refreshed best but by less than REL=0.3%]: best is updated,
            #   bad but +1. The counter was not reset at the record-breaking step.
            #   run1  step 5000 val=0.11510 < best=0.11520(improvement 0.087% < 0.3%) -> bad=3/3 stop
            #   run1b step 2800 val=0.11408   is the final best_val itself        -> bad=5/5 stop
            # fix: the reference point `ref` moves down only on a **significant** improvement,`best` only handles saving; the two are separate.
            imp_any = vl < best                                # any new record -> save .best
            imp_sig = vl < es_ref * (1.0 - CFG.EARLY_STOP_REL)  # significant improvement -> reset the counter
            if imp_sig:
                es_ref, bad = vl, 0
            else:
                bad += 1
            best = min(best, vl)
            # **early stopping is record-only, never executed** (main line 2026-08-10): the first trigger point is logged for budgeting.
            if bad >= patience and es_would_stop is None:
                es_would_stop = st + 1
                log_fn(f"    [EARLY-STOP·record only] if enabled, the first trigger happens at step {es_would_stop}"
                       f"(patience={patience}, rel={CFG.EARLY_STOP_REL}): not executed in this round; training continues")
            # ****every evaluation point is saved to disk**. The first version only saved at the end; the 2026-08-10 pilot, because of memory jitter,
            # when forced to abort at step 1000, 5.3 hours of training left no output at all.
            if ckpt_path is not None:
                ema.eval()
                torch.save(ema.state_dict(), str(ckpt_path) + ".last")
                if imp_any:
                    torch.save(ema.state_dict(), str(ckpt_path) + ".best")
                ema.train()
            pk_gb = (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else 0
            log_fn(f"    step {st+1:6d}/{steps} train={run_tr:.4f} val={vl:.4f} "
                   f"best={best:.4f} bad={bad}/{patience}"
                   + (f"  [GPU peak allocation {pk_gb:.1f} GB]" if torch.cuda.is_available() else ""))
            # **the criterion uses max_memory_allocated, not memory_reserved.**
            # enabled `expandable_segments:True` after that, reserved reports virtual address space,
            # readings like 38 GB reserved on a 24 GB card appear; using it as the criterion measures the wrong thing.
            if pk_gb > 18.0:
                log_fn(f"    [WARN] peak allocation {pk_gb:.1f} GB: not enough headroom on a 24 GB card,"
                       f"growing further triggers WDDM spilling into shared memory and slows down by an order of magnitude")
            if bad >= patience and CFG.EARLY_STOP_ENFORCE:
                stopped = st + 1
                log_fn(f"    [EARLY-STOP] {patience} consecutive val evaluations without improvement, stopped at step {stopped}")
                break
    ema.eval()
    return ema, curve, {"n_params": int(sum(p.numel() for p in ema.parameters())),
                        "stopped_at": stopped, "best_val": best, "batch": batch,
                        "T_MAX": T_MAX, "NMAX": CFG.NMAX, "clip_x0": CFG.CLIP_X0_TO_NMAX,
                        "early_stop_enforced": bool(CFG.EARLY_STOP_ENFORCE),
                        "early_stop_would_trigger_at": es_would_stop,
                        "bf16": bool(bf16), "grad_ckpt": bool(grad_ckpt),
                        "val_rng_isolated": True}
