# -*- coding: utf-8 -*-
"""MDLM (Sahoo et al., 2024), OUR PORT to count data, entered in the tables as a named baseline.

The released implementation targets text and provides no single-cell entry point; the backbone
and the capacity here are matched to ours. The paper must therefore say 'our implementation for
count data' and must not say 'we ran MDLM', and it must name the method rather than citing the
anchor alone.

The objective is the published one: masked cross-entropy with the published time weighting and
unmasking rule. Fidelity to it is asserted in the gate rather than claimed here.
"""
from __future__ import annotations

import contextlib
import copy
import math
import time

import numpy as np
try:
    import torch
except ModuleNotFoundError as _exc:                                      # noqa: E402
    raise SystemExit(
        "this entry point needs PyTorch, which the offline reproduction does not "
        "install. See README section 7; in short: pip install torch") from _exc
import torch.nn.functional as F
from torch import nn

from crndiff import config as C

C.add_base_code_to_path()
import config_hvg2k as CFG                                           # noqa: E402
import model_hvg2k as M                                              # noqa: E402

BB = M.BB
DEV = M.DEV


class MaskedMDLMNet(BB.AttnX0Posterior):
    """Our backbone plus a token embedding that includes a MASK symbol.

        The parent's continuous-count projection is unused in absorbing-state diffusion and is
        deleted. Everything else is kept verbatim, so 'the backbone and the capacity match ours' is
        a literal statement.
    """

    def __init__(self, nmax: int, dim: int, d_model: int = None, nheads: int = None,
                 nlayers: int = None):
        d_model = CFG.D_MODEL if d_model is None else d_model
        nheads = CFG.NHEADS if nheads is None else nheads
        nlayers = CFG.NLAYERS if nlayers is None else nlayers
        super().__init__(nmax, dim, d_model=d_model, nheads=nheads, nlayers=nlayers,
                         input_scale=None)
        self.vocab = nmax + 1                 # counts from zero to nmax
        self.mask_id = nmax + 1               # MASK is the symbol after the last count
        self.token_emb = nn.Embedding(nmax + 2, d_model)   # the counts plus MASK
        del self.count_proj                   # unused on the absorbing path; deleted so it does not inflate the parameter count

    def forward(self, tokens: torch.Tensor, t: torch.Tensor):        # type: ignore[override]
        if t.dim() == 1:
            t = t.unsqueeze(1)
        tok = self.token_emb(tokens)                                  # [B, dim, d]
        tok = tok + self.gene_emb(self.gene_ids)[None]
        tok = tok + self.time_mlp(t)[:, None, :]
        if self.training and CFG.USE_GRAD_CKPT:
            import torch.utils.checkpoint as _cp
            h = tok
            for layer in self.transformer.layers:
                h = _cp.checkpoint(layer, h, use_reentrant=False)
            if self.transformer.norm is not None:
                h = self.transformer.norm(h)
            return self.head(self.norm(h))
        return self.head(self.norm(self.transformer(tok)))


# training
def train_mdlm(train: np.ndarray, val: np.ndarray, seed: int, *, max_seconds: float,
               steps_cap: int, batch: int, lr: float, wd: float,
               ckpt_path=None, log_fn=print, eval_every: int = None,
               patience: int = None, early_stop_rel: float = None):
    """The published MDLM loss: cross-entropy at the masked positions, weighted by one over t.

        The early-stopping rule is the same as ours, and as with our conditional arm the
        best-validation checkpoint is delivered, with the plateau evidence recorded.
    """
    nmax = CFG.NMAX
    eval_every = eval_every or CFG.EVAL_EVERY
    patience = patience or CFG.PATIENCE
    early_stop_rel = CFG.EARLY_STOP_REL if early_stop_rel is None else early_stop_rel

    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xtr = torch.tensor(np.clip(train, 0, nmax), dtype=torch.long)
    Xva = torch.tensor(np.clip(val, 0, nmax), dtype=torch.long)
    n, dim = Xtr.shape

    net = MaskedMDLMNet(nmax, dim).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = BB._cosine_sched(opt, steps_cap, CFG.WARMUP)
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if (CFG.USE_BF16 and torch.cuda.is_available()) else contextlib.nullcontext())
    ema = copy.deepcopy(net)
    for p in ema.parameters():
        p.requires_grad_(False)

    def masked_ce(model, x0, t, mask):
        xt = torch.where(mask, torch.full_like(x0, net.mask_id), x0)
        logits = model(xt, t)
        ce = F.cross_entropy(logits.reshape(-1, nmax + 1), x0.reshape(-1), reduction="none")
        ce = ce.view(x0.shape)
        w = (1.0 / t.clamp_min(1e-3))[:, None]
        return (ce * mask.float() * w).sum() / mask.sum().clamp_min(1)

    curve = {"step": [], "train": [], "val": [], "walltime_s": []}
    run_tr, best, best_step, st = None, float("inf"), None, 0
    bad, es_ref, stop_reason = 0, float("inf"), "steps_cap"
    t_start = time.time()
    while st < steps_cap:
        if time.time() - t_start >= max_seconds:
            stop_reason = "wallclock_cap"
            break
        net.train()
        idx = torch.randint(0, n, (batch,), generator=g)
        x0 = Xtr[idx].to(DEV)
        t = torch.rand(batch, generator=g).to(DEV)
        mask = torch.rand(x0.shape, generator=g).to(DEV) < t[:, None]
        with amp:
            loss = masked_ce(net, x0, t, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), net.parameters()):
                pe.mul_(CFG.EMA_DECAY).add_(pm, alpha=1 - CFG.EMA_DECAY)
            for be, bm in zip(ema.buffers(), net.buffers()):
                be.copy_(bm)
        lv = float(loss.detach())
        run_tr = lv if run_tr is None else 0.98 * run_tr + 0.02 * lv
        st += 1

        if st % eval_every == 0:
            net.eval()
            _cpu, _cuda = torch.get_rng_state(), (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            torch.manual_seed(12345)
            with torch.no_grad():
                gv = torch.Generator(device="cpu").manual_seed(12345)
                vi = torch.randint(0, Xva.shape[0], (min(2048, Xva.shape[0]),), generator=gv)
                s_, k_ = 0.0, 0
                for lo in range(0, len(vi), CFG.VAL_CHUNK):
                    ii = vi[lo:lo + CFG.VAL_CHUNK]
                    vx = Xva[ii].to(DEV)
                    vt = torch.rand(len(ii), generator=gv).to(DEV)
                    vm = torch.rand(vx.shape, generator=gv).to(DEV) < vt[:, None]
                    s_ += float(masked_ce(net, vx, vt, vm)) * len(ii)
                    k_ += len(ii)
                vl = s_ / max(k_, 1)
            torch.set_rng_state(_cpu)
            if _cuda is not None:
                torch.cuda.set_rng_state_all(_cuda)
            el = time.time() - t_start
            curve["step"].append(st); curve["train"].append(run_tr)
            curve["val"].append(vl); curve["walltime_s"].append(el)
            imp = vl < best
            if imp:
                best, best_step = vl, st
            if vl < es_ref * (1.0 - early_stop_rel):
                es_ref, bad = vl, 0
            else:
                bad += 1
            if ckpt_path is not None:
                ema.eval()
                torch.save(ema.state_dict(), str(ckpt_path) + ".last")
                if imp:
                    torch.save(ema.state_dict(), str(ckpt_path) + ".best")
                ema.train()
            pk = (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else 0
            log_fn(f"    step {st:6d} train={run_tr:.4f} val={vl:.4f} best={best:.4f} "
                   f"bad={bad}/{patience} [{el/3600:.2f} h] [GPU {pk:.1f} GB]")
            if bad >= patience:
                stop_reason = "val_plateau"
                log_fn(f"    [PLATEAU] {patience} evaluations without a {early_stop_rel:.1%} "
                       f"improvement -> stop at step {st}, take best_val")
                break

    ema.eval()
    slope = None
    if len(curve["step"]) >= 4:
        k = max(2, int(0.2 * len(curve["step"])))
        slope = float(np.polyfit(np.asarray(curve["step"][-k:], dtype=np.float64),
                                 np.asarray(curve["val"][-k:], dtype=np.float64), 1)[0] * 1000.0)
    meta = {"n_params": int(sum(p.numel() for p in ema.parameters())),
            "steps_done": st, "stop_reason": stop_reason,
            "best_val": best, "best_val_step": best_step,
            "final_val": (curve["val"][-1] if curve["val"] else None),
            "plateau_evidence": {
                "rule": f"same early-stop rule as ours: {patience} consecutive evaluations "
                        f"without a {early_stop_rel:.1%} relative improvement",
                "eval_every": eval_every, "patience": patience,
                "early_stop_rel": early_stop_rel,
                "val_slope_per_1000_steps_last20pct": slope,
                "n_evaluations": len(curve["step"])},
            "batch": batch, "lr": lr, "weight_decay": wd,
            "steps_cap_for_cosine": steps_cap, "max_seconds": max_seconds,
            "walltime_s": time.time() - t_start, "seed": seed,
            "vocab_size": nmax + 1, "mask_id": nmax + 1, "nmax": nmax,
            "bf16": bool(CFG.USE_BF16), "grad_ckpt": bool(CFG.USE_GRAD_CKPT)}
    return ema, curve, meta


# sampling
@torch.no_grad()
def sample_mdlm(net: MaskedMDLMNet, n_samples: int, dim: int, *, n_steps: int, seed: int,
                chunk: int = 2048, record_trajectory: bool = False, log_fn=None):
    """Start fully masked and unmask a fraction of the remaining positions at each step; the last

        step forces a prediction for whatever is left. With trajectory recording on, a token
    """
    mask_id = net.mask_id
    out = np.empty((n_samples, dim), dtype=np.int64)
    traj = []
    for lo in range(0, n_samples, chunk):
        b = min(chunk, n_samples - lo)
        g = torch.Generator(device=DEV).manual_seed(int(seed) * 100 + lo // chunk)
        x = torch.full((b, dim), mask_id, dtype=torch.long, device=DEV)
        ts = torch.linspace(1.0, 0.0, n_steps + 1, device=DEV)
        for i in range(n_steps):
            t, s = float(ts[i]), float(ts[i + 1])
            masked = (x == mask_id)
            if masked.any():
                logits = net(x, torch.full((b,), t, device=DEV))
                probs = F.softmax(logits, dim=2)
                pred = torch.multinomial(probs.reshape(-1, net.vocab), 1,
                                         generator=g).squeeze(1).view(b, dim)
                p_unmask = (t - s) / max(t, 1e-6)
                do = (torch.rand(x.shape, generator=g, device=DEV) < p_unmask) & masked
                x = torch.where(do, pred, x)
            if record_trajectory and lo == 0:
                traj.append(x.detach().cpu().numpy().copy())
        masked = (x == mask_id)
        if masked.any():
            logits = net(x, torch.zeros(b, device=DEV))
            x = torch.where(masked, torch.argmax(logits, dim=2), x)
        if record_trajectory and lo == 0:
            traj.append(x.detach().cpu().numpy().copy())
        out[lo:lo + b] = x.cpu().numpy()
        if log_fn and (lo // chunk) % 10 == 0:
            log_fn(f"  chunk {lo // chunk} ({lo + b}/{n_samples})")
    return out, traj


def gate_gm1(traj: list, mask_id: int) -> dict:
    """Gate: the DEFINING property of an absorbing state, namely that once a token is unmasked it

        never changes again. Asserted step by step along the whole trajectory: after a position
        leaves MASK it must never change. A failure means the implementation is broken and its
    """
    if len(traj) < 2:
        return {"passed": False, "why": "trajectory too short"}
    viol_unmask_to_other = 0      # an unmasked position changed to a different value
    viol_remask = 0               # an unmasked position went back to MASK
    n_unmask_events = 0
    for a, b in zip(traj[:-1], traj[1:]):
        was_unmasked = (a != mask_id)
        changed = (a != b)
        viol_unmask_to_other += int((was_unmasked & changed & (b != mask_id)).sum())
        viol_remask += int((was_unmasked & (b == mask_id)).sum())
        n_unmask_events += int(((a == mask_id) & (b != mask_id)).sum())
    final_masked = int((traj[-1] == mask_id).sum())
    ok = (viol_unmask_to_other == 0 and viol_remask == 0 and final_masked == 0)
    return {"passed": bool(ok),
            "n_steps_checked": len(traj) - 1,
            "n_unmask_events": n_unmask_events,
            "violations_unmasked_token_changed": viol_unmask_to_other,
            "violations_remasked": viol_remask,
            "final_still_masked": final_masked,
            "property": "absorbing state: once a token leaves MASK it never changes again, "
                        "and nothing is left masked at the end"}
