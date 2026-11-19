# -*- coding: utf-8 -*-
"""Resumable copy of hvg.model_conditional.train_conditional. Frozen code is imported, not edited.

The numeric path is the original loop line for line. Added: full state saves (model, EMA, optimizer,
scheduler, every RNG, step, curve) written atomically, at most every `state_every_s` seconds, and a
restore at the start when the state file exists. Bitwise checks: rebuttal/resumable_check.py.
"""
import contextlib, copy, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from hvg import model_conditional as MC
CFG, M, BB, DEV = MC.CFG, MC.M, MC.BB, MC.DEV


def _save(path, blob):
    tmp = str(path) + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def train(train, labels, val, val_labels, V, iscale, seed, n_labels, *, steps_cap, resume_from=None, ckpt_path=None,
          log_fn=print, batch=None, divergence_guard=None, state_every_s=1200.0, stop_after=None, loss_log=None):
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
    model = MC.build_conditional(dim, iscale, n_labels).to(DEV)
    if resume_from is not None:
        MC.load_frozen_weights(model, Path(resume_from))
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
    elapsed0 = 0.0
    stop_reason = "steps_cap"
    guard_ratio, guard_steps = divergence_guard or (None, None)
    diverged_since = None
    guard_fired_at = None
    resumes = []
    state_path = Path(str(ckpt_path) + ".state")
    if state_path.exists():
        s = torch.load(str(state_path), map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"])
        opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        g.set_state(s["g"]); torch.set_rng_state(s["torch_rng"]); np.random.set_state(s["np_rng"])
        if torch.cuda.is_available() and s["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(s["cuda_rng"])
        run_tr, best, st, best_step, curve = s["run_tr"], s["best"], s["st"], s["best_step"], s["curve"]
        diverged_since, elapsed0 = s["diverged_since"], s["elapsed"]
        resumes = s["resumes"] + [st]
        log_fn(f"    [resume] state file restored at step {st} (resume points so far: {resumes})")
    t_start = time.time()
    t_state = time.time()

    def save_state():
        _save(state_path, {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                           "sched": sched.state_dict(), "g": g.get_state(), "torch_rng": torch.get_rng_state(),
                           "np_rng": np.random.get_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                           "run_tr": run_tr, "best": best, "st": st, "best_step": best_step, "curve": curve,
                           "diverged_since": diverged_since, "elapsed": elapsed0 + time.time() - t_start,
                           "resumes": resumes})

    while st < steps_cap:
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
        if loss_log is not None:
            loss_log.append(lv)
        if not math.isfinite(lv):
            stop_reason = "non_finite_loss"
            log_fn(f"    [HARD DIVERGENCE] non-finite training loss {lv} after step {st}; stop")
            break
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
            el = elapsed0 + time.time() - t_start
            curve["step"].append(st); curve["train"].append(run_tr)
            curve["val"].append(vl); curve["walltime_s"].append(el)
            if not math.isfinite(vl):
                stop_reason = "non_finite_val"
                log_fn(f"    [HARD DIVERGENCE] non-finite val at step {st}; stop")
                break
            imp = vl < best
            if imp:
                best, best_step = vl, st
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
                        log_fn(f"    [DIVERGENCE GUARD] val {vl:.4f} > {guard_ratio} x best {best:.4f} -> stop at {st}")
                        break
                else:
                    diverged_since = None
            pk = (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else 0
            log_fn(f"    step {st:6d} train={run_tr:.4f} val={vl:.4f} best={best:.4f} [{el/3600:.2f} h] [GPU {pk:.1f} GB]")
        if time.time() - t_state >= state_every_s or st == stop_after:
            save_state(); t_state = time.time()
            if st == stop_after:
                return None, curve, {"stopped_after": st}
    ema.eval()
    save_state()
    return ema, curve, {"n_params": int(sum(p.numel() for p in ema.parameters())), "steps_done": st, "stop_reason": stop_reason,
                        "best_val": best, "best_val_step": best_step, "final_val": curve["val"][-1] if curve["val"] else None,
                        "final_train_ema": run_tr, "divergence_guard": {"ratio": guard_ratio, "patience_steps": guard_steps, "fired_at_step": guard_fired_at},
                        "walltime_s": elapsed0 + time.time() - t_start, "resume_points": resumes, "batch": batch, "lr": CFG.LR, "wd": CFG.WD,
                        "warmup": CFG.WARMUP, "steps_cap_for_cosine": steps_cap, "T_MAX": T_MAX, "NMAX": CFG.NMAX, "ema_decay": CFG.EMA_DECAY,
                        "bf16": bool(bf16), "grad_ckpt": bool(CFG.USE_GRAD_CKPT), "seed": seed, "n_labels": n_labels}
