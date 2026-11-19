# -*- coding: utf-8 -*-
"""Resumable copy of hvg._frozen.model_hvg2k.train_ours_hvg2k, numeric path unchanged. Frozen code is
imported, not edited. Added: full state saves (atomic, at most every state_every_s), restore at start,
and a hard-divergence stop that never changes
the numbers before it fires. Early stopping is logged only, as in the frozen run. Delivered weights are
the EMA at the last step, the same rule as the OURS generators.

    python -m rebuttal.resumable_gen --seed 20260625 --steps 20000
"""
import argparse, contextlib, copy, json, math, os, sys, time
from pathlib import Path
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "4")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from hvg import model_conditional as _MC
M = _MC.M
CFG, BB, DEV = M.CFG, M.BB, M.DEV
DG = ROOT / "rebuttal/data/r13/OUT"; OUTD = ROOT / "rebuttal/out/R13/models"


def use_dg():
    meta = json.loads((DG / "meta.json").read_text(encoding="utf-8")); to = json.loads((DG / "T_O.json").read_text())
    CFG.DATA_DIR, CFG.GENE_HASH, CFG.NMAX, CFG.T_O = DG, meta["gene_hash"], 256, round(to["T_O"], 4)
    return meta


# R16: HLCA in the same array format (rebuttal/r16_src/prep_r16.py). Directory from R16_HLCA_DIR, so the same file
# runs on the 4090 and on the GPU server; NMAX and T_O are read from json computed on train.npy (HVG2K B3 rule (a); rebuttal/r13_to.py).
HLCA = Path(os.environ.get("R16_HLCA_DIR", str(ROOT / "rebuttal/data/r16_hlca/OUT"))); OUTD_HLCA = ROOT / "rebuttal/out/R16/models"


def use_hlca():
    meta = json.loads((HLCA / "meta.json").read_text(encoding="utf-8")); to = json.loads((HLCA / "T_O.json").read_text())
    nmax = int(json.loads((HLCA / "NMAX.json").read_text())["NMAX"])
    CFG.DATA_DIR, CFG.GENE_HASH, CFG.NMAX, CFG.T_O = HLCA, meta["gene_hash"], nmax, round(to["T_O"], 4)
    return meta


def _save(path, blob):
    tmp = str(path) + ".tmp"; torch.save(blob, tmp); os.replace(tmp, path)


def train(train, val, V, iscale, seed, steps, *, batch=None, log_fn=print, ckpt_path=None, guard=(1.5, 2000),
          state_every_s=1200.0, stop_after=None, loss_log=None):
    eval_every, patience = CFG.EVAL_EVERY, CFG.PATIENCE
    batch = batch or CFG.BATCH
    T_MAX = M.horizon()
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    cap = CFG.NMAX if CFG.CLIP_X0_TO_NMAX else None
    Xtr = torch.tensor(np.clip(train, 0, cap) if cap else train, dtype=torch.float32)
    Xva = torch.tensor(np.clip(val, 0, cap) if cap else val, dtype=torch.float32)
    n, dim = Xtr.shape
    bf16, grad_ckpt = CFG.USE_BF16, CFG.USE_GRAD_CKPT
    Vt = torch.tensor(np.asarray(V, np.float64), dtype=torch.float32, device=DEV)
    model = M.build_net(dim, iscale, ckpt=grad_ckpt).to(DEV)
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if (bf16 and torch.cuda.is_available()) else contextlib.nullcontext())
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
    sched = BB._cosine_sched(opt, steps, CFG.WARMUP)
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
    es_ref, es_would_stop = float("inf"), None
    st0, elapsed0, resumes, diverged_since, hard = 0, 0.0, [], None, None
    n214, first214 = 0, None
    state_path = Path(str(ckpt_path) + ".state")
    if state_path.exists():
        s = torch.load(str(state_path), map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        g.set_state(s["g"]); torch.set_rng_state(s["torch_rng"]); np.random.set_state(s["np_rng"])
        if torch.cuda.is_available() and s["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(s["cuda_rng"])
        run_tr, best, bad, es_ref, es_would_stop, curve = s["run_tr"], s["best"], s["bad"], s["es_ref"], s["es_would_stop"], s["curve"]
        st0, elapsed0, diverged_since, n214, first214 = s["st"], s["elapsed"], s["diverged_since"], s["n214"], s["first214"]; resumes = s["resumes"] + [st0]
        log_fn(f"    [resume] restored at step {st0}; resume points {resumes}")
    t_start = t_state = time.time()

    def save_state(st_next):
        _save(state_path, {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                           "g": g.get_state(), "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                           "run_tr": run_tr, "best": best, "bad": bad, "es_ref": es_ref, "es_would_stop": es_would_stop, "curve": curve,
                           "st": st_next, "elapsed": elapsed0 + time.time() - t_start, "diverged_since": diverged_since, "resumes": resumes, "n214": n214, "first214": first214})

    for st in range(st0, steps):
        model.train()
        idx = torch.randint(0, n, (batch,), generator=g)
        x0 = Xtr[idx].to(DEV)
        n_t, t = noise(x0, g)
        with amp:
            logits = model(n_t, t)
            loss = F.cross_entropy(logits.reshape(-1, CFG.NMAX + 1), x0.long().clamp(0, CFG.NMAX).reshape(-1))
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
        if loss_log is not None:
            loss_log.append(lv)
        if not math.isfinite(lv):
            hard = st + 1; stopped = st + 1
            log_fn(f"    [HARD DIVERGENCE] non-finite training loss {lv} at step {hard}; stop")
            break
        run_tr = lv if run_tr is None else 0.98 * run_tr + 0.02 * lv
        if (st + 1) % eval_every == 0 or st == steps - 1:
            model.eval()
            _cpu_rng = torch.get_rng_state()
            _cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(12345)
            with torch.no_grad():
                gv = torch.Generator(device="cpu").manual_seed(12345)
                vi = torch.randint(0, Xva.shape[0], (min(2048, Xva.shape[0]),), generator=gv)
                vx0 = Xva[vi].to(DEV)
                vnt, vt = noise(vx0, gv)
                s_, n_ = 0.0, 0
                for lo in range(0, vx0.shape[0], CFG.VAL_CHUNK):
                    hi = lo + CFG.VAL_CHUNK
                    xb, nb, tb_ = vx0[lo:hi], vnt[lo:hi], vt[lo:hi]
                    s_ += float(F.cross_entropy(model(nb, tb_).reshape(-1, CFG.NMAX + 1), xb.long().clamp(0, CFG.NMAX).reshape(-1), reduction="sum"))
                    n_ += xb.numel()
                vl = s_ / max(n_, 1)
                del vx0, vnt, vt
            torch.set_rng_state(_cpu_rng)
            if _cuda_rng is not None:
                torch.cuda.set_rng_state_all(_cuda_rng)
            curve["step"].append(st + 1); curve["train"].append(run_tr); curve["val"].append(vl)
            imp_any = vl < best
            imp_sig = vl < es_ref * (1.0 - CFG.EARLY_STOP_REL)
            if imp_sig:
                es_ref, bad = vl, 0
            else:
                bad += 1
            best = min(best, vl)
            if bad >= patience and es_would_stop is None:
                es_would_stop = st + 1
                log_fn(f"    [EARLY-STOP logged only] would trigger at step {es_would_stop}")
            ema.eval()
            torch.save(ema.state_dict(), str(ckpt_path) + ".last")
            if imp_any:
                torch.save(ema.state_dict(), str(ckpt_path) + ".best")
            ema.train()
            pk = (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else 0
            log_fn(f"    step {st+1:6d}/{steps} train={run_tr:.4f} val={vl:.4f} best={best:.4f} bad={bad}/{patience} [GPU {pk:.1f} GB] [{(elapsed0+time.time()-t_start)/3600:.2f} h]")
            # ticket 214-A breaker, recorded only: val > max(0.25, 2 x best), 3 times running
            n214 = n214 + 1 if vl > max(0.25, 2.0 * best) else 0
            if n214 >= 3 and first214 is None:
                first214 = st + 1
                log_fn(f"    [214-A logged only] would have stopped at step {first214}")
            if not math.isfinite(vl):
                hard = st + 1; stopped = st + 1
                log_fn(f"    [HARD DIVERGENCE] non-finite val at step {hard}; stop")
                break
            if vl > guard[0] * best:
                diverged_since = st + 1 if diverged_since is None else diverged_since
                if st + 1 - diverged_since >= guard[1]:
                    hard = st + 1; stopped = st + 1
                    log_fn(f"    [HARD DIVERGENCE] val {vl:.4f} > {guard[0]} x best {best:.4f} since {diverged_since}; stop at {hard}")
                    break
            else:
                diverged_since = None
        if time.time() - t_state >= state_every_s or st + 1 == stop_after:
            save_state(st + 1); t_state = time.time()
            if st + 1 == stop_after:
                return None, curve, {"stopped_after": st + 1}
    ema.eval()
    save_state(stopped)
    return ema, curve, {"n_params": int(sum(p.numel() for p in ema.parameters())), "stopped_at": stopped, "best_val": best, "batch": batch, "T_MAX": T_MAX,
                        "NMAX": CFG.NMAX, "early_stop_enforced": False, "early_stop_would_trigger_at": es_would_stop, "hard_divergence_at": hard, "breaker_214A_would_fire_at": first214,
                        "resume_points": resumes, "walltime_s": elapsed0 + time.time() - t_start, "bf16": bool(bf16), "grad_ckpt": bool(grad_ckpt), "lr": CFG.LR, "wd": CFG.WD, "warmup": CFG.WARMUP, "ema_decay": CFG.EMA_DECAY}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--mem-gb", type=float, default=None, help="cap this process, so an overrun fails here and not in another GPU task")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dataset", choices=("dg", "hlca"), default="dg")
    ap.add_argument("--print-config", action="store_true", help="print DATA_DIR, GENE_HASH, NMAX, T_O and exit (no training)")
    a = ap.parse_args(); meta = use_dg() if a.dataset == "dg" else use_hlca()
    outd = OUTD if a.dataset == "dg" else OUTD_HLCA
    if a.print_config:
        print(json.dumps({"dataset": a.dataset, "DATA_DIR": str(CFG.DATA_DIR), "GENE_HASH": CFG.GENE_HASH, "NMAX": CFG.NMAX, "T_O": CFG.T_O,
                          "BATCH_arg": a.batch, "LR": CFG.LR, "WD": CFG.WD, "WARMUP": CFG.WARMUP, "EMA_DECAY": CFG.EMA_DECAY, "outd": str(outd)}), flush=True)
        raise SystemExit(0)
    outd.mkdir(parents=True, exist_ok=True)
    if a.mem_gb:
        torch.cuda.set_per_process_memory_fraction(a.mem_gb * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
    d, _ = M.load_data(("train", "val")); der = M.derive(d["train"]); name = a.tag or f"{a.dataset}_generator_s{a.seed}"; ck = outd / f"{name}.pt"
    print(f"[r13] train {d['train'].shape} val {d['val'].shape} NMAX={CFG.NMAX} T_O={CFG.T_O} sum_V/lib={der['sum_V_over_lib']:.4f} seed={a.seed}", flush=True)
    net, curve, info = train(d["train"], d["val"], der["V"], der["iscale"], a.seed, a.steps, batch=a.batch, ckpt_path=ck, log_fn=lambda s: print(s, flush=True))
    torch.save(net.state_dict(), ck)
    (outd / f"{name}.json").write_text(json.dumps({"seed": a.seed, "steps_requested": a.steps, "dataset": "dentategyrus" if a.dataset == "dg" else "hlca_core", "NMAX": CFG.NMAX, "gene_hash": meta["gene_hash"], "T_O": CFG.T_O, "curve": curve, **info, "checkpoint_rule": "EMA at the last step, as OURS", "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None}, indent=1, default=float), encoding="utf-8")
    print("[done]", info, flush=True)
