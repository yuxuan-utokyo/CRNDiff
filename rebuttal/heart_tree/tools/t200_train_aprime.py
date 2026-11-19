# -*- coding: utf-8 -*-
"""A′ training driver (ticket 200 · AMEND-2 section 2).

all configuration fixed before running:
  start      `assets/models/generator/pilot_run1c_s20260625.pt`(frozen ckpt)
  code      `hvg/model_conditional.py`(zero-initialized label channel; frozen file unchanged)
  training data: all 361,205 training cells + `train_celltype.npy`(11 classes)
  optimizer/lr/batch  **`train_pilot.py` defaults** (AdamW 2e-3 / 1e-4, warmup 500 cosine,
                   EMA 0.999,batch = `CFG.BATCH` = 128); none is changed or swept
  budget      **5-hour wall-clock cap**; at the limit take the latest checkpoint
  forward process unchanged ⇒ T_O unchanged, grid unchanged, K=32 unchanged

**cosine the schedule needs a nominal number of steps**, while the budget is wall clock. Procedure (mechanical, reproducible, fixed before results):
run a segment first `--probe-steps` timing probe (**at `train_conditional` before seeding**, so for training
zero effect on the trajectory),`steps_cap = floor(5h / s_per_step)` rounded down to a multiple of 1000.
probe s/step and the final actual number of steps are written to json.

    python -m tools.t200_train_aprime --hours 5
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hvg import config as C                                          # noqa: E402
from hvg.io import sha256_file                                       # noqa: E402
from hvg import model_conditional as MC                              # noqa: E402

CFG = MC.CFG
M = MC.M


def probe_s_per_step(Xtr, Ytr, V, iscale, n_labels, steps: int, batch: int) -> float:
    """timing probe. Uses **another seed** and another generator, discarded after running;
    it is at `train_conditional` call `torch.manual_seed(seed)` finishes before that, so the training trajectory is unchanged."""
    g = torch.Generator(device="cpu").manual_seed(999999)
    Vt = torch.tensor(np.asarray(V, np.float64), dtype=torch.float32, device=MC.DEV)
    model = MC.build_conditional(Xtr.shape[1], iscale, n_labels).to(MC.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if (CFG.USE_BF16 and torch.cuda.is_available()) else contextlib.nullcontext())
    lt, lf = math.log(M.horizon()), math.log(CFG.F_BASE)
    n = Xtr.shape[0]
    model.train()
    t0 = None
    for st in range(steps):
        if st == 5:                       # the first 5 steps are warm-up (cudnn/allocator) and not timed
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
        idx = torch.randint(0, n, (batch,), generator=g)
        x0 = Xtr[idx].to(MC.DEV)
        y = Ytr[idx].to(MC.DEV)
        u = torch.rand(batch, generator=g).to(MC.DEV)
        t = torch.exp(lf + u * (lt - lf))
        r = torch.exp(-t).unsqueeze(1)
        n_t = (torch.binomial(x0, r.expand_as(x0))
               + torch.poisson(Vt.unsqueeze(0) * (1 - r))).clamp(0, CFG.NMAX)
        with amp:
            loss = F.cross_entropy(model(n_t, t, y).reshape(-1, CFG.NMAX + 1),
                                   x0.long().clamp(0, CFG.NMAX).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    sps = (time.time() - t0) / max(steps - 5, 1)
    del model, opt, Vt
    torch.cuda.empty_cache()
    return sps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--batch", type=int, default=None, help="default = CFG.BATCH (train_pilot default)")
    ap.add_argument("--batch-from-ckpt-json", action="store_true",
                    help="A′-v2(AMEND-5 section 1.2): batch **read from the json of the frozen ckpt itself**,"
                         "not copied by hand. That is the training configuration of this ckpt itself.")
    ap.add_argument("--take", choices=("last", "best"), default="last",
                    help="A′-v2(AMEND-5 section 1.2): take the best_val checkpoint, not the last one when wall-clock time runs out.")
    ap.add_argument("--divergence-guard-ratio", type=float, default=None,
                    help="A′-v2: stop when val > ratio * best_val for --divergence-guard-steps consecutive steps.")
    ap.add_argument("--divergence-guard-steps", type=int, default=2000)
    ap.add_argument("--probe-steps", type=int, default=55)
    ap.add_argument("--out-dir", default=str(C.MODELS / "conditional"))
    ap.add_argument("--tag", default="aprime_cond_s20260817")
    a = ap.parse_args()

    outd = Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    ck = outd / f"{a.tag}.pt"
    js = outd / f"{a.tag}.json"
    if ck.exists() or js.exists():
        raise SystemExit(f"refusing to overwrite {ck}")

    t_all = time.time()
    d, meta = M.load_data(("train", "val"))
    tr, va = d["train"], d["val"]
    der = M.derive(tr)
    ytr_raw = np.load(C.LEGACY_DATA / "all" / "train_celltype.npy", allow_pickle=True).astype(str)
    yva_raw = np.load(C.LEGACY_DATA / "all" / "val_celltype.npy", allow_pickle=True).astype(str)
    vocab = MC.label_vocabulary()
    lut = {v: i for i, v in enumerate(vocab)}
    ytr = np.asarray([lut[v] for v in ytr_raw], dtype=np.int64)
    yva = np.asarray([lut[v] for v in yva_raw], dtype=np.int64)
    print(f"[data] train {tr.shape} val {va.shape}  labels {len(vocab)}: {vocab}", flush=True)

    # ---- G-A1: must pass before training (AMEND-2 section 2.2) ----
    ga1 = MC.gate_g_a1(tr.shape[1], der["iscale"], C.GENERATOR_CKPT, len(vocab))
    print(f"[G-A1] {ga1}", flush=True)
    if not ga1["passed"]:
        raise SystemExit("G-A1 failed: the label channel is not wired as a pure addition. "
                         "Do not train. (AMEND-2 §2.3 fallback applies.)")

    # ---- batch: v1 uses the train_pilot default 128; v2 reads it from the json of the frozen ckpt (AMEND-5 section 1.2)----
    ckpt_json = Path(str(C.GENERATOR_CKPT).replace(".pt", ".json"))
    ckpt_cfg = json.loads(C.require(ckpt_json, "frozen ckpt's own json").read_text(
        encoding="utf-8"))
    batch_from_ckpt = int(ckpt_cfg["batch"])
    batch = a.batch or (batch_from_ckpt if a.batch_from_ckpt_json else CFG.BATCH)
    print(f"[batch] train_pilot default = {CFG.BATCH}; frozen ckpt's own json "
          f"({ckpt_json.name}) records batch = {batch_from_ckpt}; using {batch}", flush=True)
    cap_s = a.hours * 3600.0
    Xtr = torch.tensor(np.clip(tr, 0, CFG.NMAX), dtype=torch.float32)
    Ytr = torch.tensor(ytr, dtype=torch.long)
    sps = probe_s_per_step(Xtr, Ytr, der["V"], der["iscale"], len(vocab),
                           a.probe_steps, batch)
    steps_cap = max(1000, int(math.floor(cap_s / max(sps, 1e-9) / 1000.0) * 1000))
    print(f"[probe] {sps*1000:.1f} ms/step at batch={batch} -> steps_cap={steps_cap} "
          f"for the {a.hours:.1f} h cap", flush=True)
    del Xtr, Ytr

    guard = ((a.divergence_guard_ratio, a.divergence_guard_steps)
             if a.divergence_guard_ratio else None)
    ema, curve, mt = MC.train_conditional(
        tr, ytr, va, yva, der["V"], der["iscale"], a.seed, len(vocab),
        max_seconds=cap_s, steps_cap=steps_cap, resume_from=C.GENERATOR_CKPT,
        ckpt_path=ck, log_fn=lambda s: print(s, flush=True), batch=batch,
        divergence_guard=guard)
    # which checkpoint is delivered: v1 = the last one when wall-clock time runs out; v2 = best_val (AMEND-5 section 1.2)
    if a.take == "best" and Path(str(ck) + ".best").exists():
        sd_best = torch.load(str(ck) + ".best", map_location=MC.DEV, weights_only=False)
        ema.load_state_dict(sd_best)
        print(f"[take] delivering the best_val checkpoint "
              f"(val={mt['best_val']:.5f} at step {mt.get('best_val_step')})", flush=True)
    torch.save(ema.state_dict(), ck)

    blob = {"ticket": 200, "block": "A-prime (AMEND-2 §2)",
            "what": "training-time conditioning on the SAME count-native CRN architecture",
            "honest_disclosure": [
                "This is conditional fine-tuning on an unconditional ckpt, not joint training from scratch; a reviewer could argue that training from scratch might be better, and we accept that.",
                "its total compute = original training + 5 hours, more than the 5-hour cap of the external baselines; this bias favours it,"
                "and it is the arm that challenges our claim; being generous to it is the right direction."],
            "start_from": str(C.GENERATOR_CKPT), "start_sha256": C.GENERATOR_CKPT_SHA256,
            "gate_G_A1": ga1,
            "label_vocabulary": vocab, "n_labels": len(vocab),
            "hyperparameters_source": "train_pilot.py defaults (CFG.BATCH/LR/WD/WARMUP/EMA)",
            "version": ("A-prime-v2" if (a.batch_from_ckpt_json or a.take == "best"
                                         or guard) else "A-prime-v1"),
            "v2_changes_vs_v1": ({
                "batch": f"{CFG.BATCH} -> {batch} (read from {ckpt_json.name}::batch)",
                "checkpoint_taken": f"last -> {a.take}",
                "divergence_guard": guard,
                "everything_else": "unchanged (optimizer, LR, cosine, 5 h cap, "
                                   "zero-initialised label embedding, full data)",
                "authority": "AMEND-5 §1.2"} if a.take == "best" else None),
            "frozen_ckpt_own_config": {"json": str(ckpt_json), "batch": batch_from_ckpt,
                                       "best_val": ckpt_cfg.get("best_val"),
                                       "steps_requested": ckpt_cfg.get("steps_requested")},
            "batch": batch, "batch_note":
                "train_pilot.py default is CFG.BATCH=128. The json of the frozen ckpt itself records that it used "
                "batch=384(pilot_run1c). AMEND-2 requires keeping the train_pilot default, so 128 here;"
                "that the two differ is disclosed as is; these are not hyperparameters we picked for this arm.",
            "probe_s_per_step": sps, "probe_steps": a.probe_steps,
            "steps_cap_for_cosine": steps_cap, "wallclock_cap_hours": a.hours,
            "seed": a.seed, "training_cells": int(tr.shape[0]),
            "forward_process_unchanged": True,
            "T_horizon": M.horizon(), "K_unchanged": 32,
            "curve": curve, **mt,
            "ckpt": str(ck), "ckpt_sha256": None,
            "environment": {"python": sys.version, "numpy": np.__version__,
                            "torch": torch.__version__, "cuda": torch.version.cuda,
                            "platform": platform.platform()},
            "total_walltime_s": time.time() - t_all,
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    blob["ckpt_sha256"] = sha256_file(ck)
    js.write_text(json.dumps(blob, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[A'] steps={mt['steps_done']} stop={mt['stop_reason']} "
          f"best_val={mt['best_val']:.5f} slope/1k={mt['loss_plateau_slope_per_1000_steps_last20pct']}")
    print(f"[out] {ck}\n[out] {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
