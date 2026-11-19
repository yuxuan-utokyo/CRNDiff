# -*- coding: utf-8 -*-
"""R10 conditional arm: tools/t200_train_aprime.py (A-prime-v2 settings) on a reduced train set.

Same frozen pilot, same seed, batch from the pilot's json, take=best, 5 h cap, guard 1.5/2000.
Only difference: Neuronal train cells outside the fixed subset are removed from X and y.
V and iscale stay derived from the FULL train set: the pilot was built with them and gate G-A1
compares against the pilot bit by bit. --mem-gb caps this process so an overrun fails here
rather than taking GPU memory from the sampling queue.

    python -m rebuttal.r10_train_cond --level frac010 --hours 5
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from hvg import config as C
from hvg.io import sha256_file
from hvg import model_conditional as MC
from tools.t200_train_aprime import probe_s_per_step

CFG, M = MC.CFG, MC.M
ap = argparse.ArgumentParser()
ap.add_argument("--level", choices=("frac010", "frac003"), required=True)
ap.add_argument("--hours", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=20260817)
ap.add_argument("--mem-gb", type=float, default=None)
ap.add_argument("--probe-steps", type=int, default=55)
ap.add_argument("--tag", default=None)
ap.add_argument("--steps", type=int, default=None, help="step budget instead of the wallclock cap")
ap.add_argument("--batch", type=int, default=None, help="default: from the pilot json (384)")
a = ap.parse_args()
torch.set_num_threads(4)
if a.mem_gb:
    torch.cuda.set_per_process_memory_fraction(a.mem_gb * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
OUTD = ROOT / "rebuttal" / "out" / "R10" / "models"
tag = a.tag or f"r10_cond_{a.level}_s{a.seed}"
ck, js = OUTD / f"{tag}.pt", OUTD / f"{tag}.json"
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
sub_f = ROOT / "rebuttal" / "out" / "R10" / "subset_indices.npz"
z = np.load(sub_f)
drop = np.setdiff1d(z["neuronal_train"], z[a.level])
keep = np.ones(tr.shape[0], bool); keep[drop] = False
tr, ytr = tr[keep], ytr[keep]
n_neu = int((ytr == lut["Neuronal"]).sum())
print(f"[data] train {tr.shape} (neuronal {n_neu}) val {va.shape}", flush=True)
ga1 = MC.gate_g_a1(tr.shape[1], der["iscale"], C.GENERATOR_CKPT, len(vocab))
print(f"[G-A1] {ga1}", flush=True)
if not ga1["passed"]:
    raise SystemExit("G-A1 failed")
ckpt_json = Path(str(C.GENERATOR_CKPT).replace(".pt", ".json"))
batch = a.batch or int(json.loads(ckpt_json.read_text(encoding="utf-8"))["batch"])
if a.steps:      # 11-07 01:42: budget = the 40,985 steps A-prime-v2 reached, not wallclock
    cap_s, steps_cap, sps = 1.0e9, a.steps, None
    print(f"[budget] {a.steps} steps, no wallclock cap, batch={batch}", flush=True)
else:
    cap_s = a.hours * 3600.0
    Xtr = torch.tensor(np.clip(tr, 0, CFG.NMAX), dtype=torch.float32)
    Ytr = torch.tensor(ytr, dtype=torch.long)
    sps = probe_s_per_step(Xtr, Ytr, der["V"], der["iscale"], len(vocab), a.probe_steps, batch)
    steps_cap = max(1000, int(math.floor(cap_s / max(sps, 1e-9) / 1000.0) * 1000))
    print(f"[probe] {sps*1000:.1f} ms/step at batch={batch} -> steps_cap={steps_cap}; "
          f"peak_reserved_gb={torch.cuda.max_memory_reserved()/1024**3:.2f}", flush=True)
    del Xtr, Ytr
ema, curve, mt = MC.train_conditional(
    tr, ytr, va, yva, der["V"], der["iscale"], a.seed, len(vocab),
    max_seconds=cap_s, steps_cap=steps_cap, resume_from=C.GENERATOR_CKPT,
    ckpt_path=ck, log_fn=lambda s: print(s, flush=True), batch=batch, divergence_guard=(1.5, 2000))
if Path(str(ck) + ".best").exists():
    ema.load_state_dict(torch.load(str(ck) + ".best", map_location=MC.DEV, weights_only=False))
torch.save(ema.state_dict(), ck)
blob = {"experiment": "rebuttal_R10", "level": a.level, "n_neuronal_train": n_neu,
        "training_cells": int(tr.shape[0]), "subset_file": str(sub_f), "subset_sha256": sha256_file(sub_f),
        "start_from": str(C.GENERATOR_CKPT), "start_sha256": C.GENERATOR_CKPT_SHA256, "gate_G_A1": ga1,
        "label_vocabulary": vocab, "batch": batch, "probe_s_per_step": sps, "steps_cap_for_cosine": steps_cap,
        "wallclock_cap_hours": (None if a.steps else a.hours), "step_budget": a.steps, "cosine_steps_note": "A-prime-v2 used cosine horizon 41000 and stopped at 40985; here horizon = budget", "seed": a.seed, "curve": curve, **mt, "ckpt": str(ck),
        "ckpt_sha256": sha256_file(ck), "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "reference_aprime_v2": {"probe_s_per_step": 0.43317, "steps_done": 40985},
        "note": "pilot trained on all 3,168 neuronal cells; both R10 arms share that pilot",
        "total_walltime_s": time.time() - t_all, "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
js.write_text(json.dumps(blob, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"[done] steps={mt['steps_done']} stop={mt['stop_reason']} best_val={mt['best_val']:.5f}", flush=True)
