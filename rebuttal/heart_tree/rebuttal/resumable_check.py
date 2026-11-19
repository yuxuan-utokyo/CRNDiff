# -*- coding: utf-8 -*-
"""Bitwise checks of rebuttal/resumable_cond.py on CPU (GPU hidden so the queue is not disturbed).
A: original train_conditional vs wrapper, same seed, 200 steps. B: wrapper 200 steps straight vs
stopped and restored at step 100 in the same process. The small batch and cell subset only make CPU
feasible; the code path is the same.

    python -m rebuttal.resumable_check
"""
import json, os, shutil, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # on Windows an empty value deletes the variable
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "4"
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, torch
torch.set_num_threads(4)
from hvg import config as C
from hvg import model_conditional as MC
from rebuttal import resumable_cond as RC
assert not torch.cuda.is_available() and str(MC.DEV) == "cpu"
M = MC.M; OUT = ROOT / "rebuttal/out/R10/resume_check"; OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*"):
    f.unlink()
d, _ = M.load_data(("train", "val")); der = M.derive(d["train"])
yt = np.load(C.LEGACY_DATA / "all/train_celltype.npy", allow_pickle=True).astype(str); yv = np.load(C.LEGACY_DATA / "all/val_celltype.npy", allow_pickle=True).astype(str)
voc = MC.label_vocabulary(); lut = {v: i for i, v in enumerate(voc)}
rng = np.random.default_rng(1); it = np.sort(rng.choice(len(yt), 3000, replace=False)); iv = np.sort(rng.choice(len(yv), 64, replace=False))
tr, ytr = np.asarray(d["train"][it]), np.array([lut[v] for v in yt[it]]); va, yva = np.asarray(d["val"][iv]), np.array([lut[v] for v in yv[iv]])
del d
kw = dict(steps_cap=200, resume_from=C.GENERATOR_CKPT, batch=8, divergence_guard=(1.5, 2000), log_fn=lambda s: print(s, flush=True))
cmp = lambda a, b: {"bitwise": all(torch.equal(a[k], b[k]) for k in a), "max_abs": max(float((a[k].float() - b[k].float()).abs().max()) for k in a)}
print("[A0 original]", flush=True)
e0, c0, m0 = MC.train_conditional(tr, ytr, va, yva, der["V"], der["iscale"], 7, len(voc), max_seconds=1e9, ckpt_path=OUT / "orig.pt", **kw)
print("[A1 wrapper]", flush=True); l1 = []
e1, c1, m1 = RC.train(tr, ytr, va, yva, der["V"], der["iscale"], 7, len(voc), ckpt_path=OUT / "wrap.pt", loss_log=l1, **kw)
print("[B1 stop at 100]", flush=True); l2 = []
RC.train(tr, ytr, va, yva, der["V"], der["iscale"], 7, len(voc), ckpt_path=OUT / "res.pt", stop_after=100, loss_log=l2, **kw)
print("[B2 restore]", flush=True)
e2, c2, m2 = RC.train(tr, ytr, va, yva, der["V"], der["iscale"], 7, len(voc), ckpt_path=OUT / "res.pt", loss_log=l2, **kw)
r = {"A_weights": cmp(e0.state_dict(), e1.state_dict()), "A_curve_equal": c0["train"] == c1["train"] and c0["val"] == c1["val"],
     "B_weights": cmp(e1.state_dict(), e2.state_dict()), "B_curve_equal": c1["train"] == c2["train"] and c1["val"] == c2["val"],
     "B_step_losses_equal": l1 == l2, "n_losses": [len(l1), len(l2)], "orig_val": c0["val"], "wrap_val": c1["val"], "resume_val": c2["val"],
     "orig_train": c0["train"], "resume_points": m2["resume_points"]}
(OUT / "check.json").write_text(json.dumps(r, indent=1)); print(json.dumps(r)); print("[done]", flush=True)
