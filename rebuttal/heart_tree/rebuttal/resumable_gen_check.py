# -*- coding: utf-8 -*-
"""Bitwise checks of rebuttal/resumable_gen.py on CPU, DG data: original vs wrapper, and straight vs restored.
    python -m rebuttal.resumable_gen_check
"""
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "4"
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
torch.set_num_threads(4)
from rebuttal import resumable_gen as RG
M = RG.M; assert not torch.cuda.is_available()
OUT = ROOT / "rebuttal/out/R13/resume_check"; OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*"):
    f.unlink()
RG.use_dg(); d, _ = M.load_data(("train", "val")); der = M.derive(d["train"]); tr, va = d["train"], d["val"][:64]
cmp = lambda a, b: {"bitwise": all(torch.equal(a[k], b[k]) for k in a), "max_abs": max(float((a[k].float() - b[k].float()).abs().max()) for k in a)}
kw = dict(batch=8, log_fn=lambda s: print(s, flush=True))
print("[A0 original]", flush=True); e0, c0, m0 = M.train_ours_hvg2k(tr, va, der["V"], der["iscale"], 7, 200, ckpt_path=OUT / "orig.pt", **kw)
print("[A1 wrapper]", flush=True); l1 = []; e1, c1, m1 = RG.train(tr, va, der["V"], der["iscale"], 7, 200, ckpt_path=OUT / "wrap.pt", loss_log=l1, **kw)
print("[B1 stop at 100]", flush=True); l2 = []; RG.train(tr, va, der["V"], der["iscale"], 7, 200, ckpt_path=OUT / "res.pt", stop_after=100, loss_log=l2, **kw)
print("[B2 restore]", flush=True); e2, c2, m2 = RG.train(tr, va, der["V"], der["iscale"], 7, 200, ckpt_path=OUT / "res.pt", loss_log=l2, **kw)
r = {"A_weights": cmp(e0.state_dict(), e1.state_dict()), "A_curve_equal": c0 == c1, "B_weights": cmp(e1.state_dict(), e2.state_dict()), "B_curve_equal": c1 == c2, "B_step_losses_equal": l1 == l2, "n_losses": [len(l1), len(l2)], "val": [c0["val"], c1["val"], c2["val"]], "resume_points": m2["resume_points"]}
(OUT / "check.json").write_text(json.dumps(r, indent=1)); print(json.dumps(r)); print("[done]", flush=True)
