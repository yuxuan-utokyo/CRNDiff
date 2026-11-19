# -*- coding: utf-8 -*-
"""from `model_hvg2k.py` / `train_pilot.py` ****generates verbatim** a copy with label smoothing.

    python gen_ls_train.py      ->  model_hvg2k_ls.py  +  train_pilot_ls.py

placed in `Experiments/X/Baseline/OURS/code/` run under. Every replacement asserts it **matches exactly once**,
so"nothing changed apart from this switch"is machine-verifiable (`gen_noa4.py` the same procedure).
red line:`model_hvg2k.py` / `train_pilot.py` / `vendor_ours*.py` / `bank_dedup.py`
is left byte-identical.

what to change
------
`model_hvg2k.py:188` the hard-label CE of that line:

    loss = F.cross_entropy(logits.reshape(-1, NMAX+1), x0.reshape(-1))

replaced by CE on soft labels:

    q = (1-eps) * onehot(x0) + eps * u_g          loss = -(q * log_softmax).sum(-1).mean()

`--label-smooth 0`(default): **never enters the new branch**, identical to `model_hvg2k.py` bit-identical.

why (already settled on the 2D toy)
---------------------------
plain CE on counts with empirical frequency 0 only pushes the logit down, never up; the logit runs to -inf.
on the toy, the generated marginal at those counts was measured 1e7~1e9 times below the truth, and any **bounded** endpoint reweighting
cannot fill it (w_clip=3, 9 rounds -> dynamic range capped at 2e4). Added label smoothing (eps=3e-4,
uniform_sup) afterwards, the same model with the same PPC:

    median deviation at zero-observation points   10^7.29  ->  10^0.35
    R_mass               1.0062   ->  1.0029      (better)
    rare-bin mass ratio         1.196    ->  1.046       (the estimated limit of that bin is exactly 1.046, an exact hit)
    top three bins               0.998/1.015/1.076 -> 0.999/1.005/1.024   (all better)

rare bin of the bare model 0.640 -> 0.639 (**unchanged**): label smoothing fixes holes, PPC fixes the rare region
under-generation; the two are orthogonal and do not replace each other.

why the kernel is uniform_sup and not uniform
-------------------------------------
`uniform` u(n)=1/C is constant, so the optimum at zero observations is also constant: the generated marginal above d_g
becomes a **horizontal line**, while the truth keeps decaying over that range and is orders of magnitude higher at the far end.
`uniform_sup` floor only on n<=d_g; above d_g the training signal is exactly the same as plain CE.

hvg2k it is also much cheaper. Extra relative mass = eps * sum_g E_u_g[n] / sum_g E_true_g[n]:

    per-gene d_g quantiles [0,25,50,75,90,99,100] = [1, 17, 32, 68, 138, 512, 512]
    uniform      extra mass = eps * 695.7   ->  under a 2% budget eps <= 2.9e-5
    uniform_sup  extra mass = eps *  84.8   ->  under a 2% budget eps <= 2.4e-4

median d_g is only 32, E_u = d_g/2 ≈ 16, 8 times cheaper than uniform 256:
**so the eps=3e-4 tuned on the toy can essentially be carried over**. Suggest running 1e-4 / 3e-4.

⚠ must be trained in pairs
--------------
`pilot_run1c` was produced by **three resumed segments** run1 -> run1b -> run1c (step_offset=7800,
in each segment the optimizer and EMA cold-start and warmup/cosine restart). Comparing an LS model trained from scratch with it
mixed into the differences"different training schedule". **Each must be trained with the same script, the same number of steps and the same seed**,
differ only by `--label-smooth`. s_per_step≈0.44 (batch 384), 12000 steps take about 88 minutes.

    python train_pilot_ls.py --steps 12000 --label-smooth 0     --name cectrl
    python train_pilot_ls.py --steps 12000 --label-smooth 3e-4  --name ls3e4

val evaluation definition **is unchanged** (always plain CE), so the val curves of the two arms remain directly comparable:
changing the training objective would change the evaluation ruler, and the two curves would no longer be comparable.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = []

# ===================== model_hvg2k.py -> model_hvg2k_ls.py =====================
M_REPL = [
    ('''def train_ours_hvg2k(train, val, V, iscale, seed, steps, *, batch=None, log_fn=print,
                     eval_every=CFG.EVAL_EVERY, patience=CFG.PATIENCE, t_max=None,
                     ckpt_path=None, bf16=None, grad_ckpt=None, resume_from=None):''',
     '''def train_ours_hvg2k(train, val, V, iscale, seed, steps, *, batch=None, log_fn=print,
                     eval_every=CFG.EVAL_EVERY, patience=CFG.PATIENCE, t_max=None,
                     ckpt_path=None, bf16=None, grad_ckpt=None, resume_from=None,
                     label_smooth=0.0, ls_kernel="uniform_sup"):''', 1),

    # after the model is built and before the training loop, prepare the back-off distribution u
    ('''    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)''',
     '''    # ==== label smoothing back-off distribution u_g, [dim, NMAX+1], not back-propagated ====
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
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)''', 1),

    # of the training step loss
    ('''        with amp:
            logits = model(n_t, t)
            loss = F.cross_entropy(logits.reshape(-1, CFG.NMAX + 1),
                                   x0.long().clamp(0, CFG.NMAX).reshape(-1))''',
     '''        with amp:
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
                uf_ = ls_u.unsqueeze(0).expand(x0.shape[0], dim, CFG.NMAX + 1) \\
                          .reshape(-1, CFG.NMAX + 1)
                loss = ((1.0 - ls_w) * F.nll_loss(lp_, tg_)
                        - ls_w * (uf_ * lp_).sum(-1).mean())''', 1),

    ('''"""hvg2k · OURS: data / V / network / training. New file; does not modify `_shared/code/` and anything of 61gene.''', '''"""[generated verbatim from model_hvg2k.py by gen_ls_train.py; the only change = the training loss supports
label smoothing; with label_smooth=0 bit-identical to the original file. The val evaluation definition is **unchanged**, still plain CE,
so val curves of different eps can be compared side by side.]

hvg2k · OURS model''', 1),
]

# ===================== train_pilot.py -> train_pilot_ls.py =====================
T_REPL = [
    ('import model_hvg2k as M                                 # noqa: E402',
     'import model_hvg2k_ls as M                           # noqa: E402  ← the only import change', 1),

    ('''    ap.add_argument("--smoke", action="store_true")''',
     '''    ap.add_argument("--label-smooth", type=float, default=0.0,
                    help="★ training-side back-off. 0 = bit-identical to train_pilot.py."
                         "hvg2k eps budget: under uniform_sup, 2%% extra mass corresponds to 2.4e-4,"
                         "suggest trying 1e-4 / 3e-4 (toy optimum is 3e-4)")
    ap.add_argument("--ls-kernel", default="uniform_sup",
                    choices=["uniform_sup", "uniform"],
                    help="uniform leaves a horizontal line above d_g, and is 8 times more expensive on hvg2k")
    ap.add_argument("--smoke", action="store_true")''', 1),

    # ⚠ must be appended at the **end**: tr/va/... before it are positional; inserting keyword arguments before them is a syntax error
    ('''            resume_from=a.resume)''',
     '''            resume_from=a.resume,
            label_smooth=a.label_smooth, ls_kernel=a.ls_kernel)''', 1),

    ('''"""hvg2k · OURS pilot training.''',
     '''"""[generated verbatim from train_pilot.py by gen_ls_train.py; only --label-smooth / --ls-kernel added.]

⚠ ****must be trained in pairs**:`pilot_run1c` was produced by three resumed segments (step_offset=7800; in each segment the optimizer and
EMA cold start); comparing an LS model trained from scratch against it would mix in"different training schedule". Train one each with the same script, steps and seed:

    python train_pilot_ls.py --steps 12000 --label-smooth 0     --name cectrl
    python train_pilot_ls.py --steps 12000 --label-smooth 3e-4  --name ls3e4

s_per_step≈0.44(batch 384); 12000 steps take about 88 minutes per arm.

hvg2k · OURS pilot training.''', 1),
]

JOBS = [("model_hvg2k.py", "model_hvg2k_ls.py", M_REPL),
        ("train_pilot.py", "train_pilot_ls.py", T_REPL)]


def main():
    import ast
    for src_n, dst_n, repl in JOBS:
        src, dst = HERE / src_n, HERE / dst_n
        assert src.exists(), f"not found {src}"
        s = src.read_text(encoding="utf-8")
        print(f"\n{src_n} -> {dst_n}")
        for i, (old, new, want) in enumerate(repl, 1):
            got = s.count(old)
            assert got == want, (f"  replacement {i} matched {got} times, expected {want}: refusing to generate."
                                 f"(the source file may have changed; please check manually)")
            s = s.replace(old, new, want)
            print(f"  replacement {i}: matched {got} times  OK")
        ast.parse(s)
        dst.write_text(s, encoding="utf-8")
        print(f"  wrote {dst.name} (syntax check passed)")
    print("\nnext step (paired training, about 88 minutes each):")
    print("  python train_pilot_ls.py --steps 12000 --label-smooth 0    --name cectrl")
    print("  python train_pilot_ls.py --steps 12000 --label-smooth 3e-4 --name ls3e4")
    print("first run 200 steps each with --smoke to check the loss decreases and memory does not blow up.")


if __name__ == "__main__":
    main()
