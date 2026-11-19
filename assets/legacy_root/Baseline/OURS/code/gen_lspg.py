# -*- coding: utf-8 -*-
"""*_lspg.py is **generated verbatim** from model_hvg2k_ls.py / train_pilot_ls.py:
label smoothing eps changed from a global constant to **per gene** e_g = 2 b mu_g / d_g (--ls-kernel pergene_sup,
label_smooth parameter interpreted as the total budget b). Each replacement asserts exactly one hit; the scalar branch is kept verbatim,
so when pergene_sup is not selected, behaviour is bit-identical to the _ls version.

motivation (ANALYSIS_hvg2k.md section 4.4): relative injection of a global eps ~ eps*(d_g/2)/mu_g; low-expression genes
(single-digit d_g, mu<0.01) more than an order of magnitude above the median gene -> the lowest-expression bin is pushed up to 2.65x.
equalized relative injection: extra mass of each gene = b * its own mass; total extra mass = b.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sub1(s, old, new, tag):
    assert s.count(old) == 1, f"[{tag}] matched {s.count(old)} times, exactly 1 required"
    return s.replace(old, new)


# ---------------- model ----------------
m = (HERE / "model_hvg2k_ls.py").read_text(encoding="utf-8")

m = sub1(m, "label smoothing; with label_smooth=0 bit-identical to the original file.",
         "label smoothing(**pergene_sup: per-gene eps**, generated from the _ls version by gen_lspg.py);"
         "label_smooth=0 bit-identical to the original file.", "doc")

m = sub1(m, '''    ls_w, ls_u = float(label_smooth), None
    if ls_w > 0.0:
        C_ = CFG.NMAX + 1
        dg = np.clip(np.asarray(train), 0, CFG.NMAX).max(0).astype(int)
        u_ = np.zeros((dim, C_), np.float32)
        if ls_kernel == "uniform_sup":''',
         '''    ls_w, ls_u, ls_e = float(label_smooth), None, None
    if ls_w > 0.0:
        C_ = CFG.NMAX + 1
        Xc_ = np.clip(np.asarray(train), 0, CFG.NMAX)
        dg = Xc_.max(0).astype(int)
        u_ = np.zeros((dim, C_), np.float32)
        if ls_kernel == "pergene_sup":
            # ★ per-gene eps: equalized relative injection. label_smooth is interpreted as the total budget b,
            #   e_g = 2 b mu_g / d_g(extra mass_g = b*mu_g, total extra = b*total mass).
            #   the floor is still uniform_sup (only within the observed support); cap 1e-2 keeps the one-hot weight >= 0.99.
            mu_ = Xc_.mean(0).astype(np.float64)
            for d_ in range(dim):
                u_[d_, :dg[d_] + 1] = 1.0 / (dg[d_] + 1)
            e_ = np.where(dg > 0, 2.0 * ls_w * mu_ / np.maximum(dg, 1), 0.0)
            e_ = np.clip(e_, 0.0, 1e-2).astype(np.float32)
            ls_e = torch.tensor(e_, device=DEV)
            extra = float((e_.astype(np.float64) * (u_ * np.arange(C_)).sum(1)).sum()
                          / max(Xc_.mean() * dim, 1e-12))
            log_fn(f"    [label smoothing/pergene] b={ls_w:g}  e_g median={np.median(e_):.2e} "
                   f"max={e_.max():.2e}  capped genes={int((e_ >= 1e-2).sum())}  "
                   f"d_g median={np.median(dg):.0f}  **expected extra relative mass {extra:+.3%}**")
        elif ls_kernel == "uniform_sup":''', "eps-block-head")

m = sub1(m, '''        else:
            raise ValueError(f"unknown ls_kernel: {ls_kernel}")
        ls_u = torch.tensor(u_, device=DEV)
        extra = float(ls_w * (u_ * np.arange(C_)).sum()
                      / max(np.clip(np.asarray(train), 0, CFG.NMAX).mean() * dim, 1e-12))
        log_fn(f"    [label smoothing] eps={ls_w:g}  kernel={ls_kernel}  "
               f"d_g median={np.median(dg):.0f}  **expected extra relative mass {extra:+.3%}**")''',
         '''        else:
            raise ValueError(f"unknown ls_kernel: {ls_kernel}")
        ls_u = torch.tensor(u_, device=DEV)
        if ls_e is None:
            extra = float(ls_w * (u_ * np.arange(C_)).sum()
                          / max(np.clip(np.asarray(train), 0, CFG.NMAX).mean() * dim, 1e-12))
            log_fn(f"    [label smoothing] eps={ls_w:g}  kernel={ls_kernel}  "
                   f"d_g median={np.median(dg):.0f}  **expected extra relative mass {extra:+.3%}**")''', "eps-block-tail")

m = sub1(m, '''                uf_ = ls_u.unsqueeze(0).expand(x0.shape[0], dim, CFG.NMAX + 1) \\
                          .reshape(-1, CFG.NMAX + 1)
                loss = ((1.0 - ls_w) * F.nll_loss(lp_, tg_)
                        - ls_w * (uf_ * lp_).sum(-1).mean())''',
         '''                uf_ = ls_u.unsqueeze(0).expand(x0.shape[0], dim, CFG.NMAX + 1) \\
                          .reshape(-1, CFG.NMAX + 1)
                if ls_e is None:
                    loss = ((1.0 - ls_w) * F.nll_loss(lp_, tg_)
                            - ls_w * (uf_ * lp_).sum(-1).mean())
                else:
                    # per-gene eps: elementwise (1-e_g)*nll + e_g*(-sum u_g log p), then averaged over B*dim.
                    # e_g when it is a constant vector it is identical to the scalar formula above.
                    ef_ = ls_e.unsqueeze(0).expand(x0.shape[0], dim).reshape(-1).float()
                    nll_ = F.nll_loss(lp_, tg_, reduction="none")
                    loss = ((1.0 - ef_) * nll_ + ef_ * (-(uf_ * lp_).sum(-1))).mean()''', "loss")

(HERE / "model_hvg2k_lspg.py").write_text(m, encoding="utf-8")

# ---------------- train ----------------
t = (HERE / "train_pilot_ls.py").read_text(encoding="utf-8")
t = sub1(t, "import model_hvg2k_ls as M", "import model_hvg2k_lspg as M", "import")
t = sub1(t, 'choices=["uniform_sup", "uniform"],',
         'choices=["uniform_sup", "uniform", "pergene_sup"],', "choices")
t = sub1(t, '"""[generated verbatim from train_pilot.py by gen_ls_train.py; only --label-smooth / --ls-kernel added.]',
         '"""[generated verbatim from train_pilot_ls.py by gen_lspg.py; --ls-kernel gains pergene_sup (per-gene eps,'
         'label-smooth interpreted as the total budget b).]', "doc")
(HERE / "train_pilot_lspg.py").write_text(t, encoding="utf-8")
print("generated model_hvg2k_lspg.py / train_pilot_lspg.py")
