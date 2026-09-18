# -*- coding: utf-8 -*-
"""从 model_hvg2k_ls.py / train_pilot_ls.py **逐字生成** *_lspg.py：
label smoothing 的 eps 由全局常数改为**逐基因** e_g = 2 b mu_g / d_g（--ls-kernel pergene_sup，
label_smooth 参数解读为总预算 b）。每处替换断言只命中一次；scalar 分支逐字保留，
所以 pergene_sup 不选时行为与 _ls 版逐位相同。

动机（ANALYSIS_hvg2k.md §4.4）：全局 eps 的相对注入 ~ eps*(d_g/2)/mu_g，低表达基因
（d_g 个位数、mu<0.01）比中位基因大一个量级以上 -> 最低表达基因档被顶到 2.65x。
等化相对注入：每个基因的额外质量 = b * 自身质量，总额外质量 = b。
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sub1(s, old, new, tag):
    assert s.count(old) == 1, f"[{tag}] 命中 {s.count(old)} 次，要求恰 1 次"
    return s.replace(old, new)


# ---------------- model ----------------
m = (HERE / "model_hvg2k_ls.py").read_text(encoding="utf-8")

m = sub1(m, "label smoothing；label_smooth=0 时与原文件逐位相同。",
         "label smoothing（**pergene_sup: 逐基因 eps**，由 gen_lspg.py 从 _ls 版生成）；"
         "label_smooth=0 时与原文件逐位相同。", "doc")

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
            # ★ 逐基因 eps：等化相对注入。label_smooth 解读为总预算 b，
            #   e_g = 2 b mu_g / d_g（额外质量_g = b*mu_g，总额外 = b*总质量）。
            #   地板仍是 uniform_sup（只铺观测支撑内），cap 1e-2 保证 one-hot 权重 >= 0.99。
            mu_ = Xc_.mean(0).astype(np.float64)
            for d_ in range(dim):
                u_[d_, :dg[d_] + 1] = 1.0 / (dg[d_] + 1)
            e_ = np.where(dg > 0, 2.0 * ls_w * mu_ / np.maximum(dg, 1), 0.0)
            e_ = np.clip(e_, 0.0, 1e-2).astype(np.float32)
            ls_e = torch.tensor(e_, device=DEV)
            extra = float((e_.astype(np.float64) * (u_ * np.arange(C_)).sum(1)).sum()
                          / max(Xc_.mean() * dim, 1e-12))
            log_fn(f"    [label smoothing/pergene] b={ls_w:g}  e_g 中位={np.median(e_):.2e} "
                   f"max={e_.max():.2e}  被 cap 的基因={int((e_ >= 1e-2).sum())} 个  "
                   f"d_g 中位={np.median(dg):.0f}  **预计额外相对质量 {extra:+.3%}**")
        elif ls_kernel == "uniform_sup":''', "eps-block-head")

m = sub1(m, '''        else:
            raise ValueError(f"未知的 ls_kernel: {ls_kernel}")
        ls_u = torch.tensor(u_, device=DEV)
        extra = float(ls_w * (u_ * np.arange(C_)).sum()
                      / max(np.clip(np.asarray(train), 0, CFG.NMAX).mean() * dim, 1e-12))
        log_fn(f"    [label smoothing] eps={ls_w:g}  kernel={ls_kernel}  "
               f"d_g 中位={np.median(dg):.0f}  **预计额外相对质量 {extra:+.3%}**")''',
         '''        else:
            raise ValueError(f"未知的 ls_kernel: {ls_kernel}")
        ls_u = torch.tensor(u_, device=DEV)
        if ls_e is None:
            extra = float(ls_w * (u_ * np.arange(C_)).sum()
                          / max(np.clip(np.asarray(train), 0, CFG.NMAX).mean() * dim, 1e-12))
            log_fn(f"    [label smoothing] eps={ls_w:g}  kernel={ls_kernel}  "
                   f"d_g 中位={np.median(dg):.0f}  **预计额外相对质量 {extra:+.3%}**")''', "eps-block-tail")

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
                    # 逐基因 eps：逐元素 (1-e_g)*nll + e_g*(-sum u_g log p)，再对 B*dim 求均值。
                    # e_g 为常数向量时与上面的 scalar 公式完全相同。
                    ef_ = ls_e.unsqueeze(0).expand(x0.shape[0], dim).reshape(-1).float()
                    nll_ = F.nll_loss(lp_, tg_, reduction="none")
                    loss = ((1.0 - ef_) * nll_ + ef_ * (-(uf_ * lp_).sum(-1))).mean()''', "loss")

(HERE / "model_hvg2k_lspg.py").write_text(m, encoding="utf-8")

# ---------------- train ----------------
t = (HERE / "train_pilot_ls.py").read_text(encoding="utf-8")
t = sub1(t, "import model_hvg2k_ls as M", "import model_hvg2k_lspg as M", "import")
t = sub1(t, 'choices=["uniform_sup", "uniform"],',
         'choices=["uniform_sup", "uniform", "pergene_sup"],', "choices")
t = sub1(t, '"""[由 gen_ls_train.py 从 train_pilot.py 逐字生成，只加了 --label-smooth / --ls-kernel。]',
         '"""[由 gen_lspg.py 从 train_pilot_ls.py 逐字生成，--ls-kernel 增加 pergene_sup（逐基因 eps，'
         'label-smooth 解读为总预算 b）。]', "doc")
(HERE / "train_pilot_lspg.py").write_text(t, encoding="utf-8")
print("generated model_hvg2k_lspg.py / train_pilot_lspg.py")
