# -*- coding: utf-8 -*-
"""从 `model_hvg2k.py` / `train_pilot.py` **逐字生成**带 label smoothing 的副本。

    python gen_ls_train.py      ->  model_hvg2k_ls.py  +  train_pilot_ls.py

放在 `Experiments/X/Baseline/OURS/code/` 下跑。每处替换都断言**只命中一次**，
所以"除了这个开关什么都没动"是可机器验证的（`gen_noa4.py` 同一套做法）。
红线：`model_hvg2k.py` / `train_pilot.py` / `vendor_ours*.py` / `bank_dedup.py`
一个字节不动。

改什么
------
`model_hvg2k.py:188` 那一行硬标签 CE：

    loss = F.cross_entropy(logits.reshape(-1, NMAX+1), x0.reshape(-1))

换成对软标签的 CE：

    q = (1-eps) * onehot(x0) + eps * u_g          loss = -(q * log_softmax).sum(-1).mean()

`--label-smooth 0`（默认）时**完全不进新分支**，与 `model_hvg2k.py` 逐位相同。

为什么（2D 玩具上已经测死）
---------------------------
朴素 CE 在经验频率为 0 的 count 上只会把 logit 往下压、从不往上拉，logit 跑向 -inf。
玩具上实测生成边缘在那些 count 上比真值低 1e7~1e9 倍，而任何**有界**的端点重加权
都补不动（w_clip=3、9 轮 -> 动态范围上限 2e4）。加了 label smoothing (eps=3e-4,
uniform_sup) 之后，同一个模型同一套 PPC：

    零观测点的中位偏离   10^7.29  ->  10^0.35
    R_mass               1.0062   ->  1.0029      （变好）
    稀有档质量比         1.196    ->  1.046       （该档估计极限正是 1.046，精确命中）
    高三档               0.998/1.015/1.076 -> 0.999/1.005/1.024   （全部变好）

裸模型的稀有档 0.640 -> 0.639（**没变**）—— label smoothing 修洞，PPC 修稀有区
欠生成，两件事正交，互不替代。

为什么核是 uniform_sup 而不是 uniform
-------------------------------------
`uniform` 的 u(n)=1/C 是常数，于是零观测处的最优解也是常数 —— 生成边缘在 d_g 之上
变成一条**水平直线**，而真值在那一段一路衰减，远端高好几个数量级。
`uniform_sup` 只在 n<=d_g 上铺地板，d_g 之上给的训练信号与纯 CE 完全一样。

hvg2k 上它还便宜得多。额外相对质量 = eps * sum_g E_u_g[n] / sum_g E_true_g[n]：

    逐基因 d_g 分位数 [0,25,50,75,90,99,100] = [1, 17, 32, 68, 138, 512, 512]
    uniform      额外质量 = eps * 695.7   ->  2% 预算下 eps <= 2.9e-5
    uniform_sup  额外质量 = eps *  84.8   ->  2% 预算下 eps <= 2.4e-4

中位 d_g 只有 32，E_u = d_g/2 ≈ 16，比 uniform 的 256 便宜 8 倍 ——
**所以玩具上调出来的 eps=3e-4 基本可以直接搬**。建议跑 1e-4 / 3e-4。

⚠ 必须配对训练
--------------
`pilot_run1c` 是 run1 -> run1b -> run1c **三段续训**出来的（step_offset=7800，
每段优化器与 EMA 冷启、warmup/cosine 重来）。拿一个从头训的 LS 模型去和它比，
差异里混着"训练调度不同"。**必须用同一个脚本、同一步数、同一种子各训一个**，
只差 `--label-smooth`。s_per_step≈0.44（batch 384），12000 步约 88 分钟。

    python train_pilot_ls.py --steps 12000 --label-smooth 0     --name cectrl
    python train_pilot_ls.py --steps 12000 --label-smooth 3e-4  --name ls3e4

val 的评估口径**不动**（一律纯 CE），这样两个臂的 val 曲线仍然可以直接并排比 ——
改了训练目标就改评估尺子，两条曲线就没法比了。
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

    # 在 model 建好之后、训练循环之前，把 back-off 分布 u 备好
    ('''    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)''',
     '''    # ==== label smoothing 的 back-off 分布 u_g，[dim, NMAX+1]，不参与反传 ====
    ls_w, ls_u = float(label_smooth), None
    if ls_w > 0.0:
        C_ = CFG.NMAX + 1
        dg = np.clip(np.asarray(train), 0, CFG.NMAX).max(0).astype(int)
        u_ = np.zeros((dim, C_), np.float32)
        if ls_kernel == "uniform_sup":
            # 只在观测支撑内铺地板；d_g 之上给的训练信号与纯 CE 完全一样。
            for d_ in range(dim):
                u_[d_, :dg[d_] + 1] = 1.0 / (dg[d_] + 1)
        elif ls_kernel == "uniform":
            u_[:] = 1.0 / C_
        else:
            raise ValueError(f"未知的 ls_kernel: {ls_kernel}")
        ls_u = torch.tensor(u_, device=DEV)
        extra = float(ls_w * (u_ * np.arange(C_)).sum()
                      / max(np.clip(np.asarray(train), 0, CFG.NMAX).mean() * dim, 1e-12))
        log_fn(f"    [label smoothing] eps={ls_w:g}  kernel={ls_kernel}  "
               f"d_g 中位={np.median(dg):.0f}  **预计额外相对质量 {extra:+.3%}**")
        log_fn(f"      —— 这是 count 加权的质量代价，远尾一点点概率就是大质量；"
               f"eps 的预算就是按它定的（2% 对应 eps<=2.4e-4）")
    else:
        log_fn("    [label smoothing] 关（eps=0，与 model_hvg2k.py 逐位相同）")
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)''', 1),

    # 训练那一步的 loss
    ('''        with amp:
            logits = model(n_t, t)
            loss = F.cross_entropy(logits.reshape(-1, CFG.NMAX + 1),
                                   x0.long().clamp(0, CFG.NMAX).reshape(-1))''',
     '''        with amp:
            logits = model(n_t, t)
            if ls_u is None:
                # eps=0：**原样**，与 model_hvg2k.py 逐位相同
                loss = F.cross_entropy(logits.reshape(-1, CFG.NMAX + 1),
                                       x0.long().clamp(0, CFG.NMAX).reshape(-1))
            else:
                # q = (1-eps)*onehot + eps*u_g  =>  loss = -(q * log_softmax).sum(-1).mean()
                # .float() 与 F.cross_entropy 在 autocast 下的行为一致（它本来就上浮到 fp32）
                lp_ = F.log_softmax(logits.reshape(-1, CFG.NMAX + 1).float(), -1)
                tg_ = x0.long().clamp(0, CFG.NMAX).reshape(-1)
                uf_ = ls_u.unsqueeze(0).expand(x0.shape[0], dim, CFG.NMAX + 1) \\
                          .reshape(-1, CFG.NMAX + 1)
                loss = ((1.0 - ls_w) * F.nll_loss(lp_, tg_)
                        - ls_w * (uf_ * lp_).sum(-1).mean())''', 1),

    ('''"""hvg2k · OURS —— 数据 / V / 网络 / 训练。新文件；不改 `_shared/code/` 与 61gene 的任何东西。''', '''"""[由 gen_ls_train.py 从 model_hvg2k.py 逐字生成，唯一改动 = 训练 loss 支持
label smoothing；label_smooth=0 时与原文件逐位相同。val 评估口径**未改**，仍是纯 CE，
所以不同 eps 的 val 曲线可以直接并排比。]

hvg2k · OURS 的模型''', 1),
]

# ===================== train_pilot.py -> train_pilot_ls.py =====================
T_REPL = [
    ('import model_hvg2k as M                                 # noqa: E402',
     'import model_hvg2k_ls as M                           # noqa: E402  ← 唯一的 import 改动', 1),

    ('''    ap.add_argument("--smoke", action="store_true")''',
     '''    ap.add_argument("--label-smooth", type=float, default=0.0,
                    help="★ 训练端 back-off。0 = 与 train_pilot.py 逐位相同。"
                         "hvg2k 的 eps 预算：uniform_sup 下 2%% 额外质量对应 2.4e-4，"
                         "建议试 1e-4 / 3e-4（玩具最优是 3e-4）")
    ap.add_argument("--ls-kernel", default="uniform_sup",
                    choices=["uniform_sup", "uniform"],
                    help="uniform 会在 d_g 之上留一条水平直线，且在 hvg2k 上贵 8 倍")
    ap.add_argument("--smoke", action="store_true")''', 1),

    # ⚠ 必须追加在**末尾**：前面 tr/va/... 是位置参数，关键字参数插在它们之前是语法错误
    ('''            resume_from=a.resume)''',
     '''            resume_from=a.resume,
            label_smooth=a.label_smooth, ls_kernel=a.ls_kernel)''', 1),

    ('''"""hvg2k · OURS pilot 训练。''',
     '''"""[由 gen_ls_train.py 从 train_pilot.py 逐字生成，只加了 --label-smooth / --ls-kernel。]

⚠ **必须配对训练**：`pilot_run1c` 是三段续训出来的（step_offset=7800，每段优化器与
EMA 冷启），拿从头训的 LS 模型和它比会混进"训练调度不同"。同脚本同步数同种子各训一个：

    python train_pilot_ls.py --steps 12000 --label-smooth 0     --name cectrl
    python train_pilot_ls.py --steps 12000 --label-smooth 3e-4  --name ls3e4

s_per_step≈0.44（batch 384），12000 步约 88 分钟／臂。

hvg2k · OURS pilot 训练。''', 1),
]

JOBS = [("model_hvg2k.py", "model_hvg2k_ls.py", M_REPL),
        ("train_pilot.py", "train_pilot_ls.py", T_REPL)]


def main():
    import ast
    for src_n, dst_n, repl in JOBS:
        src, dst = HERE / src_n, HERE / dst_n
        assert src.exists(), f"找不到 {src}"
        s = src.read_text(encoding="utf-8")
        print(f"\n{src_n} -> {dst_n}")
        for i, (old, new, want) in enumerate(repl, 1):
            got = s.count(old)
            assert got == want, (f"  替换 {i} 命中 {got} 次，期望 {want} —— 拒绝生成。"
                                 f"（源文件可能已改动，请人工核对）")
            s = s.replace(old, new, want)
            print(f"  替换 {i}: 命中 {got} 次  OK")
        ast.parse(s)
        dst.write_text(s, encoding="utf-8")
        print(f"  已写出 {dst.name}（语法检查通过）")
    print("\n下一步（配对训练，各约 88 分钟）：")
    print("  python train_pilot_ls.py --steps 12000 --label-smooth 0    --name cectrl")
    print("  python train_pilot_ls.py --steps 12000 --label-smooth 3e-4 --name ls3e4")
    print("先用 --smoke 各跑 200 步确认 loss 在降、显存不炸。")


if __name__ == "__main__":
    main()
