# -*- coding: utf-8 -*-
"""从 `sample_hvg2k_la.py` **逐字生成** `sample_hvg2k_noa4.py`，只加一个 `--no-a4` 开关。

为什么要有这个脚本而不是手改：`sample_hvg2k_la.py` 本身就是这么从 `sample_hvg2k.py`
生成的（见它的文件头）。每一处替换都断言**只命中一次**，于是"除了这个开关什么都没动"
是可机器验证的，不靠人眼 review。红线 6.1：`sample_hvg2k.py` / `model_hvg2k.py` /
`bank_dedup.py` / `vendor_ours*.py` / `sample_hvg2k_la.py` 一个字节不动。

不给 `--no-a4` 时，整条数值路径与 `sample_hvg2k_la.py` **逐位相同**
（`if not _NO_A4:` 为真 -> 执行原来那一行 masked_fill，后续归一化原样保留）。

    python gen_noa4.py            # 写出 sample_hvg2k_noa4.py 并打印每处替换的命中次数
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "sample_hvg2k_la.py"
DST = SRC.with_name("sample_hvg2k_noa4.py")

REPL = [
    # ---- 1. 文件头 ----
    ('"""hvg2k 路 OURS 采样 + **logit adjustment（LA）**。T3 层。',
     '''"""hvg2k · OURS 采样 + LA/PPC + **可关闭的 A4**。由 `gen_noa4.py` 从
`sample_hvg2k_la.py` 逐字生成，唯一的改动是给 A4 那一行加了个开关。

    # 无 A4 + PPC（本轮要补的臂）
    python sample_hvg2k_noa4.py --ckpt <ckpt> --no-a4 --tau 0.5 \\
        --logpi <ppc_logc>.npy --last-only --ks 32 --n-gen 20000 --cell-chunk 500

    # 无 A4 + 裸模型（同 K 的对照，**这一支才是证明 A4 原本在干活的那个数**）
    python sample_hvg2k_noa4.py --ckpt <ckpt> --no-a4 --tau 0 --ks 32 ...

## 为什么要做这个消融

A4（`p.masked_fill(n > d_g, 0)`）是**采样时的硬截断**，它先验地规定模型不许在
"训练集里从没见过的 count"上放质量。它当初的经验依据是 D=10 多基因玩具上
**裸模型**无 A4 时总质量 3.909 倍 —— 但那个结论**早于 PPC**。同一张表里，
一旦开了重加权，无 A4 的 R_mass 就回到 1.142（LA-0.5）和 0.960（LA-1.0+WEIGHT-0.5）。
2D 玩具上更彻底：A4 开/关逐位相同，因为模型在 d_g 之上本来就只放 1e-13。

所以真正该问的是：**PPC 按实测压制之后，A4 这条先验规定还有没有必要。**
需要四臂（{A4 开/关} × {PPC 开/关}）才能回答，缺了"无 A4 + 裸模型"那一支，
"A4 冗余"就只是"A4 从来没用过"。

## 口径

`--no-a4` 会自动往 `--tag` 后面追加 `_noa4`，产物文件名必带这一段，不可能与
已有产物重名（红线：不覆盖任何已有 .npy）。`no_a4` 同时写进结果 json 和 RUN_LOG。
`n_gen` / `cell_chunk` / `batched` / seed 都是口径的一部分，**必须与被对比的臂逐字一致**。

------------------------------------------------------------------------
以下为 `sample_hvg2k_la.py` 的原始说明（逐字保留）：

hvg2k 路 OURS 采样 + **logit adjustment（LA）**。T3 层。''', 1),

    # ---- 2. 全局开关 ----
    ("@torch.no_grad()\ndef sample_chunk(",
     "_NO_A4 = False        # 由 main() 按 --no-a4 覆盖；False 时数值路径与 la 版逐位相同\n\n\n"
     "@torch.no_grad()\ndef sample_chunk(", 1),

    # ---- 3. A4 那一行加开关 ----
    ('''        p = p.masked_fill(torch.arange(p.shape[2], device=dev)[None, None, :]
                          > dmax_t[None, :, None], 0.0)          # A4 posterior-support 截断''',
     '''        # ==== A4 posterior-support 截断（本文件唯一的改动：加了个开关）==========
        # `--no-a4` 时**不执行这一行**，模型可以在 n > d_g 上保留它自己的输出。
        # 后面的归一化原样保留（不加 A4 时 p 已经是归一的，这一步是恒等操作），
        # 这样 `_NO_A4 = False` 的数值路径与 `sample_hvg2k_la.py` 逐位相同。
        if not _NO_A4:
            p = p.masked_fill(torch.arange(p.shape[2], device=dev)[None, None, :]
                              > dmax_t[None, :, None], 0.0)
        # ======================================================================''', 1),

    # ---- 4. 命令行参数 ----
    ('''    ap.add_argument("--last-only", action="store_true",''',
     '''    ap.add_argument("--no-a4", action="store_true",
                    help="★ 关掉 A4 支撑截断（不再把 n > d_g 硬置零）。会自动往 --tag "
                         "追加 _noa4，产物不可能与已有文件重名。读法：\\n"
                         "  裸模型显著变坏 + PPC 臂基本不变 => A4 冗余，可以去掉\\n"
                         "  PPC 臂也变坏                    => A4 不可替代，留着")
    ap.add_argument("--last-only", action="store_true",''', 1),

    # ---- 5. main() 里设置全局 + 追加 tag ----
    ('''    global _LA_TAU, _LA_PI_TAU, _LA_LAST_ONLY
    _LA_TAU = float(a.tau)''',
     '''    global _LA_TAU, _LA_PI_TAU, _LA_LAST_ONLY, _NO_A4
    _NO_A4 = bool(a.no_a4)
    if _NO_A4:
        a.tag = (a.tag or "") + "_noa4"      # 产物名必带，杜绝与已有臂重名
        rl("  [A4] **已关闭** —— 后验不再按 n <= d_g 截断；tag 自动追加 _noa4")
    else:
        rl("  [A4] 开（与 sample_hvg2k_la.py 逐位相同的数值路径）")
    _LA_TAU = float(a.tau)''', 1),

    # ---- 6. 结果 json 记下这一位 ----
    ('''    out = {"ckpt": Path(a.ckpt).name, "T": T, "NMAX": nmax, "dim": dim,''',
     '''    out = {"ckpt": Path(a.ckpt).name, "T": T, "NMAX": nmax, "dim": dim,
           "no_a4": bool(a.no_a4), "tau": float(a.tau), "last_only": bool(a.last_only),''', 1),

    # ---- 7. 开跑那一行的日志带上 A4 状态 ----
    ('''    rl(f"### hvg2k 采样开始 ckpt={Path(a.ckpt).name} K={a.ks} samp={a.samps} "''',
     '''    rl(f"### hvg2k 采样开始 [A4={'off' if a.no_a4 else 'on'}] "
       f"ckpt={Path(a.ckpt).name} K={a.ks} samp={a.samps} "''', 1),
]


def main():
    s = SRC.read_text(encoding="utf-8")
    for i, (old, new, want) in enumerate(REPL, 1):
        got = s.count(old)
        assert got == want, f"替换 {i} 命中 {got} 次，期望 {want} —— 拒绝生成"
        s = s.replace(old, new, want)
        print(f"  替换 {i}: 命中 {got} 次  OK")
    if DST.exists():
        print(f"[warn] {DST.name} 已存在，覆盖（它是生成物，不是手写代码）")
    DST.write_text(s, encoding="utf-8")
    import ast
    ast.parse(s)
    print(f"\n已写出 {DST}  （语法检查通过）")
    print("自检：不给 --no-a4 时应与 sample_hvg2k_la.py 逐位相同 —— "
          "差异行数应当只覆盖上面 7 处")


if __name__ == "__main__":
    main()
