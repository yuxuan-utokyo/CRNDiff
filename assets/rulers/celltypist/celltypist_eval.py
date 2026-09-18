# -*- coding: utf-8 -*-
"""celltypist_eval.py —— 用 **CellTypist** 评价条件生成的纯度，好跟已发表的数字对齐。

为什么换分类器
--------------
scDiffusion 报的是 **CellTypist 分类准确率：全类型平均 0.93**（真实细胞 0.98）。
我们本地那个 joblib 分类器给的是另一套刻度，两个数不能直接摆在一起。
换成同一个公开预训练分类器，我们的数字就能与已发表的 0.93 **直接对齐**，
不需要重跑别人的模型。

★ 必须三行一起报（这一条是整个脚本的关键）
------------------------------------------
我们的数据只有 **2000 个 HVG**，而 CellTypist 的模型是在全基因集上训的。
缺基因会让准确率整体下降 —— 我们的数会被这个 handicap 压低，别人的不会。

  所以**每一次都要跑"真实 val 细胞"那一行**：它承受完全相同的 handicap，
  于是「我们 / 真实细胞」这个比值把 handicap 约掉了，才是能跟
  scDiffusion 的「0.93 / 0.98」对比的量。**只报绝对值会误导。**

三行：空操作（无条件生成）/ 条件生成 / 真实 val 细胞。

口径
----
* CellTypist 要求输入是 **normalize_total(1e4) + log1p**（与它训练时一致），
  这与我们本地 joblib 分类器吃原始计数**不同**，两套不要混用。
* `majority_voting=False`：要逐细胞的判定。开了多数投票会按邻域抹平，把纯度虚高。

★ 细分标签必须**聚合回粗类**（2026-08-16 踩过）
------------------------------------------------
`Healthy_Adult_Heart.pkl` 预测的是亚型：内皮被拆成 EC2_cap / EC3_cap / EC5_art /
EC6_ven …… 第一版脚本拿"真实细胞里占比最高的那个标签"当目标，挑中了 EC5_art，
于是真实细胞自己也只有 0.254 —— **那不是纯度，那是某一个亚型的占比。**

聚合规则**从真实细胞学出来，不靠前缀猜**：对每个 CellTypist 标签 L，看 val 里
带标签 L 的真细胞中哪个粗类最多，就把 L 归给那个粗类。这样映射是数据定的、可审计的，
而且脚本会把映射打出来。

★ 弱支撑标签要单独报（2026-08-16 第二次踩过）
------------------------------------------------
多数投票在**样本极少**的标签上就是噪声：`Adip4` 只被判到几十个真实细胞，多数
归属却成了 Endothelial，于是它被算进靶标集合，白送所有行一点比例。
所以这一版：
* 打印每个靶标签的**支撑细胞数**和**多数占比**；
* 支撑 < `--min-support` 或多数占比 < `--min-major` 的标签标为**弱**；
* 纯度**报两个数**：`purity`（含弱标签）与 `purity_strict`（剔除弱标签）。
  两个数差得大，就说明结论建立在噪声映射上，不能用。
* 每一行的**完整标签分布**都存进 json，以后审计不用重跑。
* 模型名可用 `--list-models` 查；心脏相关的优先（数据集就是 Heart Cell Atlas）。

跑法
----
    python celltypist_eval.py --list-models
    python celltypist_eval.py --type Endothelial --arms a.npy,b.npy --model <名字>

红线：只读，不改任何已有产物；纯 CPU。
"""
import argparse, json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XR = Path(__file__).resolve().parents[3] / "assets" / "legacy_root"


def to_adata(X, genes):
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(np.asarray(X, dtype=np.float32))
    a.var_names = list(genes)
    a.var_names_make_unique()
    # CellTypist 的输入口径：每细胞归一到 1e4 再 log1p（与它训练时一致）
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-root", default=str(XR))
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--model", default="Healthy_Adult_Heart.pkl")
    ap.add_argument("--type", default=None, help="目标细胞类型（CellTypist 里的写法可能不同）")
    ap.add_argument("--arms", default=None, help="条件生成产物，逗号分隔")
    ap.add_argument("--pilot", default=None)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--n-map", type=int, default=20000,
                    help="用多少真实 val 细胞去学「细分标签 -> 粗类」的映射")
    ap.add_argument("--min-support", type=int, default=50,
                    help="一个细分标签至少要被判到这么多真实细胞，映射才算可信")
    ap.add_argument("--min-major", type=float, default=0.5,
                    help="多数粗类至少要占这么大比例，映射才算可信")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("-o", "--out", default="celltypist")
    # ★ 2026-08-17：json 文件名加后缀。此前每次运行都写 celltypist_<type>.json，
    #   后跑的单臂会**覆盖**先跑的（γ=0.85 那次覆盖了 γ=0.4 的 0.6910，记账对不上）。
    #   只改文件名，不碰任何数值路径。
    ap.add_argument("--tag", default="",
                    help="附加到 json 文件名：celltypist_<type><tag>.json，避免多次运行互相覆盖")
    a = ap.parse_args()

    import celltypist
    from celltypist import models

    if a.list_models:
        models.download_models(force_update=False)
        print(models.models_description().to_string())
        return

    R = Path(a.x_root); D = R / "_shared" / "data" / "all"
    OUT = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    genes = json.loads((D / "meta.json").read_text(encoding="utf-8"))["genes"]
    print(f"[genes] {len(genes)} 个 HVG，前 5：{genes[:5]}")

    models.download_models(model=[a.model], force_update=False)
    mdl = models.Model.load(model=a.model)
    print(f"[model] {a.model}  可预测 {len(mdl.cell_types)} 类")
    ov = len(set(genes) & set(mdl.features))
    print(f"[genes] 与模型特征的交集 {ov}/{len(genes)}"
          f"（模型共 {len(mdl.features)} 个特征）"
          f"  ← 只有 2000 个 HVG，缺基因会压低所有行，靠'真实细胞'那一行约掉")

    dists = {}

    def annotate(X, tag):
        res = celltypist.annotate(to_adata(X, genes), model=a.model,
                                  majority_voting=False)
        lab = res.predicted_labels["predicted_labels"].to_numpy().astype(str)
        u, c = np.unique(lab, return_counts=True)
        top = sorted(zip(c / c.sum(), u), reverse=True)[:4]
        print(f"  [{tag}] 前几类：" + "  ".join(f"{n} {p:.3f}" for p, n in top))
        dists[tag] = {str(k): int(v) for k, v in zip(u, c)}   # 全分布存下来备审
        return lab

    jobs = []
    pilot = Path(a.pilot) if a.pilot else (
        R / "Baseline" / "OURS" / "results" /
        "linspace_T_O_K32_spilot_run1c_s20260625_samp20260902_batched1_chunk256_F2_K32_base_noa4.npy")
    jobs.append(("空操作(无条件)", pilot))
    for f in (a.arms or "").split(","):
        if f.strip():
            jobs.append((Path(f).name[:60], Path(f.strip())))

    Xv = np.load(D / "val.npy", mmap_mode="r")
    yv = np.load(D / "val_celltype.npy", allow_pickle=True)

    out = {"model": a.model, "n_genes_overlap": int(ov), "target": a.type, "rows": []}
    labels = {}
    for tag, f in jobs:
        X = np.load(f, mmap_mode="r")
        i = np.sort(rng.choice(X.shape[0], size=min(a.n, X.shape[0]), replace=False))
        labels[tag] = annotate(np.asarray(X[i], dtype=np.float64), tag)

    if a.type:
        # ---- 先在**真实 val 的全部细胞**上跑一遍，学出「细分标签 -> 粗类」的映射 ----
        m = np.sort(rng.choice(Xv.shape[0], size=min(a.n_map, Xv.shape[0]),
                               replace=False))
        lab_map = annotate(np.asarray(Xv[m], dtype=np.float64), "真实 val（建映射用）")
        coarse = yv[m].astype(str)
        fine2coarse, support = {}, {}
        for L in np.unique(lab_map):
            sel = lab_map == L
            u, c = np.unique(coarse[sel], return_counts=True)
            k = int(np.argmax(c))
            fine2coarse[str(L)] = str(u[k])                  # 多数归属
            support[str(L)] = {"n": int(sel.sum()),
                               "major_frac": float(c[k] / c.sum())}
        tgt_labels = sorted(L for L, cc in fine2coarse.items() if cc == a.type)
        if not tgt_labels:
            raise SystemExit(f"没有任何 CellTypist 标签的多数归属是 {a.type}，停下来看映射")
        strong = [L for L in tgt_labels
                  if support[L]["n"] >= a.min_support
                  and support[L]["major_frac"] >= a.min_major]
        weak = [L for L in tgt_labels if L not in strong]
        print(f"\n[映射] '{a.type}' 对应 {len(tgt_labels)} 个 CellTypist 标签")
        print(f"  {'标签':<20}{'支撑细胞数':>10}{'多数占比':>10}   判定")
        for L in tgt_labels:
            s = support[L]
            print(f"  {L:<20}{s['n']:>10d}{s['major_frac']:>10.3f}   "
                  f"{'强' if L in strong else '弱(会被 strict 剔除)'}")
        if not strong:
            raise SystemExit("没有一个强支撑的靶标签，映射不可信，停下来看")
        out["fine_to_coarse"] = fine2coarse
        out["label_support"] = support
        out["target_labels"] = tgt_labels
        out["target_labels_strong"] = strong
        out["target_labels_weak"] = weak

        ii = np.flatnonzero(yv == a.type)
        s = np.sort(rng.choice(ii, size=min(a.n, len(ii)), replace=False))
        labels["真实 val 细胞"] = annotate(np.asarray(Xv[s], dtype=np.float64),
                                          "真实 val 细胞")
        # 稀有类型的"真实细胞"那一行可能只有几十个细胞（Mesothelial val 只有 71 个），
        # 纯度的标准误就有 0.03 以上。**n 必须跟着数一起报**，否则上限看起来像个硬数。
        print(f"\n{'群体':<44}{'n':>7}{'纯度(含弱)':>12}{'±s.e.':>9}{'纯度(strict)':>14}")
        for tag in labels:
            p = float(np.isin(labels[tag], tgt_labels).mean())
            ps = float(np.isin(labels[tag], strong).mean())
            n = int(len(labels[tag]))
            se = float(np.sqrt(max(p * (1 - p), 0.0) / max(n, 1)))
            out["rows"].append({"arm": tag, "n": n, "purity": p,
                                "purity_se": se, "purity_strict": ps})
            print(f"{tag:<44}{n:>7d}{p:>12.4f}{se:>9.4f}{ps:>14.4f}")
        base = out["rows"][0]; ceil = out["rows"][-1]
        d = max(abs(r["purity"] - r["purity_strict"]) for r in out["rows"])
        print(f"\n[弱标签敏感性] 两列最大差 {d:.4f}"
              + ("   ← 结论不依赖弱映射" if d < 0.01 else
                 "   ← 差得不小，弱映射在贡献比例，以 strict 那列为准"))
        print(f"\n参照：scDiffusion 报 CellTypist 平均 0.93，其真实细胞 0.98"
              f"（占上限 {0.93/0.98:.0%}）")
        for r in out["rows"][1:-1]:
            print(f"  {r['arm'][:44]:<44} 占上限 "
                  f"{r['purity']/max(ceil['purity'],1e-9):.0%}"
                  f" / strict {r['purity_strict']/max(ceil['purity_strict'],1e-9):.0%}"
                  f"（空操作 {base['purity']/max(ceil['purity'],1e-9):.0%}）")
    out["label_dists"] = dists
    (OUT / f"celltypist_{a.type or 'all'}{a.tag}.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {OUT}")


if __name__ == "__main__":
    main()
