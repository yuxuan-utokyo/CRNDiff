# -*- coding: utf-8 -*-
"""hvg2k · OURS —— 数据 / V / 网络 / 训练。新文件；不改 `_shared/code/` 与 61gene 的任何东西。

网络直接用 `_shared/code/backbone.py::AttnX0Posterior`（与 61gene 逐字同一个类）。
训练循环照 `EXP21::train_ours_generic` 搬，改动逐条写在 `train_ours_hvg2k` 的 docstring 里。
"""
from __future__ import annotations

import contextlib
import copy
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent                  # .../hvg2k/OURS/code
sys.path.insert(0, str(HERE))
import config_hvg2k as CFG                              # noqa: E402

# 077 债一(2026-08-24):这里原本**自己又算了一遍** `ROOT` 与 `G61`
#     ROOT = Path(__file__).resolve().parents[3]
#     G61  = ROOT.parent / "_shared" / "panel61"
# 与 config_hvg2k 里的同名常量是**两份互不相干的副本**。目录搬家后两份一起失效,
# 而只修其中一份不会让另一份跟着变——这正是它们当初能各自漂掉的原因。
# 改为**从 CFG 取唯一真源**,以后只有一个地方需要维护。
ROOT = CFG.ROOT
G61 = ROOT.parent / "_shared" / "panel61"                            # 只读

# 复用 61gene 冻结的 backbone（只读 import，不写那边任何文件）
PKG = "panel61_shared"
SHARED61 = G61 / "_shared" / "code"
if PKG not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        PKG, SHARED61 / "__init__.py", submodule_search_locations=[str(SHARED61)])
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules[PKG] = _pkg
    _spec.loader.exec_module(_pkg)
    importlib.import_module(f"{PKG}.seeds").DATA_DIR = G61 / "_shared" / "data"
    sys.modules.setdefault("_shared", sys.modules[PKG])
BB = importlib.import_module(f"{PKG}.backbone")
DEV = BB.DEV


# =========================================================================== #
# 数据
# =========================================================================== #
def quantize_log2_V(mu, lo=CFG.V_GRID_LO, hi=CFG.V_GRID_HI):
    """把逐基因均值量化到 2^lo … 2^hi 的 log2 网格（最近档）。见 config 的 B2 判据。"""
    lv = 2.0 ** np.arange(lo, hi + 1, dtype=np.float64)
    m = np.maximum(np.asarray(mu, np.float64), lv[0])
    return lv[np.abs(np.log2(m)[:, None] - np.log2(lv)[None, :]).argmin(1)], lv


def load_data(split=("train", "val")):
    meta = json.loads((CFG.DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    assert meta["gene_hash"] == CFG.GENE_HASH, \
        f"gene_hash 不符：{meta['gene_hash']} != {CFG.GENE_HASH}"      # 红线 §5.3
    out = {s: np.load(CFG.DATA_DIR / f"{s}.npy") for s in split}
    return out, meta


def derive(train):
    """从 train 估的全部派生量。**只从 train 估，val/test 不参与。**"""
    X = train.astype(np.int64)
    mu = X.mean(0)
    V, levels = quantize_log2_V(mu)
    iscale = 1.0 / (1.0 + mu)
    dmax = X.max(0).astype(int)
    if CFG.CLIP_X0_TO_NMAX:
        dmax = np.minimum(dmax, CFG.NMAX)
    return {"mu": mu, "V": V, "levels": levels, "iscale": iscale, "dmax": dmax,
            "n_distinct_V": int(len(np.unique(V))),
            "sum_V": float(V.sum()), "lib_mean": float(X.sum(1).mean()),
            "sum_V_over_lib": float(V.sum() / X.sum(1).mean())}


def horizon():
    return float(CFG.T_O if CFG.T_HORIZON == "T_O" else CFG.T_HORIZON)


def build_grid(K, T=None):
    """linspace(T, F_BASE, K)，末尾补 0（与 model.py::gfull_nfe 同一约定）。"""
    T = horizon() if T is None else T
    g = np.linspace(T, CFG.F_BASE, K)
    gf = np.concatenate([np.sort(np.unique(g[g > 1e-9]))[::-1], [0.0]])
    return gf, int(len(gf) - 1)


class AttnX0PosteriorCkpt(BB.AttnX0Posterior):
    """与 `AttnX0Posterior` **参数完全相同**的子类，只在 forward 里对 3 层 transformer
    逐层做梯度检查点。父类文件 `_shared/code/backbone.py` 是只读的，一个字节没动；
    参数名与形状不变，所以两者的 state_dict 可以互相加载（采样时用父类即可）。

    只在 `self.training and CFG.USE_GRAD_CKPT` 时走检查点；推理/采样时与父类逐字相同。
    """

    def forward(self, n_t, t):
        if not (self.training and CFG.USE_GRAD_CKPT):
            return super().forward(n_t, t)
        import torch.utils.checkpoint as _cp
        if t.dim() == 1:
            t = t.unsqueeze(1)
        sc = (n_t * self.input_scale).unsqueeze(-1)
        lg = (torch.log1p(n_t) / self._lognmax).unsqueeze(-1)
        tok = self.count_proj(torch.cat([sc, lg], -1))
        tok = tok + self.gene_emb(self.gene_ids)[None]
        tok = tok + self.time_mlp(t)[:, None, :]
        h = tok
        for layer in self.transformer.layers:            # nn.TransformerEncoder.layers
            h = _cp.checkpoint(layer, h, use_reentrant=False)
        if self.transformer.norm is not None:
            h = self.transformer.norm(h)
        return self.head(self.norm(h))


def build_net(dim, iscale, ckpt=False):
    cls = AttnX0PosteriorCkpt if ckpt else BB.AttnX0Posterior
    return cls(CFG.NMAX, dim, d_model=CFG.D_MODEL, nheads=CFG.NHEADS,
               nlayers=CFG.NLAYERS, input_scale=iscale)


# =========================================================================== #
# 训练
# =========================================================================== #
class DivergedError(RuntimeError):
    """工单 214-A 的发散熔断：训练已塌掉，调用方必须非零退出、不得进入下游。"""


def train_ours_hvg2k(train, val, V, iscale, seed, steps, *, batch=None, log_fn=print,
                     eval_every=CFG.EVAL_EVERY, patience=CFG.PATIENCE, t_max=None,
                     ckpt_path=None, bf16=None, grad_ckpt=None, resume_from=None,
                     clip_grad=None, lr=None, warmup=None, diverge_breaker=True):
    """`EXP21::train_ours_generic` 的 hvg2k 版。逐条改动：

    1. `T_MAX` 从 `ours.T_MAX`（8.0）换成本数据集的视界（默认 T_O=5.2021）；
       `T_FLOOR` 仍是 `F_BASE=0.01`。训练时 t 的采样上界**必须**与采样时的视界一致。
    2. `x0` 在喂进二项之前先 clip 到 NMAX（`CFG.CLIP_X0_TO_NMAX`）；
       61gene 是对二项+泊松的**和**才 clamp。理由见 config 的 `CLIP_X0_TO_NMAX`。
    3. batch 从 1024 降到 256：logits 是 [B, 2000, 513]，B=1024 单是前向就 4.2 GB。
    4. 加了 early stopping（主线 B5 判据：连续 `patience` 次 val 没改善就停），
       61gene 那版是固定步数跑满。
    5. val 评估用**独立的 CPU Generator**（固定 12345），不消耗训练用的 `g` ——
       EXP143 踩过这个坑（val 评估动了全局 RNG 会改训练轨迹）。

    其余（AdamW 2e-3/1e-4、warmup 500 的 cosine、EMA 0.999、交叉熵目标）逐字未改。
    """
    batch = batch or CFG.BATCH
    T_MAX = horizon() if t_max is None else t_max
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    cap = CFG.NMAX if CFG.CLIP_X0_TO_NMAX else None
    Xtr = torch.tensor(np.clip(train, 0, cap) if cap else train, dtype=torch.float32)
    Xva = torch.tensor(np.clip(val, 0, cap) if cap else val, dtype=torch.float32)
    n, dim = Xtr.shape
    bf16 = CFG.USE_BF16 if bf16 is None else bf16
    grad_ckpt = CFG.USE_GRAD_CKPT if grad_ckpt is None else grad_ckpt
    Vt = torch.tensor(np.asarray(V, np.float64), dtype=torch.float32, device=DEV)
    model = build_net(dim, iscale, ckpt=grad_ckpt).to(DEV)
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if (bf16 and torch.cuda.is_available()) else contextlib.nullcontext())
    log_fn(f"    [训练精度] bf16 autocast={bf16}（权重仍为 fp32）  梯度检查点={grad_ckpt}")
    # 工单 214-A：lr / warmup / grad-clip 变成**可选覆盖**，默认值一律回落到 CFG，
    # 所以不传这三个参数时，本函数与改动前逐字等价（训练轨迹逐位不变）。
    _lr = CFG.LR if lr is None else float(lr)
    _warmup = CFG.WARMUP if warmup is None else int(warmup)
    opt = torch.optim.AdamW(model.parameters(), lr=_lr, weight_decay=CFG.WD)
    sched = BB._cosine_sched(opt, steps, _warmup)
    if (lr is not None) or (warmup is not None) or clip_grad:
        log_fn(f"    [214A 稳定化] lr={_lr} warmup={_warmup} "
               f"clip_grad={clip_grad or 'off'}（默认配方为 lr={CFG.LR} "
               f"warmup={CFG.WARMUP} clip=off）")
    if resume_from is not None:
        sd_ = torch.load(resume_from, map_location=DEV, weights_only=False)
        model.load_state_dict(sd_)
        log_fn(f"    [RESUME] 从 {Path(resume_from).name} 续训 —— "
               f"**优化器状态与 EMA 没有保存，是从权重冷启的**，"
               f"所以 warmup/cosine 调度会重来一遍。这一点写进日志，不要当成无缝续训。")
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    lt, lf = math.log(T_MAX), math.log(CFG.F_BASE)
    curve = {"step": [], "train": [], "val": []}

    def noise(x0, gg):
        u = torch.rand(x0.shape[0], generator=gg).to(DEV)
        t = torch.exp(lf + u * (lt - lf))
        r = torch.exp(-t).unsqueeze(1)
        n_t = (torch.binomial(x0, r.expand_as(x0))
               + torch.poisson(Vt.unsqueeze(0) * (1 - r))).clamp(0, CFG.NMAX)
        return n_t, t

    run_tr, best, bad, stopped = None, float("inf"), 0, steps
    es_ref, es_would_stop = float("inf"), None   # 早停参照点 / 「若启用会停在哪一步」
    for st in range(steps):
        model.train()
        idx = torch.randint(0, n, (batch,), generator=g)
        x0 = Xtr[idx].to(DEV)
        n_t, t = noise(x0, g)
        with amp:
            logits = model(n_t, t)
            loss = F.cross_entropy(logits.reshape(-1, CFG.NMAX + 1),
                                   x0.long().clamp(0, CFG.NMAX).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad:                      # 工单 214-A：本仓库唯一真正缺的那一项
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(EMA := CFG.EMA_DECAY).add_(pm, alpha=1 - EMA)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        lv = float(loss.detach())
        run_tr = lv if run_tr is None else 0.98 * run_tr + 0.02 * lv
        if (st + 1) % eval_every == 0 or st == steps - 1:
            model.eval()
            # ===== 2026-08-10：val 必须是**权重的确定性函数**，且不能动训练 RNG =====
            # 原来只把 `noise` 里的 `torch.rand`（抽 t）交给固定的 CPU `gv`，
            # 但腐蚀本身 —— `torch.binomial(x0, r)` / `torch.poisson(...)` —— 的输入
            # 在 DEV 上，走的是**全局 CUDA RNG**，`gv` 管不到。于是两件事同时发生：
            #   1. 每次评估的腐蚀噪声都不同 -> val 曲线里混进抽样噪声（主线问的就是这个）；
            #   2. val 评估**消耗全局 CUDA RNG** -> 反过来改训练轨迹，
            #      正是 EXP143 那个坑，只是走的是 CUDA 那条流而不是 CPU 那条。
            # 修法照 EXP138：固定 val 子集 + 固定腐蚀种子 + 评估前后**存取全局 RNG 状态**。
            _cpu_rng = torch.get_rng_state()
            _cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(12345)          # 同时给 CPU 与所有 CUDA 设备播种
            with torch.no_grad():
                gv = torch.Generator(device="cpu").manual_seed(12345)
                vi = torch.randint(0, Xva.shape[0], (min(2048, Xva.shape[0]),), generator=gv)
                vx0 = Xva[vi].to(DEV)
                vnt, vt = noise(vx0, gv)
                # ★ **必须分块。** 一次前向 2048 个细胞在 dim=2000 上是 **21.6 GB**
                #   （T2 实测推理 10.8 MB/细胞），比 batch=128 的训练步（8.5 GB）还大 2.5 倍。
                #   2026-08-09 夜里抖死的真正原因就是它，不是 batch 太大 ——
                #   61gene 那份配方里 2048×61 基因微不足道，照抄到 2000 维就成了峰值。
                #   用 reduction="sum" 再除以总元素数，与一次性求 mean **数值完全相同**，
                #   所以 val 曲线与之前的运行仍然可比。
                # val 一律 fp32 评估 —— 两个精度配置的曲线要能直接并排比
                s_, n_ = 0.0, 0
                for lo in range(0, vx0.shape[0], CFG.VAL_CHUNK):
                    hi = lo + CFG.VAL_CHUNK
                    xb, nb, tb_ = vx0[lo:hi], vnt[lo:hi], vt[lo:hi]
                    s_ += float(F.cross_entropy(
                        model(nb, tb_).reshape(-1, CFG.NMAX + 1),
                        xb.long().clamp(0, CFG.NMAX).reshape(-1), reduction="sum"))
                    n_ += xb.numel()
                vl = s_ / max(n_, 1)
                del vx0, vnt, vt
            torch.set_rng_state(_cpu_rng)                       # 训练轨迹一位不动
            if _cuda_rng is not None:
                torch.cuda.set_rng_state_all(_cuda_rng)
            curve["step"].append(st + 1); curve["train"].append(run_tr); curve["val"].append(vl)
            # ================= 早停：2026-08-10 修的逻辑 bug =================
            # 旧代码（把实验 X 的两次训练都砍断了）：
            #     imp = vl < best * (1 - REL)
            #     best, bad = (vl, 0) if imp else (min(best, vl), bad + 1)
            # 错在 **`best` 与「相对阈值的参照点」是同一个变量**：
            #   `min(best, vl)` 让 best 每次刷新都往下棘轮，于是
            #   「刷新了 best、但幅度不到 REL=0.3%」的那一步 —— best 被更新，
            #   bad 却 +1。计数器在破纪录的那一步没有清零。
            #   run1  step 5000 val=0.11510 < best=0.11520（改善 0.087% < 0.3%）-> bad=3/3 停
            #   run1b step 2800 val=0.11408   就是最终 best_val 本身        -> bad=5/5 停
            # 修法：参照点 `ref` 只在**显著**改善时下移，`best` 只管存盘，两者分开。
            imp_any = vl < best                                # 任何新纪录 -> 存 .best
            imp_sig = vl < es_ref * (1.0 - CFG.EARLY_STOP_REL)  # 显著改善 -> 清零计数器
            if imp_sig:
                es_ref, bad = vl, 0
            else:
                bad += 1
            best = min(best, vl)
            # **早停只记录、不执行**（主线 2026-08-10）：第一次触发点写进日志供定预算用。
            if bad >= patience and es_would_stop is None:
                es_would_stop = st + 1
                log_fn(f"    [EARLY-STOP·仅记录] 若启用，第一次触发会在 step {es_would_stop}"
                       f"（patience={patience}, rel={CFG.EARLY_STOP_REL}）—— 本轮不执行，继续训")
            # **每个评估点都落盘**。第一版只在最后保存，2026-08-10 那次 pilot 因显存抖动
            # 被迫在 step 1000 中止时，5.3 小时训练一个产物都没留下。
            if ckpt_path is not None:
                ema.eval()
                torch.save(ema.state_dict(), str(ckpt_path) + ".last")
                if imp_any:
                    torch.save(ema.state_dict(), str(ckpt_path) + ".best")
                ema.train()
            pk_gb = (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else 0
            log_fn(f"    step {st+1:6d}/{steps} train={run_tr:.4f} val={vl:.4f} "
                   f"best={best:.4f} bad={bad}/{patience}"
                   + (f"  [GPU 峰值分配 {pk_gb:.1f} GB]" if torch.cuda.is_available() else ""))
            # **判据用 max_memory_allocated，不用 memory_reserved。**
            # 开了 `expandable_segments:True` 之后 reserved 报的是虚拟地址空间，
            # 会出现「24 GB 卡上 reserved 38 GB」这种读数，拿它当判据是量错了对象。
            if pk_gb > 18.0:
                log_fn(f"    [WARN] 峰值分配 {pk_gb:.1f} GB —— 24 GB 卡上余量不足，"
                       f"再涨会触发 WDDM 溢出到共享内存，速度掉一个数量级")
            # ===== 工单 214-A 发散熔断（必须启用，不可关）=====
            # 判据：val > max(0.25, 2.0 x best_so_far) 连续 >= 3 次 -> 立刻终止，
            # 绝不再满载空跑（2026-09-15 那次从 step 4000 发散后空转了 ~5800 步）。
            if diverge_breaker:
                if vl > max(0.25, 2.0 * best):
                    _dv = globals().setdefault("_DIVERGE_N", 0) + 1
                    globals()["_DIVERGE_N"] = _dv
                    log_fn(f"    [DIVERGE-WATCH] val={vl:.4f} > max(0.25, 2x best="
                           f"{best:.4f}) 连续 {_dv}/3")
                    if _dv >= 3:
                        globals()["_DIVERGE_N"] = 0
                        stopped = st + 1
                        log_fn(f"    [DIVERGED] val={vl:.4f} vs best={best:.4f}，"
                               f"连续 3 次越过熔断线，终止于 step {stopped}")
                        raise DivergedError(
                            f"diverged at step {stopped}: val={vl:.4f} best={best:.4f}")
                else:
                    globals()["_DIVERGE_N"] = 0
            if bad >= patience and CFG.EARLY_STOP_ENFORCE:
                stopped = st + 1
                log_fn(f"    [EARLY-STOP] 连续 {patience} 次 val 无改善，停在 step {stopped}")
                break
    ema.eval()
    return ema, curve, {"n_params": int(sum(p.numel() for p in ema.parameters())),
                        "stopped_at": stopped, "best_val": best, "batch": batch,
                        "T_MAX": T_MAX, "NMAX": CFG.NMAX, "clip_x0": CFG.CLIP_X0_TO_NMAX,
                        "early_stop_enforced": bool(CFG.EARLY_STOP_ENFORCE),
                        "early_stop_would_trigger_at": es_would_stop,
                        "bf16": bool(bf16), "grad_ckpt": bool(grad_ckpt),
                        "val_rng_isolated": True}
