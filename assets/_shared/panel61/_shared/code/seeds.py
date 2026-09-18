# 搬自 cc/experiments/EXP134_more_ckpts/run_exp134.py 的模块级常量，改动：
#   1. 把 OLD_SEEDS/NEW_SEEDS 合并成 TRAIN_SEEDS（值与 OLD+NEW 顺序完全相同，未增删）；
#   2. 新增 GENE_HASH / DATA_DIR 便于发布目录自包含（EXP134 里是 GH 与 E35 路径）；
#   3. 新增 METRICS4 之外的 EXTRA_METRICS 清单（任务书 §1.3 的附加记录项），EXP134 未显式列出。
# 其余数值逐字未动。
"""EXP136 统一口径常量。所有 arm 必须 import 这里，禁止各自硬编码。"""
from __future__ import annotations

from pathlib import Path

# ---- 数据（唯一一份，见 _shared/data/DATA.md）----
SHARED = Path(__file__).resolve().parent
DATA_DIR = SHARED / "data"
GENE_HASH = "47ef02be9e74e8a84cc3bf67ae5b4cbefba53553ce5cb05f0f7f74cb8c1c7750"

# ---- 种子（EXP134 逐字一致，8 训练 × 2 采样）----
TRAIN_SEEDS = [20260625, 20260626, 20260627, 20260628, 20260629, 20260701, 20260702, 20260703]
SAMP_SEEDS = [20260901, 20260902]

# ---- 生成与评测 ----
N_GEN = 20000
MHALF = 5000
NMAX = 128
SHUF_SEED = 20260970          # FIX-1：spearmanr 取样前打乱 train 用的固定种子
REF_PERM_SEED = 2026          # REF = val[permv[MHALF:2*MHALF]]
DIM = 61

# ---- 指标 ----
METRICS4 = ["resid_corr_sqrt2", "mean_abs_pairwise_corr_err", "fano_rel_err_mean", "FD_frechet"]
EXTRA_METRICS = ["joint_mmd", "joint_mmd_dh", "energy_dh", "integrality", "lowcount_tv_mean",
                 "pergene_histTV_mean", "tail_survivalL1_mean", "kurtosis_abserr_mean",
                 "zero_calib_err", "maxcount_relerr_mean", "W2_sliced", "MMD_rbf"]

# ---- OURS / Countsdiff 的 K ladder（EXP134 BUDGETS 的超集，见各 arm sweep.json）----
K_LADDER = [8, 16, 32, 64, 128, 256, 512, 1024]
