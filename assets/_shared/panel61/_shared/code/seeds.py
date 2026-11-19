# moved from the module-level constants of cc/experiments/EXP134_more_ckpts/run_exp134.py; changes:
#   1. merge OLD_SEEDS/NEW_SEEDS into TRAIN_SEEDS (values identical to OLD+NEW in the same order, nothing added or removed);
#   2. added GENE_HASH / DATA_DIR so the release directory is self-contained (in EXP134 these were GH and the E35 path);
#   3. added the EXTRA_METRICS list beyond METRICS4 (additional records of spec section 1.3), not listed explicitly in EXP134.
# all other values unchanged verbatim.
"""EXP136 unified definition constants. Every arm must import from here; hard-coding is forbidden."""
from __future__ import annotations

from pathlib import Path

# ---- data (single copy, see _shared/data/DATA.md)----
SHARED = Path(__file__).resolve().parent
DATA_DIR = SHARED / "data"
GENE_HASH = "47ef02be9e74e8a84cc3bf67ae5b4cbefba53553ce5cb05f0f7f74cb8c1c7750"

# ---- seeds (identical to EXP134, 8 training × 2 sampling)----
TRAIN_SEEDS = [20260625, 20260626, 20260627, 20260628, 20260629, 20260701, 20260702, 20260703]
SAMP_SEEDS = [20260901, 20260902]

# ---- generation and evaluation ----
N_GEN = 20000
MHALF = 5000
NMAX = 128
SHUF_SEED = 20260970          # FIX-1: fixed seed for shuffling train before sampling in spearmanr
REF_PERM_SEED = 2026          # REF = val[permv[MHALF:2*MHALF]]
DIM = 61

# ---- metric ----
METRICS4 = ["resid_corr_sqrt2", "mean_abs_pairwise_corr_err", "fano_rel_err_mean", "FD_frechet"]
EXTRA_METRICS = ["joint_mmd", "joint_mmd_dh", "energy_dh", "integrality", "lowcount_tv_mean",
                 "pergene_histTV_mean", "tail_survivalL1_mean", "kurtosis_abserr_mean",
                 "zero_calib_err", "maxcount_relerr_mean", "W2_sliced", "MMD_rbf"]

# ---- OURS / Countsdiff K ladder (a superset of EXP134 BUDGETS; see each arm sweep.json)----
K_LADDER = [8, 16, 32, 64, 128, 256, 512, 1024]
