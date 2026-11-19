# -*- coding: utf-8 -*-
"""hvg2k · All tunable parameters of OURS are collected here, each with what it is and why it has this value.

changing the experiment only touches this file. Trigger values of all criteria are recorded in `hvg2k/RUN_LOG.md`.
"""
from __future__ import annotations

from pathlib import Path

# 077 debt 1 (2026-08-24): ROOT was originally `Path(__file__).resolve().parents[3]`.
# HVG2K the tree was later moved from `CRNDIFF/Experiments/X/...` moved to
# `CRNDIFF/Conditional generation experiments/HVG2K/...`,`parents[3]` so it points to
# [Conditional generation experiments], **both derived paths break at once**:
#     DATA_DIR = ROOT/_shared/data/all          -> does not exist
#     G61      = ROOT.parent/_shared/panel61    -> does not exist (actual location Experiments/_shared/panel61)
# the consequence is that the pure-tilt sampling pipeline cannot even start in the current checkout (= reproducibility gap, filed as 077).
# fix: pin ROOT back to the directory it actually pointed to at the time. Checked: with this value both DATA_DIR and G61 exist.
# acceptance per 077 debt 1.2: reproduce an archived tilt arm bit for bit; sha256 must match the archive.
ROOT = Path(r".")
if not (ROOT / "_shared" / "data").is_dir():        # stop immediately if it moves again, instead of silently computing wrong values
    raise SystemExit(f"config_hvg2k: ROOT does not resolve to the data tree: {ROOT}")

# =========================================================================== #
# data
# =========================================================================== #
DATASET = "all"          # "all"(11 classes, 451,513 cells) or "endo"(Endothelial subset)
                         # Main line 2026-08-10: cancelled"first endo pilot", T2 runs directly on all/
DATA_DIR = ROOT / "_shared" / "data" / DATASET
GENE_HASH = "6f7963523c7a4c5bf1df7264c4c11e0a42fb0a9c4d52ca91d5724ae5e1eadb12"
# ↑ T0e hash of the gene set after filtering (MT-/RPL/RPS/MTRNR2L/HB* removed before HVG).
#   asserted at load time; mismatch stops immediately (red line section 5.3).

# =========================================================================== #
# NMAX — B3 criterion
# =========================================================================== #
NMAX = 512
# criterion (a): truncated **total count mass** <= 0.5%:
#     NMAX=128 -> 3.073%(fails)  256 -> 0.888% (fails)  **512 -> 0.284% (passes)**
#     1024 -> 0.149%   2048 -> 0.093%
#   take the **smallest** value satisfying (a) = 512. 36 genes are truncated; mass actually lost after clipping 0.148%.
#
# criterion (b) e^-T · NMAX <= 0.1 **not adopted, see [DEVIATION] in RUN_LOG**:
#   at T=T_O=5.2021 it requires NMAX <= 18, in direct conflict with (a); and our **published** 10D main-text row
#   (linspace + T_O) itself already runs at e^-T·NMAX = 13.09 (D1) / 5.82 (D2),
#   i.e. this criterion is violated 130 times over by our own best configuration. Use instead an upper bound calibrated from the published configuration (see below).

CLIP_X0_TO_NMAX = True
# clip the train counts to NMAX **before feeding them into the forward process**.
# 61gene of `train_ours_generic` is `binomial(original x0, r)` then clamp the **sum**,
#   i.e. the binomial term uses the untruncated n0. On 61gene max_n0=481, NMAX=128, ratio 3.8, harmless;
#   hvg2k is 9037 / 512 = 17.6, an order of magnitude off.
# clip afterwards the supports of the forward process, target and class head all coincide as [0, NMAX],
#   and in the criterion e^-T·max_n0 for whether the start has forgotten n0, max_n0 equals NMAX,
#   = e^-5.2021 × 512 = **2.81**, within the published upper bound 6.52 (10D D1 main-text row).
# cost: 0.284% of count mass (the share criterion (a) already accepts), tails of 36 genes.

# =========================================================================== #
# V — B2 criterion
# =========================================================================== #
V_GRID_LO, V_GRID_HI = -10, 4      # V quantized to the log2 grid 2^-10 … 2^4
# criterion sum V / lib_mean in [0.95, 1.10]: all three candidate lower bounds pass; take the one closest to 1
#     2^-10 -> 1.0386(chosen)   2^-8 -> 1.0410   2^-6 -> 1.0573
#   2^-10 also the fewest genes lifted by the floor (219, 0.02% of total mass).
#   actually uses 15 distinct V values; bank memory is cached by (tau, V), so this column is the hard constraint.
# the current `max(1, nonzero median)` deprecated: on hvg2k it gives sum V/lib = 6.22,
#   98.1% genes off by sqrt(2) (only 6.6% on 61gene, since the nonzero rate there is 0.668 versus 0.059 here).

# =========================================================================== #
# horizon T and time grid
# =========================================================================== #
GRID = "linspace"        # Main line 2026-08-10: all experiments use linspace; wake-sigma retired
F_BASE = 0.01            # consistent with 61gene/OURS/code/model.py
T_HORIZON = "T_O"        # "T_O" horizon estimated from the spectrum; a number can also be given directly
T_O = 5.2021             # = 0.5*ln(sigma_1/sigma_noise), computed on the **filtered new gene set**
                         #   sigma_1 = 5093.6712,sigma_noise = 0.15436
# the main line overruled my earlier recommendation of T=8, which was based on two proxies (step allocation / sd-mean),
# the reason is that the direct 10D measurement (with linspace T_O beats T8 by 37% at K=8) contradicts the proxies; the measurement wins.

# =========================================================================== #
# network (`_shared/code/backbone.py::AttnX0Posterior`, the very same class as in 61gene)
# =========================================================================== #
D_MODEL, NHEADS, NLAYERS = 128, 4, 3
# B4 criterion head parameters <= 5 times backbone parameters: the head is **shared across tokens**
#   `Linear(d_model, nmax+1)`; the parameter count does not depend on dim.
#   dim=2000 / nmax=512 gives head 66,177 / backbone 611,968 = **0.108×**, passes by a wide margin.
#   total parameters 636,673 (61gene) -> 934,401, only 1.47 times larger.
# what spec B4 calls Linear(hidden -> dim×(nmax+1)), 33 million to 130 million, is
#   `vendor_ours.py::X0PosteriorJoint`(the class used by 10Dtoy) property; measured 33,571,280 at dim=2000 —
#   is not the same class as the attention backbone used for panel/hvg2k.** So the architecture is not changed in this round.**

# =========================================================================== #
# training (recipe as for 61gene: EXP21::train_ours_generic + EXP134::STEPS)
# =========================================================================== #
BATCH = 128
# 2026-08-10 lesson: **batch=256 will hang.**
# 200 the 16.9 GB peak in the step smoke test looks safe, but that is `max_memory_allocated`;
# what really decides whether it stalls is **reserved** (the number nvidia-smi shows), which follows allocator fragmentation
# climbs to 23.6 / 24.5 GB, then WDDM starts spilling into shared memory:
#   first 400 steps 1.9 s/step -> then about 20 s/step (GPU reports 100% utilization at only 103 W: it is waiting on memory, not computing).
# 5.3 hours only reached step 1000. Reduced to 128 (logits [128,2000,513] float32 = 0.52 GB),
# and set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the launch script against fragmentation.
# 128×2000 = 25.6 ten thousand gene-tokens per step, still 4 times the 61-gene case (1024×61 = 62k).
LR, WD, WARMUP = 2e-3, 1e-4, 500

# ---- EXP158-T3: training speed-up. **Applies only to the training loop.** ----
USE_BF16 = True
# `torch.autocast(bfloat16)`. **Weights are still saved in fp32**; at sampling time they are loaded in fp32 and run forward in fp32,
# `post_fn` output is still float32 (as in the reference implementation); kernel bank / bridge weights remain float64 throughout.
# so zero effect on the numerical behaviour of the sampler; explicitly approved by main line EXP158-T3.
# `vendor_ours*.py` / `ours_fast_v2.py` / `bank_dedup.py` not a single byte changed.
USE_GRAD_CKPT = True
# 3 transformer layers, layer by layer `torch.utils.checkpoint(use_reentrant=False)`.
# T2 measured: the attention scores are **not** materialized (slope 68.3 MB/batch; materialized would be 183),
# most memory goes to FFN activations (3 layers × B×2000×512×4B) and the [B,2000,513] logits,
# is exactly what the checkpoint saves.
EMA_DECAY = 0.999
STEPS_SMOKE = 200
STEPS_PILOT = 4000
# smoke measured **1.80 s/step** (batch=256, peak memory 16.9 GB), so 8000 steps take 4 hours.
# at least 1 hour must be left for T3 sampling tonight, so changed to 4000 steps (about 2 hours). The criterion is still"3 consecutive val
# stop if no improvement"; not reaching the full count is normal.
#
# why so slow: **the bottleneck is self-attention in the backbone, not the head**. Attention is quadratic in the number of genes:
#   61gene   61 tokens -> 61² = 3.7k attention scores
#   hvg2k  2000 tokens -> 2000² = 4.0M, **1075 times**
# this is the real scaling wall from 61 to 2000, the B4"head too large"worry does not hold (the head is shared).
EVAL_EVERY = 200
VAL_CHUNK = 256
# val evaluation chunk size. **This is the real bottleneck found on 2026-08-10**:
# originally copied the 61gene approach, 2048 cells per forward pass; at dim=2000 that is **21.6 GB**
# (T2 measured 10.8 MB/cell at inference), 2.5 times more than a training step at batch=128 (8.5 GB).
# chunked to 256 -> 2.7 GB. Use reduction="sum" accumulate then divide by the total number of elements; numerically identical to taking the mean at once.
PATIENCE = 5
EARLY_STOP_REL = 0.003
EARLY_STOP_ENFORCE = False
# ★★ 2026-08-10 Main-line instruction: **early stopping is always record-only, never executed** (`EARLY_STOP_ENFORCE=False`).
#    first time training a 2000-dim model; information matters more than saving time; the log still prints
#    [if enabled, the first trigger happens at step ____], for budgeting later.
# ★ lesson 1: the first version of the criterion was `val < best - 1e-4` **absolute** threshold + patience=3,
#   at val≈0.115, 1e-4 is only 0.09% relative, drowned in noise.
#   changed to a **relative** threshold + patience=5 (an observation window of about 1000 steps).
# ★ lesson 2 (the real bug, which cut both run1 and run1b): after switching to a relative threshold,
#   `best` serves both as the best for saving and as the reference point of the relative threshold, while `best=min(best,vl)`
#   every refresh ratchets downward, so **the step that refreshed best by less than 0.3%**
#   best is updated but bad +1; the counter was not reset at the record-breaking step:
#       run1  step 5000  val=0.11510 < best=0.11520(+0.087%)-> bad=3/3
#       run1b step 2800  val=0.11408 == the final best_val itself      -> bad=5/5
#   fixed in `model_hvg2k.py`: reference point `es_ref` and the one used for saving `best` ****split into two variables**,
#   `es_ref` moves down only on a significant improvement. Not enabled even after the fix (see above).
SEED_PILOT = 20260625

# =========================================================================== #
# sampling / evaluation
# =========================================================================== #
N_GEN = 20000
K_LADDER_PILOT = [8, 32, 128]
SAMP_SEEDS = [20260901, 20260902]
