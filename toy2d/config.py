# -*- coding: utf-8 -*-
"""Constants and paths for the 2-D toy package.
Purpose: constants and paths. Numerical constants are carried over once from
verified/primitives.py; every path resolves inside this repository; TOY2D_OUT redirects the
output directory, which the self-tests use.
"""
from __future__ import annotations

import os
from pathlib import Path

from .verified import primitives as _v

ROOT = Path(__file__).resolve().parents[1]           # the repository root
ASSETS = ROOT / "assets" / "toy"                     # everything the toy experiment reads
DATA = ASSETS / "data"                               # the frozen ring dataset
MODELS = ASSETS / "frozen" / "models"                # net2d_v1_*.pt/.json, discriminators_v1.joblib, tilts, chain caches
REFERENCES = ASSETS / "frozen" / "references"        # per-request role-0 / role-1 reference draws
OUT = Path(os.environ.get("TOY2D_OUT", ROOT / "out"))
RUNS = OUT / "runs"

# --- caliber constants (verified module) ---
EPS_T = _v.EPS_T
F_BASE = _v.F_BASE
K_REV = _v.K_REV
CF_MULT = _v.CF_MULT
S_DIMS = _v.S_DIMS

# --- verified numerical primitives, re-exported ---
kmat = _v.kmat
tgrid = _v.tgrid
trunc_pois_pmf = _v.trunc_pois_pmf
feats = _v.feats
nfeat = _v.nfeat
build_net = _v.build_net
self_check = _v.self_check
draw_rows = _v.draw_rows
draw_iid = _v.draw_iid
softmax_np = _v.softmax_np
write_json = _v.write_json

# --- sampler defaults (the deployed toy configuration; prereg section 1) ---
DEFAULT_ESS_FRAC = 0.5          # theta: resample when ESS/M < theta
DEFAULT_INTERVAL = 8            # scheduled checkpoints every 8 reverse steps
DEFAULT_REWARD_FLOOR = 1.0e-8
TWIST_ROOT = 20260905           # root of the SEPARATE RNG stream for the twist's candidate endpoints
NOISE_ROOT = 20260821           # root of the corruption noise stream (prereg section 4)
