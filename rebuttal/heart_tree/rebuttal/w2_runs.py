# -*- coding: utf-8 -*-
"""Shared list of the Table 2 runs (5 arms x 3 train x 3 sample seeds) for all three targets."""
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "out" / "runs"
DATA = ROOT / "assets" / "legacy_root" / "_shared" / "data" / "all"
TARGETS = ["Endothelial", "Myeloid", "Neuronal"]
N_OUT = {"Endothelial": 10057, "Myeloid": 2500, "Neuronal": 2500}
TAU = {"Endothelial": "0p4", "Myeloid": "0p2", "Neuronal": "0p2"}
M = {"Endothelial": 20114, "Myeloid": 5000, "Neuronal": 5000}
ARMS = ["OURS", "CFGen", "MDLM", "scVI", "scANVI", "CountsDiff"]   # CountsDiff added 11-08
COL = dict(zip(ARMS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a4fc4"]))
# CountsDiff: the selected cell per target, train 20260625, sample seed 20261061
# (Endothelial 2,500 at 20261061 + 7,557 top-up at 20261062 = 10,057). One run per target.
CD_DIR = {"Endothelial": "Endothelial_K256_g1p0_n10057", "Myeloid": "Myeloid_K32_g1p0_n2500", "Neuronal": "Neuronal_K256_g1p0_n2500"}
CD_FILE = {"Endothelial": "CountsDiff_Endothelial_K256_g1p0_n10057_train20260625_samp20261061_20261062.npy",
           "Myeloid": "CountsDiff_Myeloid_K32_g1p0_n2500_train20260625_samp20261061.npy",
           "Neuronal": "CountsDiff_Neuronal_K256_g1p0_n2500_train20260625_samp20261061.npy"}
SUBSAMPLE_SEED = 20261107


# Runs behind Table 1 of ICLR2027_final-1.tex, recovered by matching all four printed metrics on all
# three targets. The manuscript does not list them.
TABLE1 = {"OURS": [(20260625, 20260989), (20260627, 20260987), (20260630, 20260989)],
          "MDLM": [(20260625, 20260933), (20260934, 20260932), (20260935, 20260932)],
          "CFGen": [("run1", 20260932), ("run2", 20260932), ("run3", 20260933)],
          "scVI": [(20261050, 20261051), (20261053, 20261055), (20261054, 20261055)],
          "scANVI": [(20261060, 20261065), (20261063, 20261061), (20261064, 20261061)],
          "CountsDiff": [(20260625, 20261061)]}
FIG7D_RUN = (20260625, 20260987)


def in_table1(arm, train, samp):
    return (train, samp) in TABLE1[arm]


def ours_stem(t, s):
    return (f"{t}_residual_hvgtwist_fk_tiltevery_trigany_K32_M{M[t]}_N{N_OUT[t]}_alpha1p0_J16_int8_th0p5_"
            f"tau{TAU[t]}_cc256_odds_seed{s}.npy")


def run_list(t):
    r, n = [], N_OUT[t]
    for s in (20260987, 20260988, 20260989):
        r.append(dict(arm="OURS", train=20260625, samp=s, path=RUNS / "table1_ours" / f"{t}_a1" / ours_stem(t, s)))
    r.append(dict(arm="OURS", train=20260627, samp=20260987,
                  path=RUNS / "table2_seeds_s20260627" / f"{t}_s20260627" / ours_stem(t, 20260987)))
    for s in (20260988, 20260989):
        r.append(dict(arm="OURS", train=20260627, samp=s,
                      path=RUNS / f"table2_ours_s20260627_samp{s}" / t / f"ours_{t}_train20260627_samp{s}.npy"))
    for s in (20260987, 20260988, 20260989):
        r.append(dict(arm="OURS", train=20260630, samp=s,
                      path=RUNS / f"table2_ours_s20260630_samp{s}" / t / f"ours_{t}_train20260630_samp{s}.npy"))
    for s in (20260931, 20260932, 20260933):
        d = RUNS / "cfgen_cond" / f"{t}_cfgen_cond"
        r.append(dict(arm="CFGen", train="run1", samp=s, path=d / f"cfgen_cond_{t}_seed{s}_n{n}.npy"))
        for run in ("run2", "run3"):
            r.append(dict(arm="CFGen", train=run, samp=s, path=d / f"cfgen_cond_{t}_{run}_seed{s}_n{n}.npy"))
        r.append(dict(arm="MDLM", train=20260625, samp=s,
                      path=RUNS / "mdlm_cond" / f"{t}_mdlm_cond" / f"mdlm_cond_K128_{t}_seed{s}_n{n}.npy"))
        for tr in (20260934, 20260935):
            r.append(dict(arm="MDLM", train=tr, samp=s,
                          path=RUNS / f"mdlm_cond_216_s{tr}" / f"{t}_mdlm_cond_s{tr}"
                          / f"mdlm_cond_K128_{t}_train{tr}_samp{s}_n{n}.npy"))
    for arm, low, trs, sps in (("scVI", "scvi", (20261050, 20261053, 20261054), (20261051, 20261055, 20261056)),
                               ("scANVI", "scanvi", (20261060, 20261063, 20261064), (20261061, 20261065, 20261066))):
        for tr in trs:
            for s in sps:
                r.append(dict(arm=arm, train=tr, samp=s,
                              path=RUNS / f"table2_{low}_s{tr}_samp{s}" / t / f"{low}_{t}_train{tr}_samp{s}.npy"))
    # r3_j: R3 resample seed offset WITHIN the arm; the old arms keep
    # their position in this list, unchanged, because CountsDiff is appended last
    r.append(dict(arm="CountsDiff", train=20260625, samp=20261061, r3_j=0, path=RUNS / "rebuttal_CD_final" / CD_DIR[t] / CD_FILE[t]))
    miss = [str(x["path"]) for x in r if not x["path"].is_file()]
    if miss:
        raise SystemExit("missing runs:\n" + "\n".join(miss))
    return r


def load_cells(r, t):
    x = np.load(r["path"], mmap_mode="r")
    n = N_OUT[t]
    if x.shape[0] > n:
        train = r["train"] if isinstance(r["train"], int) else 0
        s = np.random.SeedSequence([SUBSAMPLE_SEED, r["samp"], train]).generate_state(1)
        idx = np.sort(np.random.default_rng(int(s[0])).choice(x.shape[0], n, replace=False))
        return np.asarray(x[idx], dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def real_cells(split, t):
    x = np.load(DATA / f"{split}.npy", mmap_mode="r")
    y = np.load(DATA / f"{split}_celltype.npy", allow_pickle=True).astype(str)
    return np.asarray(x[np.flatnonzero(y == t)], dtype=np.float64)


def log_cp10k(a):
    lib = a.sum(1, keepdims=True); lib[lib == 0] = 1.0
    return np.log1p(a / lib * 1e4)
