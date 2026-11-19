# -*- coding: utf-8 -*-
"""R1 (reviewer z7bw, Fig 7d): where do generated neuronal cells fall relative to real ones?

Compute stage only; figures are drawn by rebuttal/r1_figures.py from the outputs here.

Spaces for the k-NN (k=15), all on log-CP10K, index = all real val cells (n=45,145):
  pca2  frozen atlas basis (frozen_atlas_pca_basis.npz has ONLY 2 components)
  pca50 pca50_val.npz, the PCA the frozen UMAP reducer takes as input
  full  the 2,000 log-CP10K genes, no projection
Metric for a query cell q:
  neuronal_frac(q) = (1/15) * #{j in kNN(q): label(j) == Neuronal}
References (no generated cells): real val neuronal (self removed from its own neighbours)
and real train neuronal (not in the index, so queried exactly like a generated cell).
The test split is not read anywhere in this file.

    cd Experiments/HVG && python -m rebuttal.r1_fig7d
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = os.environ.get("R_THREADS", "8")   # R12 CPU window: 1
os.environ["PYTHONUTF8"] = "1"

import hashlib                                                       # noqa: E402
import json                                                          # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402
from sklearn.neighbors import NearestNeighbors                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from figures import hvgfig as H                                      # noqa: E402

RUNS = ROOT / "out" / "runs"
OUT = ROOT / "rebuttal" / "out" / "R1"
OUT.mkdir(parents=True, exist_ok=True)
K = 15
N_MATCH = 2500
SUBSAMPLE_SEED = 20261107          # only used to thin the 20,000-cell scVI / scANVI deliveries
T = "Neuronal"
OURS_STEM = ("Neuronal_residual_hvgtwist_fk_tiltevery_trigany_K32_M5000_N2500_alpha1p0_J16_"
             "int8_th0p5_tau0p2_cc256_odds_seed{s}.npy")


def run_list() -> list[dict]:
    """The Table 2 grid (final results/README.md): 5 arms x 3 train x 3 sample seeds."""
    r = []
    for s in (20260987, 20260988, 20260989):
        r.append(dict(arm="OURS", train=20260625, samp=s,
                      path=RUNS / "table1_ours" / "Neuronal_a1" / OURS_STEM.format(s=s)))
    r.append(dict(arm="OURS", train=20260627, samp=20260987,
                  path=RUNS / "table2_seeds_s20260627" / "Neuronal_s20260627"
                  / OURS_STEM.format(s=20260987)))
    for s in (20260988, 20260989):
        r.append(dict(arm="OURS", train=20260627, samp=s,
                      path=RUNS / f"table2_ours_s20260627_samp{s}" / T
                      / f"ours_Neuronal_train20260627_samp{s}.npy"))
    for s in (20260987, 20260988, 20260989):
        r.append(dict(arm="OURS", train=20260630, samp=s,
                      path=RUNS / f"table2_ours_s20260630_samp{s}" / T
                      / f"ours_Neuronal_train20260630_samp{s}.npy"))
    for s in (20260931, 20260932, 20260933):
        d = RUNS / "cfgen_cond" / "Neuronal_cfgen_cond"
        r.append(dict(arm="CFGen", train="run1", samp=s,
                      path=d / f"cfgen_cond_Neuronal_seed{s}_n2500.npy"))
        for run in ("run2", "run3"):
            r.append(dict(arm="CFGen", train=run, samp=s,
                          path=d / f"cfgen_cond_Neuronal_{run}_seed{s}_n2500.npy"))
        r.append(dict(arm="MDLM", train=20260625, samp=s,
                      path=RUNS / "mdlm_cond" / "Neuronal_mdlm_cond"
                      / f"mdlm_cond_K128_Neuronal_seed{s}_n2500.npy"))
        for tr in (20260934, 20260935):
            r.append(dict(arm="MDLM", train=tr, samp=s,
                          path=RUNS / f"mdlm_cond_216_s{tr}" / f"Neuronal_mdlm_cond_s{tr}"
                          / f"mdlm_cond_K128_Neuronal_train{tr}_samp{s}_n2500.npy"))
    for arm, low, trains, samps in (
            ("scVI", "scvi", (20261050, 20261053, 20261054), (20261051, 20261055, 20261056)),
            ("scANVI", "scanvi", (20261060, 20261063, 20261064), (20261061, 20261065, 20261066))):
        for tr in trains:
            for s in samps:
                r.append(dict(arm=arm, train=tr, samp=s,
                              path=RUNS / f"table2_{low}_s{tr}_samp{s}" / T
                              / f"{low}_Neuronal_train{tr}_samp{s}.npy"))
    r.append(dict(arm="CountsDiff", train=20260625, samp=20261061,   # R12
                  path=RUNS / "rebuttal_CD_final" / "Neuronal_K256_g1p0_n2500"
                  / "CountsDiff_Neuronal_K256_g1p0_n2500_train20260625_samp20261061.npy"))
    for x in r:
        if not x["path"].is_file():
            raise SystemExit(f"missing run: {x['path']}")
    return r


def is_umap_run(r: dict) -> bool:
    """UMAP transform is slow: all 3 sample seeds of the OURS reference generator, one per arm."""
    first = {"CFGen": ("run1", 20260931), "MDLM": (20260625, 20260931), "CountsDiff": (20260625, 20261061),
             "scVI": (20261050, 20261051), "scANVI": (20261060, 20261061)}
    if r["arm"] == "OURS":
        return r["train"] == 20260625
    return (r["train"], r["samp"]) == first[r["arm"]]


def tag(r: dict) -> str:
    return f"{r['arm']}_{r['train']}_{r['samp']}"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_cells(r: dict) -> np.ndarray:
    x = np.load(r["path"], mmap_mode="r")
    if x.shape[0] > N_MATCH:      # scVI / scANVI: thin to OURS' n, reproducibly per run
        s = np.random.SeedSequence([SUBSAMPLE_SEED, r["samp"], int(r["train"])]).generate_state(1)
        idx = np.sort(np.random.default_rng(int(s[0])).choice(x.shape[0], N_MATCH, replace=False))
        x = x[idx]
    return np.asarray(x, dtype=np.float64)


def knn_stats(nbr_idx: np.ndarray, y_int: np.ndarray, nt: int, neu: int) -> tuple[dict, np.ndarray]:
    lab = y_int[nbr_idx]
    frac = (lab == neu).mean(1)
    comp = np.stack([(lab == t).mean() for t in range(nt)])
    return {"n": int(len(frac)),
            "mean_neuronal_frac": float(frac.mean()),
            "median_neuronal_frac": float(np.median(frac)),
            "share_frac_ge_0p5": float((frac >= 0.5).mean()),
            "share_frac_eq_0": float((frac == 0).mean()),
            "nn1_is_neuronal": float((lab[:, 0] == neu).mean()),
            "neighbour_type_share": comp.tolist()}, frac


def main() -> None:
    t0 = time.time()
    xval, yval = H.real_cells()
    xval = np.asarray(xval, dtype=np.float64)
    types = sorted(set(yval.tolist()))
    y_int = np.array([types.index(t) for t in yval])
    neu = types.index(T)
    mask_val = yval == T

    xtr = np.load(H.DATA / "train.npy", mmap_mode="r")
    ytr = np.load(H.DATA / "train_celltype.npy", allow_pickle=True).astype(str)
    idx_tr = np.flatnonzero(ytr == T)
    x_tr_neu = np.asarray(xtr[idx_tr], dtype=np.float64)
    print(f"[data] val={xval.shape} val_neuronal={mask_val.sum()} train_neuronal={len(idx_tr)}",
          flush=True)

    pca2 = H.FrozenPCA()
    gate_pca = pca2.gate_against_archive(xval)
    um = H.FrozenUMAP()
    gate_umap = um.gate_against_archive()

    spaces = {
        "pca2": lambda c: pca2.project(c),
        "pca50": lambda c: (H.cpm_log1p(c) - um.mean) @ um.comp.T,
        "full": lambda c: H.cpm_log1p(c),
    }
    index, nn = {}, {}
    for s, f in spaces.items():
        index[s] = f(xval).astype(np.float32)
        nn[s] = NearestNeighbors(n_neighbors=K, algorithm="brute", n_jobs=int(os.environ.get("R_THREADS", "8"))).fit(index[s])
        print(f"[index] {s} {index[s].shape} ({time.time() - t0:.0f}s)", flush=True)

    rows, cell_frac = [], {}
    # --- references
    for s, f in spaces.items():
        q = index[s][mask_val]
        ii = nn[s].kneighbors(q, n_neighbors=K + 1, return_distance=False)
        own = np.flatnonzero(mask_val)
        keep = np.array([[j for j in row if j != o][:K] for row, o in zip(ii, own)])
        st, fr = knn_stats(keep, y_int, len(types), neu)
        rows.append(dict(arm="real_val", train=None, samp=None, space=s, **st))
        cell_frac[f"real_val__{s}"] = fr
        ii = nn[s].kneighbors(f(x_tr_neu).astype(np.float32), return_distance=False)
        st, fr = knn_stats(ii, y_int, len(types), neu)
        rows.append(dict(arm="real_train", train=None, samp=None, space=s, **st))
        cell_frac[f"real_train__{s}"] = fr
    print(f"[ref] done ({time.time() - t0:.0f}s)", flush=True)

    # --- generated
    runs = run_list()
    prov, umap_xy, cells_for_labels = [], {}, {}
    for r in runs:
        x = load_cells(r)
        prov.append(dict(arm=r["arm"], train=r["train"], samp=r["samp"], path=str(r["path"]),
                         sha256=sha(r["path"]), n_file=int(np.load(r["path"], mmap_mode="r").shape[0]),
                         n_used=int(x.shape[0]), integer_valued=bool(np.all(x[:200] == np.round(x[:200])))))
        for s, f in spaces.items():
            ii = nn[s].kneighbors(f(x).astype(np.float32), return_distance=False)
            st, fr = knn_stats(ii, y_int, len(types), neu)
            rows.append(dict(arm=r["arm"], train=r["train"], samp=r["samp"], space=s, **st))
            cell_frac[f"{tag(r)}__{s}"] = fr
        if is_umap_run(r):
            umap_xy[tag(r)] = um.project(x)
            cells_for_labels[tag(r)] = x
        print(f"[run] {tag(r)} ({time.time() - t0:.0f}s)", flush=True)
        (OUT / "knn_rows.partial.json").write_text(json.dumps(rows), encoding="utf-8")

    # gate: the OURS reference run must land where the paper's panel (d) put it
    cached = np.load(H.FIG / "fig_umap_panels.coords.npz")["Neuronal_umap"]
    delta = float(np.abs(umap_xy["OURS_20260625_20260987"] - cached).max())
    print(f"[gate] OURS s987 vs paper fig coords: max |delta| = {delta:.3e}", flush=True)

    # --- per-cell CellTypist labels: the stored runs keep only histograms, and the clump
    #     question needs labels per cell. Same call as the purity ruler.
    from analysis.score_purity import annotate
    genes = json.loads((H.DATA / "meta.json").read_text(encoding="utf-8"))["genes"]
    labels = {"real_val": annotate(xval[mask_val], genes)}
    for t, x in cells_for_labels.items():
        labels[t] = annotate(x, genes)
        print(f"[celltypist] {t} ({time.time() - t0:.0f}s)", flush=True)

    # --- stored label histograms (read, not recomputed)
    d214 = json.loads((ROOT / "out" / "summary" / "214_neuronal_dual_calibre.json").read_text(encoding="utf-8"))
    stored = {"real_val_composition": d214["real_val_composition"],
              "frozen_calibre": d214["frozen_calibre"], "cells": d214["cells"]}

    np.savez_compressed(OUT / "r1_cells.npz",
                        real_xy=um.real_xy, yval=yval, types=np.array(types),
                        **{f"xy__{k}": v for k, v in umap_xy.items()},
                        **{f"frac__{k}": v for k, v in cell_frac.items()},
                        **{f"label__{k}": v for k, v in labels.items()})
    (OUT / "r1_numbers.json").write_text(json.dumps({
        "k": K, "types": types, "rows": rows, "provenance": prov,
        "n_val": int(xval.shape[0]), "n_val_neuronal": int(mask_val.sum()),
        "n_train_neuronal": int(len(idx_tr)),
        "prevalence_neuronal_val": float(mask_val.mean()),
        "gate_pca": gate_pca, "gate_umap": {k: v for k, v in gate_umap.items()},
        "gate_ours_vs_paper_panel_max_abs_delta": delta,
        "paper_panel_coords": str(H.FIG / "fig_umap_panels.coords.npz"),
        "stored_label_hist_214": stored,
        "subsample_seed": SUBSAMPLE_SEED, "walltime_s": time.time() - t0,
    }, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
