# -*- coding: utf-8 -*-
"""Build conditional weight tables log c_g(n) for arbitrary reference groups.

Sibling of `condition/build_cond_table.py`, not a modification of it. That file
builds `log c_g(n) = log P*_g(n) - log Q_g(n)` where the reference group is a
**cell type**. The only thing that has to change to condition on something else
is *which cells the reference marginal is computed from* — the operator, the
smoothing, the clamping and the sampler are all untouched.

Recipe reproduced line for line from `build_cond_table.py` (verified against the
source, not from memory):

  * calibration denominator `Q_g` = the marginal of the SAME unconditional pilot
    arm (K=32 base, chunk256, no-A4);
  * `eps = 0.5` count smoothing, applied as `eps/N` on each side with that
    side's own N;
  * log-ratio clamped to `+-log(1e3)`;
  * counts never seen in the reference group are left uncorrected (`logc = 0`),
    the `keep_zero_target=False` branch;
  * output `[2000, 513]` float64, fed to the sampler as `--logpi <file> --tau 0.4`.

Reference group specifications
------------------------------
    TYPE:<name>                     all train cells of that cell type
    MIX:<A>:<B>:<wA>                mixture at ratio wA : (1-wA)
    DONOR:<id>                      all train cells of that donor
    INTERSECT:<type>:<gene_index>   cells of <type> with n_gene > 0

MIX is handled **exactly**, not by drawing cells. The empirical marginal of a
population that is a wA fraction of A and (1-wA) of B is, per count bin,
`wA*h_A(n) + (1-wA)*h_B(n)` — the marginal is linear in the population exactly
as the moments are. Sampling cells to build it would only add noise to a
quantity available in closed form. The effective reference size recorded for the
estimability diagnostics is `min(N_A/wA, N_B/(1-wA))`, the size of the largest
real population with that composition, which is the honest N for a ratio the
data does not literally contain.

    python -m scripts.build_tilt_reference_table \
        --ref MIX:Ventricular_Cardiomyocyte:Endothelial:0.5 -o tables_ref
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XR = Path(__file__).resolve().parents[1] / "assets" / "legacy_root"
NMAX = 512


def marg_from_rows(X, rows, C, chunk=20000):
    """Identical to build_cond_table.marg_from_rows."""
    G = X.shape[1]
    off = (np.arange(G, dtype=np.int64) * C)[None, :]
    H = np.zeros(G * C, dtype=np.int64)
    for k in range(0, len(rows), chunk):
        b = np.asarray(X[np.sort(rows[k:k + chunk])], dtype=np.int64).clip(0, C - 1)
        H += np.bincount((b + off).ravel(), minlength=G * C)
    return H.reshape(G, C).astype(np.float64) / len(rows)


def marg_from_gen(path, C, chunk=20000):
    """Identical to build_cond_table.marg_from_gen."""
    A = np.load(path, mmap_mode="r")
    N, G = A.shape
    off = (np.arange(G, dtype=np.int64) * C)[None, :]
    H = np.zeros(G * C, dtype=np.int64)
    for i in range(0, N, chunk):
        b = np.asarray(A[i:i + chunk], dtype=np.int64).clip(0, C - 1)
        H += np.bincount((b + off).ravel(), minlength=G * C)
    return H.reshape(G, C).astype(np.float64) / N, N


def build_reference(spec, X, y, dn, C):
    """-> (name, reference marginal [G,C], N_ref, provenance dict)"""
    parts = spec.split(":")
    kind = parts[0].upper()

    if kind == "TYPE":
        t = parts[1]
        ii = np.flatnonzero(y == t)
        if not len(ii):
            raise SystemExit(f"no cells of type {t}")
        return (f"TYPE_{t}", marg_from_rows(X, ii, C), len(ii),
                {"kind": "type", "type": t, "n_ref_cells": int(len(ii))})

    if kind == "MIX":
        A, B, w = parts[1], parts[2], float(parts[3])
        ia, ib = np.flatnonzero(y == A), np.flatnonzero(y == B)
        if not len(ia) or not len(ib):
            raise SystemExit(f"missing a component of {spec}")
        ha, hb = marg_from_rows(X, ia, C), marg_from_rows(X, ib, C)
        h = w * ha + (1.0 - w) * hb
        neff = int(round(min(len(ia) / max(w, 1e-9), len(ib) / max(1 - w, 1e-9))))
        # The mixture marginal must still be a probability distribution per gene.
        chk = float(np.abs(h.sum(1) - 1.0).max())
        assert chk < 1e-9, f"mixture marginal not normalised: {chk}"
        # And it must be exactly reproducible by pooling integer cell counts,
        # which is the property that makes "a ratio the data does not contain"
        # a legitimate request rather than an extrapolation.
        na = 4000
        nb = int(round(4000 * (1 - w) / w))
        pooled = (na * ha + nb * hb) / (na + nb)
        lin = float(np.abs(h - pooled).max())
        return (f"MIX{int(round(w*1000)):04d}_{A}_{B}", h, neff,
                {"kind": "mixture", "A": A, "B": B, "w_A": w,
                 "n_cells_A": int(len(ia)), "n_cells_B": int(len(ib)),
                 "n_ref_cells_effective": neff,
                 "normalisation_max_abs_err": chk,
                 "linearity_vs_pooled_max_abs_err": lin,
                 "note": "marginal is linear in the population; built in closed "
                         "form, not by drawing cells"})

    if kind == "DONOR":
        d = parts[1]
        ii = np.flatnonzero(dn == d)
        if not len(ii):
            raise SystemExit(f"no cells for donor {d}")
        comp = {}
        for t in sorted(set(y[ii].tolist())):
            comp[str(t)] = int((y[ii] == t).sum())
        return (f"DONOR_{d}", marg_from_rows(X, ii, C), len(ii),
                {"kind": "donor", "donor": d, "n_ref_cells": int(len(ii)),
                 "n_celltypes_spanned": len(comp), "celltype_composition": comp,
                 "note": "donor cross-cuts every cell type, so this partition is "
                         "orthogonal to the label the 11 published arms use"})

    if kind == "INTERSECT":
        t, g = parts[1], int(parts[2])
        it = np.flatnonzero(y == t)
        col = np.asarray(X[:, g])
        ii = it[col[it] > 0]
        if not len(ii):
            raise SystemExit(f"empty intersection for {spec}")
        return (f"INTERSECT_{t}_g{g}", marg_from_rows(X, ii, C), len(ii),
                {"kind": "intersection", "type": t, "gene_index": g,
                 "n_ref_cells": int(len(ii)), "n_cells_type": int(len(it)),
                 "detection_rate_in_type": float(len(ii) / len(it))})

    raise SystemExit(f"unknown reference spec: {spec}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-root", default=str(XR))
    ap.add_argument("--ref", required=True, help="reference group specification")
    ap.add_argument("--pilot", default=None)
    ap.add_argument("--ratio-cap", type=float, default=1e3)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--keep-zero-target", action="store_true")
    ap.add_argument("-o", "--out", default="tables_ref")
    a = ap.parse_args()

    R = Path(a.x_root)
    OUT = Path(a.out)
    if not OUT.is_absolute():
        OUT = HERE / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    C = NMAX + 1
    D = R / "_shared" / "data" / "all"
    RES = R / "Baseline" / "OURS" / "results"
    pilot = Path(a.pilot) if a.pilot else (
        RES / ("linspace_T_O_K32_spilot_run1c_s20260625_samp20260902"
               "_batched1_chunk256_F2_K32_base_noa4.npy"))
    if not pilot.exists():
        raise SystemExit(f"pilot not found: {pilot}")

    Qg, n_gen = marg_from_gen(pilot, C)
    print(f"[pilot] {pilot.name}  n={n_gen}  "
          f"Q_g mean {(Qg * np.arange(C)).sum(1).mean():.4f}")

    X = np.load(D / "train.npy", mmap_mode="r")
    y = np.load(D / "train_celltype.npy", allow_pickle=True)
    dn = np.load(D / "train_donor.npy", allow_pickle=True)

    name, ht, Nref, prov = build_reference(a.ref, X, y, dn, C)
    print(f"[ref] {name}  N={Nref}")

    e_g = a.eps / n_gen
    e_t = a.eps / Nref
    cap = np.log(a.ratio_cap)
    nn = np.arange(C, dtype=np.float64)

    logc = np.log(ht + e_t) - np.log(Qg + e_g)
    n_clip = int((np.abs(logc) > cap).sum())
    logc = np.clip(logc, -cap, cap)
    if not a.keep_zero_target:
        logc = np.where(ht > 0, logc, 0.0)

    pred = {}
    for g in (0.2, 0.4, 0.6, 0.8, 1.0):
        q = Qg * np.exp(g * logc)
        q /= q.sum(1, keepdims=True)
        ach = (q * nn).sum(1)
        m1 = (ht * nn).sum(1)
        pos = m1 > 0
        pred[f"gamma_{g}"] = {
            "rel_median": float(np.median(np.abs(ach - m1)[pos] / m1[pos])),
            "gen_mean": float(ach.mean())}
    q0 = (Qg * nn).sum(1)
    m1 = (ht * nn).sum(1)
    pos = m1 > 0
    base = float(np.median(np.abs(q0 - m1)[pos] / m1[pos]))

    rse = np.sqrt(np.clip(ht * (1.0 - ht), 0.0, None) / Nref).sum(1)
    lowmass = float((ht * (ht * Nref < 10)).sum(1).mean())

    f = OUT / f"logc_{name}.npy"
    if f.exists():
        raise SystemExit(f"refusing to overwrite {f}")
    np.save(f, logc.astype(np.float64))
    meta = {"reference_spec": a.ref, "name": name, "provenance": prov,
            "n_ref_cells_for_eps": int(Nref),
            "calib_gen": pilot.name, "n_gen_calib": int(n_gen),
            "ratio_cap": a.ratio_cap, "eps": a.eps,
            "keep_zero_target": bool(a.keep_zero_target),
            "n_entries_clipped": n_clip,
            "span_median": float(np.median(logc.max(1) - logc.min(1))),
            "abs_max": float(np.abs(logc).max()),
            "frac_corrected": float((logc != 0).mean()),
            "target_m1_mean": float(m1.mean()),
            "est_rse_median": float(np.median(rse)),
            "est_rse_p90": float(np.percentile(rse, 90)),
            "mass_lowcount_lt10": lowmass,
            "baseline_rel_median_do_nothing": base,
            "static_endpoint_prediction": pred,
            "note": "sampling: sample_hvg2k_noa4.py --logpi <this> --tau 0.4 --no-a4"}
    (OUT / f"logc_{name}.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  span median {meta['span_median']:.2f}  |max| {meta['abs_max']:.2f}  "
          f"clipped {n_clip}  corrected {meta['frac_corrected']:.3f}")
    print(f"  target mean {m1.mean():.4f}   do-nothing rel_median {base:.3f} "
          f"-> static gamma=0.4 {pred['gamma_0.4']['rel_median']:.3f}")
    print(f"  estimability: rse median {meta['est_rse_median']:.4f}  "
          f"p90 {meta['est_rse_p90']:.4f}  low-support mass {lowmass:.4f}")
    print(f"[out] {f}")


if __name__ == "__main__":
    main()
