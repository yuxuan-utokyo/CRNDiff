# -*- coding: utf-8 -*-
"""Rebuttal R5 / R9: alternative product-tilt tables, built with the exact recipe of
assets/tables/tables_kz (eps=0.5 smoothing with eps/N per side, log-ratio clipped at
+-log(1e3), keep_zero_target=True, calibration pool = unconditional pilot arm).

  --check            rebuild the full-data table and assert it equals tables_kz/logc_<T>.npy
  --topk k1,k2,...   keep only the k genes with the largest E_{P*}[logc_g]; other rows = 0
  --subsample f1,..  rebuild from a random fraction f of the target's train cells
  --draws n          independent subsample draws per fraction
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np

HVG = Path(__file__).resolve().parents[1]
DATA = HVG / "assets/legacy_root/_shared/data/all"
KZ = HVG / "assets/tables/tables_kz"
PILOT = HVG / "out/gates/200_basepool_ref/linspace_T_O_K32_spilot_run1c_s20260625_samp20260902_batched1_chunk256_F2_K32_base_noa4.npy"
OUT = Path(__file__).resolve().parent / "tables"
C = 513

def marg(A, rows=None, chunk=5000):
    G = A.shape[1]; off = (np.arange(G, dtype=np.int64) * C)[None, :]
    H = np.zeros(G * C, dtype=np.int64)
    idx = np.arange(A.shape[0]) if rows is None else np.sort(rows)
    for k in range(0, len(idx), chunk):
        sl = idx[k:k + chunk]
        b = np.asarray(A[sl] if rows is not None else A[sl[0]:sl[-1] + 1], dtype=np.int64).clip(0, C - 1)
        H += np.bincount((b + off).ravel(), minlength=G * C)
    return H.reshape(G, C).astype(np.float64) / len(idx)

def logc_of(ht, Nref, Qg, n_gen, eps=0.5, cap=np.log(1e3)):
    return np.clip(np.log(ht + eps / Nref) - np.log(Qg + eps / n_gen), -cap, cap)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--topk", default="")
    ap.add_argument("--subsample", default="")
    ap.add_argument("--draws", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    P = np.load(PILOT, mmap_mode="r"); n_gen = P.shape[0]
    Qg = marg(P)
    X = np.load(DATA / "train.npy", mmap_mode="r")
    y = np.load(DATA / "train_celltype.npy", allow_pickle=True)
    ii = np.flatnonzero(y == a.type)
    ht = marg(X, ii)
    full = logc_of(ht, len(ii), Qg, n_gen)
    ref = np.load(KZ / f"logc_{a.type}.npy")
    err = float(np.abs(full - ref).max())
    print(f"[{a.type}] n_ref={len(ii)} n_gen={n_gen}  max|rebuilt - tables_kz| = {err:.3e}")
    assert err < 1e-12, "rebuild does not match the deployed table; stop"
    if a.check:
        return
    log = {"type": a.type, "n_ref_full": int(len(ii)), "rebuild_max_abs_err": err, "tables": []}
    if a.topk:
        score = (ht * full).sum(1)                      # E_{P*}[log c_g]  ~ KL(P*_g || Q_g)
        order = np.argsort(-score)
        np.save(OUT / f"gene_rank_{a.type}.npy", order)
        for k in [int(v) for v in a.topk.split(",")]:
            t = np.zeros_like(full); t[order[:k]] = full[order[:k]]
            f = OUT / f"logc_{a.type}_top{k}.npy"; np.save(f, t)
            log["tables"].append({"kind": "topk", "k": k, "file": f.name, "sha256": sha(f),
                                  "score_mass_kept": float(score[order[:k]].sum() / score[score > 0].sum())})
            print(" ", f.name)
    if a.subsample:
        for fr in [float(v) for v in a.subsample.split(",")]:
            for d in range(a.draws):
                rng = np.random.default_rng(20261107 + 1000 * d + int(fr * 100))
                sub = rng.choice(ii, size=max(1, int(round(fr * len(ii)))), replace=False)
                t = logc_of(marg(X, sub), len(sub), Qg, n_gen)
                tag = f"sub{int(round(fr*100)):03d}_d{d}"
                f = OUT / f"logc_{a.type}_{tag}.npy"; np.save(f, t)
                log["tables"].append({"kind": "subsample", "fraction": fr, "draw": d, "n_ref": int(len(sub)),
                                      "file": f.name, "sha256": sha(f),
                                      "max_abs_diff_vs_full": float(np.abs(t - full).max()),
                                      "mean_abs_diff_vs_full_where_target_mass":
                                          float((ht * np.abs(t - full)).sum(1).mean())})
                print(" ", f.name, len(sub))
    (OUT / f"tables_{a.type}.json").write_text(json.dumps(log, indent=1))

if __name__ == "__main__":
    main()
