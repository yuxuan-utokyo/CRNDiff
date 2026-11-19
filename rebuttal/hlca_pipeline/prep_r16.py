#!/usr/bin/env python
"""R16: HLCA (Human Lung Cell Atlas core, CFGen's preprocessed files, Zenodo 15031364) in the HVG2K array format.
Adapted from rebuttal/r13_src/prep_r13.py (the R13 Dentate Gyrus converter); differences are listed here and in the data card.

  * Gene list is an INPUT.
    Without --genes it falls back to the R13 rule (cell filter, gene filter, human T0e, seurat_v3 2,000 HVG, no labels),
    computed on the training cells only (val and test never enter the selection).
  * Split:
      test  = CFGen's held-out part (the test file(s)), untouched;
      val   = whole donors drawn from CFGen's training part, seed 20261300, until >= 10% of its cells; every cell type with
              >= 200 cells in that part must have >= 20 cells in val, else redraw with seed + 1 (at most 10 tries; record all);
      train = the rest of CFGen's training part. Donors are disjoint between train and val (asserted).
  * Vg (per-gene median of the non-zero values, floor 1) from TRAIN cells only (vice 18:2x); HVG2K / R13 used train + val.
    The per-gene means used by the generator are derived from train.npy at training time, so they are train-only too.
  * Counts must be non-negative integers (asserted) and fit int32 (asserted) after the gene subset. Stored as int32 (vice 11-09 19:1x): one train entry, SCGB1A1 = 34,950, exceeds int16; no value is changed.
Outputs in --out: {train,val,test}.npy (int32), *_celltype.npy, *_donor.npy, meta.json, genes.json, data_card.json.
Read-only on the inputs. No GPU.
    python prep_r16.py --train <h5ad ...> --test <h5ad ...> --out OUT --label-key <obs key> --donor-key <obs key> [--genes genes.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

VAL_SEED, VAL_FRAC, VAL_MIN_TYPE, VAL_MIN_CELLS, VAL_TRIES = 20261300, 0.10, 200, 20, 10
N_HVG, MIN_COUNTS, MIN_CELLS = 2000, 10, 3
T0E_HUMAN = {"mito": r"^MT-", "ribo": r"^RP[LS]", "mtrnr": r"^MTRNR2L", "haem": r"^HB[ABDGQZ]\d*$"}   # HVG2K spelling


def sha(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()
def array_hash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read_part(files, label_key, donor_key, layer):
    import anndata as ad
    Xs, cells, labels, donors, genes = [], [], [], [], None
    for f in files:
        a = ad.read_h5ad(f)
        if layer not in a.layers:
            raise SystemExit(f"{f}: no '{layer}' layer (layers: {list(a.layers)})")
        X = sp.csr_matrix(a.layers[layer])
        if X.nnz and (X.data.min() < 0 or not np.all(X.data == np.round(X.data))):
            raise SystemExit(f"{f}: '{layer}' is not non-negative integers")
        g = a.var_names.astype(str).to_numpy()
        if genes is not None and not np.array_equal(genes, g):
            raise SystemExit(f"{f}: gene list differs from the first file")
        genes = g
        Xs.append(X); cells.append(a.obs_names.astype(str).to_numpy())
        labels.append(a.obs[label_key].astype(str).to_numpy()); donors.append(a.obs[donor_key].astype(str).to_numpy())
    return sp.vstack(Xs).tocsr(), np.concatenate(cells), np.concatenate(labels), np.concatenate(donors), genes


def donor_val(labels, donors):
    """Whole donors from CFGen's training part until >= VAL_FRAC of its cells; every type with >= VAL_MIN_TYPE cells
    there needs >= VAL_MIN_CELLS in val. Seed VAL_SEED, then +1 on failure, at most VAL_TRIES tries."""
    n = len(labels); big = {t for t, c in zip(*np.unique(labels, return_counts=True)) if c >= VAL_MIN_TYPE}
    uniq = np.array(sorted(set(donors))); tries = []
    for k in range(VAL_TRIES):
        seed = VAL_SEED + k; rng = np.random.Generator(np.random.PCG64(seed)); order = rng.permutation(len(uniq))
        chosen, cnt = [], 0
        for i in order:
            if cnt >= VAL_FRAC * n:
                break
            chosen.append(uniq[i]); cnt += int((donors == uniq[i]).sum())
        m = np.isin(donors, chosen)
        short = {t: int((labels[m] == t).sum()) for t in sorted(big) if (labels[m] == t).sum() < VAL_MIN_CELLS}
        tries.append({"seed": seed, "n_donors_val": len(chosen), "n_val": int(m.sum()), "frac": float(m.mean()), "types_short": short})
        if not short:
            return m, sorted(map(str, chosen)), tries
    raise SystemExit(f"no valid donor-level val split in {VAL_TRIES} tries: {tries}")


def cell_val(labels, seed=VAL_SEED, frac=VAL_FRAC):
    """Final pre-registration: within CFGen's training part, each cell_type separately, a random
    round(frac * n_type) cells go to val (PCG64(seed), classes in sorted order, as R13's stratified_split); the rest is train.
    Train and val share donors (as CFGen's own train / test)."""
    rng = np.random.Generator(np.random.PCG64(seed)); m = np.zeros(len(labels), dtype=bool)
    for lab in sorted(set(labels)):
        idx = np.flatnonzero(labels == lab); rng.shuffle(idx); m[idx[:int(round(len(idx) * frac))]] = True
    return m


def select_hvg(X, genes):
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(X=X.astype(np.float32).tocsr()); a.var_names = genes
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=N_HVG)
    chosen = set(a.var_names[a.var["highly_variable"].to_numpy()])
    return [str(g) for g in genes if g in chosen]


def per_type(y, X):
    rows = []
    for lab in sorted(set(y)):
        m = y == lab
        rows.append({"cell_type": str(lab), "n": int(m.sum()), "share": float(m.mean()), "zero_frac": float((X[m] == 0).mean()),
                     "median_counts_per_cell": float(np.median(X[m].sum(1)))})
    return sorted(rows, key=lambda r: -r["n"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", type=Path, required=True, help="CFGen training-part h5ad file(s)")
    ap.add_argument("--test", nargs="+", type=Path, required=True, help="CFGen held-out h5ad file(s)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label-key", required=True); ap.add_argument("--donor-key", required=True)
    ap.add_argument("--layer", default="X_counts")
    ap.add_argument("--val-rule", choices=("cell", "donor"), default="cell",
                    help="cell = final pre-registration rule C (default); donor = the first rule, kept reachable, off by default")
    ap.add_argument("--genes", type=Path, help="genes.json with {'genes': [...]}; the gene space is an input")
    ap.add_argument("--expect-meta", type=Path)
    a = ap.parse_args()

    Xtr, ctr, ytr, dtr, genes = read_part(a.train, a.label_key, a.donor_key, a.layer)
    Xte, cte, yte, dte, g2 = read_part(a.test, a.label_key, a.donor_key, a.layer)
    if not np.array_equal(genes, g2):
        raise SystemExit("train and test parts disagree on the gene list")
    if len(set(ctr) | set(cte)) != len(ctr) + len(cte):
        raise SystemExit("duplicate cell names across the parts")
    n_in = {"train_part": list(Xtr.shape), "test_part": list(Xte.shape)}
    o = np.argsort(ctr, kind="stable"); Xtr, ctr, ytr, dtr = Xtr[o], ctr[o], ytr[o], dtr[o]
    o = np.argsort(cte, kind="stable"); Xte, cte, yte, dte = Xte[o], cte[o], yte[o], dte[o]
    k = np.asarray(Xtr.sum(1)).ravel() >= MIN_COUNTS; Xtr, ctr, ytr, dtr = Xtr[k], ctr[k], ytr[k], dtr[k]
    k = np.asarray(Xte.sum(1)).ravel() >= MIN_COUNTS; Xte, cte, yte, dte = Xte[k], cte[k], yte[k], dte[k]

    if a.val_rule == "donor":
        mval, val_donors, tries = donor_val(ytr, dtr)
    else:
        mval = cell_val(ytr); val_donors = None
        tries = [{"rule": "C: per-cell_type random 10% of CFGen's training part", "seed": VAL_SEED, "n_val": int(mval.sum()), "frac": float(mval.mean())}]
    itr, iva = np.flatnonzero(~mval), np.flatnonzero(mval)
    if a.val_rule == "donor":
        assert not set(dtr[itr]) & set(dtr[iva]), "train and val share a donor"

    removed, overlap = None, None
    if a.genes:
        chosen = json.loads(a.genes.read_text(encoding="utf-8"))["genes"]; how = f"from --genes {a.genes.name}"
    else:
        # vice 11-09 18:3x: as the pre-registration and HVG2K / R13, on ALL cells (train + val + test), no labels
        Xa = sp.vstack([Xtr, Xte]).tocsr(); keep = np.asarray((Xa > 0).sum(0)).ravel() >= MIN_CELLS
        removed = {k2: [str(g) for g in genes[keep] if re.search(p, str(g))] for k2, p in T0E_HUMAN.items()}
        drop = set(sum(removed.values(), [])); cand = np.array([keep[i] and str(genes[i]) not in drop for i in range(len(genes))])
        chosen = select_hvg(Xa[:, cand], genes[cand]); how = "seurat_v3 2000 HVG on all cells, no labels (HVG2K / R13 rule)"
    pos = {str(g): i for i, g in enumerate(genes)}
    missing = [g for g in chosen if g not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} genes not in the files, e.g. {missing[:5]}")
    col = np.array([pos[g] for g in chosen])

    a.out.mkdir(parents=True, exist_ok=True)
    parts = {"train": (Xtr[itr], ytr[itr], dtr[itr]), "val": (Xtr[iva], ytr[iva], dtr[iva]), "test": (Xte, yte, dte)}
    a.out.mkdir(parents=True, exist_ok=True)   # vice 11-09 19:1x: obs names per split (row order of the arrays), one per line
    for s, names in (("train", ctr[itr]), ("val", ctr[iva]), ("test", cte)):
        (a.out / f"{s}_obs_names.txt").write_bytes(("\n".join(map(str, names)) + "\n").encode("utf-8"))
    arrays, card = {}, {}
    for s, (X, y, d) in parts.items():
        dense = np.asarray(X[:, col].todense())
        if dense.max() > np.iinfo(np.int32).max:
            raise SystemExit(f"{s}: a count exceeds int32 ({dense.max()})")
        arr = dense.astype(np.int32); arrays[s] = arr
        np.save(a.out / f"{s}.npy", arr)
        np.save(a.out / f"{s}_celltype.npy", y.astype(object), allow_pickle=True)
        np.save(a.out / f"{s}_donor.npy", d.astype(object), allow_pickle=True)
        lib = arr.sum(1)
        card[s] = {"n_cells": int(len(y)), "n_donors": len(set(d)), "zero_frac": float((arr == 0).mean()),
                   "counts_per_cell": {"mean": float(lib.mean()), "median": float(np.median(lib)),
                                       "p05": float(np.percentile(lib, 5)), "p95": float(np.percentile(lib, 95))},
                   "max_count": int(arr.max()), "per_type": per_type(y.astype(str), arr)}
    tr = arrays["train"]
    vg = [max(1.0, float(np.median(c[c > 0]))) if (c > 0).any() else 1.0 for c in tr.T]
    meta = {"dataset": "hlca_core", "gene_hash": sha("\n".join(chosen)), "n_genes": len(chosen), "Vg": vg, "Vg_cells": "train only (vice 11-09 18:2x); HVG2K and R13 computed Vg on train + val",
            "genes": chosen, "n_train": len(itr), "n_val": len(iva), "n_test": int(len(yte)), "dtype": "int32", "dtype_reason": "int16 would overflow: one train entry, SCGB1A1 = 34,950 (> 32,767); stored as int32, no value changed; the generator clips to NMAX before training",
            "cell_type_key": a.label_key, "donor_key": a.donor_key, "layer": a.layer, "gene_selection": how,
            "val_rule": {"used": a.val_rule, "seed0": VAL_SEED, "frac": VAL_FRAC, "min_type_train": VAL_MIN_TYPE, "min_val": VAL_MIN_CELLS,
                         "max_tries": VAL_TRIES, "tries": tries, "val_donors": val_donors},
            "test_rule": "CFGen held-out part as given", "min_counts_per_cell": MIN_COUNTS,
            "train_stats": {"mean": float(tr.mean()), "zerofrac": float((tr == 0).mean()), "lib_mean": float(tr.sum(1).mean()), "max": int(tr.max())},
            "gene_hash_definition": "sha256 of the newline-joined gene names, in file order",
            "source": "CFGen preprocessed data, Zenodo record 15031364, CC-BY-4.0",
            "input_files": {"train": [p.name for p in a.train], "test": [p.name for p in a.test]},
            "array_sha256": {s: array_hash(v) for s, v in arrays.items()}}
    (a.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (a.out / "genes.json").write_text(json.dumps({"genes": chosen}, ensure_ascii=False), encoding="utf-8")
    from importlib import metadata as _md   # vice 11-09 18:5x: record the environment so the conversion is reproducible
    env = {"python": sys.version.split()[0],
           "pip_freeze": sorted(f"{d.metadata['Name']}=={d.version}" for d in _md.distributions())}
    (a.out / "data_card.json").write_text(json.dumps({"input_shape": n_in, "gene_selection": how, "environment": env, "dtype": "int32", "dtype_reason": meta["dtype_reason"],
        "test_donor_overlap": {"n_test_donors_also_in_train": len(set(dte) & set(dtr[itr])),
                               "n_test_donors_also_in_val": len(set(dte) & set(dtr[iva])),
                               "note": "test is CFGen's held-out file as given (vice 11-09 19:0x); CFGen's split is not by donor, so test shares donors with train and val"},
        "differences_from_hvg2k_r13": ["Vg on train only (HVG2K / R13: train + val)",
                                       f"val = {'whole donors' if a.val_rule == 'donor' else 'per-cell_type random 10% of cells (rule C)'} from CFGen's training part, test = CFGen's held-out part (R13: 8:1:1 stratified by type over all cells)"], "t0e_regex": T0E_HUMAN if removed else None,
        "t0e_removed": {k2: {"n": len(v), "genes": v} for k2, v in removed.items()} if removed else None,
        "donors": {s: sorted(set(map(str, p[2]))) for s, p in parts.items()}, "splits": card}, ensure_ascii=False, indent=1), encoding="utf-8")
    for s, v in arrays.items():
        print(f"{s}: {v.shape} donors {card[s]['n_donors']} sha256={meta['array_sha256'][s]}", flush=True)
    print("gene_hash", meta["gene_hash"], "| val tries", [(t["seed"], t["frac"], len(t.get("types_short", {}))) for t in tries], flush=True)
    if a.expect_meta:
        want = json.loads(a.expect_meta.read_text(encoding="utf-8"))
        bad = [s for s in arrays if want["array_sha256"][s] != meta["array_sha256"][s]] + (["gene_hash"] if want["gene_hash"] != meta["gene_hash"] else [])
        if bad:
            raise SystemExit(f"hash mismatch: {bad}")
        print("all hashes match", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
