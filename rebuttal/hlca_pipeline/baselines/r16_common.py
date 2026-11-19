"""R16 shared helpers for baseline generation inputs (no model code here; testable without cfgen / scldm installed).
Pre-registration: STATUS 11-09 19:0x. Five target classes, val = cell-type-stratified 10% of the train file (seed 20261300),
baselines generated on all genes and cut to CFGen's 2,000 HVGs (var["highly_variable"] of the released files).
"""
import json
import numpy as np

TARGETS = ["type II pneumocyte", "classical monocyte", "multi-ciliated epithelial cell", "plasma cell", "ionocyte"]


def read_obs_var(path, label_key="cell_type"):
    """Labels, obs names, var names (Ensembl IDs), HVG mask, from an h5ad without loading the matrix."""
    import anndata as ad
    a = ad.read_h5ad(path, backed="r")
    out = (np.asarray(a.obs[label_key].astype(str), dtype=str), np.asarray(a.obs_names, dtype=str), np.asarray(a.var_names, dtype=str),
           np.asarray(a.var["highly_variable"]).astype(bool) if "highly_variable" in a.var else None)
    a.file.close()
    return out


def read_rows(path, rows, layer="X_counts"):
    """CSR rows (sorted unique row indices) of layers[layer] (or X), read with h5py one block at a time; returns scipy CSR."""
    import h5py, scipy.sparse as sp
    rows = np.asarray(rows)
    with h5py.File(path, "r") as h5:
        g = h5[f"layers/{layer}"] if layer != "X" else h5["X"]
        shape = tuple(g.attrs["shape"]); ip = g["indptr"][...]
        data, ind, ptr = [], [], [0]
        for r in rows:
            s, e = ip[r], ip[r + 1]
            data.append(g["data"][s:e]); ind.append(g["indices"][s:e]); ptr.append(ptr[-1] + (e - s))
    return sp.csr_matrix((np.concatenate(data) if data else np.zeros(0), np.concatenate(ind) if ind else np.zeros(0, int), np.asarray(ptr)),
                         shape=(len(rows), shape[1]))


def train_pool(labels, obs_names, val_ids_file):
    """Boolean mask of train-file cells that are NOT val (val list = one obs name per line)."""
    keep = np.ones(len(obs_names), bool)
    if val_ids_file:
        val = set(l.strip() for l in open(val_ids_file) if l.strip())
        keep = ~np.isin(obs_names, list(val))
        assert len(obs_names) - keep.sum() == len(val & set(obs_names)), "val list does not match the train file"
    return keep


def pick_rows(labels, pool, classes, n, seed, replace_if_short=True):
    """For each class: n row indices from the pool (without replacement when enough cells, else with replacement and flagged)."""
    rng = np.random.default_rng(seed); out, info = [], {}
    for c in classes:
        idx = np.flatnonzero(pool & (labels == c)); assert len(idx) > 0, f"no train cells for {c!r}"
        rep = len(idx) < n
        if rep and not replace_if_short:
            raise ValueError(f"{c}: {len(idx)} < {n}")
        sel = rng.choice(idx, n, replace=rep); out.append(sel)
        info[c] = {"n_pool": int(len(idx)), "n": int(n), "with_replacement": bool(rep)}
    return np.concatenate(out), info


def pick_uncond(labels, pool, n, seed):
    """n row indices drawn uniformly from the pool (so labels follow the train class proportions)."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.flatnonzero(pool), n, replace=False))


def write_manifest(path, d):
    json.dump(d, open(path, "w"), indent=1)
