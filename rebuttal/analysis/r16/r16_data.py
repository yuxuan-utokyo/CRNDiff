"""Reader for Windows' R16 export (_data_exchange/r16_data/, written by r16_export.py): per split, int32 CSR row blocks
<split>_rows<lo>-<hi>.npz (scipy save_npz), <split>_celltype.npy, <split>_obs_names.txt; genes.json = the 2,000 HVG IDs.
Splits: train (train file minus val, 421,159), val (46,796), test (116,989), pca_ref (50,000 train rows).
"""
import glob, json, os, re
import numpy as np
import scipy.sparse as sp


def split_files(D, split):
    fs = glob.glob(os.path.join(D, f"{split}_rows*.npz"))
    fs = sorted((f for f in fs if not f.endswith(".part.npz")), key=lambda f: int(re.search(r"_rows(\d+)-", f).group(1)))
    lo = 0
    for f in fs:  # blocks must tile the rows with no gap
        a, b = map(int, re.search(r"_rows(\d+)-(\d+)\.npz$", f).groups()); assert a == lo, f"gap before {f}"; lo = b
    return fs + [os.path.join(D, f"{split}_celltype.npy"), os.path.join(D, f"{split}_obs_names.txt")]


def genes(D):
    g = json.load(open(os.path.join(D, "genes.json")))
    return np.asarray(g["genes"] if isinstance(g, dict) else g, dtype=str)


def load_split(D, split, with_names=False):
    """(X int32 CSR, labels str, genes str[, obs names])."""
    fs = split_files(D, split)
    X = sp.vstack([sp.load_npz(f) for f in fs[:-2]], format="csr")
    assert X.dtype == np.int32, X.dtype
    y = np.load(fs[-2], allow_pickle=True).astype(str)
    names = np.array([l.rstrip("\n") for l in open(fs[-1], encoding="utf-8") if l.strip()])
    g = genes(D)
    assert X.shape == (len(y), len(g)) == (len(names), len(g)), (X.shape, len(y), len(names), len(g))
    return (X, y, g, names) if with_names else (X, y, g)
