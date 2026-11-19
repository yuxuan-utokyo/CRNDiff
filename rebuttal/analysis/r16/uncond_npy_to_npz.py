"""R16 checkpoint 1: wrap a generator's unconditional pool (.npy, cells x 2,000 counts, int16, Windows' sampler output) into the npz that
table2_metrics.py v3 reads (counts, labels, genes). Nothing in the counts is changed (cast to int32 only; checked element by element).
Genes: the generator was trained on data/r16_hlca_fit/train.npy (byte-identical to the GPU server's), whose column order is meta.json "genes";
that list must equal r16_data's genes.json (the order table2_metrics uses for the real cells), else stop. labels = "uncond" for every
cell (unconditional mode ignores labels).
    python uncond_npy_to_npz.py POOL.npy R16_DATA_DIR FIT_META.json OUT.npz
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from r16_data import genes

pool, D, meta, out = sys.argv[1:5]
x = np.load(pool, allow_pickle=False); assert x.ndim == 2 and x.shape[1] == 2000 and x.dtype == np.int16, (x.shape, x.dtype)
g = np.asarray(genes(D)).astype(str); gm = np.asarray(json.load(open(meta))["genes"]).astype(str)
assert len(g) == 2000 and np.array_equal(g, gm), "generator gene order differs from r16_data genes.json"
c = x.astype(np.int32); assert np.array_equal(c, x) and c.min() >= 0
np.savez(out, counts=c, labels=np.full(len(c), "uncond"), genes=g)
z = np.load(out, allow_pickle=False); assert np.array_equal(z["counts"], x) and np.array_equal(z["genes"], g)
print(json.dumps({"pool": pool, "pool_sha256": hashlib.sha256(open(pool, "rb").read()).hexdigest(), "n_cells": int(len(c)),
                  "out": out, "out_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(),
                  "gene_hash": hashlib.sha256("\n".join(g).encode()).hexdigest(),
                  "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}, indent=1))
