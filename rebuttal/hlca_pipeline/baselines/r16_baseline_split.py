#!/usr/bin/env python
"""R16 baseline delivery split (Windows, 11-10; vice 10:09 naming). Reads the npz written by cfgen_generate.py (v2 0779f510) or by
scldm_output_to_npz.py (2f94ee0f) -- keys counts (int32, cells x 2000 HVG), labels, genes -- and writes, without changing any value:
  cond:   <method>_cond_<slug>_n<n>_seed<seed>.npy  one file per target class (rows in npz order), asserting exactly n rows each
  uncond: <method>_uncond_n<n>_seed<seed>.npy       all rows, asserting exactly n rows
plus manifest_<mode>.json: source npz + sha256, gene order (var_names), per-file shape/sha256, label counts, and every scalar key of the
npz (seed, ckpt/script sha, ...). Each output npy is re-read and compared to the npz rows before the manifest is written.
    python r16_baseline_split.py --npz X.npz --method cfgen --mode cond --n 2500 --seed 20260987 --outdir DIR
"""
import argparse, hashlib, json, os
import numpy as np
SLUG = {"type II pneumocyte": "typeIIpneumocyte", "classical monocyte": "classicalmonocyte",
        "multi-ciliated epithelial cell": "multiciliated", "plasma cell": "plasmacell", "ionocyte": "ionocyte"}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--method", required=True)
ap.add_argument("--mode", choices=("cond", "uncond"), required=True); ap.add_argument("--n", type=int, required=True)
ap.add_argument("--seed", type=int, required=True); ap.add_argument("--outdir", required=True); a = ap.parse_args()
z = np.load(a.npz, allow_pickle=False); X, lab, genes = z["counts"], z["labels"].astype(str), z["genes"].astype(str)
assert X.dtype == np.int32 and X.ndim == 2 and X.shape[1] == 2000 == len(genes) and len(lab) == len(X), (X.dtype, X.shape, len(genes))
os.makedirs(a.outdir, exist_ok=True); files = {}
u, c = np.unique(lab, return_counts=True); counts = {k: int(v) for k, v in zip(u, c)}
if a.mode == "cond":
    assert set(counts) == set(SLUG) and all(v == a.n for v in counts.values()), counts
    parts = [(f"{a.method}_cond_{SLUG[k]}_n{a.n}_seed{a.seed}.npy", np.flatnonzero(lab == k)) for k in SLUG]
else:
    assert len(X) == a.n, len(X); parts = [(f"{a.method}_uncond_n{a.n}_seed{a.seed}.npy", np.arange(len(X)))]
for name, rows in parts:
    p = os.path.join(a.outdir, name); np.save(p, X[rows]); assert np.array_equal(np.load(p), X[rows])
    files[name] = {"shape": list(X[rows].shape), "dtype": "int32", "sha256": sha(p), "label": None if a.mode == "uncond" else str(lab[rows[0]])}
m = {"method": a.method, "mode": a.mode, "n": a.n, "seed": a.seed, "source_npz": os.path.abspath(a.npz), "source_npz_sha256": sha(a.npz),
     "split_script_sha256": sha(__file__), "label_counts": counts, "files": files, "var_names": genes.tolist(),
     "npz_scalars": {k: z[k].item() for k in z.files if z[k].ndim == 0}}
json.dump(m, open(os.path.join(a.outdir, f"manifest_{a.mode}.json"), "w"), indent=1)
print("split", a.npz, "->", {k: v["shape"] for k, v in files.items()})
