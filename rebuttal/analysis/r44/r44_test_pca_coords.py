#!/usr/bin/env python
r"""R44: HVG2K test cells in the frozen atlas PCA of Fig. 6 (frozen_atlas_pca_basis.npz cd0c9728,
PCA(30, rs 20261031) fit on cpm_log1p(val), first two components; never refit). The transform is hvgfig.FrozenPCA.project, copied verbatim:
(cpm_log1p(x) - mean_) @ comp2.T with cpm_log1p = log1p(x / lib * 1e4), lib == 0 -> 1, float64.
Gate (before any test cell is touched): projecting val.npy reproduces the v4 grid's real_xy within 1e-10 (scripts/figs/pca_src/pca_grid_coords_k128.npz)
and val_celltype.npy equals its real_type. Then test.npy (ad800326) is projected once -> runs/r44/test_pca_coords.npz (xy, type) + json.
One test read, logged in runs/r19/READ_LOG.tsv.
    python r44_test_pca_coords.py
"""
import hashlib, json, os, time
import numpy as np
W = "./work"; D = f"{W}/data/hvg2k"; O = f"{W}/runs/r44"; os.makedirs(O, exist_ok=True)
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
BAS = f"{W}/from_windows/rebuttal_code/assets/frozen_atlas_pca_basis.npz"; assert sha(BAS)[:8] == "cd0c9728"
assert sha(f"{D}/test.npy")[:8] == "ad800326" and sha(f"{D}/val.npy")[:8] == "49ce6c22"
def cpm_log1p(a):                                   # hvgfig.cpm_log1p, verbatim
    a = np.asarray(a, dtype=np.float64); lib = a.sum(1, keepdims=True); lib[lib == 0] = 1.0
    return np.log1p(a / lib * 1e4)
z = np.load(BAS); MEAN = z["mean_"].astype(np.float64); COMP2 = z["comp2"].astype(np.float64)
project = lambda x: (cpm_log1p(x) - MEAN) @ COMP2.T  # hvgfig.FrozenPCA.project, verbatim
V4 = np.load(f"{W}/scripts/figs/pca_src/pca_grid_coords_k128.npz")
zv = project(np.load(f"{D}/val.npy", mmap_mode="r")); yv = np.load(f"{D}/val_celltype.npy", allow_pickle=True).astype(str)
gate = float(np.abs(zv.astype(V4["real_xy"].dtype) - V4["real_xy"]).max()); gate64 = float(np.abs(zv - V4["real_xy"].astype(np.float64)).max())
assert np.array_equal(yv, V4["real_type"].astype(str)), "val labels differ from v4 real_type"
TOL = 1e-10   # dry run (val only, 17:2x): max |d| 2.84e-14 on |coords| <= 36, float64 BLAS rounding Windows vs Mac; not bit-exact
assert gate64 <= TOL, ("val gate", gate, gate64)
print(f"gate passed: val projection == v4 real_xy within {TOL} (max |d| {gate64:.2e}, float64)", flush=True)
# ---- test read (once) ----
xt = np.load(f"{D}/test.npy", mmap_mode="r"); yt = np.load(f"{D}/test_celltype.npy", allow_pickle=True).astype(str)
zt = project(xt).astype(V4["real_xy"].dtype); assert len(zt) == len(yt) == 45163
np.savez(f"{O}/test_pca_coords.npz", real_xy=zt, real_type=yt)
cnt = {t: int((yt == t).sum()) for t in ("Endothelial", "Myeloid", "Neuronal")}
json.dump({"what": "HVG2K test cells in the frozen atlas PCA of Fig. 6 (R44)", "basis_sha256": sha(BAS), "test_npy_sha256": sha(f"{D}/test.npy"),
           "test_celltype_sha256": sha(f"{D}/test_celltype.npy"), "n_test": len(yt), "target_counts": cnt, "dtype": str(zt.dtype),
           "gate": {"check": "project(val.npy) == pca_grid_coords_k128.npz::real_xy (v4)", "max_abs_delta_stored_dtype": gate, "max_abs_delta_float64": gate64, "tolerance": TOL, "bit_exact": gate64 == 0.0, "passed": True},
           "script_sha256": sha(__file__), "out_sha256": sha(f"{O}/test_pca_coords.npz"), "time": time.strftime("%m-%d %H:%M:%S")},
          open(f"{O}/test_pca_coords.json", "w"), indent=1)
with open(f"{W}/runs/r19/READ_LOG.tsv", "a") as f:
    f.write(f"{time.strftime('%m-%d_%H:%M:%S')}\tR44 HVG2K test cells projected into the frozen atlas PCA of Fig. 6 (r44_test_pca_coords.py {sha(__file__)[:8]})\truns/r44/test_pca_coords.npz\t{sha(f'{O}/test_pca_coords.npz')[:8]}\n")
print("test projected:", cnt, "out", sha(f"{O}/test_pca_coords.npz")[:8])
