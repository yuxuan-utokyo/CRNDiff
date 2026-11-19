#!/usr/bin/env python
r"""R19: refit R1's two PCA spaces on TRAIN cells with their original recipes.
  pca2  : PCA(n_components=30, random_state=20261031) on cpm_log1p (float64) -> first 2 components (frozen_atlas_pca_basis.json recipe)
  pca50 : PCA(n_components=50, svd_solver="full") on cpm_log1p cast to float32 (the recipe that reproduces pca50_val.npz on val: mean to
          1.2e-7, min |cos| 0.999997; the original is not bit-reproducible on this machine)
Check first: the same code on VAL must reproduce the stored val bases (pca2 exactly, pca50 to |cos| >= 0.99999), else stop.
Fit set: ALL train cells (every type), as the val bases used all val cells; R1 counts neighbour types over the whole atlas.
    python r1_bases_train.py DATA_DIR ASSETS_DIR OUT.npz
"""
import hashlib, json, sys
import numpy as np
from sklearn.decomposition import PCA
D, A, out = sys.argv[1:4]
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
def cpm(x):
    x = np.asarray(x, dtype=np.float64); lib = x.sum(1, keepdims=True); return np.log1p(x / np.maximum(lib, 1) * 1e4)
def fit(split):
    x = cpm(np.load(f"{D}/{split}.npy", mmap_mode="r"))
    p2 = PCA(n_components=30, random_state=20261031).fit(x)
    p50 = PCA(n_components=50, svd_solver="full").fit(x.astype(np.float32))
    return {"mean_": p2.mean_, "comp2": p2.components_[:2], "evr2": p2.explained_variance_ratio_[:2], "mean50": p50.mean_, "components50": p50.components_,
            "explained50": p50.explained_variance_ratio_, "n_fit": len(x)}
cos = lambda a, b: float(np.min(np.abs((a * b).sum(1)) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)))
v = fit("val"); P2 = np.load(f"{A}/frozen_atlas_pca_basis.npz"); P50 = np.load(f"{A}/pca50_val.npz")
chk = {"pca2_mean_maxdiff": float(np.abs(v["mean_"] - P2["mean_"]).max()), "pca2_min_abs_cos": cos(v["comp2"], P2["comp2"]),
       "pca2_comp_maxdiff_signfixed": float(np.abs(v["comp2"] * np.sign((v["comp2"] * P2["comp2"]).sum(1))[:, None] - P2["comp2"]).max()),
       "pca50_mean_maxdiff": float(np.abs(v["mean50"] - P50["mean"]).max()), "pca50_min_abs_cos": cos(v["components50"], P50["components"])}
print("val check", json.dumps(chk))
assert chk["pca2_comp_maxdiff_signfixed"] < 1e-12 and chk["pca50_min_abs_cos"] >= 0.99999, chk
t = fit("train")
np.savez(out, mean_=t["mean_"], comp2=t["comp2"], evr2=t["evr2"], mean=t["mean50"].astype(np.float32), components=t["components50"].astype(np.float32),
         explained=t["explained50"], n_fit=t["n_fit"])
rec = {"val_check": chk, "n_fit_train": t["n_fit"], "evr_pc1_pc2_train": t["evr2"].tolist(), "out": out, "out_sha256": sha(out), "script_sha256": sha(__file__),
       "data_sha256": {f: sha(f"{D}/{f}")[:16] for f in ("train.npy", "val.npy")}}
json.dump(rec, open(out.replace(".npz", ".json"), "w"), indent=1); print(json.dumps(rec, indent=1))
