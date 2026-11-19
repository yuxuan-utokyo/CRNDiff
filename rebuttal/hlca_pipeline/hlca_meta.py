# the GPU server, venv-hlca, read-only metadata of the HLCA h5ad files (backed mode: the matrices are not loaded).
import sys, json, anndata as ad, numpy as np, pandas as pd
for f in sys.argv[1:]:
    a = ad.read_h5ad(f, backed="r")
    print(f"=== {f}\nshape {a.shape}\nX type {type(a.X).__name__} dtype {getattr(a.X, 'dtype', None)}")
    print("layers", list(a.layers.keys())); print("obsm", list(a.obsm.keys())); print("varm", list(a.varm.keys())); print("uns", list(a.uns.keys()))
    print("obs columns", list(a.obs.columns)); print("var columns", list(a.var.columns)); print("var index name", a.var.index.name, "first genes", list(a.var_names[:5]))
    for c in a.obs.columns:
        s = a.obs[c]; nu = s.nunique()
        print(f"  obs {c}: dtype {s.dtype} n_unique {nu}" + (f" values {s.value_counts().to_dict()}" if nu <= 60 else f" e.g. {list(s.astype(str).unique()[:5])}"))
    for c in a.var.columns:
        s = a.var[c]; nu = s.nunique()
        print(f"  var {c}: dtype {s.dtype} n_unique {nu}" + (f" values {s.value_counts().to_dict()}" if nu <= 10 else f" e.g. {list(s.astype(str).unique()[:5])}"))
    a.file.close()
