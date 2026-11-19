#!/usr/bin/env python
"""R16: build the data view r13_score.py / score_with_npz.py read (train.npy, train_celltype.npy, val.npy, val_celltype.npy,
meta.json) for HLCA. Train is a symlink to data/r16_hlca_fit (the same arrays the discriminators and the purity ruler used);
val is densified from data/r16_hlca/val_rows0-46796.npz (int32 CSR) and checked element by element against the CSR. Val only;
test is not touched.
    python make_score_view.py   (writes data/r16_hlca_score/)
"""
import hashlib, json, os
from pathlib import Path
import numpy as np, scipy.sparse as sp

W = Path(os.path.expanduser("~/research/crndiff-rebuttal")); S = W / "data/r16_hlca"; F = W / "data/r16_hlca_fit"; O = W / "data/r16_hlca_score"
if O.exists():
    raise SystemExit(f"refusing to overwrite {O}")
O.mkdir()
for f in ("train.npy", "train_celltype.npy", "meta.json"):
    (O / f).symlink_to(F / f)
z = np.load(S / "val_rows0-46796.npz")
m = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
x = m.toarray(); assert x.dtype == np.int32 and (x >= 0).all()
assert (sp.csr_matrix(x) != m).nnz == 0, "dense val differs from CSR"
y = np.load(S / "val_celltype.npy", allow_pickle=True); assert len(y) == x.shape[0]
assert json.loads((S / "genes.json").read_text())["genes"] == json.loads((F / "meta.json").read_text())["genes"]
np.save(O / "val.npy", x); (O / "val_celltype.npy").symlink_to(S / "val_celltype.npy")   # the original file, as the purity freeze read it
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec = {"val_source": str(S / "val_rows0-46796.npz"), "val_source_sha256": sha(S / "val_rows0-46796.npz"), "val_npy_sha256": sha(O / "val.npy"),
       "val_celltype_sha256": sha(O / "val_celltype.npy"), "train_link": str(F), "shape_val": list(x.shape), "script_sha256": sha(__file__)}
(O / "view_record.json").write_text(json.dumps(rec, indent=1)); print(rec)
