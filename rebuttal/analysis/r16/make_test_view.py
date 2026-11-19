#!/usr/bin/env python
"""R16: the TEST view for r13_score.py / score_with_npz.py (R13 convention: the test cells sit in the val slot, so val.npy /
val_celltype.npy are the test arrays and the purity ceiling is computed on the test cells). Train is the same symlink as the val
view. Test is densified from data/r16_hlca/test_rows*.npz (int32 CSR) and checked element by element against the CSR.
Built 11-10 only after the first R16 setting (CountsDiff K) was in team STATUS (04:31).
    python make_test_view.py   (writes data/r16_hlca_test_view/)
"""
import hashlib, json, os
from pathlib import Path
import numpy as np, scipy.sparse as sp

W = Path(os.path.expanduser("~/research/crndiff-rebuttal")); S = W / "data/r16_hlca"; F = W / "data/r16_hlca_fit"; O = W / "data/r16_hlca_test_view"
if O.exists():
    raise SystemExit(f"refusing to overwrite {O}")
O.mkdir()
for f in ("train.npy", "train_celltype.npy", "meta.json"):
    (O / f).symlink_to(F / f)
parts = sorted(S.glob("test_rows*.npz"), key=lambda p: int(p.stem.split("rows")[1].split("-")[0]))
ms = []
for p in parts:
    z = np.load(p); ms.append(sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])))
m = sp.vstack(ms).tocsr()
x = m.toarray(); assert x.dtype == np.int32 and (x >= 0).all()
assert (sp.csr_matrix(x) != m).nnz == 0, "dense test differs from CSR"
y = np.load(S / "test_celltype.npy", allow_pickle=True); assert len(y) == x.shape[0], (len(y), x.shape)
assert json.loads((S / "genes.json").read_text())["genes"] == json.loads((F / "meta.json").read_text())["genes"]
np.save(O / "val.npy", x); (O / "val_celltype.npy").symlink_to(S / "test_celltype.npy")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec = {"split_in_val_slot": "test", "test_sources": {p.name: sha(p) for p in parts}, "val_npy_sha256": sha(O / "val.npy"),
       "test_celltype_sha256": sha(S / "test_celltype.npy"), "train_link": str(F), "shape": list(x.shape), "script_sha256": sha(__file__)}
(O / "view_record.json").write_text(json.dumps(rec, indent=1)); print(rec)
