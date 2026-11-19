#!/usr/bin/env python
"""R16 matched-n subsets. Reads ONLY cell ids and labels (val_celltype.npy, val_obs_names.txt,
test_celltype.npy, test_obs_names.txt), never expression values.
  val : type II pneumocyte, 2,500 of its val cells, without replacement, np.random.default_rng(20261330).
  test: type II pneumocyte and classical monocyte, 2,500 each of their test cells, without replacement; each class draws with
        a fresh np.random.default_rng(20261331), so neither draw depends on the other or on the order.
idx = sorted row positions in the split's arrays (the order of *_celltype.npy / *_obs_names.txt); obs_names = the matching cell ids.
    python make_subsets.py OUT_DIR
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

D = Path.home() / "research/crndiff-rebuttal/data/r16_hlca"; O = Path(sys.argv[1]); O.mkdir(parents=True, exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = [("val", "type II pneumocyte", 20261330), ("test", "type II pneumocyte", 20261331), ("test", "classical monocyte", 20261331)]
rec = {"rule": "the task notes 11-10 04:06", "n": 2500, "script_sha256": sha(__file__), "inputs": {}, "files": {}}
for split, t, seed in plan:
    y = np.load(D / f"{split}_celltype.npy", allow_pickle=True).astype(str)
    names = np.asarray((D / f"{split}_obs_names.txt").read_text().split("\n")); names = names[names != ""]
    assert len(names) == len(y), (split, len(names), len(y))
    for f in (f"{split}_celltype.npy", f"{split}_obs_names.txt"):
        rec["inputs"][f] = sha(D / f)
    pool = np.flatnonzero(y == t)
    idx = np.sort(np.random.default_rng(seed).choice(pool, 2500, replace=False))
    assert len(np.unique(idx)) == 2500 and (y[idx] == t).all()
    out = O / f"{split}_{t.replace(' ', '')}_n2500_s{seed}.npz"
    np.savez(out, idx=idx.astype(np.int64), obs_names=names[idx], target=np.asarray(t), split=np.asarray(split), seed=np.asarray(seed), n_class=np.asarray(len(pool)))
    rec["files"][out.name] = {"split": split, "target": t, "seed": seed, "n_class": int(len(pool)), "n": 2500, "sha256": sha(out)}
    print(out.name, len(pool), sha(out))
(O / "subsets_record.json").write_text(json.dumps(rec, indent=1))
