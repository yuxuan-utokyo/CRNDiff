#!/usr/bin/env python
"""R16 baselines: put a baseline's generated cells into the r16 layout. Nothing in the counts
is changed except the column order: columns are reordered BY GENE ID to data/r16_hlca/genes.json (the order table2_metrics.py and
score_with_npz.py use for the real cells); the gene sets must be equal, else stop. Counts must be integer-valued and >= 0; they are
cast to int32 and checked element by element against the input.
v2 (11-10 10:1x; v1 0bdfa783 archived in _archive_r16/): also reads Windows' delivery layout:
  _data_exchange/r16_baselines/<method>/<method>_uncond_n25000_seed20260987.npy, <method>_cond_<slug>_n2500_seed20260987.npy,
  with the gene order from --genes-file: a JSON list, a JSON object with exactly one of genes / var_names / gene_order / gene_ids,
  or a text file with one gene ID per line. An npz input (counts, labels, genes) carries its own genes.
  uncond: --gen FILE.npy|.npz; must have exactly --uncond-n rows (pre-registered 25,000). Writes OUTDIR/<method>_uncond.npz
          (counts, labels "uncond", genes) for table2_metrics.py v3.
  cond:   --gen DIR (the 5 per-class npy) or FILE.npz (labels); each class must have exactly --n rows (pre-registered 2,500); else stop
         . Writes OUTDIR/<method>_cond_<slug>_n<N>.npy.
Writes OUTDIR/<method>_<mode>_record.json with input/output sha256.
    python base_to_r16.py --gen DIR/cfgen_uncond_n25000_seed20260987.npy --genes-file DIR/manifest.json --mode uncond --method cfgen --outdir runs/r16/ckpt1
    python base_to_r16.py --gen DIR --genes-file DIR/manifest.json --mode cond --method cfgen --outdir from_windows/r16_baselines/cfgen/r16
"""
import argparse, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r16_data import genes

W = "./work"
TARGETS = [("classical monocyte", "classicalmonocyte"), ("multi-ciliated epithelial cell", "multiciliated"), ("plasma cell", "plasmacell"),
           ("ionocyte", "ionocyte"), ("type II pneumocyte", "typeIIpneumocyte")]
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
# v3 (11-10 12:3x; v2 924d2bc9 archived): R16 replies use 4 classes. The 4 reply classes stay required at exactly
# --n cells; multi-ciliated is optional (converted if delivered, skipped and recorded if not).
OPTIONAL = "multiciliated"
ap = argparse.ArgumentParser(); ap.add_argument("--gen", required=True); ap.add_argument("--mode", choices=("uncond", "cond"), required=True)
ap.add_argument("--method", required=True); ap.add_argument("--outdir", required=True); ap.add_argument("--n", type=int, default=2500)
ap.add_argument("--uncond-n", type=int, default=25000); ap.add_argument("--genes-file"); ap.add_argument("--seed", type=int, default=20260987)
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)


def read_genes(f):
    if f.endswith(".json"):
        j = json.load(open(f))
        if isinstance(j, dict):
            k = [x for x in ("genes", "var_names", "gene_order", "gene_ids") if x in j]
            assert len(k) == 1, f"{f}: expected exactly one gene key, found {k} (keys {list(j)[:20]})"; j = j[k[0]]
        return np.asarray(j, dtype=str)
    return np.asarray([l.strip() for l in open(f) if l.strip()], dtype=str)


g = genes(f"{W}/data/r16_hlca"); assert len(g) == 2000 and len(set(g)) == 2000
inputs = {}
if a.gen.endswith(".npz"):
    z = np.load(a.gen, allow_pickle=False); X, lab, gg = z["counts"], np.asarray(z["labels"]).astype(str), np.asarray(z["genes"]).astype(str)
    inputs[a.gen] = sha(a.gen)
else:
    assert a.genes_file, "--genes-file is required for npy input"; gg = read_genes(a.genes_file); inputs[a.genes_file] = sha(a.genes_file)
    if a.mode == "uncond":
        X = np.load(a.gen, allow_pickle=False); lab = np.full(len(X), "uncond"); inputs[a.gen] = sha(a.gen)
    else:
        X, lab = [], []
        for t, tok in TARGETS:
            p = f"{a.gen}/{a.method}_cond_{tok}_n{a.n}_seed{a.seed}.npy"
            if tok == OPTIONAL and not os.path.exists(p): continue   # multi-ciliated optional since the task notes 11-10 12:25
            x = np.load(p, allow_pickle=False); inputs[p] = sha(p)
            X.append(x); lab += [t] * len(x)
        X = np.concatenate(X); lab = np.asarray(lab)
assert X.ndim == 2 and X.shape == (len(lab), len(gg)), (X.shape, len(lab), len(gg))
assert len(set(gg)) == len(gg) and set(gg) == set(g), f"gene sets differ: {len(set(gg) - set(g))} extra, {len(set(g) - set(gg))} missing"
pos = {x: i for i, x in enumerate(gg)}; idx = np.array([pos[x] for x in g])
Y = np.asarray(X)[:, idx]; C = Y.astype(np.int32)
assert np.array_equal(C, Y) and C.min() >= 0, "counts are not non-negative integers"
rec = {"method": a.method, "mode": a.mode, "inputs_sha256": inputs, "input_dtype": str(X.dtype), "n_cells_in": int(len(C)),
       "gene_order_changed": bool(not np.array_equal(gg, g)), "gene_hash_r16": hashlib.sha256("\n".join(g).encode()).hexdigest(),
       "script_sha256": sha(__file__), "outputs": {}}
if a.mode == "uncond":
    assert len(C) == a.uncond_n, f"uncond pool has {len(C)} cells, expected {a.uncond_n}"
    out = f"{a.outdir}/{a.method}_uncond.npz"; np.savez(out, counts=C, labels=lab, genes=g)
    r = np.load(out, allow_pickle=False); assert np.array_equal(r["counts"], Y) and np.array_equal(r["genes"], g)
    rec["outputs"][out] = {"n": int(len(C)), "sha256": sha(out)}
else:
    u, k = np.unique(lab, return_counts=True); rec["labels_in"] = {str(x): int(y) for x, y in zip(u, k)}
    for t, tok in TARGETS:
        m = lab == t
        if tok == OPTIONAL and m.sum() == 0: rec["outputs"][f"{tok} (not delivered)"] = None; continue
        assert m.sum() == a.n, f"{t}: {int(m.sum())} cells, expected {a.n}"
        out = f"{a.outdir}/{a.method}_cond_{tok}_n{a.n}.npy"; np.save(out, C[m])
        assert np.array_equal(np.load(out), Y[m]); rec["outputs"][out] = {"target": t, "n": int(m.sum()), "sha256": sha(out)}
    rec["other_labels_ignored"] = sorted(set(u) - {t for t, _ in TARGETS})
json.dump(rec, open(f"{a.outdir}/{a.method}_{a.mode}_record.json", "w"), indent=1); print(json.dumps(rec, indent=1))
