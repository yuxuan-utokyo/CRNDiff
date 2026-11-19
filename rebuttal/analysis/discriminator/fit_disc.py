#!/usr/bin/env python
"""Fit a residual discriminator with the unchanged tools/fit_apply_residual_hvg.py, then export the portable npz.

Run in the crndiff-fit env (python 3.11.5, numpy 1.24.3, sklearn 1.3.0), the versions of the existing discriminators.
The fit script reads <x-root>/_shared/data/all, so a link to the data folder is made inside --out. The export is
export_portable from tools/fit_selection_classifier.py with the same identity check; that module needs hvg.config,
which is Windows-only, so portable_predict is copied here.

    python fit_disc.py --data HVG2K --proposal POOL.npy --target Neuronal --fit-pos 1800 --val-pos 1000 --n-features 192 --out OUT
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True); ap.add_argument("--proposal", type=Path, required=True)
ap.add_argument("--target", required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--fit-script", type=Path, default=Path(__file__).resolve().parents[2] / "from_windows/rebuttal_code/tools/fit_apply_residual_hvg.py")
ap.add_argument("--threads", default="4")
a, passthrough = ap.parse_known_args()   # fit-pos, val-pos, n-features, seed and so on go to the fit script unchanged
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = a.threads
if a.out.exists():
    raise SystemExit(f"refusing to overwrite: {a.out}")
a.out.mkdir(parents=True)
(a.out / "_xroot/_shared/data").mkdir(parents=True)
(a.out / "_xroot/_shared/data/all").symlink_to(a.data.resolve(), target_is_directory=True)
cmd = [sys.executable, "-I", str(a.fit_script), "--x-root", str(a.out / "_xroot"), "--proposal", str(a.proposal),
       "--target", a.target, "-o", str(a.out), *passthrough]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

import joblib, numpy as np, sklearn, scipy

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def portable_predict(nodes, offsets, baseline, x):
    raw = np.full(len(x), baseline, dtype=np.float64)
    rows = np.arange(len(x))
    for start, end in zip(offsets[:-1], offsets[1:]):
        tree = nodes[int(start):int(end)]
        index = np.zeros(len(x), dtype=np.int64)
        for _ in range(len(tree)):
            node = tree[index]
            active = node["is_leaf"] == 0
            if not np.any(active):
                break
            rr = rows[active]
            nn = node[active]
            value = x[rr, nn["feature_idx"]]
            go_left = np.where(np.isnan(value), nn["missing_go_to_left"] != 0, value <= nn["num_threshold"])
            index[active] = np.where(go_left, nn["left"], nn["right"])
        else:
            raise RuntimeError("tree traversal did not reach leaves")
        raw += tree[index]["value"]
    return np.where(raw >= 0, 1.0 / (1.0 + np.exp(-raw)), np.exp(raw) / (1.0 + np.exp(raw)))


jl = a.out / "residual_classifier.joblib"
model = joblib.load(jl)
if model.n_trees_per_iteration_ != 1 or len(model.classes_) != 2:
    raise SystemExit("only binary one-tree-per-iteration HGBT is supported")
trees = [row[0].nodes for row in model._predictors]
if any(np.any(t["is_categorical"] != 0) for t in trees):
    raise SystemExit("categorical tree nodes are not supported")
offsets = np.concatenate(([0], np.cumsum([len(t) for t in trees]))).astype(np.int64)
nodes = np.concatenate(trees)
baseline = float(np.asarray(model._baseline_prediction).reshape(-1)[0])
x = np.random.default_rng(20260817).uniform(0.0, 10.0, size=(2048, model.n_features_in_)).astype(np.float32)
x[0, 0] = np.nan
error = float(np.max(np.abs(model.predict_proba(x)[:, 1] - portable_predict(nodes, offsets, baseline, x))))
if error > 1.0e-12:
    raise SystemExit(f"portable predictor identity failed: max abs error={error}")
port = a.out / "residual_classifier_portable.npz"
np.savez_compressed(port, nodes=nodes, offsets=offsets, baseline=np.asarray([baseline], dtype=np.float64),
                    n_features=np.asarray([model.n_features_in_], dtype=np.int64),
                    source_sha256=np.asarray([sha(jl)]), identity_max_abs_error=np.asarray([error], dtype=np.float64))
log = json.loads((a.out / "residual_log.json").read_text())
rec = {"portable_sha256": sha(port), "joblib_sha256": sha(jl), "proposal_sha256": sha(a.proposal), "fit_script_sha256": sha(a.fit_script),
       "identity_max_abs_error": error, "n_trees": len(trees), "n_nodes": int(len(nodes)), "heldout_roc_auc": log["classifier"]["heldout_roc_auc"],
       "n_iter": log["classifier"]["n_iter"], "n_features": log["classifier"]["n_features"], "seed": log["seed"], "splits": log["splits"],
       "command": cmd, "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}}
(a.out / "fit_record.json").write_text(json.dumps(rec, indent=1))
print({k: rec[k] for k in ("heldout_roc_auc", "n_iter", "n_features", "n_nodes", "identity_max_abs_error", "versions")}, flush=True)
