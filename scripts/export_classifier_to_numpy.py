"""Export frozen sklearn HistGradientBoosting classifiers to a NumPy tree format.

Run this with any local environment that can unpickle the original joblib.
The export is inference-only and is checked against ``predict_proba`` before it
is written.  It does not fit or alter the classifier.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

try:
    import joblib
except ModuleNotFoundError as _exc:                                      # noqa: E402
    raise SystemExit(
        "this entry point needs joblib, which the offline reproduction does not "
        "install. See README section 7; in short: pip install joblib scikit-learn") from _exc
import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def predict(nodes: np.ndarray, offsets: np.ndarray, baseline: float,
            x: np.ndarray) -> np.ndarray:
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
            go_left = np.where(np.isnan(value), nn["missing_go_to_left"] != 0,
                               value <= nn["num_threshold"])
            index[active] = np.where(go_left, nn["left"], nn["right"])
        else:
            raise RuntimeError("tree traversal did not reach leaves")
        raw += tree[index]["value"]
    return np.where(raw >= 0, 1.0 / (1.0 + np.exp(-raw)),
                    np.exp(raw) / (1.0 + np.exp(raw)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("joblib")
    ap.add_argument("output")
    ap.add_argument("--seed", type=int, default=20260822)
    a = ap.parse_args()
    src, out = Path(a.joblib), Path(a.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite: {out}")
    model = joblib.load(src)
    if model.n_trees_per_iteration_ != 1 or len(model.classes_) != 2:
        raise SystemExit("only binary one-tree-per-iteration HGBT is supported")
    trees = [row[0].nodes for row in model._predictors]
    if any(np.any(t["is_categorical"] != 0) for t in trees):
        raise SystemExit("categorical tree nodes are not supported")
    offsets = np.concatenate(([0], np.cumsum([len(t) for t in trees]))).astype(np.int64)
    nodes = np.concatenate(trees)
    baseline = float(np.asarray(model._baseline_prediction).reshape(-1)[0])

    rng = np.random.default_rng(a.seed)
    x = rng.uniform(0.0, 10.0, size=(2048, model.n_features_in_)).astype(np.float32)
    x[0, 0] = np.nan
    expected = model.predict_proba(x)[:, 1]
    actual = predict(nodes, offsets, baseline, x)
    error = float(np.max(np.abs(expected - actual)))
    if error > 1.0e-12:
        raise SystemExit(f"portable predictor identity failed: max abs error={error}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        nodes=nodes,
        offsets=offsets,
        baseline=np.asarray([baseline], dtype=np.float64),
        n_features=np.asarray([model.n_features_in_], dtype=np.int64),
        source_sha256=np.asarray([sha256_file(src)]),
        identity_max_abs_error=np.asarray([error], dtype=np.float64),
    )
    print(f"[gate] sklearn vs portable max abs error = {error:.3e}")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
