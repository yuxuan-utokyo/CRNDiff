# -*- coding: utf-8 -*-
"""Extract everything the toy figures need from the frozen runs, once.

The shipped copy of the extract is results/toy/figdata.npz with results/toy/figmeta.json
beside it, and the figure scripts read only those two files, so a figure's data source is
traceable and no figure touches the experiment tree. A rebuild writes into out/toy/ and
compares itself against the shipped copy rather than overwriting it; the comparison is
reported and nothing under results/ is ever written.

The configuration is frozen: the dataset stem, the request, the particle count, the delivered
count, the number of twist candidates, alpha, the ESS threshold and the resampling interval are
all written out into figmeta.json. The dataset stem is HARD-CODED here rather than read from the
toy package, because the package's default follows whatever experiment is running while the
figures must always come from the frozen configuration.

    python scripts/reproduce/build_toy_figure_data.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
TOY_DATA = REPO / "assets" / "toy" / "data"
TOY_RUNS = REPO / "results" / "toy" / "runs"
TOY_FIG  = REPO / "results" / "toy"
FIG_OUT  = REPO / "out" / "figures"
BUILD_OUT = REPO / "out" / "toy"          # a rebuild lands here; results/toy/ is never overwritten


from scripts.reproduce._array_compare import compare_npz
from toy2d.chain import request_weight_vector   # noqa: E402
from toy2d.metrics import request_pmf           # noqa: E402

DATA_STEM = "data2d_ring8_sym_v2"               # the configuration the paper uses; hard-coded on purpose
RUNS = TOY_RUNS
SEED = 20260901
MTAG = "M10000"
REQUEST = "diag_rare"

ARMS = ["tilt_only", "tilt_off_interval8", "tilt_on_interval8"]
KS = [32, 16, 8, 4, 0]                          # the columns of the strip figure; the last is the terminal resampling step
N_BG = 20000                                    # how many training points to draw for the grey background
BG_SEED = 7


def _run(arm: str) -> Path:
    hits = [p for p in sorted(RUNS.joinpath(arm).glob("*.json"))
            if MTAG in p.name and f"seed{SEED}.json" in p.name]
    if len(hits) != 1:
        raise SystemExit(f"{arm}: expected exactly one run, found {len(hits)}")
    return hits[0]


def main() -> int:
    data_dir = TOY_DATA
    cfg = json.loads((data_dir / f"{DATA_STEM}.json").read_text(encoding="utf-8"))
    pmf_components = np.load(data_dir / f"{DATA_STEM}_pmf.npz", allow_pickle=False)["components"]
    rp = request_pmf(pmf_components, request_weight_vector(cfg, REQUEST))

    out = {"target_dim0": rp.sum(1), "target_dim1": rp.sum(0)}
    meta = {"data_stem": DATA_STEM, "request": REQUEST, "seed": SEED,
            "ks": KS, "arms": ARMS, "runs": {}, "fired": {}, "is_fk": {}}

    # the grey background: a fixed subsample of the training data
    Xtr = np.load(data_dir / f"{DATA_STEM}.npz", allow_pickle=False)["X_train"]
    rng = np.random.default_rng(BG_SEED)
    out["bg"] = Xtr[rng.choice(len(Xtr), size=N_BG, replace=False)].astype(np.int16)

    for arm in ARMS:
        f = _run(arm)
        d = json.loads(f.read_text(encoding="utf-8"))
        z = np.load(f.with_suffix(".npz"), allow_pickle=False)
        meta["runs"][arm] = f.name
        meta["fired"][arm] = [int(k) for k in (d.get("resample_steps") or [])]
        meta["is_fk"][arm] = (d.get("sampler") == "fk")

        for k in KS:
            out[f"{arm}_k{k}_X"] = np.asarray(z[f"traj_x_{k}"], np.int16)
            key = f"traj_logw_{k}"
            if key in z.files:                       # the tilt-only arm carries no weights
                lw = np.asarray(z[key], np.float64)
                w = np.exp(lw - lw.max()); w /= w.sum()
            else:
                w = np.full(len(out[f"{arm}_k{k}_X"]), 1.0 / len(out[f"{arm}_k{k}_X"]))
            out[f"{arm}_k{k}_w"] = w

        Xd = np.asarray(z["X"], np.int16)
        out[f"{arm}_out_X"] = Xd
        # each delivered particle's ancestor, that is the index of its initial particle; the
        out[f"{arm}_out_anc"] = (np.asarray(z["traj_lineage_0"], np.int32)
                                 if "traj_lineage_0" in z.files
                                 else np.arange(len(Xd), dtype=np.int32))

    # the first coordinate before and after: the unweighted particles before the terminal
    for arm in ("tilt_off_interval8", "tilt_on_interval8"):
        out[f"{arm}_pre_dim0"] = out[f"{arm}_k0_X"][:, 0].copy()
        out[f"{arm}_post_dim0"] = out[f"{arm}_out_X"][:, 0].copy()

    BUILD_OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BUILD_OUT / "figdata.npz", **out)
    (BUILD_OUT / "figmeta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                            encoding="utf-8")
    print(f"[figdata] {DATA_STEM}  seed {SEED}  arms {ARMS}")
    for arm in ARMS:
        print(f"  {arm:22s} fired at {meta['fired'][arm]}  "
              f"ancestors {len(np.unique(out[f'{arm}_out_anc']))}")
    print(f"[out] {BUILD_OUT / 'figdata.npz'}")

    shipped = TOY_FIG / "figdata.npz"
    if shipped.exists():
        a = hashlib.sha256(shipped.read_bytes()).hexdigest()
        b = hashlib.sha256((BUILD_OUT / "figdata.npz").read_bytes()).hexdigest()
        ok, detail = compare_npz(shipped, BUILD_OUT / "figdata.npz")
        print(f"[compare] shipped {a}\n[compare] rebuilt {b}\n"
              f"[compare] {detail}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
