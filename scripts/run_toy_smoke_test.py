"""Exercise the shipped toy network and 32-step samplers (CPU, no atlas required).

Requires torch, scikit-learn and joblib in addition to requirements.txt.
This is a runtime and reproducibility check, not a quality benchmark or training run.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import torch
    from toy2d.chain import load_everything, reward_model_for, theta_for
    from toy2d.sampler import run_fk, run_tilt_only

    torch.set_num_threads(2)
    d = load_everything("cpu")
    args = (d["net"], "cpu", d["chain"], d["terminal"], theta_for("diag_rare"))
    reward = reward_model_for(d["discriminators"], "diag_rare", "proposal_residual")
    budget = dict(n=64, n_out=32, seed=20260918)
    cases = {
        "tilt_only": lambda: run_tilt_only(*args, **budget),
        "fk_default": lambda: run_fk(*args, reward, alpha=1.0, J=16, **budget),
        # Exercise the existing unconditional-resampling switch to cover this branch
        # even when a small particle cloud never crosses the default ESS threshold.
        "fk_resampling_branch": lambda: run_fk(*args, reward, alpha=1.0, J=16,
                                               ess_frac=1.0, **budget),
        "endpoint_control": lambda: run_fk(*args, reward, alpha=1.0, J=16,
                                           enable_intermediate_resampling=False, **budget),
    }
    summary = {}
    for name, run in cases.items():
        x, diag = run()
        repeat, _ = run()
        assert x.shape == (32, 2), (name, x.shape)
        assert np.issubdtype(x.dtype, np.integer) and (x >= 0).all(), name
        assert np.array_equal(x, repeat), f"{name}: same-seed output changed"
        if name == "fk_resampling_branch":
            assert diag["n_intermediate_resamples"] > 0, diag
        if name == "endpoint_control":
            assert diag["telescope_max_abs_error"] < 1e-10, diag
        summary[name] = dict(shape=list(x.shape), same_seed_exact=True,
                             NFE=diag["NFE"],
                             intermediate_resamples=diag["n_intermediate_resamples"])
        print(f"PASS {name}")
    out = ROOT / "out" / "runtime" / "toy_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
