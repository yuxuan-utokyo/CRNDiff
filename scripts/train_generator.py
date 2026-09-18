"""Public launcher for the unchanged frozen unconditional trainer.

Example: python scripts/train_generator.py --name smoke --steps 2 --batch 4
Small budgets check execution only; they do not reproduce the trained paper model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="unique checkpoint name, without directory")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", type=Path, help="weight-only restart, as in the frozen trainer")
    a = ap.parse_args()
    if not a.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in a.name):
        ap.error("--name must contain only ASCII letters, digits, underscores or hyphens")
    for key in ("steps", "batch"):
        if getattr(a, key) is not None and getattr(a, key) <= 0:
            ap.error(f"--{key} must be positive")

    from crndiff import config as C
    for name in ("meta.json", "train.npy", "val.npy"):
        C.require(C.LEGACY_DATA / "all" / name, "atlas training input; see README section 6")
    if a.resume is not None:
        C.require(a.resume, "weight-only restart checkpoint")
    C.add_base_code_to_path()
    import config_hvg2k as CFG
    import train_pilot

    seed = CFG.SEED_PILOT if a.seed is None else a.seed
    models = CFG.ROOT / "Baseline" / "OURS" / "models"
    stem = f"{a.name}_s{seed}"
    if any((models / (stem + suffix)).exists() for suffix in (".pt", ".pt.last", ".pt.best", ".json")):
        raise SystemExit(f"refusing to overwrite training outputs: {models / stem}")
    # The frozen loop saves .last/.best before train_pilot creates these directories.
    models.mkdir(parents=True, exist_ok=True)
    (CFG.ROOT / "Baseline" / "OURS" / "loss").mkdir(parents=True, exist_ok=True)
    return train_pilot.main()  # original argv, defaults, optimiser and numerical loop


if __name__ == "__main__":
    raise SystemExit(main())
