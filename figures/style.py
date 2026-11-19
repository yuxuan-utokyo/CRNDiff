# -*- coding: utf-8 -*-
"""Shared figure style: the colour-blind-safe palette, summary loading, and figure saving.

Every figure saved through this module gets a sidecar json recording each number drawn on it, so
that no number in a figure can be transcribed by hand without showing up in the sidecar.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff.config import OUT        # noqa: E402

FIG = OUT / "figures"
SUMMARY = OUT / "summary"

C = {"blue": "#0072B2", "sky": "#56B4E9", "orange": "#E69F00", "vermilion": "#D55E00",
     "green": "#009E73", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000"}
MS, LW = 5.0, 1.8


def load_summary(name: str) -> dict:
    p = SUMMARY / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"summary missing: {p}  "
                         f"(run: python -m analysis.summarize --experiment {name})")
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, stem: str, numbers: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    numbers = dict(numbers)
    numbers["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (FIG / f"{stem}.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=1) + "\n",
                                      encoding="utf-8")
    print(f"[fig] {FIG / (stem + '.pdf')}")
