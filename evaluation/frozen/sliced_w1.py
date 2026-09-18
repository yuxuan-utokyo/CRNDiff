# -*- coding: utf-8 -*-
"""Corrected sliced-W1, built on the project's own verified 1-D W1.

WHY THIS FILE EXISTS
--------------------
`Experiments/Toy/_shared/code/metrics.py` has a latent bug in `swd`:

    v = rng.standard_normal((n_proj, dim))   # make_projections -> [n_proj, dim]
    ...
    xp = x @ projections.T                   # [n, n_proj]   correct
    ds = [w1_1d(xp[:, j], yp[:, j]) for j in range(projections.shape[1])]
                                             #      ^^^^^^^^^^^^^^^^^^^^^
                                             # shape[1] is DIM, not n_proj

The loop bound is the data dimension instead of the projection count. Two
consequences, both verified by instantiating the function:

  * dim < n_proj (the 10-D toy, dim=10, n_proj=100): it silently averages the
    FIRST 10 projections and reports that as a 100-projection estimate. It does
    not crash, so every 10-D sliced-W1 in this project so far used 10 slices.
  * dim > n_proj (HVG, dim=2000, n_proj=512): IndexError.

`metrics.py` is a read-only reference and is NOT edited. This module reuses its
verified 1-D estimator `w1_1d` unchanged and fixes only the loop bound, so the
numerical definition of the 1-D distance is identical to every earlier table.

`sliced_w1_shared_slices` additionally subsamples BOTH inputs to a common
`max_points` with one RNG, because the parent applies its cap to each side
independently.
"""

from __future__ import annotations

import numpy as np


def sliced_w1(x, y, projections, w1_1d, *, max_points: int = 8000,
              seed: int = 42) -> float:
    """Mean 1-D W1 over ALL `projections` rows. `projections` is [n_proj, dim]."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) > max_points:
        x = x[rng.choice(len(x), size=max_points, replace=False)]
    if len(y) > max_points:
        y = y[rng.choice(len(y), size=max_points, replace=False)]
    xp = x @ projections.T          # [n, n_proj]
    yp = y @ projections.T
    n_proj = projections.shape[0]   # <- the fix: rows, not columns
    return float(np.mean([w1_1d(xp[:, j], yp[:, j]) for j in range(n_proj)]))
