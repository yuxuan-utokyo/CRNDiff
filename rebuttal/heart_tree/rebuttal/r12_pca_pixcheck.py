"""Pixel comparison of rebuttal/out/R12/pca_grid/v3_replica.png against out/figures/fig_pca_baselines_all_v3.png (read-only)."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
a = Image.open(ROOT / "out/figures/fig_pca_baselines_all_v3.png"); b = Image.open(ROOT / "rebuttal/out/R12/pca_grid/v3_replica.png")
print("size", a.size, b.size, "info", {k: str(v)[:60] for k, v in a.info.items()}, {k: str(v)[:60] for k, v in b.info.items()})
if a.size == b.size:
    x = np.asarray(a.convert("RGB"), dtype=int); y = np.asarray(b.convert("RGB"), dtype=int); d = np.abs(x - y)
    print("max_abs_pixel_diff", int(d.max()), "n_pixels_differ", int((d.sum(2) > 0).sum()), "of", d.shape[0] * d.shape[1])
    m = d.sum(2) > 0; ys, xs = np.nonzero(m)
    print("diff bbox x", xs.min(), xs.max(), "y", ys.min(), ys.max())
    H, W = m.shape; gy, gx = 6, 10
    print("diff counts on a 6x10 grid:"); print(np.array([[int(m[i*H//gy:(i+1)*H//gy, j*W//gx:(j+1)*W//gx].sum()) for j in range(gx)] for i in range(gy)]))
    Image.fromarray((m * 255).astype(np.uint8)).save(ROOT / "rebuttal/out/R12/pca_grid/v3_replica_diffmask.png")
