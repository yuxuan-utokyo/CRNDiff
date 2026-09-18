# -*- coding: utf-8 -*-
"""E21: SCC and exact Gaussian-kernel MMD^2 for Table 1 arms.

The metric contract is frozen in ticket 099 and the finite-sample details in
`HVG2K/PREREG_E21_scc_mmd_implementation.md`, both written before this script
produced any metric.  Every number is emitted to E21_scc_mmd.json.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import spearmanr
import torch


HVG = Path(__file__).resolve().parents[2]
BASE = HVG.parent
XROOT = BASE.parent / "Experiments" / "X"
DATA = XROOT / "_shared" / "data" / "all"
XRES = XROOT / "Baseline" / "OURS" / "results"
RUNS = HVG / "results" / "runs"
OUT = HVG / "results" / "diagnostics" / "E21_scc_mmd.json"
PREREG = HVG / "PREREG_E21_scc_mmd_implementation.md"

SEED = 20261071
TARGET_SUM = 10000.0
BLOCK = 1024
TYPES = {"Endothelial": 10057, "Myeloid": 2302, "Neuronal": 396}


def one(pattern_root: Path, pattern: str) -> Path:
    hits = sorted(pattern_root.glob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"expected one source for {pattern_root / pattern}, got {hits}")
    return hits[0]


def method_sources(cell_type: str) -> dict[str, Path]:
    return {
        "scVI": one(HVG / "baseline_scvi" / "out" / "counts",
                    f"scvi_cond_{cell_type}_*.npy"),
        "CFGen": one(HVG / "baseline_cfgen" / "out" / "counts",
                     f"cfgen_cond_{cell_type}_*.npy"),
        "scANVI": one(HVG / "baseline_scanvi" / "out" / "counts",
                      f"scanvi_cond_{cell_type}_*.npy"),
        "OURS (tilt)": XRES / (
            "linspace_T_O_K32_spilot_run1c_s20260625_samp20260902_batched1_"
            f"chunk256_CONDFULL_{cell_type}_tables_kz_n20000_g0p4_noa4.npy"),
        "OURS (tilt+FK)": {
            "Endothelial": RUNS / (
                "Endothelial_clean_fkv2_fk_no_trigger_tiltevery_trigscheduled_"
                "K32_M40000_N20000_alpha1p0_sc0p25_int8_seed20260920.npy"),
            "Myeloid": RUNS / (
                "Myeloid_residual_fkv2_fk_tiltevery_trigscheduled_K32_M40000_"
                "N20000_alpha2p0_sc0p5_int8_seed20260994.npy"),
            "Neuronal": RUNS / (
                "Neuronal_residual_fkv2_fk_tiltevery_trigscheduled_K32_M10000_"
                "N2500_alpha2p0_sc0p5_int8_seed20261005.npy"),
        }[cell_type],
    }


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cp10k_log1p(raw: np.ndarray) -> tuple[np.ndarray, int]:
    x = np.asarray(raw, dtype=np.float32)
    lib = x.sum(axis=1, dtype=np.float64)
    zero = int(np.count_nonzero(lib == 0))
    scale = np.zeros_like(lib, dtype=np.float32)
    nz = lib > 0
    scale[nz] = (TARGET_SUM / lib[nz]).astype(np.float32)
    out = np.log1p(x * scale[:, None], dtype=np.float32)
    if not np.isfinite(out).all():
        raise SystemExit("non-finite CP10K+log1p values")
    return out, zero


def exact_bandwidth(real: torch.Tensor) -> float:
    n = real.shape[0]
    d = torch.cdist(real, real, p=2,
                    compute_mode="use_mm_for_euclid_dist")
    mask = ~torch.eye(n, dtype=torch.bool, device=real.device)
    sigma = float(torch.median(d[mask]).item())
    del d, mask
    torch.cuda.empty_cache()
    if not math.isfinite(sigma) or sigma <= 0:
        raise SystemExit(f"invalid median bandwidth {sigma}")
    return sigma


@torch.no_grad()
def kernel_mean(a: torch.Tensor, b: torch.Tensor, sigma: float) -> float:
    total = torch.zeros((), dtype=torch.float64, device=a.device)
    denom = 2.0 * sigma * sigma
    for i in range(0, a.shape[0], BLOCK):
        ai = a[i:i + BLOCK]
        an = (ai * ai).sum(1)[:, None]
        for j in range(0, b.shape[0], BLOCK):
            bj = b[j:j + BLOCK]
            bn = (bj * bj).sum(1)[None, :]
            d2 = (an + bn - 2.0 * (ai @ bj.T)).clamp_min_(0.0)
            total += torch.exp(-d2 / denom).sum(dtype=torch.float64)
    return float((total / (a.shape[0] * b.shape[0])).item())


@torch.no_grad()
def mmd2_biased(real: torch.Tensor, gen: torch.Tensor, sigma: float,
                k_real_real: float) -> tuple[float, dict[str, float]]:
    k_gg = kernel_mean(gen, gen, sigma)
    k_rg = kernel_mean(real, gen, sigma)
    value = k_real_real + k_gg - 2.0 * k_rg
    if value < -1e-6:
        raise SystemExit(f"MMD^2 numerical failure: {value}")
    value = max(value, 0.0)
    return value, {"mean_Krr": k_real_real,
                   "mean_Kgg": k_gg, "mean_Krg": k_rg}


def select_rows(arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if arr.shape[0] < n:
        raise SystemExit(f"source has {arr.shape[0]} rows but matched n is {n}")
    if arr.shape[0] == n:
        return np.asarray(arr)
    idx = np.sort(rng.choice(arr.shape[0], n, replace=False))
    return np.asarray(arr[idx])


def score(ref_np: np.ndarray, ref_gpu: torch.Tensor, gen_raw: np.ndarray,
          sigma: float, k_rr: float) -> dict:
    transformed, zero = cp10k_log1p(gen_raw)
    scc = float(spearmanr(ref_np.mean(0, dtype=np.float64),
                          transformed.mean(0, dtype=np.float64)).statistic)
    if not math.isfinite(scc):
        raise SystemExit("non-finite SCC")
    gen_gpu = torch.from_numpy(transformed).to(ref_gpu.device)
    mmd2, terms = mmd2_biased(ref_gpu, gen_gpu, sigma, k_rr)
    del gen_gpu, transformed
    torch.cuda.empty_cache()
    return {"scc": scc, "mmd2_rbf_biased": mmd2,
            "zero_library_rows": zero, "kernel_terms": terms}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite authoritative output: {OUT}")
    if not PREREG.exists():
        raise SystemExit(f"missing implementation freeze: {PREREG}")
    if not torch.cuda.is_available():
        raise SystemExit("exact matched-n E21 evaluator requires CUDA block acceleration")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0")

    Xv = np.load(DATA / "val.npy", mmap_mode="r")
    yv = np.load(DATA / "val_celltype.npy", allow_pickle=True).astype(str)
    Xt = np.load(DATA / "train.npy", mmap_mode="r")
    yt = np.load(DATA / "train_celltype.npy", allow_pickle=True).astype(str)

    out = {
        "ticket": "099 s1 E21",
        "status": "PASS",
        "preregistration": str(PREREG.relative_to(BASE)),
        "definitions": {
            "representation": "library-size normalize to 10000, then log1p",
            "scc": "Spearman correlation of 2000-gene population mean vectors",
            "kernel": "exp(-squared_euclidean/(2*sigma^2))",
            "bandwidth": "exact median of all non-diagonal real-real Euclidean distances",
            "mmd2": "biased empirical Gaussian-kernel MMD squared, exact all pairs",
        },
        "seed": SEED,
        "target_sum": TARGET_SUM,
        "block_size": BLOCK,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "torch": torch.__version__,
                     "cuda_device": torch.cuda.get_device_name(0)},
        "types": {},
    }

    started = time.perf_counter()
    for cell_type, n in TYPES.items():
        t0 = time.perf_counter()
        print(f"\n=== {cell_type}: matched n={n} ===", flush=True)
        vi = np.flatnonzero(yv == cell_type)
        ti = np.flatnonzero(yt == cell_type)
        if len(vi) != n:
            raise SystemExit(f"{cell_type}: expected {n} validation cells, found {len(vi)}")
        rng = np.random.default_rng(SEED)
        ref_raw = np.asarray(Xv[vi])
        floor_raw = select_rows(Xt[ti], n, rng)
        ref_np, ref_zero = cp10k_log1p(ref_raw)
        ref_gpu = torch.from_numpy(ref_np).to(device)
        sigma = exact_bandwidth(ref_gpu)
        k_rr = kernel_mean(ref_gpu, ref_gpu, sigma)
        print(f"[bandwidth] sigma={sigma:.10f}  gamma={1/(2*sigma*sigma):.12g}",
              flush=True)

        floor = score(ref_np, ref_gpu, floor_raw, sigma, k_rr)
        floor.update({"n": n, "source": str((DATA / "train.npy").resolve()),
                      "source_selection": f"cell_type={cell_type}, without replacement seed {SEED}"})
        rows = {"floor (real train vs real val)": floor}

        sources = method_sources(cell_type)
        for method, path in sources.items():
            if not path.exists():
                raise SystemExit(f"missing {method} source: {path}")
            arr = np.load(path, mmap_mode="r")
            raw = select_rows(arr, n, rng)
            result = score(ref_np, ref_gpu, raw, sigma, k_rr)
            result.update({"n": n, "n_source": int(arr.shape[0]),
                           "source": str(path.resolve())})
            rows[method] = result
            print(f"[{method}] SCC={result['scc']:.10f}  "
                  f"MMD2={result['mmd2_rbf_biased']:.10g}", flush=True)

        best = min(sources, key=lambda k: rows[k]["mmd2_rbf_biased"])
        out["types"][cell_type] = {
            "n": n, "n_val": int(len(vi)), "n_train": int(len(ti)),
            "underpowered": n < 2000,
            "reference": str((DATA / "val.npy").resolve()),
            "reference_zero_library_rows": ref_zero,
            "bandwidth_sigma": sigma,
            "kernel_gamma": 1.0 / (2.0 * sigma * sigma),
            "shared_bandwidth_across_all_methods": True,
            "rows": rows,
            "mechanical_best_mmd2_method": best,
            "walltime_s": time.perf_counter() - t0,
        }
        del ref_gpu, ref_np
        torch.cuda.empty_cache()

    out["walltime_s"] = time.perf_counter() - started
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n[out] {OUT}", flush=True)
    print(f"[md5] {md5(OUT)}", flush=True)


if __name__ == "__main__":
    main()
