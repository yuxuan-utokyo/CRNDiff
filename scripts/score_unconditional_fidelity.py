# -*- coding: utf-8 -*-
"""The unconditional fidelity table. CPU only.

The reference is a fixed subsample of the validation set, drawn once on a fixed seed and shared
by every generator. From each generator and seed, the first cells of the pool are taken, with no
filtering.

Every metric is imported rather than rewritten: W1 and sliced-W1 as functions from the
distance scorer with the reference swapped, the dispersion from the frozen figure module, and
the correlations from the frozen ruler. Each is labelled in the output with the module it came
from.

    python scripts/score_unconditional_fidelity.py --help
"""
from __future__ import annotations

# the thread count must be capped BEFORE numpy is imported: on a machine with many logical
# cores an uncapped BLAS saturates it (observed twice, with the machine locking up).
# this ran alongside training). The thread count changes nothing but parallelism, and the
# W1 convention here is NOT the frozen ruler, so eight threads are used rather than one.
import os                                                            # noqa: E402

_T = os.environ.get("T200_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse                                                      # noqa: E402
import importlib.util                                                # noqa: E402
import json                                                          # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from crndiff.io import sha256_file                                       # noqa: E402

VAL_SUB_SEED = 20261200
N_SUB = 20000
N_PROJ = 512
PROJ_SEED = 20260906           # the distance scorer's own projection seed, reused
ARCHIVE = C.ARCHIVE
F8 = ARCHIVE / "viz" / "make_F8_dispersion_baselines.py"


def heterogeneity(raw: np.ndarray) -> dict:
    """Cell-to-cell heterogeneity diagnostics.

        The existing diagnostics cannot see 'a crowd of average cells': the aggregate marginals
        can fit well while every cell looks the same. These columns look at exactly that.

        These columns were added AFTER seeing a result and are NOT pre-registered; no criterion
        was changed. They are descriptive only and are computed symmetrically for every
    """
    x = np.asarray(raw, dtype=np.float64)
    lib = x.sum(axis=1)
    det = (x > 0).sum(axis=1).astype(np.float64)

    def cv(v):
        m = float(v.mean())
        return float(v.std(ddof=1) / m) if m else None

    def skew(v):
        m, s = float(v.mean()), float(v.std(ddof=1))
        return float((((v - m) / s) ** 3).mean()) if s else None

    return {
        "libsize_mean": float(lib.mean()), "libsize_sd": float(lib.std(ddof=1)),
        "libsize_CV": cv(lib), "libsize_skew": skew(lib),
        "libsize_p50": float(np.quantile(lib, 0.5)),
        "libsize_p99": float(np.quantile(lib, 0.99)),
        "libsize_p99_over_p50": float(np.quantile(lib, 0.99) / max(np.quantile(lib, 0.5), 1e-12)),
        "detected_genes_mean": float(det.mean()), "detected_genes_CV": cv(det),
        "detected_genes_skew": skew(det),
        "detected_genes_p1_p50_p99": [float(np.quantile(det, q)) for q in (0.01, 0.5, 0.99)],
    }


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True,
                    help="<generator>:<seed>:<npy> triples of generator, seed and array")
    ap.add_argument("--out", default=str(C.OUT / "summary" / "200_uncond_fidelity.json"))
    ap.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda",
                    help="where to compute the MMD^2 kernel matrix. A 20000x20000 kernel is about "
                         "8e12 flops on a cpu and only the GPU can do it in reasonable time, so this "
                         "block must run on its own while the GPU is idle and never alongside the "
                         "training queue.")
    a = ap.parse_args()

    import evaluation.score_distances_logcp10k as SD                          # noqa: PLC0415
    import torch                                                      # noqa: PLC0415
    torch.set_num_threads(int(os.environ.get("T200_THREADS", "8")))
    from scipy.stats import spearmanr                                 # noqa: PLC0415
    e21 = load_module(C.FROZEN_E21, "_t200_e21")
    f8 = load_module(F8, "_t200_f8") if F8.exists() else None

    D = C.LEGACY_DATA / "all"
    xv = np.load(D / "val.npy", mmap_mode="r")
    rng = np.random.default_rng(VAL_SUB_SEED)
    idx = np.sort(rng.choice(xv.shape[0], N_SUB, replace=False))
    ref_raw = np.asarray(xv[idx], dtype=np.float64)
    ref_log = SD.cp10k_log1p(ref_raw)
    ref_lib = ref_raw.sum(1)
    ref_gene_mean = ref_raw.mean(0)
    ref_gene_mean_log = ref_log.mean(0)
    dev = ("cuda" if (a.device == "cuda" or
                      (a.device == "auto" and torch.cuda.is_available())) else "cpu")
    ref_e21, _ = e21.cp10k_log1p(np.asarray(xv[idx]))
    ref_gpu = torch.from_numpy(ref_e21).to(dev)
    sigma = e21.exact_bandwidth(ref_gpu)
    k_rr = e21.kernel_mean(ref_gpu, ref_gpu, sigma)
    ref_fano = f8.dispersion(ref_raw)[0] if f8 else None
    print(f"[ref] val subsample n={N_SUB} seed={VAL_SUB_SEED} lib_mean={ref_lib.mean():.2f} "
          f"sigma={sigma:.6f}", flush=True)

    out = {"ticket": 200, "block": "M1-6", "table": "B1 unconditional fidelity",
           "reference": {"source": str(D / "val.npy"), "n": N_SUB, "seed": VAL_SUB_SEED,
                         "rule": "np.sort(rng.choice(N_val, 20000, replace=False))",
                         "library_size_mean": float(ref_lib.mean()),
                         "library_size_median": float(np.median(ref_lib)),
                         "zero_fraction": float((ref_raw == 0).mean()),
                         "median_fano": (ref_fano["fano_median"] if ref_fano else None),
                         "heterogeneity": heterogeneity(ref_raw)},
           "heterogeneity_columns_added_after_seeing_dcm": True,
           "heterogeneity_note":
               "these diagnostics were added after seeing the DCM result and are not pre-registered. "
               "Not one word of the B-F1 criterion changed; they are descriptive only, and they "
               "are computed symmetrically for every generator and for the real data.",
           "real_reference_heterogeneity": {},
           "calibers": {
               "w1": {"tag": "hvg_score_distribution_caliber",
                      "NOT_a_frozen_ruler": True,
                      "scale": "CP10K + log1p",
                      "functions": ["analysis/score_distribution.py::w1_equal_size",
                                    "analysis/score_distribution.py::sliced_w1"],
                      "n_proj": N_PROJ, "proj_seed": PROJ_SEED,
                      "why_not_table1_ruler":
                          "Table 1's W1 is the score_w1_paper raw-counts ruler, whose reference is the "
                          "real validation cells at the per-type matched n. The reference here is "
                          "one mixed 20000-cell validation subsample, so the two are not "
                          "interchangeable and this one does not borrow that ruler's name"},
               "e21": {"source": str(C.FROZEN_E21), "sigma": float(sigma),
                       "reference": "the same 20000-cell val subsample (NOT Table 1's "
                                    "per-type matched n) -- calibre stated, not borrowed"},
               "fano": {"source": str(F8), "definition": "var(ddof=1)/mean over genes with "
                                                         "mean>0; median across genes",
                        "available": bool(f8)}},
           "B_F1": {"library_mean_within": 0.20, "log_gene_mean_pearson_min": 0.99,
                    "symmetric_for_all_generators": True,
                    "prior_disclosure": "an earlier measurement of our generator's library-size mean and gene-mean correlation is 311 "
                                        "(on record; no threshold was adjusted because of it"},
           "rows": []}

    # the real training and validation heterogeneity, computed symmetrically; the training set
    for split in ("train", "val"):
        p = D / f"{split}.npy"
        if not p.exists():
            continue
        xs = np.load(p, mmap_mode="r")
        rr = np.random.default_rng(VAL_SUB_SEED)
        idx_s = (np.sort(rr.choice(xs.shape[0], N_SUB, replace=False))
                 if xs.shape[0] > N_SUB else np.arange(xs.shape[0]))
        out["real_reference_heterogeneity"][split] = {
            "n": int(len(idx_s)), "subsample_seed": VAL_SUB_SEED,
            **heterogeneity(np.asarray(xs[idx_s], dtype=np.float64))}
        h = out["real_reference_heterogeneity"][split]
        print(f"[het ] real {split:<5} libsize_CV={h['libsize_CV']:.4f} "
              f"skew={h['libsize_skew']:.3f} p99/p50={h['libsize_p99_over_p50']:.3f} "
              f"detected_CV={h['detected_genes_CV']:.4f}", flush=True)

    for spec in a.pools:
        gen, seed, path = spec.split(":", 2)
        p = Path(path)
        if not p.exists():
            out["rows"].append({"generator": gen, "seed": int(seed), "pool": str(p),
                                "status": "NOT_RUN_missing_pool"})
            print(f"[skip] {gen} seed {seed}: missing {p}")
            continue
        t0 = time.time()
        x = np.load(p, mmap_mode="r")
        if x.shape[0] < N_SUB:
            out["rows"].append({"generator": gen, "seed": int(seed), "pool": str(p),
                                "status": f"NOT_RUN_pool_only_{x.shape[0]}"})
            continue
        g_raw = np.asarray(x[:N_SUB], dtype=np.float64)
        g_log = SD.cp10k_log1p(g_raw)
        g_lib = g_raw.sum(1)
        pw = SD.w1_equal_size(g_log, ref_log)
        sw = SD.sliced_w1(g_log, ref_log, N_PROJ, PROJ_SEED)
        g_e21, zero_rows = e21.cp10k_log1p(np.asarray(x[:N_SUB]))
        g_gpu = torch.from_numpy(g_e21).to(dev)
        mmd2, terms = e21.mmd2_biased(ref_gpu, g_gpu, sigma, k_rr)
        scc = float(spearmanr(ref_e21.mean(0, dtype=np.float64),
                              g_e21.mean(0, dtype=np.float64)).statistic)
        pcc = float(np.corrcoef(ref_e21.mean(0, dtype=np.float64),
                                g_e21.mean(0, dtype=np.float64))[0, 1])
        del g_gpu
        fano = f8.dispersion(g_raw)[0] if f8 else None
        gm = g_raw.mean(0)
        gml = g_log.mean(0)
        pear_raw = float(np.corrcoef(ref_gene_mean, gm)[0, 1])
        pear_log = float(np.corrcoef(ref_gene_mean_log, gml)[0, 1])
        spear = float(spearmanr(ref_gene_mean, gm).statistic)
        lib_rel = abs(g_lib.mean() - ref_lib.mean()) / ref_lib.mean()
        bf1 = bool(lib_rel <= 0.20 and pear_log >= 0.99)
        sidecar = p.with_suffix(".json")
        sc = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        row = {
            "generator": gen, "seed": int(seed), "pool": str(p),
            "pool_sha256": sha256_file(p), "n_scored": N_SUB,
            "W1_per_gene_mean": float(pw.mean()), "W1_per_gene_median": float(np.median(pw)),
            "sliced_W1": sw["mean"], "sliced_W1_detail": sw,
            "mmd2_rbf_biased": float(mmd2), "scc": scc, "pcc": pcc,
            "median_fano": (fano["fano_median"] if fano else "NOT_RUN"),
            "fano_q25_q75": ([fano["fano_q25"], fano["fano_q75"]] if fano else None),
            "NFE_per_sample": sc.get("NFE_chain") or sc.get("NFE") or sc.get("NFE_per_sample"),
            "library_size_source": sc.get("library_size_source",
                                          "generated" if gen == "OURS" else None),
            "count_legality": {
                "fraction_negative": float((g_raw < 0).mean()),
                "fraction_non_integer": float((g_raw != np.round(g_raw)).mean())},
            "diagnostic_not_ruler": {
                "library_size_mean": float(g_lib.mean()),
                "library_size_median": float(np.median(g_lib)),
                "library_size_mean_rel_diff_vs_real": float(lib_rel),
                "gene_mean_pearson_raw": pear_raw,
                "gene_mean_pearson_log": pear_log,
                "gene_mean_spearman_raw": spear,
                "zero_fraction": float((g_raw == 0).mean()),
                "e21_zero_library_rows": int(zero_rows)},
            "heterogeneity": heterogeneity(g_raw),
            "vocab_size": sc.get("vocab_size"),
            "training_budget": sc.get("training_budget"),
            "B_F1_passed": bf1,
            "walltime_s": time.time() - t0}
        hh = row["heterogeneity"]
        rh = (out["real_reference_heterogeneity"].get("val") or {})
        row["heterogeneity_vs_real_val"] = {
            "libsize_CV_ratio": (hh["libsize_CV"] / rh["libsize_CV"]
                                 if rh.get("libsize_CV") else None),
            "detected_genes_CV_ratio": (hh["detected_genes_CV"] / rh["detected_genes_CV"]
                                        if rh.get("detected_genes_CV") else None)}
        if not bf1:
            row["generator_pathology"] = True
        out["rows"].append(row)
        print(f"[B1] {gen:<8} seed {seed}  W1 {pw.mean():.5f}  sW1 {sw['mean']:.5f}  "
              f"mmd2 {float(mmd2):.4g}  fano {row['median_fano']}  lib {g_lib.mean():.1f} "
              f"({lib_rel:.1%})  pearson_log {pear_log:.4f}  B-F1 "
              f"{'PASS' if bf1 else 'FAIL -> generator_pathology'}", flush=True)
        print(f"[het ] {gen:<8} seed {seed}  libsize_CV={hh['libsize_CV']:.4f} "
              f"(real {rh.get('libsize_CV', float('nan')):.4f})  skew={hh['libsize_skew']:.3f}  "
              f"p99/p50={hh['libsize_p99_over_p50']:.3f}  "
              f"detected_CV={hh['detected_genes_CV']:.4f}", flush=True)

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest = dest.with_name(dest.stem + f"_{int(time.time())}.json")
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
