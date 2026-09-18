# -*- coding: utf-8 -*-
"""Shared machinery for the atlas figures: frozen bases, the palette, sampling rules, sidecars.

Nothing here is fitted. Both embeddings are files frozen by earlier figures, and this module
only loads them and self-checks on the spot:

* PCA: the frozen atlas basis (mean and components). After loading, it must reproduce the
  cached real-cell embedding, or the run stops. That assertion is repeated on every load.
* UMAP: the frozen reducer fitted once on the validation set. It is only ever used to
  transform, never to refit.

Every figure written through `save()` gets a sidecar json carrying each number drawn on it, so
no number in a figure script is ever transcribed by hand.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]                 # .../Experiments/HVG
sys.path.insert(0, str(ROOT))
from crndiff.config import OUT                                 # noqa: E402

from crndiff import config as C                            # noqa: E402
HVG2K = C.ARCHIVE
VIZ = HVG2K / "viz"
FD = HVG2K / "results" / "figure_data"
LEGACY = ROOT / "assets" / "legacy_root"
DATA = LEGACY / "_shared" / "data" / "all"
PDATA = LEGACY / "Processing_data" / "data"
PDIAG = LEGACY / "Processing_data" / "diagnostics"

FIG = OUT / "figures"
SUMMARY = OUT / "summary"
RUNS = OUT / "runs"
PAPER_FIGS = ROOT / "out" / "paper_figs"

PCA_BASIS = FD / "frozen_atlas_pca_basis.npz"
PCA_PROOF = FD / "F9_embedding.npz"
UMAP_REDUCER = PDATA / "umap_reducer.pkl"
UMAP_REAL_XY = PDATA / "umap_xy_val_real.npy"
UMAP_PCA50 = PDIAG / "pca50_val.npz"

# the convention of the archived embedding figures, carried over unchanged
SUBSAMPLE_SEED = 20261031
N_PANEL = 2500
GENE_UAP1 = 128            # make_F22_request_shapes.py::GENE
TYPES3 = ["Endothelial", "Myeloid", "Neuronal"]

# the delivered arrays of this round's deployed arm
SEED_ROUND = 20260987
DELIVERED = {
    "Endothelial": RUNS / "table1_ours" / "Endothelial_a1" / (
        "Endothelial_residual_hvgtwist_fk_tiltevery_trigany_K32_M20114_N10057_"
        "alpha1p0_J16_int8_th0p5_tau0p4_cc256_odds_seed20260987.npy"),
    "Myeloid": RUNS / "table1_ours" / "Myeloid_a1" / (
        "Myeloid_residual_hvgtwist_fk_tiltevery_trigany_K32_M5000_N2500_"
        "alpha1p0_J16_int8_th0p5_tau0p2_cc256_odds_seed20260987.npy"),
    "Neuronal": RUNS / "table1_ours" / "Neuronal_a1" / (
        "Neuronal_residual_hvgtwist_fk_tiltevery_trigany_K32_M5000_N2500_"
        "alpha1p0_J16_int8_th0p5_tau0p2_cc256_odds_seed20260987.npy"),
}
ABLATION = {
    "tilt_only": RUNS / "ablation_table10" / "Endothelial_tilt_only" / (
        "Endothelial_hvgtilt_only_tilt_K32_M20114_N10057_tau0p4_cc256_seed20260987.npy"),
    "fk_only": RUNS / "ablation_table10" / "Endothelial_fk_only" / (
        "Endothelial_residual_hvgtwist_fk_notilt_tiltevery_trigany_K32_M20114_N10057_"
        "alpha1p0_J16_int8_th0p5_tau0p4_cc256_odds_seed20260987.npy"),
    "tilt_fk": DELIVERED["Endothelial"],
}
INTERSECT = {
    "a1": RUNS / "intersect_ladder" / "a1" / (
        "INTERSECT_Endothelial_g128_residual_hvgtwist_fk_tiltevery_trigany_K32_M5000_N2500_"
        "alpha1p0_J16_int8_th0p5_tau0p4_cc256_odds_seed20260987.npy"),
    "tilt_only": RUNS / "intersect_ladder" / "tilt_only" / (
        "INTERSECT_Endothelial_g128_hvgtilt_only_tilt_K32_M5000_N2500_"
        "tau0p4_cc256_seed20260987.npy"),
}


# the archived palette
def load_style():
    """Import the archived style registry and run the drift gate it ships with."""
    spec = importlib.util.spec_from_file_location("_hvg2k_viz_style", VIZ / "style.py")
    st = importlib.util.module_from_spec(spec)
    sys.modules["_hvg2k_viz_style"] = st
    spec.loader.exec_module(st)
    st.assert_palette(Endothelial="#6aa6ee", Myeloid="#eb6834", Neuronal="#1baf7a",
                      Mesothelial="#00695c", Ventricular_Cardiomyocyte="#d13b8a",
                      UAP1_positive="#e69f00", neither="#9db4c0")
    return st


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cpm_log1p(a) -> np.ndarray:
    """The same convention the three archived basis builders use."""
    a = np.asarray(a, dtype=np.float64)
    lib = a.sum(1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(a / lib * 1e4)


# real cells
def real_cells():
    x = np.load(DATA / "val.npy", mmap_mode="r")
    y = np.load(DATA / "val_celltype.npy", allow_pickle=True).astype(str)
    if x.shape[0] != 45145:
        raise SystemExit(f"real background is {x.shape[0]} cells, the caliber says 45,145")
    return x, y


# the frozen PCA basis and its gate
class FrozenPCA:
    def __init__(self):
        if not PCA_BASIS.is_file():
            raise SystemExit(f"frozen PCA basis missing: {PCA_BASIS} -- STOP (ticket s0.1 "
                             f"forbids refitting one)")
        z = np.load(PCA_BASIS)
        self.mean_ = z["mean_"].astype(np.float64)
        self.comp2 = z["comp2"].astype(np.float64)
        self.evr = [float(v) for v in z["evr"]]
        self.recorded_proof = float(z["proof_max_delta"])
        self.fit_seed = int(z["seed"])
        self.n_fit = int(z["n_fit"])
        self.sha = sha256(PCA_BASIS)
        self.gate = None

    def project(self, counts) -> np.ndarray:
        return (cpm_log1p(counts) - self.mean_) @ self.comp2.T

    def gate_against_archive(self, real_counts) -> dict:
        """Must reproduce the cached archived coordinates, or stop."""
        if not PCA_PROOF.is_file():
            raise SystemExit(f"archived proof coordinates missing: {PCA_PROOF} -- STOP")
        z9 = np.load(PCA_PROOF, allow_pickle=True)["real_pca"]
        Z = self.project(real_counts)
        d = float(np.abs(Z - z9).max())
        ok = bool(np.allclose(Z, z9, atol=1e-4))
        self.gate = {
            "basis_file": str(PCA_BASIS), "basis_sha256": self.sha,
            "proof_file": str(PCA_PROOF), "proof_sha256": sha256(PCA_PROOF),
            "check": "(cpm_log1p(val) - mean_) @ comp2.T  ==  F9_embedding.npz::real_pca",
            "max_abs_delta": d, "atol": 1e-4, "passed": ok,
            "recorded_proof_max_delta_when_frozen": self.recorded_proof,
            "fit": "PCA(n_components=30, random_state=%d).fit(cpm_log1p(val.npy)), n=%d"
                   % (self.fit_seed, self.n_fit),
            "evr_pc1_pc2": self.evr,
            "refitted_here": False,
        }
        if not ok:
            raise SystemExit(f"FROZEN PCA GATE FAIL: max |delta| = {d:.3e} > 1e-4. "
                             f"The axes are not the archived ones -- STOP, do not draw.")
        print(f"[pca ] frozen basis reproduces the archived Figure 6 coordinates "
              f"(max |delta| = {d:.3e})")
        return self.gate


# the frozen UMAP basis and its gate
class FrozenUMAP:
    """The named frozen reducer; only ever used to transform, never to refit.

        These coordinates differ from the archived embedding figure's, and the sidecar says so: that
        one was fitted inside the figure script and its reducer object was NEVER written to disk,
        only the coordinates were cached, so new cells cannot be projected into it. The only frozen
        reducer that can take new cells is this one.
    """

    def __init__(self):
        for p in (UMAP_REDUCER, UMAP_REAL_XY, UMAP_PCA50):
            if not p.is_file():
                raise SystemExit(f"frozen UMAP input missing: {p} -- STOP (ticket s0.1 "
                                 f"forbids refitting one)")
        import pickle
        with UMAP_REDUCER.open("rb") as fh:
            self.red = pickle.load(fh)
        z = np.load(UMAP_PCA50)
        self.mean = z["mean"].astype(np.float64)
        self.comp = z["components"].astype(np.float64)
        self.real_xy = np.load(UMAP_REAL_XY)
        self.gate = None

    def project(self, counts) -> np.ndarray:
        """The archived embedding path followed by transform, word for word as in the original script."""
        z = (cpm_log1p(counts) - self.mean) @ self.comp.T
        return np.asarray(self.red.transform(z))

    def gate_against_archive(self) -> dict:
        E = np.asarray(self.red.embedding_)
        d = float(np.abs(E - self.real_xy).max())
        ok = bool(np.allclose(E, self.real_xy, atol=1e-5))
        self.gate = {
            "reducer_file": str(UMAP_REDUCER), "reducer_sha256": sha256(UMAP_REDUCER),
            "real_xy_file": str(UMAP_REAL_XY), "real_xy_sha256": sha256(UMAP_REAL_XY),
            "pca50_file": str(UMAP_PCA50), "pca50_sha256": sha256(UMAP_PCA50),
            "fitted_by": "assets/legacy_root/Processing_data/code/umap_expX.py "
                         "(fit once on val, red line: never refit)",
            "n_neighbors": int(self.red.n_neighbors), "min_dist": float(self.red.min_dist),
            "random_state": int(self.red.random_state), "n_components": int(self.red.n_components),
            "check": "reducer.embedding_ == umap_xy_val_real.npy",
            "max_abs_delta": d, "passed": ok, "refitted_here": False,
            "NOT_the_archived_figure7_embedding": {
                "why": "Figure 7's UMAP was fitted inside make_F9_embedding.py "
                       "(PCA30 -> UMAP, random_state 20261031) and the reducer object was "
                       "never written to disk -- only the coordinates were cached in "
                       "F9_embedding.npz. New cells therefore cannot be projected into it "
                       "without refitting, which s0.1 forbids.",
                "what_is_used_instead": "the reducer the ticket names in s0.1 "
                                        "(umap_expX.py, PCA50 -> UMAP, random_state 0), "
                                        "which does support transform()",
                "consequence": "F3's coordinates are NOT comparable point-for-point with the "
                               "archived Figure 7; the layout is.",
                "max_abs_delta_between_the_two_real_embeddings": None,
            },
        }
        if not ok:
            raise SystemExit(f"FROZEN UMAP GATE FAIL: max |delta| = {d:.3e} -- STOP")
        print(f"[umap] frozen reducer reproduces its own cached real coordinates "
              f"(max |delta| = {d:.3e}); n_neighbors={self.red.n_neighbors} "
              f"min_dist={self.red.min_dist} random_state={self.red.random_state}")
        return self.gate


def umap_vs_figure7_delta() -> float:
    """The distance between the two real embeddings, recorded in the sidecar to quantify the difference."""
    a = np.load(PCA_PROOF, allow_pickle=True)["real_umap"]
    b = np.load(UMAP_REAL_XY)
    return float(np.abs(a - b).max())


# panel size and subsampling
def take_panel(counts, rng, n: int = N_PANEL):
    """The archived convention: a sorted draw without replacement."""
    a = np.load(counts, mmap_mode="r") if isinstance(counts, (str, Path)) else counts
    N = a.shape[0]
    idx = np.sort(rng.choice(N, size=min(n, N), replace=False))
    return np.asarray(a[idx]), idx


def array_record(path: Path, n_used: int, idx=None, seed=None) -> dict:
    a = np.load(path, mmap_mode="r")
    return {"path": str(path), "sha256": sha256(path),
            "n_source": int(a.shape[0]), "n_genes": int(a.shape[1]),
            "n_plotted": int(n_used), "subsample_seed": seed,
            "subsampled": bool(idx is not None and len(idx) < a.shape[0])}


# saving and the sidecar
def save(fig, stem: str, sidecar: dict, dpi: int = 400) -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    out = {}
    for ext in ("pdf", "png"):
        p = FIG / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="#ffffff")
        out[ext] = p
    plt.close(fig)
    sidecar = dict(sidecar)
    sidecar["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    sidecar["outputs"] = {e: {"path": str(p), "sha256": sha256(p)} for e, p in out.items()}
    (FIG / f"{stem}.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=1, default=float) + "\n",
        encoding="utf-8")
    for e, p in out.items():
        print(f"[out] {p}")
    print(f"[out] {FIG / (stem + '.json')}")
    return out


def copy_pdf_to_paper(stem: str, name: str) -> Path:
    """Copy the pdf only; the png and the sidecar stay in the output directory."""
    import shutil
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    src, dst = FIG / f"{stem}.pdf", PAPER_FIGS / name
    shutil.copy2(src, dst)
    if sha256(src) != sha256(dst):
        raise SystemExit(f"copy differs: {dst}")
    print(f"[paper] {dst}  sha256={sha256(dst)}")
    return dst


def read_json(p: Path):
    p = Path(p)
    if not p.is_file():
        raise SystemExit(f"authoritative json missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))
