# -*- coding: utf-8 -*-
"""Paths and constants for the HVG package, plus explicit checks on every frozen input.
Purpose: path constants plus an explicit existence check, with a clear message, for every
frozen input. This package imports nothing from the retired tree except
`assets/legacy_root/Baseline/OURS/code` (the frozen sampling code, kept where it was) and
`crndiff/frozen/`. Every path resolves inside the repository; HVG_OUT redirects the output
directory, which the self-tests and gates use. Layout (see README and PROVENANCE.json):

    assets/legacy_root/             the retired tree, directory shape preserved verbatim
      Baseline/OURS/code/           frozen sampling code (sample_hvg2k_noa4, model_hvg2k, ...)
      Baseline/OURS/models/         the frozen generator checkpoint and its marginal prior
      _shared/data/all/             1.7 GB of counts and labels (not redistributed)
    assets/_shared/panel61/         the gene-panel package, rebuilt and validated by gate H0
    assets/models/{generator,clean_reward,residual}/
    assets/tables/tables_ref/       the product-tilt table of each request, logc_*.npy

One hard rule: this file only RESOLVES paths and checks that they exist. It never creates or
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]              # the repository root
ASSETS = ROOT / "assets"
FROZEN = ROOT / "crndiff" / "frozen"
ARCHIVE = ROOT / "archive"      # byte-identical copies of the archived-tree inputs we still read
BASELINES = ROOT / "baselines"  # scVI / scANVI / CFGen / MDLM

LEGACY_ROOT = ASSETS / "legacy_root"                    # the retired Experiments\X tree
BASE_CODE = LEGACY_ROOT / "Baseline" / "OURS" / "code"  # frozen sampling code, imported as-is
LEGACY_DATA = LEGACY_ROOT / "_shared" / "data"          # DATASET subdirs live under here
PANEL61 = ASSETS / "_shared" / "panel61"                # G61 (rebuilt; gate H0)
PANEL61_CODE = PANEL61 / "_shared" / "code"

MODELS = ASSETS / "models"
RULERS = ASSETS / "rulers"                              # purity label sets + CellTypist histograms
GENERATOR = ASSETS / "generator"                        # frozen ckpt + the LA prior
CLEAN_REWARD = MODELS / "clean_reward"
RESIDUAL = MODELS / "residual"
TABLES = ASSETS / "tables" / "tables_ref"               # DONOR / INTERSECT / MIX tilt tables
# --- topped up 2026-09-06 by ruling 164 section 3 (all bitwise-checked; see PROVENANCE.json) ---
TABLES_KZ = ASSETS / "tables" / "tables_kz"             # logc_<celltype>.npy: the BARE cell-type
                                                        # tilt tables (11); this round runs on these
MASTER_PURITY_V7 = RULERS / "master_purity_v7.json"     # the frozen canonical label sets
MASTER_PURITY_V8 = RULERS / "master_purity_v8_lineage.json"      # the lineage-stem ruler
CELLTYPIST = RULERS / "celltypist"                      # celltypist_eval.py + 27 archived label
                                                        # histograms (the ruler gate G5 checks)
# the tau = 0.0 UNTILTED 20,000-cell pool: the negative pool a proposal_untilted discriminator
# must be fitted against (ruling 164 / the ablation batch). Written by the frozen sampler, so it
# lives where that sampler writes.
BASE_POOL = (LEGACY_ROOT / "Baseline" / "OURS" / "results" /
             "linspace_T_O_K32_spilot_run1c_s20260625_samp20260902_batched1_chunk256"
             "_F2_K32_base_noa4.npy")
BASE_POOL_SHA256 = "486835ee9f25def8e1b9c15e61d9ce830c40635e64c870b4395d54a1b00fe0bf"

OUT = Path(os.environ.get("HVG_OUT", ROOT / "out"))
RUNS = OUT / "runs"
GATES = OUT / "gates"

# --- the frozen generator (authority value in HVG2K/README.md and PROVENANCE.json) ---
GENERATOR_CKPT = GENERATOR / "pilot_run1c_s20260625.pt"
GENERATOR_CKPT_SHA256 = "2b07bf5ee19b342473f8222b75b9be8d791378b325681e39036bce38b052fe77"
LOGPI_LA = GENERATOR / "logpi_nmax512_eps0.5.npy"       # the LA prior used by the unconditional arm

# --- deployed sampler configuration (frozen; see the task book section B1) ---
DEFAULT_K = 32                  # reverse steps
DEFAULT_M = 5000                # particles
DEFAULT_N_OUT = 2500            # delivered cells
DEFAULT_CELL_CHUNK = 256        # changes the RNG consumption order -> part of the caliber
DEFAULT_ESS_FRAC = 0.5          # resample at a scheduled step iff ESS/M < this
DEFAULT_INTERVAL = 8            # scheduled checkpoints
DEFAULT_REWARD_FLOOR = 1.0e-8   # r clipped to [floor, 1-floor]; NO upper clip (odds may exceed 1)
DEFAULT_J = 16                  # twist candidate endpoints per particle per step
DEFAULT_MIN_RESAMPLE_GAP = 2    # only bites for trigger='any'
DEFAULT_TAU = 0.4               # product tilt strength of the deployed unconditional proposal
TWIST_SEED_OFFSET = 2           # torch.Generator seed for the J candidates = seed + this
                                # (the frozen transition generator uses seed + 1; the twist must
                                #  NOT share it -- see _frozen/e38_twist.py docstring point 2)


# --- 167-AMEND-3: N and M are a RULE, not a per-type judgement call -------------------------
#     N = max(2500, n_matched(type))      2500 is what the purity ruler needs;
#                                         n_matched is what the distance ruler needs
#     M = 2 * N                           fixed compute budget: two proposals per delivered cell
# `n_matched` is READ from the frozen E21 scorer, never transcribed -- that file is the ruler
# that produced the paper's Table 1/6, so it is the only authority on matched n.
DEPLOYED_N_FLOOR = 2500
FROZEN_E21 = ROOT / "evaluation" / "frozen" / "e21_scc_mmd.py"


def matched_n() -> dict:
    """The distance ruler's matched n per cell type, read out of `_frozen/e21_scc_mmd.py`.

    Parsed with `ast` rather than imported: reading the literal has no side effects (the frozen
    module imports torch/scipy and builds paths into a retired tree), and if TYPES ever stops
    being a plain literal this raises loudly instead of silently returning something else.
    """
    import ast
    require(FROZEN_E21, "frozen E21 scorer (the authority on matched n)")
    tree = ast.parse(FROZEN_E21.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "TYPES" for t in node.targets):
            return dict(ast.literal_eval(node.value))
    raise SystemExit(f"no TYPES assignment found in {FROZEN_E21}")


def n_out_for(request: str) -> int:
    """N for a request. Bare cell types take the rule; every other request (INTERSECT / MIX)
    has a distance ruler whose matched n is below the floor, so it takes the floor."""
    return max(DEPLOYED_N_FLOOR, matched_n().get(request, 0))


def mn_for(request: str) -> dict:
    """{n_particles, n_out} for a request, by the AMEND-3 rule. draws is 2x for every run."""
    n = n_out_for(request)
    return {"n_particles": 2 * n, "n_out": n}


def require(path: Path, what: str, hint: str = "") -> Path:
    """Explicit existence check with a clear error. Never creates anything."""
    if not Path(path).exists():
        msg = f"missing required input ({what}): {path}"
        if hint:
            msg += f"\n  -> {hint}"
        raise SystemExit(msg)
    return Path(path)


def add_base_code_to_path() -> Path:
    """Put the frozen sampling code on sys.path exactly where the tree keeps it.

    `sample_hvg2k_noa4.py` resolves `ROOT = Path(__file__).resolve().parents[3]`, which under
    `assets/legacy_root/Baseline/OURS/code/` lands back on `legacy_root` -- that is why the
    inherited tree keeps the old shape verbatim and the frozen code needs no edit.
    """
    require(BASE_CODE, "frozen sampling code",
            "assets/legacy_root/Baseline/OURS/code must hold the inherited Experiments\\X code")
    require(PANEL61_CODE / "backbone.py", "panel61 backbone (G61)",
            "assets/_shared/panel61/_shared/code/backbone.py -- rebuilt, verified by gate H0")
    p = str(BASE_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)
    return BASE_CODE


def tilt_table(request: str) -> Path:
    """The product tilt table for a request.

    Bare cell types live in `tables_kz/` (topped up 2026-09-06); the DONOR / INTERSECT / MIX
    requests live in `tables_ref/`. Looked up in that order, and the resolved path plus its
    sha256 go into every run json (ruling 164 section 4: the archived sample_log never recorded
    which table it used, and that cost gate H0 an attempt).
    """
    kz = TABLES_KZ / f"logc_{request}.npy"
    if kz.exists():
        return kz
    return TABLES / f"logc_{request}.npy"


def require_release_assets() -> dict:
    """The four inputs ruling 164 section 3 topped up. Called by check_assets and by any run
    that needs them; raises with the expected path rather than failing deep in a loop."""
    require(MASTER_PURITY_V7, "canonical purity label sets (master_purity_v7.json)")
    require(TABLES_KZ, "bare cell-type product tilt tables (tables_kz)")
    require(CELLTYPIST, "CellTypist eval script + archived label histograms")
    require(BASE_POOL, "untilted tau=0 base pool (negative pool for proposal_untilted)")
    return {"master_purity_v7": str(MASTER_PURITY_V7), "tables_kz": str(TABLES_KZ),
            "celltypist": str(CELLTYPIST), "base_pool": str(BASE_POOL)}


def clean_reward_paths(target: str) -> tuple[Path, Path]:
    return (CLEAN_REWARD / f"{target}_clean_logistic.npz",
            CLEAN_REWARD / f"{target}_clean_logistic.json")


def residual_paths(residual_dir: str | Path) -> dict:
    """The residual reward directory holds the portable classifier, its log and the split.

    NOTE the frozen `fk_hvg_v2.main` also verifies `residual_classifier.joblib` against the
    portable file's recorded `source_sha256`. That joblib was NOT part of the migration; the
    check therefore cannot run here and `tools/check_assets.py` reports it as missing rather
    than silently skipping it.
    """
    d = Path(residual_dir)
    if not d.is_absolute():
        d = RESIDUAL / d
    return {"dir": d,
            "portable": d / "residual_classifier_portable.npz",
            "log": d / "residual_log.json",
            "split": d / "split_indices.npz",
            "joblib": d / "residual_classifier.joblib"}
