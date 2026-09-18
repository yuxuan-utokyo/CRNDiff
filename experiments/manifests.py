# -*- coding: utf-8 -*-
"""The pre-registered run grids, written out as manifests.
Purpose: write the gates and the experiment grids as manifests (json lists, one stem per entry).
The stem is recomputed by `crndiff.io.run_stem` and the queue checks it, so editing a manifest

Running only the gates is the normal first step: all four must pass before any grid is released.
`--all` also writes the experiment grids, but writing them is not running them.

    python -m experiments.manifests --gates          #     python -m experiments.manifests --gates          # write the gates only
    python -m experiments.manifests --all            #     python -m experiments.manifests --all            # gates plus the experiment grids
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                          # noqa: E402
from crndiff.io import run_stem, sha256_file             # noqa: E402

TICKET = 163
OUT = C.OUT / "manifests"

# The archived fkv2 run gate G2 reproduces. Its tilt table is the BARE per-celltype
# `logc_Myeloid.npy`, which ruling 164 section 3 topped up into `assets/tables/tables_kz/`.
# The path moved; the FILE must not. `assert_g2_table_unchanged()` re-checks that against the
# old tree on every manifest write -- G2's reference cannot be allowed to drift because a path
# changed (that is exactly the failure mode H0 attempt 1 was).
OLD_TREE_TABLES = (C.ARCHIVE / "old_scRNA" / "tables_kz")
G2_TABLE = C.TABLES_KZ / "logc_Myeloid.npy"


def assert_g2_table_unchanged() -> dict:
    """G2's tilt table now resolves inside assets/; require it to be byte-identical to the old
    tree's copy it was migrated from. If the old tree is gone, record that and carry on with
    the migrated file's own sha256 (the top-up was bitwise-verified -- see PROVENANCE.json)."""
    C.require(G2_TABLE, "G2 reference tilt table (bare Myeloid)")
    here = sha256_file(G2_TABLE)
    old = OLD_TREE_TABLES / "logc_Myeloid.npy"
    if old.exists():
        there = sha256_file(old)
        if here != there:
            raise SystemExit(
                "G2 reference tilt table changed when the path moved:\n"
                f"  assets/tables/tables_kz/logc_Myeloid.npy  {here}\n"
                f"  old tree tables_kz/logc_Myeloid.npy       {there}\n"
                "G2's reference must not move because a path moved. Stop and report.")
        return {"path": str(G2_TABLE), "sha256": here, "old_tree": str(old),
                "old_tree_sha256": there, "bitwise_identical": True}
    return {"path": str(G2_TABLE), "sha256": here, "old_tree": str(old),
            "old_tree_sha256": None,
            "bitwise_identical": "old tree copy not present; top-up was verified at migration"}


G2_ARCHIVED = dict(
    request="Myeloid", reward_model="residual", residual_dir="Myeloid",
    sampler="plugin", reward_mode="probability",      # port fidelity caliber
    alpha=2.0, J=C.DEFAULT_J, K=32, tau=0.2, tilt_mode="every", trigger="scheduled",
    n_particles=5000, n_out=2500, cell_chunk=256, ess_frac=0.5, resample_interval=8,
    intermediate_resampling=True, seed=20260987,
    logpi=str(G2_TABLE),
)


def entry(**kw) -> dict:
    r = dict(kw)
    r.setdefault("ticket", TICKET)
    r.setdefault("role", "experiment")
    r.setdefault("sampler", "twist")
    r.setdefault("reward_mode", "odds")
    r.setdefault("tilt_mode", "every")
    r.setdefault("trigger", "scheduled")
    r.setdefault("J", C.DEFAULT_J)
    r.setdefault("K", C.DEFAULT_K)
    r.setdefault("tau", C.DEFAULT_TAU)
    r.setdefault("n_particles", C.DEFAULT_M)
    r.setdefault("n_out", C.DEFAULT_N_OUT)
    r.setdefault("cell_chunk", C.DEFAULT_CELL_CHUNK)
    r.setdefault("ess_frac", C.DEFAULT_ESS_FRAC)
    r.setdefault("resample_interval", C.DEFAULT_INTERVAL)
    r.setdefault("intermediate_resampling", True)
    r["stem"] = run_stem(r)
    return r


def gates() -> list[dict]:
    """G1 and G2's runs. G2b re-executes G2's run; G3 is gate H0's artefact; G4 reads the
    frozen discriminator logs. So the queue only ever spawns these two."""
    return [
        # G1 telescope: no in-chain resample, so the delivered log-weight must be EXACTLY
        # alpha * log rho(n0). Small and cheap -- the identity does not depend on M.
        entry(experiment="gate", arm="telescope", role="gate_telescope",
              request="DONOR_D1", reward_model="residual", residual_dir="UNCOND",
              sampler="twist", reward_mode="odds", alpha=1.0, J=4, seed=20260901,
              n_particles=1000, n_out=500, intermediate_resampling=False),
        # G2 port fidelity: reproduce the archived fkv2 run bit for bit.
        entry(experiment="gate", arm="port_fidelity", role="gate_port", **G2_ARCHIVED),
    ]


# ============================================================================================
# The grid of this round, which replaces the earlier 45-entry one.
#
# These are NEW runs, not a version comparison: the old numbers are what is being replaced, not a
# baseline. So there is no new-versus-old column here. A new number has to stand against the
#
# But the other rows of the table (scVI, CFGen, scANVI, product tilt, the ceiling) are NOT re-run,
# since the sampler change does not affect them, so the new row must land on the same ruler: the
# same CellTypist model (gate G5), the same frozen label set, the same matched n.
#
# The request is always a bare cell type (logc_<type>.npy), never a type-and-gene intersection:
# an intersection request has far fewer reference cells, and the two are not comparable.
#
# M and N are set by the rule below. The reason is not alignment with the old runs: neither the
# matched-n purity nor the diversity reading is scale free in M, so an order of magnitude less
# gives a different duplication rate and a different ESS behaviour, which would not be comparable
#
# alpha in {1.0, 2.0} is TWO EXPERIMENTS, not a sweep: the old two-parameter form split the
# intermediate and endpoint potentials into alpha*sc and alpha, and the single-parameter form ties
#
# This round has no resample_interval dimension: the trigger is always `any`, so the ESS is
# ============================================================================================

# tau is INHERITED per request from the archived runs, frozen, not swept and not touched here. The
# counts below are the mode read out of the archived run jsons; they are recorded to make clear
# that this is inheritance rather than selection. Freezing tau keeps the alpha axis clean.
TAU_INHERITED = {
    "Endothelial": {"tau": 0.4, "archived_counts": {"0.4": 6},
                    "note": "6/6 archived residual fkv2 arms used tau=0.4"},
    "Myeloid": {"tau": 0.2, "archived_counts": {"0.2": 14, "0.4": 2, "0.1": 2},
                "note": "mode of 18 archived residual fkv2 arms"},
    "Neuronal": {"tau": 0.2, "archived_counts": {"0.2": 1},
                 "note": "the only archived residual fkv2 arm for this target"},
}
MAIN_TARGETS = ("Myeloid", "Endothelial", "Neuronal")
MAIN_SEED = 20260987
MAIN_ALPHAS = (1.0, 2.0)
# the common configuration, spelled out so a reader never has to chase a default
MAIN_COMMON = dict(sampler="twist", reward_model="residual", reward_mode="odds",
                   tilt_mode="every", trigger="any", J=16, K=32,
                   n_particles=40000, n_out=20000, cell_chunk=256, ess_frac=0.5,
                   resample_interval=8,   # inert under trigger='any'; kept for the stem/schema
                   intermediate_resampling=True)


def main_single_type() -> list[dict]:
    """First batch, the main result: three requests x alpha in {1.0, 2.0}, one seed each."""
    rows = []
    for target in MAIN_TARGETS:
        tau = TAU_INHERITED[target]["tau"]
        for alpha in MAIN_ALPHAS:
            rows.append(entry(experiment="main_single_type", arm=f"{target}_a{alpha:g}",
                              request=target, residual_dir=target, tau=tau,
                              alpha=alpha, seed=MAIN_SEED, **MAIN_COMMON))
    return rows


def ablation_single_type() -> list[dict]:
    """Second batch, the ablation: identical to the main result except in one place.

        tilt_only     tilt alone, with no reward, no twist and no resampling (also the alpha = 0 arm)
        residual_only tilt off, residual layer applied at every step as usual. This needs a
                                    discriminator fitted against the UNTILTED proposal: the existing one was fitted
                                    against the tilted proposal, and turning the tilt off changes the negative pool,
        The alpha is the one the first batch settles on, so this manifest is finalised after it.
    """
    rows = []
    for target in MAIN_TARGETS:
        tau = TAU_INHERITED[target]["tau"]
        common = dict(MAIN_COMMON)
        rows.append(entry(experiment="ablation", arm=f"{target}_tilt_only",
                          request=target, residual_dir=None, tau=tau,
                          alpha=None, seed=MAIN_SEED,
                          **{**common, "sampler": "tilt_only", "reward_model": "none",
                             "reward_mode": "none"}))
        rows.append(entry(experiment="ablation", arm=f"{target}_residual_only",
                          request=target, residual_dir=f"UNTILTED_{target}", tau=tau,
                          alpha=None,       # filled from batch 1's chosen alpha
                          seed=MAIN_SEED, tilt_on=False, **common))
    return rows


# ============================================================================================
# The full re-run, arranged by the tables of the paper.
#
# M and N are the deployed configuration frozen in `crndiff/config.py`, which is also the n the
# figures report. The reasons:
#   1. the paper's own operating point is here;
#   2. the main table reports a mean and standard deviation over sampling seeds, and without seeds
#      there is no such column: one large run gives no error bar and several smaller ones do, and
#      the number of seeds matters more than the size of any single run;
#   3. the draws column is the ratio M/N, which is unchanged.
# The cost, stated plainly: the matched n for purity equals N here, so there is exactly one block
# and no block spread. What is bought instead is a real seed spread, which is the error bar the
#
# The inherited tau values, the main seed and the thresholds are untouched.
# ============================================================================================
# 167-AMEND (2026-09-06 18:45): Table 1 goes from 5 seeds to 3. It must be EXACTLY these
# three -- Table 10's `tilt+FK (deployed)` row borrows Stage 1's results, so the seeds have to
# line up on both sides. 20260990 / 20260991 are de-registered (kept here only as a record of
# what the 5-seed plan was).
PROGRAMME_SEEDS = (20260987, 20260988, 20260989)
SEEDS3 = PROGRAMME_SEEDS[:3]            # the 3-seed arms (ablation / ladder / composition …)
DEREGISTERED_SEEDS_167AMEND = (20260990, 20260991)
# 167-AMEND-3: M/N is a RULE applied per request, not a constant --
#   N = max(2500, n_matched(type)),  M = 2N,  draws = 2x for EVERY run, no exceptions.
# `C.mn_for(request)` reads n_matched out of the frozen E21 scorer. This supersedes
# AMEND-2's per-type table (which gave Endothelial 40000/20000; the rule gives 20114/10057).
# Note for the paper (AMEND-3 section 1): the ARCHIVED batch was inconsistent on this --
# Neuronal was 4x (M10000/N2500) while the other two were 2x. This round fixes that.
PROG_COMMON = {k: v for k, v in MAIN_COMMON.items() if k not in ("n_particles", "n_out")}
PROG_ALPHA = 1.0            # AMEND-3 section 3: HVG runs at alpha=1.0 this round, no exceptions

# Stage 3: the intersection request. gene 128 of meta.json is UAP1, so this request IS the
# paper's "endothelial and UAP1+" one. tau and the residual dir are INHERITED, not chosen:
# the tilt table's own json records --tau 0.4, and the residual there is the TILTED Endothelial
# one (the proposal really is tilted in this arm).
INTERSECT_REQUEST = "INTERSECT_Endothelial_g128"
INTERSECT_TAU = 0.4
# CORRECTION (2026-09-07): the FK arm's discriminator is the request's OWN, not the bare
# Endothelial one. The archived arm
#   universality_fk/out/refFK_INTERSECT_Endo_g128_..._alpha1p0_..._seed20261045.json
# records residual_dir = universality_fk/out/disc_INTERSECT_g128, fitted with positives =
# the INTERSECT reference group and negatives = the INTERSECT tilt proposal pool
# (UNI_INTERSECT_Endo_g128_n10000_g0p4), held-out AUC 0.9888864923747277, gate_passed true.
# It has been migrated verbatim (all four files, sha256 src == dst) to
# assets/models/residual/INTERSECT_Endothelial_g128/. Using assets/models/residual/
# Endothelial/ instead -- which is what this grid said before the correction -- would put a
# discriminator fitted for a DIFFERENT request on this arm, i.e. exactly the substitution
# the ticket forbids. residual_dir is not part of the stem, so the fix changes no stem.
INTERSECT_RESIDUAL = "INTERSECT_Endothelial_g128"
# The tilt-off arm's discriminator, fitted against the UNTILTED base pool (AUC 0.995872).
INTERSECT_UNTILTED_RESIDUAL = "UNTILTED_INTERSECT_Endothelial_g128"
# 167-AMEND-3 section 3: the alpha ladder {0.5, 1.0, 2.0, 4.0} is CANCELLED this round.
# Stage 3 keeps only tilt_only and tilt+FK at alpha=1.0 -> 6 runs (was 15). Table 9's alpha
# rows stay empty this round and get their own ticket. The first batch's alpha in {1,2} runs
# at 40000/20000 are kept untouched for that future alpha study.
INTERSECT_ALPHAS = (PROG_ALPHA,)
# Stage 4: the composition requests, tau inherited from their own table jsons
COMPOSITION_REQUESTS = ("MIX0500_Ventricular_Cardiomyocyte_Endothelial",
                        "MIX0615_Ventricular_Cardiomyocyte_Endothelial",
                        "MIX0700_Ventricular_Cardiomyocyte_Endothelial")
COMPOSITION_TAU = 0.4


def table1_ours() -> list[dict]:
    """Stage 1 · Table 1 / 6 / 12 — single type, 3 types x alpha=1.0 x 3 seeds = 9 (167-AMEND)."""
    rows = []
    for target in MAIN_TARGETS:
        for seed in PROGRAMME_SEEDS:
            rows.append(entry(experiment="table1_ours", arm=f"{target}_a1",
                              request=target, residual_dir=target,
                              tau=TAU_INHERITED[target]["tau"], alpha=PROG_ALPHA, seed=seed,
                              **C.mn_for(target), **PROG_COMMON))
    return rows


def ablation_table10() -> list[dict]:
    """Stage 2 · Table 10 — component ablation, 3 types x 3 seeds x 2 arms = 18.

    The `tilt+FK (deployed)` row is BORROWED from Stage 1's first three seeds, not re-run.
    `FK only` needs a discriminator fitted against the UNTILTED proposal: the existing
    residual/<type> was fitted against the TILTED pool, and turning the tilt off changes the
    negative pool, so reusing it would measure something nobody can name.
    """
    rows = []
    for target in MAIN_TARGETS:
        tau = TAU_INHERITED[target]["tau"]
        for seed in SEEDS3:
            rows.append(entry(experiment="ablation_table10", arm=f"{target}_tilt_only",
                              request=target, residual_dir=None, tau=tau,
                              alpha=None, seed=seed, **C.mn_for(target),
                              **{**PROG_COMMON, "sampler": "tilt_only",
                                 "reward_model": "none", "reward_mode": "none"}))
            rows.append(entry(experiment="ablation_table10", arm=f"{target}_fk_only",
                              request=target, residual_dir=f"UNTILTED_{target}", tau=tau,
                              alpha=PROG_ALPHA, seed=seed, tilt_on=False,
                              **C.mn_for(target), **PROG_COMMON))
    return rows


# The intersection arm gains three seeds. Only seeds are added; no other parameter moves.
# Why: the three distance columns of that arm were extremely dispersed. One seed was checked and
# is not a fault (the three run jsons agree field by field and their wall-clock times agree to
# within a second), so it is a real sampler collapse. The sampler is deterministic and re-running
# the same seed only reproduces the same sha256, so instead of re-running, seeds are added to
INTERSECT_SEEDS_174 = (20260990, 20260991, 20260992)


def intersect_ladder() -> list[dict]:
    """Stage 3, Tables 3 and 9: endothelial intersected with UAP1.

        Rows 1 to 6 are the three original seeds and their artefacts are untouched;
        rows 7 to 12 are three added seeds, and both arms use the same seed set so they can be read
        in pairs. Only the seed changes: the discriminator, the tilt table, alpha, J, K, tau, the
        trigger, the ESS fraction and M/N all reuse the same constants, so a field-by-field
    """
    rows = []
    for seed in tuple(SEEDS3) + INTERSECT_SEEDS_174:
        rows.append(entry(experiment="intersect_ladder", arm="tilt_only",
                          request=INTERSECT_REQUEST, residual_dir=None, tau=INTERSECT_TAU,
                          alpha=None, seed=seed, **C.mn_for(INTERSECT_REQUEST),
                          **{**PROG_COMMON, "sampler": "tilt_only",
                             "reward_model": "none", "reward_mode": "none"}))
        for alpha in INTERSECT_ALPHAS:
            rows.append(entry(experiment="intersect_ladder", arm=f"a{alpha:g}",
                              request=INTERSECT_REQUEST, residual_dir=INTERSECT_RESIDUAL,
                              tau=INTERSECT_TAU, alpha=alpha, seed=seed,
                              **C.mn_for(INTERSECT_REQUEST), **PROG_COMMON))
        # The third arm of that panel, residual correction only, with the tilt off.
        # It is isomorphic to the fk_only arm of the ablation: turning the tilt off changes the negative
        # class, so the discriminator has to be the one fitted against the UNTILTED proposal and the one
        # fitted against the tilted pool cannot be reused. The seeds match the other two arms exactly.
        rows.append(entry(experiment="intersect_ladder", arm="fk_only",
                          request=INTERSECT_REQUEST,
                          residual_dir=INTERSECT_UNTILTED_RESIDUAL,
                          tau=INTERSECT_TAU, alpha=PROG_ALPHA, seed=seed, tilt_on=False,
                          **C.mn_for(INTERSECT_REQUEST), **PROG_COMMON))
    return rows


def composition() -> list[dict]:
    """Stage 4 · Table 2 / 7 — ventricular:endothelial at three ratios, 3 seeds = 9.

    ★ PURE TILT arm, and that is INHERITED, not chosen. `PREREG_E17_mixture_perrequest_fk.md`
    fitted one per-request residual discriminator per mixture level (positives = that level's
    reference population, negatives = that level's PURE TILT proposal pool), on the verbatim
    disc_DONOR_D1 recipe with the same AUC gate 0.80, and ruled:

                A gate that does not pass marks its level FAIL; refitting to try again is not allowed

    All three archived discriminators record `gate_passed: false`
    (`disc_MIX0500_e17` 0.7563 / `disc_MIX0615_e17` 0.7393 / `disc_MIX0700_e17` 0.7645),
    so every level was marked FAIL and the tilt+FK composition arm was never validly
    established. Fitting a replacement would be exactly the refitting that the gate forbids, and
    `run_one.resolve_reward` refuses a `gate_passed: false` classifier anyway (as it should).
    Substituting a component-type discriminator (e.g. Endothelial) would be an invention: the
    target here is a MIXTURE population, not either component.

    So this stage runs the arm that DOES have an accepted artifact -- the pure tilt -- which is
    also the very pool E17 used as its negatives. Table 2/7's readouts (per-cell classifier
    shares, deconvolved population share, the in-chain share that is exact by construction) are
    all properties of whether the tilt achieves the requested composition.
    """
    rows = []
    for request in COMPOSITION_REQUESTS:
        for seed in SEEDS3:
            rows.append(entry(experiment="composition", arm=request.split("_")[0],
                              request=request, residual_dir=None,
                              tau=COMPOSITION_TAU, alpha=None, seed=seed,
                              **C.mn_for(request),
                              **{**PROG_COMMON, "sampler": "tilt_only",
                                 "reward_model": "none", "reward_mode": "none"}))
    return rows


# Stage 6 · Table 8 — the second type pair (endothelial + myeloid 50/50).
# The MIX tilt table was built for this round with the verbatim archived builder
# (`tools/build_ref_table.py --ref MIX:Endothelial:Myeloid:0.5`), N=36844 reference cells.
TYPE_PAIR_REQUEST = "MIX0500_Endothelial_Myeloid"
TYPE_PAIR_TAU = 0.4          # same as every other MIX table's own json records


def type_pair() -> list[dict]:
    """Stage 6 · Table 8 — `equal weight` arm x 3 seeds = 3.

    The task book asks for two arms, `equal weight` and `quota+FK`. Only the first is runnable:

    CORRECTION (2026-09-07): the paper's `equal weight` row is NOT the MIX product tilt. Both
    of Table 8's rows come from `multicluster/`: arm A is a mixture REWARD
    `logsumexp_k(log w_k + log p_k)` over the frozen per-type clean rewards, with a proposal
    that mixes the components' TEMPERED tilted marginals per gene and runs the engine at
    tau=1.0; D2 adds stratified-quota resampling. The implementation exists --
    `old/scRNA/condition/residual_v3/fk_hvg_multicluster.py` -- so porting it is a PORT, not
    an invention, and the earlier claim to the contrary is withdrawn. It is still not in this
    package, and it is not an A-class fix: it needs several clean rewards combined at the
    probability level, a different proposal construction, and a RULING on how the archived
    arms' two-parameter `--intermediate-scale 0.25` maps onto this round's single-parameter
    alpha. Reported, not built.

    * `mix_tilt_only` = the MIX product tilt alone (pure tilt), 3 seeds -- registered here.
      This is a THIRD thing: it mixes the reference populations' marginals and then tempers
      the result, where arm A tempers first and mixes after. It answers "can the MIX product
      tilt deliver 50/50 on its own", which is a real question, but it must never be filled
      into Table 8 as the `equal weight` row.
    * `quota+FK` needs the in-chain per-particle component quota mechanism. E17 section 0 is
      explicit that this is a DIFFERENT mechanism from per-request FK, and that the two are not
      it requires per-component quantities, and that E13 was BLOCKED on exactly it. This
      package's sampler has no quota arm, and inventing one is not a port. Reported, not built.
    """
    return [entry(experiment="type_pair", arm="mix_tilt_only",
                  request=TYPE_PAIR_REQUEST, residual_dir=None,
                  tau=TYPE_PAIR_TAU, alpha=None, seed=seed,
                  **C.mn_for(TYPE_PAIR_REQUEST),
                  **{**PROG_COMMON, "sampler": "tilt_only",
                     "reward_model": "none", "reward_mode": "none"})
            for seed in SEEDS3]


# ============================================================================================
# The two multi-component arms.
#
# These are PORTED from the retired tree rather than newly invented; see the header of
# `crndiff/multicluster_sampler.py`. The mixture product-tilt arm run elsewhere is a third thing
# and is not merged with these two.
#
# The single-parameter form removed `intermediate_scale`, so the archived rows have no
# corresponding value and these two arms run at alpha = 1.0, the same as every other table of this
TABLE8_REQUEST = "MIX0500_Endothelial_Myeloid"
TABLE8_TYPES = ("Endothelial", "Myeloid")
TABLE8_WEIGHTS = (0.5, 0.5)
# The quota target is the requested proportion itself. That is not another knob: the arm is
TABLE8_QUOTA = (0.5, 0.5)
# The component tempering temperature is read from the mixture table's own json; run_one checks it
# again and refuses to run on a mismatch. The engine runs at tau = 1.0, the temperature having
TABLE8_COMPONENT_TAU = 0.4


def table8_multicluster() -> list[dict]:
    """    The two multi-component arms, `equal weight` and `quota+FK`, three seeds each.

        They differ in one place: `quota+FK` replaces the resampling inside `run_fk` with quota
        resampling by dominant component, used both for the intermediate resampling and for the
    """
    rows = []
    for seed in SEEDS3:
        for arm, quota in (("equal_weight", None), ("quota_fk", list(TABLE8_QUOTA))):
            rows.append(entry(
                experiment="table8", arm=arm, request=TABLE8_REQUEST, ticket=170,
                reward_model="mixture", types=list(TABLE8_TYPES),
                weights=list(TABLE8_WEIGHTS), stratify=quota,
                stratified=bool(quota), residual_dir=None,
                # tau in the row is the ENGINE temperature; the CLI gets component_tau
                tau=1.0, component_tau=TABLE8_COMPONENT_TAU,
                alpha=PROG_ALPHA, seed=seed,
                **C.mn_for(TABLE8_REQUEST),
                **{k: v for k, v in PROG_COMMON.items() if k != "reward_model"}))
    return rows


def schedule_table11() -> list[dict]:
    """Stage 5 · Table 11 — CANCELLED by 167-AMEND (2026-09-06 18:45). NOT RUN this round.

        The earlier default disposition is withdrawn: the paper drops that table and section.
    The function is kept (the amendment says to keep it) but is NOT registered in
    PROGRAMME_167, so `--programme` never writes it and the driver never queues it.
    """
    rows = []
    for target in MAIN_TARGETS:
        for seed in SEEDS3:
            rows.append(entry(experiment="schedule_table11", arm=f"{target}_scheduled",
                              request=target, residual_dir=target,
                              tau=TAU_INHERITED[target]["tau"], alpha=PROG_ALPHA, seed=seed,
                              **C.mn_for(target),
                              **{**PROG_COMMON, "trigger": "scheduled",
                                 "resample_interval": 8}))
    return rows


def deployed_grid() -> list[dict]:
    """The single-parameter twist grid. WRITTEN ONLY -- section B4 forbids running it tonight."""
    rows = []
    for request, residual in (("INTERSECT_Myeloid_g1188", "Myeloid"),
                              ("INTERSECT_Neuronal_g607", "Neuronal"),
                              ("INTERSECT_Endothelial_g128", "Endothelial")):
        for alpha in (0.5, 1.0, 2.0):
            for seed in (20260987, 20260988, 20260989):
                rows.append(entry(experiment="alpha_sweep", arm=f"twist_{request}",
                                  request=request, reward_model="residual",
                                  residual_dir=residual, alpha=alpha, seed=seed))
    return rows


def J_convergence() -> list[dict]:
    """J -> infinity recovers the exact twist; J is a Monte-Carlo budget, not a model parameter."""
    rows = []
    for J in (1, 4, 16, 64):
        for seed in (20260987, 20260988, 20260989):
            rows.append(entry(experiment="J_convergence", arm=f"J{J}",
                              request="INTERSECT_Myeloid_g1188", reward_model="residual",
                              residual_dir="Myeloid", alpha=1.0, J=J, seed=seed))
    return rows


# ============================================================================================
# The plug-in comparison arm.
#
# Purpose: split the gap between this round and the originally published row into two parts, the
# configuration changes (alpha, intermediate_scale, clean to residual, M and N) and the
# single-parameter change (the plug-in potential to the twist potential). The method is to keep
# this round's specification and swap only `--sampler twist` for `--sampler plugin`.
# The header of `crndiff/sampler.py` is that contract: `sampler='plugin'` reproduces the frozen
# intermediate potential and `sampler='twist'` replaces only it; the twist candidates use an
# independent generator, so both arms follow the same proposal trajectory, asserted by a gate.
# J is meaningless for the plug-in arm: `run_stem` writes the J token only for the twist sampler,
# so a plug-in stem never carries one. The row records J = 1 rather than inheriting the default.
PLUGIN172_J = 1


def plugin_172() -> list[dict]:
    """    The plug-in arm on the three single-type requests: three requests x three seeds.

        Identical to the main arm except for the sampler. The discriminators are reused, not refitted.
    """
    rows = []
    for target in MAIN_TARGETS:
        for seed in PROGRAMME_SEEDS:
            rows.append(entry(experiment="plugin_172", arm=f"{target}_plugin", ticket=172,
                              request=target, residual_dir=target,
                              tau=TAU_INHERITED[target]["tau"], alpha=PROG_ALPHA, seed=seed,
                              **C.mn_for(target),
                              **{**PROG_COMMON, "sampler": "plugin", "J": PLUGIN172_J}))
    return rows


def plugin_vs_twist() -> list[dict]:
    """The one comparison the single parameterisation is for: same proposal, same alpha, the
    plug-in intermediate potential vs the twist.

        Two corrections to the original grid:

        1. the request moves to the endothelial intersection and the discriminator moves from the
              BARE type to the one fitted for THAT request. The original used a bare-type discriminator
              on an intersection request, which is the error another machine made at the same stage;
              and this package has no discriminator for the other intersection at all, so the original
           grid could not run here. 2. The twist arm is NOT re-run: it is already the a1 arm of the
              intersection ladder (same request, discriminator, alpha, M and N, and seed; only the
              sampler differs), so this grid registers the three plug-in rows and points the rest at
    """
    return [entry(experiment="plugin_vs_twist", arm="plugin", sampler="plugin", ticket=172,
                  request=INTERSECT_REQUEST, reward_model="residual",
                  residual_dir=INTERSECT_RESIDUAL, tau=INTERSECT_TAU,
                  alpha=PROG_ALPHA, seed=seed, J=PLUGIN172_J,
                  **C.mn_for(INTERSECT_REQUEST),
                  **{k: v for k, v in PROG_COMMON.items()
                     if k not in ("sampler", "reward_model", "J")})
            for seed in PROGRAMME_SEEDS]


# the twist counterpart of `plugin_vs_twist`, NOT re-run -- it is Stage 3's a1
PLUGIN_VS_TWIST_TWIST_ARM = ("intersect_ladder", "a1")


GRIDS = {"gates": gates, "main_single_type": main_single_type,
         "table8": table8_multicluster,
         # ---- ticket 167: the full HVG programme, one grid per paper table ----
         "table1_ours": table1_ours,                # Stage 1 · Table 1 / 6 / 12
         "ablation_table10": ablation_table10,      # Stage 2 · Table 10
         "intersect_ladder": intersect_ladder,      # Stage 3 · Table 3 / 9
         "composition": composition,                # Stage 4 · Table 2 / 7
         "type_pair": type_pair,                    # Stage 6 · Table 8
         "schedule_table11": schedule_table11,      # Stage 5 · CANCELLED by 167-AMEND
         # ---- Stage 6 (Table 8) needs a MIX table built first; registered when it exists ----
         # ---- queued behind Q's go; NOT part of ticket 167 ----
         "alpha_sweep": deployed_grid,
         "J_convergence": J_convergence, "plugin_vs_twist": plugin_vs_twist,
         # ---- ticket 172 ----
         "plugin_172": plugin_172}
# ticket 167 stages, in the order section 5 fixes. Stages are independent (except that
# Table 10 / 11 borrow their deployed row from Stage 1), so a stuck stage is skipped, not fatal.
# 167-AMEND: Stage 5 (schedule_table11) removed from the execution sequence.
# Order is unchanged otherwise: 1 (9) -> 2 (discriminators + 18) -> 3 (15) -> 4 (9)
# -> 6 (Table 8, needs a MIX table built first). 57 runs total.
PROGRAMME_167 = ("table1_ours", "ablation_table10", "intersect_ladder", "composition",
                 "type_pair")
THIS_ROUND = ("gates", "main_single_type")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="write every grid (writing != running)")
    ap.add_argument("--gates", action="store_true", help="write only the gate manifest")
    ap.add_argument("--round", action="store_true",
                    help="write this round's manifests (gates + main_single_type)")
    ap.add_argument("--programme", action="store_true",
                    help="write ticket 167's five stage manifests")
    a = ap.parse_args(argv)
    if not (a.all or a.gates or a.round or a.programme):
        a.round = True
    names = (list(GRIDS) if a.all else
             list(PROGRAMME_167) if a.programme else
             list(THIS_ROUND) if a.round else ["gates"])
    OUT.mkdir(parents=True, exist_ok=True)
    chk = assert_g2_table_unchanged()
    print(f"[manifest] G2 tilt table {chk['sha256'][:16]}... bitwise_identical="
          f"{chk['bitwise_identical']}")
    for name in names:
        rows = GRIDS[name]()
        p = OUT / f"{name}.json"
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"[manifest] {name:18s} {len(rows):4d} runs -> {p}")
    if a.round:
        print("[manifest] queued behind Q's go, NOT written by --round: "
              "alpha_sweep(INTERSECT) / J_convergence / plugin_vs_twist / interval ladder / "
              "seeds 20260988, 20260989 / any baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
