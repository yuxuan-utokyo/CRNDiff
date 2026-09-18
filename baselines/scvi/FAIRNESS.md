# FAIRNESS — scVI baseline on hvg2k

Red-line document for STEP A of `SPEC_baselines_scvi_scdiffusion.md`.
Written before training.

## 1. Official package, official defaults

| item | value |
|---|---|
| package | `scvi-tools` (scverse official, pip) |
| version | **1.3.3** |
| model | `scvi.model.SCVI`, constructor defaults untouched |
| training | `model.train()` with **no** `max_epochs` — the library's own schedule |
| conditioning | `cell_type` as `batch_key` |

No hyper-parameter is set by us. Nothing is swept. If the arm underperforms,
that is the result.

## 2. Environment: reusing `crn_diffusion`, and why

The cfDiffusion arm got its own conda env because it needed a pinned, awkward
stack (cu111-era code on an sm_89 GPU). scVI does not: `scvi-tools 1.3.3` is
**already installed and working** in `crn_diffusion`, on the exact pin this
project previously had to fight for —

* `scvi-tools 1.3.3`
* `orbax-checkpoint 0.5.23` (the Windows long-path workaround this repo hit
  before; a fresh `pip install scvi-tools` tends to pull a newer orbax and
  reintroduce it)

Rebuilding that from scratch would risk breaking a known-good stack for no
benefit. Verified at the start of this arm: scvi-tools 1.3.3, torch 2.11.0+cu128
with CUDA available, anndata 0.11.4, scanpy 1.11.5, numpy 1.26.4,
lightning 2.6.5. Recorded in `logs/env.txt`.

scVI is only ever *imported* here — no upstream source file is edited, so there
is no patch to record and no `work/` copy to keep pristine against.

## 3. Same data as every other arm

Same `all` 2000-HVG split, gene_hash asserted equal to
`6f7963523c7a4c5b…eadb12` at load time. Train split for training, val split as
the evaluation reference. No re-selection of HVGs.

## 4. The one modelling choice, and which way it leans

scVI needs a library size to decode. It is drawn from the **real val cells of
the target type** (seed 20261052), exactly as the cfDiffusion arm does. This is
**generous to the baseline** — it is handed the real library distribution rather
than having to produce one — and it is applied identically to both baselines so
they stay comparable. Stated wherever a library-dependent quantity is reported.

## 5. A structural advantage scVI has, stated plainly

scVI samples counts from a fitted **ZINB likelihood**, so its output is integer
and non-negative **by construction**. It cannot commit the count-legality
failures the latent-Gaussian route commits (cfDiffusion needed expm1 + clip +
round, and produced counts up to 1e16 against a real maximum of ~1240).

That is a genuine architectural strength of scVI and is reported as such. It
also means the count-legality columns are trivially clean for this arm, and a
clean column there is **not** evidence of overall quality.

## 6. Evaluation is the project's, unchanged

`celltypist_eval.py`, the `cond_w1` caliber, `diag_fkv2.py` and
`make_dispersion.py` are reused read-only. Purity is reported as three rows
(unconditional / conditional / real val) with the ratio to real as the headline,
because the 2000-HVG panel depresses every row and only the ratio is comparable
to published numbers.
