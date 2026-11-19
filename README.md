# CRNDiff

Code for **CRNDiff: Count-Native Diffusion Framework via Chemical Reaction Networks**.

CRNDiff is a continuous-time diffusion model on non-negative integer vectors. The forward process
is a stochastic chemical reaction network with a closed-form transition kernel, so the reverse
process can be sampled exactly. Conditional generation needs no retraining: a product marginal
tilt is applied to the reverse proposal, and a Feynman-Kac particle layer corrects what the tilt
cannot express. The strength of the conditioning is one scalar per request.

This repository contains **code only**. It ships no data, no trained model weights, no generated
samples and no result tables. Everything that the code reads but that is not in the repository is
listed in `assets/README.md` with the command that rebuilds it.

## Layout

| Path | Contents |
|---|---|
| `crndiff/` | the model: forward chain, tilted reverse proposal, twist, particle layer, metrics, run I/O |
| `crndiff/frozen/` | the sampler and ruler modules exactly as used for the paper; their sha256 is asserted at run time (see `PROVENANCE.json`) |
| `evaluation/` | scorers: purity, Wasserstein-1 on raw counts, MMD² and PCC on log-CP10K, ancestry, downstream DE and TSTR, acceptance gates |
| `figures/` | panel helpers and figure builders |
| `toy2d/` | the two-dimensional count mixture used for the toy experiments |
| `horizon/` | the closed-form reverse kernel used to choose the noising horizon `T_O` |
| `experiments/` | run one delivery, or a queue of deliveries from a manifest |
| `scripts/` | entry points: training, sampling, discriminator fitting, tilt tables, scoring |
| `baselines/` | drivers for scVI, scANVI, CFGen and an MDLM port; see `baselines/README.md` |
| `assets/` | code of the frozen generator package that the model imports at load time (no data or weights) |
| `rebuttal/` | code for the additional experiments of the revised paper; see `rebuttal/README.md` |

## Installation

```bash
python -m venv .venv && . .venv/bin/activate      # Python 3.10 or newer
pip install -r requirements.txt                   # numpy, scipy, matplotlib, pandas, pillow
pip install -r requirements-runtime.txt           # torch and the rest of the training stack
python -m unittest discover -s tests -v
python scripts/check_assets.py                    # lists every input, present or missing
```

`check_assets.py` reports missing inputs (the normal state of a fresh clone) and fails only when
a file that is present has the wrong sha256.

## Data

The single-cell experiments use

* the Heart Cell Atlas (Litvinukova et al., *Nature*, 2020, https://www.heartcellatlas.org),
  restricted to 2,000 highly variable genes (451,513 cells, 11 cell types), and
* the Human Lung Cell Atlas core, in the preprocessed version released with CFGen
  (Zenodo record 15031364), restricted to 2,000 highly variable genes.

Prepare the heart arrays and place them under `assets/legacy_root/_shared/data/all/` as
`train.npy`, `val.npy`, `test.npy`, with the matching `*_celltype.npy`, `*_donor.npy` and a
`meta.json` that carries the gene list. Every loader asserts the gene-list hash. The lung
preprocessing is in `rebuttal/hlca_pipeline/prep_r16.py`.

Hyperparameters are selected on the validation split; the test split is used only for the
reported numbers.

## Training, sampling, evaluation

```bash
python scripts/train_generator.py --help                 # unconditional generator
python scripts/train_conditional_generator.py --help     # label-embedding variant (ablation)
python scripts/build_tilt_reference_table.py --help      # log c_g(n) for a requested reference group
python scripts/fit_residual_discriminator.py --help      # ratio estimator used by the particle layer
python scripts/export_classifier_to_numpy.py --help      # a fitted discriminator as a portable npz
python experiments/run_one.py --help                     # one conditional delivery
python experiments/queue.py --help                       # a manifest of deliveries
python scripts/score_deliveries.py --help                # purity, W1, MMD², PCC
```

`scripts/run_ours_training_seed.py` and `scripts/run_ours_downstream_chain.py` run the full
sequence for one training seed. The deployed configuration is `K = 32` reverse steps, `J = 16`
twist candidates per particle per step, reward exponent `alpha = 1`, resampling checkpoints every
8 steps with ESS threshold 0.5, `M = 2 n_out` particles, and a per-request tilt exponent `tau`
chosen on the validation split.

The toy dataset is generated rather than downloaded:

```bash
python scripts/make_toy_dataset.py
```

## Baselines

`baselines/README.md` records, for each baseline, where the implementation comes from, which
commit, and what was changed. No third-party source is vendored: scVI and scANVI run through
`scvi-tools`, CFGen and CountsDiff through their upstream repositories at pinned commits.

## Notes

* `.gitattributes` sets `* -text`, because the sha256 of several files is asserted at run time and
  line-ending conversion would break those checks.
* The frozen modules carry English comments in this release; their code is unchanged, and the
  hashes in `PROVENANCE.json` and in the scorers refer to the released files.
* Some scripts keep the directory names of the original experiment tree (for example
  `Baseline/OURS/results/`) as output locations.

## Licence and citation

The source code is under the MIT licence (`LICENSE`). Data used with it remain under their own
licences.

```bibtex
@inproceedings{crndiff,
  title     = {CRNDiff: Count-Native Diffusion Framework via Chemical Reaction Networks},
  author    = {Qiu, Yuxuan and Gagrani, Praful and Kobayashi, Tetsuya J.},
  booktitle = {International Conference on Learning Representations},
  year      = {2027}
}
```
