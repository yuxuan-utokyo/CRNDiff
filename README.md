# CRNDiff

Reference implementation of **CRNDiff**, a count-native diffusion framework based on
continuous-time chemical reaction networks (CRNs), with inference-time conditioning
through marginal tilting and Feynman--Kac (FK) steering.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone <this-repository-url>
cd CRNDiff

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

For training and sampling components that require additional runtime dependencies:

```bash
pip install -r requirements-runtime.txt
```

## Repository structure

```text
crndiff/       Core CRNDiff model and sampling implementation
toy2d/         Two-dimensional synthetic count-mixture experiment
horizon/       Terminal-noising-time and birth--death utilities
experiments/   Experiment launchers
evaluation/    Evaluation metrics and scoring utilities
scripts/       Training, sampling, and evaluation entry points
baselines/     Baseline interfaces
tests/         Basic runtime tests
```

## Method

CRNDiff models nonnegative integer-valued data directly in count space using a
continuous-time birth--death process.

Conditional generation is performed at inference time with a frozen generator:

1. marginal tilting adapts the reverse proposal toward target coordinate marginals;
2. Feynman--Kac steering applies a residual correction through particle weighting
   and resampling.

## Training and sampling

Training and sampling entry points are provided under `scripts/`.

Examples:

```bash
python scripts/train_generator.py --help
python scripts/sample_unconditional_pool.py --help
python scripts/sample_best_of_n.py --help
python scripts/sample_value_guided.py --help
```

The tilted FK sampler is implemented in `crndiff/`.

## Synthetic experiment

The two-dimensional synthetic experiment is implemented in `toy2d/`.
It does not require the single-cell dataset.

## Single-cell experiments

The single-cell experiments use scRNA-seq data from the adult human heart atlas.

The dataset is not redistributed in this repository. Please obtain it from the
original source:

Litvinukova et al., *Cells of the adult human heart*, Nature, 2020.

This repository also does not redistribute trained checkpoints, generated-cell
pools, cached experiment outputs, or result files used to construct the reported
tables and figures.

## Evaluation

Evaluation utilities are provided in `evaluation/` and include the metrics used in
the paper, such as purity, Wasserstein-1 distance, MMD, Pearson correlation,
genealogical diversity, differential-expression agreement, and downstream
classification.

## Baselines

Baseline interfaces are provided in `baselines/`.

Third-party source code is not redistributed. Please obtain the corresponding
implementations from their original repositories or packages.

## Testing

```bash
python -m unittest discover -s tests -v
```

## License

The source code is released under the MIT License.

## Citation

```bibtex
@inproceedings{crndiff,
  title     = {CRNDiff: Count-Native Diffusion Framework via Chemical Reaction Networks},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026}
}
```
