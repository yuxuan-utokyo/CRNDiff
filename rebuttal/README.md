# Code for the additional experiments of the revised paper

These scripts were written during the review period, after the code in the repository root was
frozen. They are kept close to how they were run, so they use their own directory layout and
read inputs through a working directory, `./work/`, that is not part of the repository. Paths to
data, generated arrays and intermediate files must be adapted before running. No data, model
weights, generated samples or results are included.

## Directories

| Directory | Contents |
|---|---|
| `heart_tree/` | the experiment tree used for the heart (HVG2K) experiments. `hvg/` is the model package under its original name (the same code as `crndiff/` in the repository root), with `analysis/`, `figures/`, `tools/` and `experiments/` holding only the modules the rebuttal scripts import. `heart_tree/rebuttal/` contains the experiment scripts. |
| `hlca_pipeline/` | the Human Lung Cell Atlas pipeline: data preparation (`prep_r16.py`, `nmax.py`, `r16_export.py`), tilt tables (`r16_tables.py`), generation and Feynman-Kac sampling (`fk/`), and the CFGen baseline split (`baselines/`). |
| `analysis/` | scoring, tables and figures. Folder names follow the internal experiment numbers listed below. |

## Paper section → code

| Paper | Code |
|---|---|
| Table 1 (main table, heart): CountsDiff rows, coverage and 1-NN columns | `heart_tree/rebuttal/cd_*.py`, `heart_tree/rebuttal/r12_src/`, `analysis/r12/`, `analysis/r12_countsdiff/`, `analysis/r12_extra/`, `heart_tree/rebuttal/w2_runs.py`, `analysis/r19/` |
| Table 2, Table 6, Appendix B.4 and B.8 (heart test metrics, neighbour fraction, subtypes, library size) | `analysis/r19/`, `heart_tree/rebuttal/r19_src/`, `analysis/r38/`, `analysis/r42/`, `analysis/r46/`, `heart_tree/rebuttal/r1_*.py`, `r2_*.py`, `r3_*.py` |
| Table 3 and Appendix C (lung) | `hlca_pipeline/`; scoring in `analysis/r16/`, `analysis/r24_26/`, `analysis/r28/`, `analysis/r29_30/`, `analysis/r31/`, `analysis/r34/` to `analysis/r41/` |
| Lung baselines (CFGen, CountsDiff, scVI, scANVI) | `hlca_pipeline/baselines/`, `heart_tree/rebuttal/r16_baselines/`, `analysis/r16_baselines_trainlib.py` |
| Table 14 (cells and genes used for the marginal ratios, including top-k tilt) | `heart_tree/rebuttal/run_capped.py` with tilt tables restricted to the top-k genes; commands in `heart_tree/rebuttal/RUN_EXAMPLES.md` |
| Appendix B.5, Figure 10 (choice of τ) | `analysis/r17/`, `analysis/r32/`, `analysis/r33/`, `heart_tree/rebuttal/r6_tau.py`, `r6b_split.py` |
| Appendix B.6 and B.7 (discriminator diagnostics, 10% labels, intersection target UAP1⁺) | `analysis/discriminator/`, `heart_tree/rebuttal/r4_*.py`, `heart_tree/rebuttal/r10_*.py`, `analysis/r10/`, `heart_tree/rebuttal/r14*.py`, `analysis/r14/` |
| Appendix B.9 (runtime) | `heart_tree/rebuttal/r18/`, `analysis/r18/` |
| Figure 6 (PCA grid on the test split) and Figure 7 | `analysis/figs/`, `analysis/r44/`, `heart_tree/rebuttal/fig7_test_overlay.py` |

## Environments

`heart_tree/rebuttal/r12_src/README.md` and the requirement files in the repository root list the
packages. The scoring scripts were run with Python 3.11, numpy 2.x, scikit-learn 1.9 and
scvi-tools 1.4; discriminators were fitted with scikit-learn 1.3.0, the version used for the
original submission.
