# CRN-Diffusion

Reference implementation and reproduction package for a continuous-time discrete diffusion model
on `Z_{>=0}^S` with an inference-time conditioner: a product marginal tilt applied to the reverse
proposal, followed by a Feynman-Kac particle layer that corrects what the tilt cannot express.
The conditioner needs no retraining, and its strength is one scalar per request.

The repository contains the model, every baseline used in the paper, every number behind the ten
tables and nine figures, and one script per table and figure that rebuilds it from those numbers.

## Quickstart

```bash
git clone <this repository>
cd CRN-Diffusion
python -m venv .venv && . .venv/bin/activate     # Python 3.10 or newer
pip install -r requirements.txt                  # numpy, scipy, matplotlib, pandas, pillow
python scripts/run_smoke_test.py
```

That last command runs all fifteen offline entry points and takes about half a minute. It should
end with:

```
PASS  table_01_toy_quality.py                       11.5s
...
PASS  figure_horizon_criterion.py                    4.1s
------------------------------------------------------------------------------
tables written to out/tables/   (8 files)
figures written to out/figures/ (8 pdf)
toy figure extract rebuilt: all arrays exactly equal
------------------------------------------------------------------------------
[smoke] all 15 entry points reproduced from the files in this repository
```

The extract comparison checks array names, shapes, dtypes and values. Discrete arrays must
match exactly; floating arrays permit only eight machine epsilons of relative rounding
(no absolute tolerance). Frozen source and checkpoint SHA-256 checks remain strict.

No GPU, no dataset download and no training are involved. The ten tables and five of the nine
figures come out of the files in the clone. The other four figures are atlas scatter plots and
need the count matrices, which are not redistributed; section 6 says how to obtain them.

## Runtime checks and training

The offline reproduction above does not train or sample a model. To exercise the shipped toy
network and 32-step samplers, including intermediate resampling and same-seed checks:

```bash
pip install -r requirements-runtime.txt
python scripts/run_toy_smoke_test.py
python -m unittest discover -s tests -v
```

After preparing the atlas files in section 6, use the public launcher for unconditional training:

```bash
python scripts/train_generator.py --name runtime_smoke --steps 2 --batch 4 --seed 20260918
```

This delegates to the frozen trainer unchanged and creates its output directories before its
first checkpoint save. The example is a runtime test, not a converged model. Outputs are in
`assets/legacy_root/Baseline/OURS/{models,loss}/`. Use a unique `--name` for each run.
Omitting `--steps`, `--batch` and `--seed` preserves the frozen trainer's defaults; the paper's
long-run orchestration additionally requires its archived experiment manifests.

For canonical purity scoring, run `python -m evaluation.gate_celltypist` first, with the
original atlas, base pool and Heart CellTypist model installed. The gate must pass without
changing its histogram comparisons. On Windows, `PYTHONUTF8=1` keeps redirected logs readable.

## 1. Layout

| Path | Contents |
|---|---|
| `crndiff/` | the model: forward chain, tilted reverse proposal, twist, particle layer, metrics, run I/O |
| `crndiff/frozen/` | byte-identical copies of the sampler and ruler modules that produced the published numbers. Never edited; their sha256 is asserted at run time, so current code can be diffed against them |
| `evaluation/` | the scorers: purity, Wasserstein on raw counts, MMD^2 and PCC on log-CP10K, ancestry, downstream DE and TSTR, acceptance gates |
| `evaluation/frozen/` | the frozen metric implementations the scorers import rather than reimplement |
| `figures/` | panel helpers and the atlas figure builders |
| `toy2d/` | the two-dimensional count mixture: chain, kernels, exact component responsibilities |
| `horizon/` | the closed-form reverse kernel used to place the noising horizon `T_O` |
| `experiments/` | one delivery, one manifest, one queue |
| `scripts/` | entry points, one per operation; `scripts/reproduce/` has one per paper table and figure |
| `baselines/` | scVI, scANVI, CFGen drivers and our MDLM port |
| `assets/` | the frozen generator, the purity rulers, the toy dataset and its network, the 61-gene panel |
| `archive/` | the small set of files from the previous experiment tree that the gates and frozen engines still read |
| `results/` | every number behind the paper, at per-run granularity |
| `docs/` | the paper map, the copy manifest, the patch logs, the pre-registrations |

## 2. What runs with what

| You have | You can run |
|---|---|
| the clone and `requirements.txt` | the ten tables, the five offline figures, `check_assets.py`, `run_smoke_test.py` |
| the above plus `torch`, `scikit-learn`, `joblib` | every entry point's argument parsing, the toy sampler, the portable-classifier export |
| the above plus the atlas counts (section 6) | training, sampling, scoring, the four atlas figures, the acceptance gates |
| the above plus the fitted models and tilt tables (`assets/README.md`) | the full pipeline end to end, from a request to a scored delivery |

The main training and sampling launchers report missing inputs explicitly. Some historical
experiment drivers also require archived manifests or evaluation gates. See
`assets/README.md` for the inputs that are not included and how to obtain them.

## 3. Reproducing the tables

```bash
python scripts/reproduce/table_01_toy_quality.py
python scripts/reproduce/table_02_conditional_fidelity.py
python scripts/reproduce/table_03_hyperparameters.py
python scripts/reproduce/table_04_differential_expression.py
python scripts/reproduce/table_05_tstr.py
python scripts/reproduce/table_06_ablation.py
python scripts/reproduce/table_07_particle_diagnostics.py
python scripts/reproduce/table_08_tilt_exponent.py
python scripts/reproduce/table_09_unconditional_fidelity.py
python scripts/reproduce/table_10_sampler_axis.py
```

Each prints its LaTeX and writes it to `out/tables/`. All ten read `results/` alone.

`docs/paper_map.md` states, table by table, which script builds it, which files it reads, and
which cells differ from the printed paper. Table 3 is a special case: it is read out of
`crndiff/config.py` at run time, so the reported configuration cannot drift from the one the
samplers use.

## 4. Reproducing the figures

```bash
python scripts/reproduce/build_toy_figure_data.py         # the toy figure extract
python scripts/reproduce/figure_01_toy_delivered.py
python scripts/reproduce/figure_toy_dataset.py
python scripts/reproduce/figure_toy_chain_and_resampling.py
python scripts/reproduce/figure_horizon_criterion.py
```

`build_toy_figure_data.py` writes into `out/toy/` and compares its own sha256 against the copy
shipped in `results/toy/`, rather than overwriting it. The two are identical, which is the one
check in the package that compares a rebuild against a published artefact byte for byte.

The four atlas figures (`figures/pca_grid_swap_cfgen_pool.py`, `figures/umap_panels.py`,
`figures/ablation_pca_grid_3x3.py`, `figures/ancestry_collapse.py`) read delivered count arrays
and the atlas embedding, so they need section 6. The published PDFs are in `results/figures/`.

## 5. Where the numbers live

| Path | What |
|---|---|
| `results/summary_json/` | the scored summaries each table is built from, one file per scoring run |
| `results/by_table/` | the per-run records behind each table, grouped by table |
| `results/table02_seeds/` | the three training seeds per arm of the main conditional-fidelity table, one run each |
| `results/toy/` | the toy runs, the figure extract and the toy table source |
| `results/figures/` | `as_published/` holds the PDFs the paper embeds, `from_scripts/` the outputs of the scripts here |

Every scored run stores its full classifier label histogram, so a purity ruler can be changed and
every number recomputed without sampling again:

```bash
python scripts/rescore_purity_lineage_ruler.py     # no arguments; writes out/summary/
```

The scoring and table-building scripts outside `scripts/reproduce/` read `out/summary/`, which is
the working directory those runs wrote to. `results/summary_json/` is the published copy of it. To
re-run one of those builders on the published numbers, copy the directory across first:

```bash
mkdir -p out/summary && cp results/summary_json/* out/summary/
```

## 6. Data

The single-cell experiments use the Heart Cell Atlas (Litvinukova et al., *Nature*, 2020,
"Cells of the adult human heart", https://www.heartcellatlas.org), restricted to 2,000 highly
variable genes: 451,513 cells in 11 cell types. The counts are not redistributed here, with
one exception noted below.

Prepare the arrays and place them under `assets/legacy_root/_shared/data/all/` as `train.npy`,
`val.npy` and `test.npy`, with the matching `*_celltype.npy`, `*_donor.npy` and `meta.json`.
`meta.json` carries the gene list, and every loader asserts its hash, so a different gene
selection fails at load instead of quietly producing different numbers.

The exception: `assets/_shared/panel61/_shared/data/` holds an endothelial 61-gene panel
(80,463 training and 10,057 validation cells) that the frozen model imports at load time. It is
Heart Cell Atlas data under CC-BY-4.0 and is redistributed here under that licence.

The toy dataset is generated rather than downloaded:

```bash
python scripts/make_toy_dataset.py       # rebuilds assets/toy/data/data2d_ring8_sym_v2.* bitwise
```

```bash
python scripts/check_assets.py           # every input, present or absent, with its expected path
```
`check_assets.py` exits non-zero only when a present file's sha256 disagrees with
`PROVENANCE.json`. Missing inputs are the normal state of a fresh clone; pass `--strict` to make
them fail too. `assets/README.md` lists each missing input and how to rebuild it.

## 7. Training, sampling, evaluation

Beyond the offline reproduction these need `torch`, `scikit-learn`, `joblib`, `anndata`, `scanpy`,
`celltypist`, and `scvi-tools` for the scVI and scANVI baselines. The commented block in
`requirements.txt` lists them.

```bash
python scripts/train_conditional_generator.py --help     # label-embedding variant
python scripts/fit_residual_discriminator.py --help      # the ratio estimator the particle layer uses
python scripts/fit_selection_classifier.py --help        # the best-of-N selector
python scripts/fit_noised_value_classifier.py --help     # the value-guided scorer
python scripts/build_tilt_reference_table.py --help      # log c_g(n) for a requested reference group
python scripts/export_classifier_to_numpy.py --help      # a fitted discriminator as a portable npz
```

`scripts/run_ours_training_seed.py` and `scripts/run_ours_downstream_chain.py` run that sequence
end to end for one training seed, which is how the three seeds of the main table were produced.

```bash
python experiments/run_one.py --help                     # one delivery
python experiments/queue.py --help                       # a manifest of deliveries
python scripts/sample_best_of_n.py --help
python scripts/sample_value_guided.py --help
python scripts/sample_unconditional_pool.py --help
```

The frozen generator used throughout the paper ships at
`assets/generator/pilot_run1c_s20260625.pt`, and `crndiff/config.py` asserts its sha256 on load.
The deployed configuration is `K = 32` reverse steps, `J = 16` twist candidates per particle per
step, reward exponent `alpha = 1`, resampling checkpoints every 8 steps with ESS threshold 0.5,
`n_out = max(2500, n_matched)` delivered cells and `M = 2 n_out` particles, and a per-request tilt
exponent `tau` of 0.4 for endothelial and 0.2 for myeloid and neuronal.

```bash
python scripts/score_deliveries.py --help                # purity, W1, MMD^2, PCC on the frozen rulers
python scripts/score_unconditional_fidelity.py --help
python scripts/score_pool_composition.py --help
python scripts/collect_ess_at_resampling.py             # no arguments: reads out/runs/
python scripts/collect_ancestry_diagnostics.py          # no arguments: reads out/runs/
```

Thirteen scripts take no arguments at all and read `out/` directly; `--help` on those runs the
script. `grep -L ArgumentParser scripts/*.py` lists them.

Purity is a classifier-assigned label histogram intersected with a fixed label set per request.
Both label sets are in `assets/rulers/`: `master_purity_v7.json` is the original one and
`master_purity_v8_lineage.json` the lineage-broadened one the paper reports.

## 8. Baselines

`baselines/README.md` records, for each baseline, where the implementation comes from, which
commit, what was changed and what was not. No third-party source is vendored: scVI and scANVI run
through `scvi-tools`, and CFGen through its own upstream repository at a pinned commit.

```bash
python scripts/train_and_sample_scvi_scanvi.py --help
python scripts/sample_baseline_cfgen.py --help
python scripts/run_mdlm_baseline.py --mode {gate,train,sample}
```

Every baseline is trained on the same split, scored by the same frozen rulers and given the same
delivery budget as the sampler here.

## 9. Provenance

Nothing in this repository is normalised by git (`.gitattributes` sets `* -text`) because the
sha256 of many files is asserted at run time and line-ending conversion would break those guards.

* `PROVENANCE.json` records the frozen modules, the generator checkpoint, the gene panel and the
  one edit made to an inherited file, each with its sha256 before and after.
* `docs/COPY_MANIFEST.json` records every file: where it came from, its size and its sha256.
* `docs/PATCH_LOG.json`, `docs/PATCH_LOG_reproduce.json` and `docs/PATCH_LOG_repairs.json` record
  every import and path rewrite applied while the tree was reorganised, rule by rule, with counts.
  No numerical value, seed, hyperparameter or metric definition was changed in any of them.
* `.github/workflows/smoke.yml` runs `check_assets.py`, the smoke test and `compileall` on every
  push, on Python 3.10 and 3.12, so a change that breaks a fresh clone shows up here first.
* `docs/prereg/` holds the pre-registration the tilt-exponent sweep was registered against, in its
  original wording and language.

Result files carry the absolute paths of the machine the runs were made on. They are kept as
recorded rather than rewritten, because they are part of the artefacts whose hashes are asserted.

## 10. Known limitations

* Four cells of Tables 6, 8 and 10 and one column of Table 1 differ from the printed paper. Both
  values are listed in `docs/paper_map.md`. The scripts print what the artefacts contain, and
  nothing is adjusted to match the paper.
* Five of the nine published figure files carry a `-1` suffix from a download collision and no
  script in this tree writes them. Both versions are shipped, under `results/figures/`.
* The counts, the fitted discriminators, the tilt tables and the unconditional pools are not
  redistributed. `assets/README.md` lists each one and the command that rebuilds it.
* The multi-seed downstream evaluation (`evaluation/downstream_multiseed.py`,
  `evaluation/downstream_deployed_arm.py`) reads the delivered count arrays listed in
  `archive/universality_fk/out/e30plus/E30plus_manifest.json`, about 7.6 GB that are not
  redistributed. The manifest itself ships, so the array list is public.
* `scripts/audit_paper_numbers.py` needs the paper source; point `CRNDIFF_PAPER_TEX` at it.
* Continuous integration runs the offline reproduction and launcher regression tests. The
  CPU toy runtime test is a separate command; atlas training and sampling require the
  additional inputs listed above. Passing the offline checks does not establish full-budget
  scientific reproduction or validate every historical experiment driver.

## 11. Licence and citation

The source code is under the MIT licence. The redistributed 61-gene panel arrays stay under
CC-BY-4.0, the licence of the Heart Cell Atlas they are derived from. See `LICENSE`.

```bibtex
@inproceedings{crndiffusion,
  title     = {TITLE},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026}
}
```

Fill in the title, and the author and venue fields once the submission is no longer anonymous.
