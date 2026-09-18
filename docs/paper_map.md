# Paper map

One row per table and per figure: what builds it, what it reads, and whether it rebuilds from
this repository alone. "Offline" means no GPU, no dataset, no retraining.

## Tables

| # | Label | Script | Reads | Offline |
|---|---|---|---|---|
| 1 | `tab:toy-quality` | `scripts/reproduce/table_01_toy_quality.py` | `results/toy/runs/`, `assets/toy/data/` | yes |
| 2 | `tab:hvg_single_type_main` | `scripts/reproduce/table_02_conditional_fidelity.py` | `results/table02_seeds/`, `215_paper_numbers_v8.json` | yes |
| 3 | `tab:hparams` | `scripts/reproduce/table_03_hyperparameters.py` | `crndiff/config.py` | yes |
| 4 | `tab:hvg_de` | `scripts/reproduce/table_04_differential_expression.py` | `212_cfgen_de.json` | yes |
| 5 | `tab:hvg_tstr` | `scripts/reproduce/table_05_tstr.py` | `212_tstr11.json` | yes |
| 6 | `tab:hvg_ablation` | `scripts/reproduce/table_06_ablation.py` | `215_paper_numbers_v8.json`, `w1_paper_ablation_table10.json`, `e21_ablation_table10.json`, `201_tau_archived_raw_scored.json` | yes |
| 7 | `tab:hvg_sampling_cost` | `scripts/reproduce/table_07_particle_diagnostics.py` | `ess_pre_resample.json`, `ancestry_trace.json` | yes |
| 8 | `tab:hvg_tau` | `scripts/reproduce/table_08_tilt_exponent.py` | `215_paper_numbers_v8.json`, `201_tau_*_scored.json` | yes |
| 9 | `tab:hvg_uncond_fidelity` | `scripts/reproduce/table_09_unconditional_fidelity.py` | `202_tableB1_full.json`, `20*_pool_composition_*.json`, `assets/rulers/master_purity_v8_lineage.json` | yes |
| 10 | `tab:hvg_sampler_axis` | `scripts/reproduce/table_10_sampler_axis.py` | `215_paper_numbers_v8.json`, `200_bestofn_*`, `200_vgr_FINAL_scored.json`, `200_aprime_v2_scored.json` | yes |

## Figures

| Figure | Script | Reads | Offline |
|---|---|---|---|
| 1, toy deliveries | `scripts/reproduce/figure_01_toy_delivered.py` | `results/toy/figdata.npz` | yes |
| toy dataset | `scripts/reproduce/figure_toy_dataset.py` | `assets/toy/data/`, `results/toy/figdata.npz` | yes |
| toy reverse chain, toy resampling marginal | `scripts/reproduce/figure_toy_chain_and_resampling.py` | `results/toy/figdata.npz` | yes |
| horizon criterion (a) and (b) | `scripts/reproduce/figure_horizon_criterion.py` | closed form, no inputs | yes |
| atlas PCA grid | `figures/pca_grid_swap_cfgen_pool.py` | delivered arrays, atlas basis | no |
| atlas UMAP panels | `figures/umap_panels.py` | delivered arrays, fitted UMAP reducer | no |
| ablation PCA 3x3 | `figures/ablation_pca_grid_3x3.py` | delivered arrays, atlas basis | no |
| ancestry collapse | `figures/ancestry_collapse.py` | `ancestry_trace.json` | no |

The four atlas figures need the counts of section 3 of the README and, for the UMAP panels, the
fitted reducer, which is 73 MB and not redistributed.

## Where the scripts and the printed paper differ

These are recorded, not resolved. Each script prints what the artefacts contain.

| Cell | Printed in the paper | Computed from the artefacts in this repository |
|---|---|---|
| Table 1, energy distance, all three rows | 0.255 / 0.036 / 0.017 | 0.274 / 0.068 / 0.021 |
| Table 1, mean marginal TV, i.i.d. row | 0.050 | 0.051 |
| Tables 6, 8, 10, tilted FK, endothelial `W1` | 0.058 | 0.060 |
| Tables 6, 8, tilted FK, myeloid `PCC` | 0.991 | 0.983 |
| Tables 6, 8, tilted FK, neuronal `MMD^2` | 0.0118 | 0.0121 |
| Tables 6, 8, tilted FK, neuronal `PCC` | 0.954 | 0.951 |

For the four tilted-FK cells the standard deviations agree with the artefacts (0.002, 0.008,
0.0003, 0.001); only the means differ, and the printed values each occur in the artefacts as a
single-seed value. For Table 1 the other two columns agree exactly, and the repository's energy
distance is taken against the exact request law while an earlier estimator used a sampled
reference; `results/toy/table_toy_main.json` carries both the per-seed values and the estimator
definition.

Five of the nine figures in the submitted paper are files whose names carry a `-1` suffix from a
download collision and which no script in this tree writes. Both versions are kept:
`results/figures/as_published/` holds the files the paper embeds, `results/figures/from_scripts/`
the corresponding script outputs. For the toy deliveries figure the script output rebuilds
byte-for-byte; for the other four the two differ in size and the difference has not been
explained.

## Provenance

`docs/COPY_MANIFEST.json` records every file in this repository: where it came from in the working
tree, its size and its sha256, verified after the copy. `docs/PATCH_LOG.json` and
`docs/PATCH_LOG_reproduce.json` record every import and path rewrite applied during the move,
rule by rule, with counts. `docs/PATCH_LOG_repairs.json` records a later pass that resolved every
asset path, module name and subprocess invocation in the repository and repaired the ones the
restructure had made stale, together with the three files that had to be added because something
in the tree referenced them. No numerical value, seed, hyperparameter or metric definition was
changed in any of these passes.
