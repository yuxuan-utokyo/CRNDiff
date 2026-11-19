# Assets

Only code is shipped under `assets/`: the frozen generator package
(`legacy_root/Baseline/OURS/code/`), the gene-panel package it imports at load time
(`_shared/panel61/_shared/code/`) and the CellTypist wrapper (`rulers/celltypist/`). Their
directory layout is load-bearing and must not be flattened.

No data, model weights, tilt tables, discriminators, label sets or generated arrays are
included. `python scripts/check_assets.py` lists every expected input with its path, and whether
it is present.

| Input | Rebuild |
|---|---|
| `legacy_root/_shared/data/all/` (heart counts, 2,000 HVGs) | prepare from the Heart Cell Atlas; see the main README |
| `_shared/panel61/_shared/data/` (61-gene endothelial panel) | derived from the same atlas |
| `generator/` (unconditional generator checkpoint and marginal prior) | `python scripts/train_generator.py` |
| `models/residual/` (residual discriminators) | `python scripts/fit_residual_discriminator.py` |
| `models/selection/` (best-of-N selector) | `python scripts/fit_selection_classifier.py` |
| `models/noised_clf/` (value-guided scorer) | `python scripts/fit_noised_value_classifier.py` |
| `models/conditional/` (label-embedding generator) | `python scripts/train_conditional_generator.py` |
| `models/mdlm/` (MDLM port) | `python scripts/run_mdlm_baseline.py --mode train` |
| `tables/tables_kz/`, `tables/tables_ref/` (product-tilt tables) | `python scripts/build_tilt_reference_table.py` |
| unconditional proposal pools | `python scripts/sample_unconditional_pool.py` |
| `rulers/` label sets for purity | `python -m evaluation.gate_celltypist` and `crndiff/frozen/canonical_purity.py` |
| `toy/` toy dataset and network | `python scripts/make_toy_dataset.py`, then train with `toy2d/` |
