# Assets

## Shipped

| Path | What | Size |
|---|---|---|
| `generator/pilot_run1c_s20260625.pt` | the frozen CRN generator used in every experiment; sha256 asserted by `crndiff/config.py` | 3.8 MB |
| `generator/logpi_nmax512_eps0.5.npy` | the marginal prior the unconditional arm uses | 8.2 MB |
| `rulers/master_purity_v7.json` | original per-request label set | |
| `rulers/master_purity_v8_lineage.json` | lineage-broadened label set; the one the paper reports | |
| `rulers/celltypist/` | the classifier wrapper and the archived label histograms the ruler gate checks against | 416 KB |
| `toy/data/` | the toy count mixture, its component pmf and its config | 1.5 MB |
| `toy/frozen/` | the toy network, tilt tables, discriminators and reference pools | 17 MB |
| `legacy_root/Baseline/OURS/code/` | the frozen sampling code, imported as-is; its directory shape is load-bearing and must not be flattened | 320 KB |
| `_shared/panel61/_shared/code/` | the gene-panel package the frozen model imports at load time | 1 MB |
| `_shared/panel61/_shared/data/` | the endothelial 61-gene panel counts that package reads: `train.npy` (80,463 x 61) and `val.npy` (10,057 x 61), integer counts derived from the Heart Cell Atlas and redistributed under its CC-BY-4.0 licence. `meta.json` carries the gene hash `47ef02be...7750` | 44 MB |

## Not shipped, and how to rebuild

| Missing | Why | Rebuild |
|---|---|---|
| `legacy_root/_shared/data/all/` | 1.7 GB of third-party atlas counts | prepare from the public atlas; README section 6 |
| `models/residual/` the fitted residual discriminators | 594 MB | `python scripts/fit_residual_discriminator.py` |
| `models/selection/` the best-of-N selector | | `python scripts/fit_selection_classifier.py` |
| `models/noised_clf/` the value-guided scorer | | `python scripts/fit_noised_value_classifier.py` |
| `models/conditional/` the label-embedding generator | | `python scripts/train_conditional_generator.py` |
| `models/mdlm/` the MDLM port's checkpoint | | `python scripts/run_mdlm_baseline.py --mode train` |
| `tables/tables_kz/` and `tables/tables_ref/`, the product-tilt tables | 150 MB, derived from the counts | `python scripts/build_tilt_reference_table.py` |
| the unconditional proposal pools | 1.1 GB | `python scripts/sample_unconditional_pool.py` |
| the fitted UMAP reducer | 73 MB, only the UMAP figure needs it | refit from the validation counts |
| the delivered count arrays | about 105 GB across all arms | `python experiments/queue.py` on the manifests |

`python scripts/check_assets.py` lists every one of these with its expected path and whether it
is present, so a missing input is a clear message rather than a stack trace. It exits non-zero
only when a file that IS present has the wrong sha256; missing inputs are the normal state of a
fresh clone. Pass `--strict` to require everything.

One more input is not shipped and has no rebuild command: the 40 MB archived reference array that
gate G3 compares a reproduction against, `archive/results/runs/linspace_T_O_K32_spilot_run1c_...
_UNI_DONOR_D1_n10000_g0p4_noa4.npy`. Its sha256 is asserted in `evaluation/acceptance_gates.py`,
and the gate needs the atlas counts anyway, so only someone who has already prepared the data can
run it.

## Attribution

`_shared/panel61/_shared/data/` is derived from the Heart Cell Atlas (Litvinukova et al., Nature
2020, https://www.heartcellatlas.org), which is distributed under CC-BY-4.0. Those two arrays keep
that licence; the MIT terms in `LICENSE` cover the code, not them.
