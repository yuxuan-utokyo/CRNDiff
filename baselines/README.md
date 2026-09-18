# Baselines

Four baselines appear in the paper. For each one we record where the implementation comes from,
what we changed and what we did not.

| Baseline | Implementation | Ours? | Changes |
|---|---|---|---|
| scVI | `scvi-tools`, driven by `scvi/code/train_scvi.py` and `generate_scvi.py` | driver only | none to the library; the driver fixes the split, the seed and the output layout |
| scANVI | `scvi-tools`, driven by `scanvi/code/train_scanvi.py` and `generate_scanvi.py` | driver only | as above |
| CFGen | upstream `https://github.com/theislab/CFGen.git` at commit `e347d1a981944e1a66bee3f913e74081d5793c27` | wrappers and dataset configs only | none to the upstream Python sources, verified by `git diff` against that commit. Our additions are the eight dataset configs in `cfgen/configs/` |
| MDLM | our port of masked discrete diffusion to count data; the released reference implementation targets text | ours | documented at the top of `mdlm/model_masked_mdlm.py` |

CFGen upstream is not vendored here. Clone the commit above into `baselines/cfgen/upstream`,
install its own dependencies and install it with `pip install -e baselines/cfgen/upstream`.
Copy the contents of `baselines/cfgen/configs/` into the upstream `configs/` directory. Prepared
H5AD files belong in `baselines/cfgen/data/`. The wrapper sets `CRNDIFF_CFGEN_DATA_DIR` for the
configs; set that environment variable yourself to use another data directory. Optionally set
`CRNDIFF_CFGEN_UPSTREAM` to another checkout of the same pinned commit. No upstream Python
sources are modified.

```bash
python baselines/cfgen/code/train_cfgen.py --tag smoke --smoke-steps 2
python baselines/cfgen/code/generate_cfgen.py --tag smoke --n 8
```

The H5AD preparation pipeline is not included in this release; the runtime audit used the
existing local `hvg2k_smoke_cfgen.h5ad`. The example requires that input. Archived multiseed
sampling wrappers additionally require their original sampling-state artifacts.

The scVI/scANVI training drivers accept `--max-epochs 1` for an explicit runtime budget;
omitting it retains the original library-default schedule. Use a separate test copy and small
real-data fixtures for smoke runs, as these drivers retain their original output locations.

Every baseline is trained on the same split, scored by the same frozen rulers and given the same
delivery budget as our sampler. Their per-run results are in `results/`.

`frozen_viz` note: two modules of the previous tree that our figure and fidelity code imports
rather than reimplements live in `archive/viz/`, not here, because they are rulers rather than
baselines.
