# PRE-REGISTRATION — scVI baseline (STEP A)

Written **before scVI was constructed or trained**: at the time of writing this
folder contains only `code/ data/ logs/ out/`, no model, no sample, no metric.
Governed by `HVG2K/SPEC_baselines_scvi_scdiffusion.md` §0 and §1.

## What this arm is for

It replaces "we only compared against one setup-incompatible cfDiffusion" with
"we compared against scVI on the *same* 2000-HVG split, on the same ruler". So
every degree of freedom that could make scVI look better or worse than it is
gets fixed here, before any number exists.

## 1. Data — identical split, asserted not assumed

The same `all` split ours and cfDiffusion use:
`Experiments/X/_shared/data/all/{train,val}.npy` + `*_celltype.npy`,
2000 HVGs, gene_hash **`6f7963523c7a4c5bf1df7264c4c11e0a42fb0a9c4d52ca91d5724ae5e1eadb12`**.

The loader **asserts** that hash and stops if it differs. No HVG selection is
redone here. scVI is trained on the **train** split only; `val` is reference and
never trained on.

## 2. Model and training — official defaults, nothing tuned

* `scvi.model.SCVI` with library defaults: `n_latent=10`, `n_hidden=128`,
  `n_layers=1`, `dropout_rate=0.1`, `dispersion="gene"`,
  `gene_likelihood="zinb"`, `latent_distribution="normal"`.
* Conditioning axis: **`cell_type` as `batch_key`** in `setup_anndata`. This is
  the covariate scVI conditions its decoder on, which is what makes
  "generate cells of type T" well defined (`transform_batch`-style decoding).
  Alternatives (`categorical_covariate_keys`) do reach the decoder too, but
  `batch_key` is the documented conditioning slot and is what the generative
  path uses; the choice is fixed here rather than after seeing purity.
* **Training length: the library default.** `model.train()` is called with **no**
  `max_epochs`, so scvi-tools computes it from the cell count
  (`min(round((20000/n_cells)*400), 400)`). The resolved value is recorded in
  `logs/train_scvi.json`. It is not overridden in either direction.
* **Early stopping: off** (the library default). Unlike the cfDiffusion arm,
  where Q approved a monitored early stop for a 800k-step budget, scVI's default
  schedule is short and self-limiting, so there is nothing to stop early. The
  ELBO/reconstruction history is still logged for the appendix.
* Seed: **20261050** via `scvi.settings.seed`, recorded.

No hyper-parameter is tuned. If scVI performs poorly, that is recorded as the
result.

## 3. Conditional generation — the one place a choice must be made

scVI is a latent-variable model, not a conditional sampler with a guidance
knob. To produce **new** cells of type T (rather than reconstruct existing
ones):

1. draw `z ~ N(0, I)` in the 10-D latent, one per requested cell;
2. decode with the batch covariate set to **T**;
3. draw counts from the fitted **ZINB** likelihood with those parameters.

Step 3 uses the model's own likelihood, so counts are integer **by
construction** — scVI cannot emit a negative or fractional count, and this is a
structural property to report, not a pass we award it.

**Library size.** scVI's decoder needs one. Fixed choice, matching the
cfDiffusion arm exactly so the two baselines are comparable: draw library sizes
i.i.d. from the **real val cells of that same type**, with
`numpy.random.default_rng(20261052)`, sampled with replacement. As there, this
hands the baseline the real library distribution and is generous to it.

**n = 20000 per type**, matching the main table's generated-arm size.

**Types: Endothelial, Myeloid, Neuronal.** Mesothelial is excluded by the
spec, and independently by the fact that CellTypist's own guard refuses a
mapping at 71 real val cells.

**Sampling seed 20261051.** No guidance scale exists in scVI, so there is no
grid to sweep and nothing to select on.

## 4. Unconditional row

The required "unconditional" row is produced from the **same trained model** by
decoding `z ~ N(0, I)` with the batch covariate drawn from the empirical
training mix rather than fixed to T. That isolates the effect of conditioning
within one model, which is what the three-row protocol is for.

## 5. Evaluation — reused, not rewritten

* **Purity**: `code/eval/celltypist_eval.py` unchanged, `Healthy_Adult_Heart.pkl`,
  n=2500, tag `_scvi`. **Three rows reported: unconditional / conditional /
  real val**, and the headline is the **ratio to the real-val row**, because the
  2000-HVG panel handicaps every row equally and only the ratio cancels it.
* **W1 family**: `cond_w1` caliber verbatim — same reference (that type's real
  val), same floor (real train vs real val), 512 projections at seed
  **20261070**, per-type common-n subsampling at seed **20261071**. This puts
  scVI on one ruler with the cfDiffusion rows and with OURS.
* **Correlation error and dispersion**: `code/eval/diag_fkv2.py` and
  `code/eval/make_dispersion.py`, unchanged.

No evaluator is modified, and no evaluator output participates in choosing any
setting above.

## 6. What would falsify a favourable reading

If scVI's conditional purity ratio is not clearly above its own unconditional
row, then conditioning is not working in this setup and the arm is reported that
way — a high absolute purity with a flat conditional/unconditional gap does not
count as success.
