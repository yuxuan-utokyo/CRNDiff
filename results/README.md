# Results

Every number the paper prints, at the granularity of a single run.

| Path | Contents |
|---|---|
| `summary_json/` | the authoritative summary artefacts, copied verbatim from the experiment tree. Every reproduction script reads these and never recomputes a metric |
| `table02_seeds/<arm>/` | Table 2: three training seeds per arm, one sampling run each, one file per run |
| `by_table/table06..table10/` | Tables 6 to 10: one file per run, split out of the summaries above |
| `toy/` | the toy figure data, the toy table record and the frozen toy runs the table reads |
| `figures/as_published/` | the figure files the submitted paper embeds |
| `figures/from_scripts/` | the corresponding outputs of the scripts in this repository |
| `all_runs_index.csv` | one row per run across all tables, with its metrics and its source file |

## Seeds

Table 2 varies the training seed: three independently trained generators per arm, sampled once
each.

| Arm | Training seeds | Sampling seeds |
|---|---|---|
| Ours | 20260625, 20260627, 20260630 | 20260987, 20260989 |
| MDLM | 20260625, 20260934, 20260935 | 20260932, 20260933 |
| CFGen | three independent training runs (the upstream code exposes no seed) | 20260932, 20260933 |
| scVI | 20261050, 20261053, 20261054 | 20261051, 20261055 |
| scANVI | 20261060, 20261063, 20261064 | 20261061, 20261065 |

Tables 6 to 10 hold the generator fixed and vary only the sampling seed, three runs per arm.
MDLM's 20260625 and ours are the same number in two unrelated training pipelines.

## Metrics

| Field | Meaning |
|---|---|
| `purity_abs_v8` | classifier-assigned share of the requested lineage; the label sets are in `assets/rulers/master_purity_v8_lineage.json` |
| `purity_rel_v8` | the same divided by the ceiling that real cells of that type reach on the same ruler |
| `purity_abs_v7` | the earlier single-label ruler, kept for comparison; not what the paper reports |
| `W1` | Wasserstein-1 on raw counts |
| `mmd2` | biased Gaussian-kernel MMD squared on log-CP10K |
| `pcc`, `scc` | Pearson and Spearman correlation of the 2,000-gene population mean vectors |
| `label_dist_whole` | the full label histogram of the delivery, so any purity ruler can be recomputed without re-sampling |

The purity columns of Tables 4 and 5 repeat Table 2. Their other columns are five evaluation
resamplings of fixed deliveries, so their spread is evaluation noise, not generation noise.
