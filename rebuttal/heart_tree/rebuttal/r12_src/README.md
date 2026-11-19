# R12 CountsDiff, class-conditional, 2,000 genes

`countsdiff2k.py` trains and samples the conditional CountsDiff with the OURS attention trunk. It needs only numpy and
torch, and it does not read or change any frozen file. It writes only under `--out`.

Check first (about 2 minutes, CPU). It must print `resume test passed`:

    python test_resume.py DATA

Smoke test on the GPU, then look at the s/step and the peak memory (expected near OURS, about 0.44 s/step at batch 384):

    python countsdiff2k.py train --data DATA --out OUT_SMOKE --seed 20260625 --steps 40 --batch 384 --nmax NMAX --eval-every 20

Full training, one command per seed. All three seeds are reported; none is dropped without a note first. A stopped run
continues from `s<seed>.last.pt` when the same command is run again:

    python countsdiff2k.py train --data DATA --out OUT --seed 20260625 --nmax NMAX
    python countsdiff2k.py train --data DATA --out OUT --seed 20260626 --nmax NMAX
    python countsdiff2k.py train --data DATA --out OUT --seed 20260627 --nmax NMAX

On Dentate Gyrus (DG) use `--nmax 256` and the same recipe, for example:

    python countsdiff2k.py train --data DG_DATA --out OUT_DG --seed 20260625 --nmax 256

Sample the DG targets GlialProg, RadialGlia2, OPC and Granule (`--type`). The checkpoint stores NMAX, so `sample` needs no `--nmax`. The arrays go to Mac for scoring; all DG scoring is done there.

`NMAX` is 512 on HVG2K and 256 on Dentate Gyrus (the smallest value that meets the HVG2K rule on each dataset). Defaults
follow the authors' public scRNA configuration (`configs/scrna_heart.yaml`, `trainer.py`): Adam (betas 0.9, 0.999, wd 1e-4),
lr 2e-4 constant after a linear warmup of 1000 steps, loss averaged over genes, gradient clipping at 1.0, EMA 0.999,
p_uncond 0.2, sampling eta 0.005. Budget as OURS: 20,000 steps, batch 384, bf16 and gradient checkpointing on CUDA.

Note on the recipes: OURS trains for a fixed 20,000 steps (seed 20260630 for 19,000) with lr 2e-3 and no gradient clipping. This baseline uses the CountsDiff authors' recipe above with the same step budget. State this difference in any report.

Differences from the authors that stay, to be stated in any report: the OURS trunk (4 heads, dropout 0, two-channel input, MLP time embedding, against 8 heads, dropout 0.1, log1p input, sinusoidal time embedding); the label is added to every gene token, not a separate token; cell type is the only condition (the authors also use batch, gender, age); 20,000 steps at batch 384, against 50,000 at batch 512; K is chosen among 32, 128, 256, against 100.

Version 2 (11-07) changed p_uncond 0.1 to 0.2, eta 0 to 0.005, cosine decay to constant lr and AdamW to Adam, after a comparison with the authors' code. The earlier README called p_uncond 0.1 the authors' value, which was wrong. Nothing was trained for a report with version 1.

Divergence guard: the run stops with exit code 3 when val stays above 1.5 x best for 2,000 steps (the rule of the Blackout
runs) and writes `stopped_reason: divergence_guard` to `s<seed>.json`. The EMA weights it evaluates lag the raw weights, so
it reacts late. Report a stopped seed; do not rerun or drop it without a note.

Sampling, for one cell type:

    python countsdiff2k.py sample --ckpt OUT/s20260625.best.pt --type Neuronal --n 2500 --steps-k 128 --guidance 1.0 --seed 20261061 --out-npy F.npy

Choose K among 32, 128 and 256, and the guidance among 1.0, 1.5 and 2.0, on val only. If K=128 is clearly worse than
K=256, use K=256 and record the cost. NFE is K, doubled when guidance differs from 1.0; it is written to `F.json`.

Validation loss: the EMA weights lag the raw weights early in training, so a large train-val gap early on need not be
overfitting. `check_gap.py` scores both sets of weights on the same cells.
