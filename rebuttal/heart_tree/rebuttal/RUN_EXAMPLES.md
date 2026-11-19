# Example commands

One command per experiment arm, as run from the root of the experiment tree (`heart_tree/`, with the
frozen generator and data in place). Seeds used: 20260987, 20260988, 20260989.

## rebuttal_R9B_Neuronal_top32_matched

```bash
python -m rebuttal.run_capped --mem-gb 12 -- --experiment rebuttal_R9B --arm Neuronal_top32_matched --n-particles 5000 --role experiment --request Neuronal --seed 20260988 --sampler twist --K 32 --tau 0.2 --tilt-mode every --n-out 2500 --cell-chunk 256 --reward-mode odds --reward-model residual --alpha 1.0 --J 16 --trigger any --ess-frac 0.5 --resample-interval 8 --residual-dir <HVG_ROOT>/rebuttal/out/R9B/disc/Neuronal_top32 --logpi <HVG_ROOT>/rebuttal/tables/logc_Neuronal_top32.npy
```

## rebuttal_R9B_Neuronal_k128

```bash
python -m rebuttal.run_capped --mem-gb 12 -- --experiment rebuttal_R9B --arm Neuronal_k128 --n-particles 5000 --role experiment --request Neuronal --seed 20260988 --sampler twist --K 32 --tau 0.2 --tilt-mode every --n-out 2500 --cell-chunk 256 --reward-mode odds --reward-model residual --alpha 1.0 --J 16 --trigger any --ess-frac 0.5 --resample-interval 8 --residual-dir <HVG_ROOT>/rebuttal/out/R9B/disc/Neuronal_top128 --logpi <HVG_ROOT>/rebuttal/tables/logc_Neuronal_top128.npy
```

## rebuttal_R10_Neuronal_frac010_fk

```bash
python -m rebuttal.run_capped --mem-gb 12 -- --experiment rebuttal_R10 --arm Neuronal_frac010_fk --n-particles 5000 --role experiment --request Neuronal --seed 20260988 --sampler twist --K 32 --tau 0.2 --tilt-mode every --n-out 2500 --cell-chunk 256 --reward-mode odds --reward-model residual --alpha 1.0 --J 16 --trigger any --ess-frac 0.5 --resample-interval 8 --residual-dir <HVG_ROOT>/rebuttal/out/R10/disc/Neuronal_frac010 --logpi <HVG_ROOT>/rebuttal/out/R10/logc_Neuronal_frac010.npy
```

## clip_rebuttal_R4clip_Neuronal_clip_train20260627

```bash
python -m rebuttal.run_clip --U 22.657017784245227 -- --experiment rebuttal_R4clip --arm Neuronal_clip_train20260627 --n-particles 5000 --role experiment --request Neuronal --seed 20260987 --sampler twist --K 32 --tau 0.2 --tilt-mode every --n-out 2500 --cell-chunk 256 --reward-mode odds --reward-model residual --alpha 1.0 --J 16 --trigger any --ess-frac 0.5 --resample-interval 8 --ckpt <HVG_ROOT>/assets/legacy_root/Baseline/OURS/models/pilot_run214_s20260627.pt --logpi <HVG_ROOT>/assets/tables/tables_kz_s20260627/logc_Neuronal.npy --residual-dir <HVG_ROOT>/assets/models/residual/Neuronal_s20260627
```

## clip_rebuttal_R4clip_Neuronal_clip_train20260630

```bash
python -m rebuttal.run_clip --U 22.657017784245227 -- --experiment rebuttal_R4clip --arm Neuronal_clip_train20260630 --n-particles 5000 --role experiment --request Neuronal --seed 20260989 --sampler twist --K 32 --tau 0.2 --tilt-mode every --n-out 2500 --cell-chunk 256 --reward-mode odds --reward-model residual --alpha 1.0 --J 16 --trigger any --ess-frac 0.5 --resample-interval 8 --ckpt <HVG_ROOT>/assets/legacy_root/Baseline/OURS/models/pilot_run214_s20260630.pt --logpi <HVG_ROOT>/assets/tables/tables_kz_s20260630/logc_Neuronal.npy --residual-dir <HVG_ROOT>/assets/models/residual/Neuronal_s20260630
```

