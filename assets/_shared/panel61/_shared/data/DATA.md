# _shared/data —— 唯一一份锁定数据

- **gene_hash** `47ef02be9e74e8a84cc3bf67ae5b4cbefba53553ce5cb05f0f7f74cb8c1c7750`
- **panel**：Heart Cell Atlas · Endothelial · med_band · **61 基因**
- `train.npy` (80463, 61) int64 · `val.npy` (10057, 61) int64 · `meta.json`（含 `Vg`）
- **来源**：`cc/experiments/EXP35_generalization/data/`（逐元素验证一致，见 P0 闸门输出）
- **test 不在此目录，也不在 `CRNDIFF/baseline/` 任何位置。** 本轮一次未碰 test。
- 所有派生统计（`Vg` / `NMAX=128` / `iscale` / `dmax` / early-stop 判据）**只从 train 估**；
  `val` 只用于评测与模型选择。
