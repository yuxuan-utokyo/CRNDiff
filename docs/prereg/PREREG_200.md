# PREREG_200 —— 工单 200 预注册（任何运行之前落盘；落盘后本文件只增不改，修订用 AMENDMENT 文件）

来源：`Paper\adversarial\comms\cloud_to_cc\200_external_baselines_two_machines.md` §1.4、§1.5、§7 原文抄录（下方），加「派生常数」节由 cc 在 M1-0 填写。
落盘时间与 sha256 由 cc 写进 `200_receipt_machine1.md` 第一节。

---

### 1.4 NFE 口径与预算（compute-matched 的定义）

沿用 `Conditional generation experiments\HVG2K\universality_fk\out\e29\E29_nfe_definition.json`：**NFE = 生成器网络前向调用次数**。
判别器 / 分类器 / J 候选抽样的代价**另计**，每条 run 的 json 记 `aux_calls = {model, n_calls, device}` 与 `walltime_s`，
预算曲线**双横轴**（NFE 与 1 号机墙钟）。

- tilted FK 部署臂：`NFE_fk(型) = n_particles × NFE_per_particle`，两者都从部署 run json 读
  （`out/summary/e21_table1_ours.json::types.<型>.rows.<method>.source` 指向的 run json 的 `n_particles`、`NFE` 字段；
  183 的做法）。不许假设 32 / 36 / 63。**注**：`NFE_per_particle` 若含检查点前向拆分的记账（36 而非 32），会抬高 B₁，
  对我方**不利**，照用并在回执写一句。
- 无条件链：`NFE_chain` 从无条件 run 的 sidecar json 读（应为 K=32）。
- **基础预算** `B₁(型) = floor(NFE_fk(型) / NFE_chain)`。
- 预算阶梯 `b ∈ {1, 2, 4, 8, 16}`，第 b 档 = 该 seed 池中**按生成顺序的前 b·B₁ 条**（嵌套，不重新生成）。
- **池上限** `N_pool ≤ 160,000`（硬上限，见 M1-1）。`b·B₁ > N_pool` 的档记 `not_run_budget_cap`，**不许静默截断**。
- **解析天花板（必须先算、写进 PREREG 派生常数节）**：forced-top-N 的 purity 上界
  `ceil(型, b) = min(1, b·B₁(型)·π̂_pool(型) / N(型))`，其中 `π̂_pool` = 该 seed 池按 CellTypist 打出的该型比例（诊断用，不是筛选）。
  它说明 b=1 的胜负在稀有型上由算术决定。**右删失**：`π̂ < N/(16·B₁)` 的型，追平倍数报「≥16×」，不写「永不」。
- 一个无条件池服务三个请求（分摊有利于 best-of-N；FK 每个请求单独跑），在论文里明说。


### 1.5 种子（开跑前登记进 `PREREG_200.md`，不许增删、不许挑）

| 用途 | 种子 |
|---|---|
| 我方无条件池 / 基线无条件池 / 外部生成器采样 | **20260931, 20260932, 20260933** |
| 无条件采样器复现闸门 G-U | 20260902（= `BASE_POOL` 的采样种子） |
| 价值引导采样 | 预期 **20260987, 20260988, 20260989**（= 部署臂 Table 1 前三个种子）；**以 M1-0 第 5 条从 `e21_table1_ours.json` 读出的为准** |
| S1 / 噪声态分类器拟合 | 20260817 |
| 噪声态分类器的加噪流 | 20260818（`torch.Generator`，与拟合种子分开） |
| 无条件保真度的 val 子样本（n=20000） | **20261200** |
| toy best-of-N | 20260901–20260905（toy 现有五个） |


---

## 7 预注册判据（`PREREG_200.md` 落盘即冻结）

**主张 S（表 A，我方生成器上）**

- **P-A1（一致性检查，不是主判）**：b=1 时 `purity(部署臂)` 对 `purity(best-of-N forced, S2)` 3-vs-3 全排列，报差值原值；同时打印 `ceil(型,1)`。
  **预期在 Myeloid / Neuronal 上由算术决定**；在 Endothelial 上可能不成立（先例：丰度型上 untilted 臂距离指标更好）——不成立照报。
- **P-A2（主判之一）**：best-of-N forced（S2）的 **3-seed 均值 purity** 首次 ≥ 部署臂 **3-seed 均值 purity** 的最小 b，每型一个数，旁注该档 3-vs-3 九对里 ≥ 的对数；右删失写「≥16×」或「≥ 上限档」。
- **P-A3（主判之一）**：generate-until-N 的 `NFE_needed / NFE_fk`，每型 3 seed 原值。**不预注册它对稀有度单调**（勘误 14）。
- **P-A4（诚实项）**：b=16（或最高可跑档）best-of-N 的 sliced-W1 / MMD² / PCC 对部署臂，照报；预期 best-of-N 不差。
- **P-A5（tilt + 终点筛选）**：b=1 下该臂与部署臂的 purity、sliced-W1、`unique_cell_fraction` 3-vs-3；这是「重采样有没有用」的直接读数，无方向预言。
- **P-A6（VGR）**：两种打分器各取三档 γ 中 **3-seed 均值 purity 最高**的那档（选择规则只看 purity，先于数字写死；对基线宽松），
  在该档报 purity 与 sliced-W1 与部署臂的 3-vs-3 差值；三档全部进附录；报 NFE 差异；无预言。
  VGR-noised 与 VGR-clean 的差即「打分对象在噪声态还是干净态」的效应，单列。
- **P-A7**：S1 与 S2 的 best-of-N 结果差即「选择器质量」的效应，单列；论文只用 S2 行作主对比，S1 行进附录。

**主张 G（表 B）**

- **P-B1**：无条件保真度各指标排名照报，无预言；训练预算、NFE/sample、库大小来源、计数合法性四列**必须**在同一张表里。
- **P-B2**：同一 best-of-N（S2）、C=25000 与期望目标数配平两种口径，各生成器三型 purity 排名照报，标 `descriptive`。
- **B-F1** 对称适用；不过者标 `generator_pathology`。

**Toy**

- **P-T1**：b=1 下 tilted FK 的 ED 对 oracle-SIR 与 learned 两个 best-of-N（五 seed 原值）。对 oracle-SIR **预期不占优**（它是 ED 上界），照报；
  对 learned 无预言。oracle top-N 只作 purity 上界参照。

**事先写好的措辞（§11 用）**：若 P-A2 在丰度型上 ≤2× —— "on abundant requests a matched-budget rejection sampler is competitive; the advantage of tilted FK is concentrated on rare requests, where rejection requires ≥k× the budget"。若 Neuronal 也 ≤4× —— 论文的稀有优势句**降级为成本句**，不再写 "particularly effective for rare requests"。

---


---

## 派生常数（cc 在 M1-0 第 5–6 条填写；每个数旁写读自哪个文件哪个字段）

| 量 | Endothelial | Myeloid | Neuronal | 来源文件 / 字段 |
|---|---|---|---|---|
| 部署臂 run json 路径 | | | | `e21_table1_ours.json::types.<型>.rows.<key>.source` |
| `n_out` = N(型) | | | | |
| `n_particles` = M(型) | | | | |
| `NFE`（每粒子）| | | | |
| `NFE_fk(型) = M × NFE` | | | | |
| `tau`（断言 0.4 / 0.2 / 0.2）| | | | |
| 部署臂实际种子（升序前三 → VGR 种子）| | | | |
| `NFE_chain`（G-U 的 sidecar json）| | | | |
| `B₁(型) = floor(NFE_fk / NFE_chain)` | | | | |
| `NFE_per_chain_tilt`（M1-1b smoke）| | | | |
| `T_b(型)`，b = 1 / 2 / 4 | | | | |
| smoke（n=2000）墙钟 → 外推 16× 池墙钟 | | | | |
| **N_pool（定后不改）** | | | | |
| `not_run_budget_cap` 的档 | | | | |
| `π̂_pool(型)`（seed 31 池，CellTypist 诊断）| | | | 填于池生成后，但 `ceil(型,b)` 公式在此已冻结 |
| `ceil(型, b)`，b = 1 … 16 | | | | |

先验测量披露（写于任何运行之前）：我方部署生成器无条件池 2026-09-05 cloud 测得库大小均值 311（真实 val 375）、基因均值 Pearson 0.997。B-F1 阈值（±20%，log 尺度 Pearson ≥ 0.99）**不因此调整**，且对所有生成器对称适用。
