# Stage 4：FLOWR structure reward-guided fine-tuning 训练接入报告

本阶段把基于 PLIF Tanimoto similarity、strain energy、Vina score 的 reward-guided fine-tuning 以**默认关闭**方式接入 FLOWR pocket training path。新增代码不改变默认训练、采样、evaluation 行为；只有显式传入 `--enable_rl_finetune` 且 `--rl_loss_weight > 0` 时才会执行 RL 链路。

## 1. 修改文件

新增：

```text
flowr/flowr/rl/training.py
migration_notes/09_stage4_structure_rl_training_integration_report.md
```

修改：

```text
flowr/flowr/train.py
flowr/flowr/models/fm_pocket.py
flowr/flowr/rl/structure_rewards.py
flowr/flowr/rl/structure_selection.py
flowr/flowr/rl/__init__.py
```

说明：

- `train.py` 只新增默认关闭 CLI/hparams；
- `fm_pocket.py` 只在 `LigandPocketCFM.training_step` 中加一个 gated hook；
- `structure_rewards.py` 新增 `compute_structure_rewards_from_records(...)`，方便训练时从内存 record 计算 reward；
- `structure_selection.py` 调整为：单目标不使用目标硬阈值过滤 top，多目标才使用硬门槛防补偿。

## 2. 新增配置参数

第一版只新增必要参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--enable_rl_finetune` | `false` | 基础开关；未开启时不执行 RL 链路。 |
| `--rl_loss_weight` | `0.0` | RL surrogate loss 权重；默认 0。 |
| `--rl_objective_mode` | `strain` | `plif/strain/vina/plif_strain/plif_vina/strain_vina/plif_strain_vina`。 |
| `--rl_multiobjective_strategy` | `constrained_weighted_sum` | 第一版只实现该策略；单目标忽略。 |
| `--rl_plif_weight` | `1.0` | 通过门槛后的排序/组合权重。 |
| `--rl_strain_weight` | `1.0` | 通过门槛后的排序/组合权重。 |
| `--rl_vina_weight` | `1.0` | 通过门槛后的排序/组合权重。 |
| `--rl_plif_min_threshold` | `0.3` | 多目标 top PLIF 门槛；单目标忽略。 |
| `--rl_strain_max_threshold` | `10.0` | 多目标 top strain 最大值；单目标忽略。 |
| `--rl_vina_max_threshold` | `-6.0` | 多目标 top Vina 最大值；单目标忽略。 |
| `--rl_strain_good_threshold` | `0.0` | strain 归一化 good threshold。 |
| `--rl_strain_bad_threshold` | `20.0` | strain 归一化 bad threshold。 |
| `--rl_vina_good_threshold` | `-10.0` | Vina 归一化 good threshold。 |
| `--rl_vina_bad_threshold` | `0.0` | Vina 归一化 bad threshold。 |
| `--rl_top_ratio` | `0.1` | top selection ratio。 |
| `--rl_bottom_ratio` | `0.1` | bottom selection ratio。 |
| `--rl_metric_source` | `compute` | `compute/cached/existing_eval_output`。 |
| `--rl_metric_cache_path` | `None` | JSON/JSONL reward cache 或 existing reward JSON 路径。 |

## 3. 默认关闭策略

默认情况下：

```text
enable_rl_finetune = false
rl_loss_weight = 0.0
```

`flowr.rl.training.rl_enabled(...)` 同时检查这两个条件。只有二者都满足时才执行 reward、selection 和 surrogate loss；否则 training step 仅返回原始 FLOWR loss。

## 4. FLOWR 原始 loss 位置

当前接入的是 pocket 模型：

```text
flowr/flowr/models/fm_pocket.py::LigandPocketCFM.training_step
flowr/flowr/models/fm_pocket.py::LigandPocketCFM._loss
```

原始训练逻辑为：

```python
losses = self._loss(lig_data, lig_interp, predicted, times=ligand_times)
loss = sum(list(losses.values()))
```

Stage 4 在这之后执行：

```python
loss, rl_logs = maybe_apply_rl_finetune_loss(self, loss, lig_data, predicted)
```

如果 RL 关闭，该函数直接返回原始 loss。

## 5. RL surrogate loss 设计

PLIF / Vina / PoseBusters / strain reward 不可微，因此不对 reward backward。第一版 surrogate 为：

```text
total_loss = original_flowr_loss + rl_loss_weight * rl_surrogate_loss
```

其中：

```text
rl_surrogate_loss = positive_top_loss + bounded_bottom_repulsion_loss
```

- positive top loss：top 样本对监督 ligand target 做 coordinate imitation；
- bottom repulsion：bottom 样本使用 bounded margin `relu(1 - per_sample_coord_mse)`，鼓励低 reward 样本不要过度贴近其监督 target；
- reward 只用于 selection，不进入计算图；
- RDKit / PLIF / Vina / PoseBusters 过程均在非可微路径中执行；
- 若 bottom 不足，negative branch 为 0，只保留 top positive branch；
- 若 top 不足，跳过 RL loss。

这是 FLOWR 架构下的最小可行 surrogate，不是逐行迁移 sotmol-rl 的 flow matching surrogate。

## 6. PLIF / strain / Vina reward 如何接入

`flowr/flowr/rl/training.py` 支持三种 metric source：

1. `compute`：训练 step 中从当前 model prediction 构造 RDKit mol，写 pocket PDB，调用 `compute_structure_rewards_from_records(...)`；
2. `cached`：先按 ligand MolBlock hash + system id + objective 查 JSONL cache，miss 时计算并回写；
3. `existing_eval_output`：从 `rl_metric_cache_path` 指向的 reward JSON 中循环读取 reward dict，适合 smoke test 或对齐好的离线候选。

训练中 compute 模式会：

1. 用 `predicted` tensor 构造候选 ligand；
2. 从 batch `complex` 取 native ligand 作为 reference ligand；
3. 将 pocket 写入 `save_dir/train_rl_ref_pdbs/`；
4. 生成 per-sample record；
5. 调用 Stage 1 reward wrapper 计算 PLIF / strain / Vina / validity / PoseBusters validity。

## 7. validity / PoseBusters gating

训练中沿用 Stage 1 / Stage 2 规则：

1. `valid == false`：main score 为 0，不能进 top；
2. `posebusters_valid == false`：main score 为 0，不能进 top；
3. 启用 metric failure：main score 为 0，不能进 top；
4. failed / invalid / PoseBusters failed 样本可以进入 bottom。

`RewardConfig` 在训练中固定使用 `require_posebusters_validity=True` 和 `compute_posebusters_validity=True`，不新增开关，满足本阶段固定规则。

## 8. fallback 规则

训练时任何 reward/metric 失败都不会让训练崩溃：

- 单个 molecule metric 失败：Stage 1 reward dict 中 success flag 为 false，`main_score=0`；
- 一个 batch 全部 metric 失败：selection 无 top，跳过 RL loss；
- reward 计算抛异常：捕获并跳过 RL loss；
- RL surrogate NaN：跳过 RL loss；
- 所有 skip 都记录 `train-rl-skip-count` 和 numeric `train-rl-skip-reason`。

## 9. 多目标非补偿 selection

训练中调用 Stage 2 `select_top_middle_bottom(...)`。

规则：

- 单目标：不使用 `rl_*_threshold` 硬过滤 top，只要求 valid、PoseBusters valid、启用 metric success，并按 normalized/main score 排序；
- 多目标：启用每个目标硬门槛，任一目标未达标则 `eligible_top=false`，不能进 top；
- 权重只影响通过门槛后的排序或 constrained weighted sum；
- invalid / PoseBusters failed / metric failed 仍可进入 bottom。

## 10. expensive metric 计算策略

第一版实现：

- cheap-first：Stage 1 reward wrapper 先检查 RDKit validity / PoseBusters validity，失败时不会作为 top；
- cached metrics：`rl_metric_source=cached` 可读写 JSONL cache，成功/失败都缓存；
- train-time fallback：异常或无 top 时跳过 RL loss。

第一版暂不新增 `every_n_steps`、warmup、max failure ratio 等复杂调度参数，避免参数膨胀。PLIF/Vina 低频调度建议后续阶段再加。

## 11. Cache 策略

`rl_metric_cache_path` 在 `cached` 模式下作为 JSONL cache：

```json
{"cache_key": "...", "reward": {...}}
```

cache key 包含：

- generated ligand MolBlock hash；
- system id；
- objective mode。

失败 reward 也会写入 cache，避免重复失败。

## 12. skip RL loss 的条件

RL loss 会在以下情况跳过：

1. `enable_rl_finetune=false`；
2. `rl_loss_weight <= 0`；
3. 没有 reward dict；
4. reward/metric 计算异常；
5. selection 后没有 top 样本；
6. `rl_surrogate_loss` 非 finite / NaN。

bottom 不足时不跳过；negative branch 置 0，只保留 top positive branch。

## 13. logging key

新增日志统一使用 `train-rl-*` 前缀：

```text
train-rl-enabled
train-rl-loss
train-rl-loss-weight
train-rl-objective-mode
train-rl-num-candidates
train-rl-num-valid
train-rl-num-posebusters-valid
train-rl-num-metric-success
train-rl-num-top
train-rl-num-bottom
train-rl-plif-mean
train-rl-strain-mean
train-rl-vina-mean
train-rl-main-score-mean
train-rl-main-score-max
train-rl-invalid-count
train-rl-posebusters-failed-count
train-rl-plif-failed-count
train-rl-strain-failed-count
train-rl-vina-failed-count
train-rl-skip-count
train-rl-skip-reason
train-rl-cache-hit-rate
train-rl-positive-loss
train-rl-negative-loss
```

注意：Lightning scalar logger 不能直接记录字符串，因此 `train-rl-objective-mode` 和 `train-rl-skip-reason` 使用 numeric id。

## 14. 最小训练命令

### 14.1 strain-only smoke test（相对便宜）

```bash
cd /data/bhli/Project/repo-rl/flowr
export PYTHONPATH="$PWD"

python -m flowr.train \
  --dataset spindr \
  --data_path /path/to/spindr \
  --save_dir /path/to/rl_strain_smoke \
  --arch pocket \
  --pocket_noise fix \
  --trial_run \
  --enable_rl_finetune \
  --rl_loss_weight 0.01 \
  --rl_objective_mode strain \
  --rl_metric_source cached \
  --rl_metric_cache_path /path/to/rl_strain_cache.jsonl \
  --rl_strain_good_threshold 0 \
  --rl_strain_bad_threshold 20 \
  --rl_top_ratio 0.1 \
  --rl_bottom_ratio 0.1
```

### 14.2 Vina-only smoke test（慢，建议小规模）

```bash
python -m flowr.train \
  --dataset spindr \
  --data_path /path/to/spindr \
  --save_dir /path/to/rl_vina_smoke \
  --arch pocket \
  --pocket_noise fix \
  --trial_run \
  --enable_rl_finetune \
  --rl_loss_weight 0.001 \
  --rl_objective_mode vina \
  --rl_metric_source cached \
  --rl_metric_cache_path /path/to/rl_vina_cache.jsonl \
  --rl_vina_good_threshold -10 \
  --rl_vina_bad_threshold 0 \
  --rl_top_ratio 0.1 \
  --rl_bottom_ratio 0.1
```

### 14.3 existing reward JSON smoke test

适合只验证训练 hook / logging / loss 组合，不重新计算 heavy metrics：

```bash
python -m flowr.train \
  --dataset spindr \
  --data_path /path/to/spindr \
  --save_dir /path/to/rl_existing_reward_smoke \
  --arch pocket \
  --pocket_noise fix \
  --trial_run \
  --enable_rl_finetune \
  --rl_loss_weight 0.01 \
  --rl_objective_mode strain_vina \
  --rl_metric_source existing_eval_output \
  --rl_metric_cache_path /path/to/reward_strain_vina.json \
  --rl_strain_max_threshold 10 \
  --rl_vina_max_threshold -6
```

## 15. 当前已验证内容

已验证：

1. 新增 Python 文件可 `py_compile`；
2. `enable_rl_finetune=false` + `rl_loss_weight=0` 时 `rl_enabled=false`；
3. 单目标 `strain` 下，hard threshold 不用于过滤 top；
4. selection 可以根据 normalized/main score 排序 top；
5. `git diff --check` 通过。

## 16. 当前未验证内容

未在当前环境验证：

1. 真实 FLOWR dataloader + Lightning 训练跑通；
2. train-time RDKit mol reconstruction 的真实成功率；
3. PoseBusters / PLIF / Vina 在训练 step 中的真实速度；
4. distributed training 下 cache 并发写入；
5. full RL fine-tuning 是否提升 evaluation 指标。

这些需要真实数据、GPU、ADFR/Vina、PoseBusters、GenBench3D 数据和完整 FLOWR 环境。

## 17. 与 sotmol-rl 原始实现的差异

| 项目 | sotmol-rl | FLOWR Stage 4 |
|---|---|---|
| 任务 | 无条件 3D molecule generation | protein pocket conditioned ligand generation |
| reward | MPO / property objective | PLIF / strain / Vina + validity/PoseBusters gating |
| selection | top/bottom reward partition | Stage 2 structure top/middle/bottom selection |
| surrogate | flow matching reward-guided surrogate，可包含 positive/negative 分支 | 最小 FLOWR surrogate：top imitation + bounded bottom repulsion |
| reference model | sotmol-rl 可用 current/reference 模型 | Stage 4 暂不引入 reference model |
| reward 频率 | 方法内管理 | Stage 4 先支持 compute/cached/existing，低频调度后续再加 |

## 18. 对 FLOWR 原始 baseline 是否有影响

默认情况下没有影响：

- 新增参数默认关闭；
- `rl_loss_weight=0.0`；
- `LigandPocketCFM.training_step` 中 gated helper 立即返回原始 loss；
- 原始 sampling / evaluation 脚本未修改；
- 原始 checkpoint monitor 和 logger 未修改。

因此原始 FLOWR baseline 训练、采样、evaluation 命令应保持不变。
