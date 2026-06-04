# 06. Stage 1 — FLOWR structure reward/objective layer report

> 范围：本阶段在 `flowr/` 中新增 default-off 的 structure reward/objective wrapper。该层不接入训练 loop，不修改 FLOWR 原始 training、sampling、evaluation 入口；后续 RL / reward-guided fine-tuning 可显式 import 该层。

## A. `eval_spindr.sh` 调用链

FLOWR 原始 evaluation shell 脚本是：

```text
flowr/scripts/eval_spindr.sh
```

调用链：

```text
scripts/eval_spindr.sh
  ├─ python -m flowr.eval.evaluate_metrics
  │    ├─ gather_predictions(...) / gather_predictions_pilot(...)
  │    ├─ evaluate_validity(...)
  │    ├─ evaluate_mol_metrics(...)
  │    ├─ evaluate_gb3_validity(...)
  │    ├─ evaluate_gbsb3(...)
  │    ├─ evaluate_pb_validity(...)
  │    ├─ evaluate_posecheck(...)
  │    └─ compute_interaction_recovery_parallel(...)
  └─ python -m flowr.eval.evaluate_interactions
       ├─ gather_predictions(...)
       └─ compute_interaction_recovery_parallel(...)
            └─ evaluate_interaction_recovery(...)
                 └─ interaction_recovery_per_complex(...)
                      └─ _get_plif_recovery(...)
```

原始脚本输入：

- `--data_path`: SPINDR data directory；evaluation 需要 `train_mols.pkl` 和 processed statistics。
- `--save_dir`: FLOWR sampling output directory。
- `--multiple_files`: 读取 `predictions_multi_*.pt`。
- `--remove_hs`: evaluation 中传给 PLIF/interaction 处理。

原始脚本输出：

- `metrics.pt` 或 `metrics_valid_unique.pt`。
- `interaction_recovery_list.pt` / `interaction_recovery.pt`，以及 valid_unique variants。

## B. PLIF、strain、Vina、validity 的原始代码位置

| Metric | 方向 | 原始代码位置 | Stage 1 wrapper 复用方式 |
|---|---|---|---|
| PLIF Tanimoto similarity | 越高越好 | `flowr/flowr/eval/evaluate_interactions.py::compute_interaction_recovery_parallel()` 聚合 `PLIF Tanimoto similarity`；底层为 `flowr/flowr/util/metrics.py::evaluate_interaction_recovery()`, `interaction_recovery_per_complex()`, `_get_plif_recovery()` | `flowr.rl.structure_rewards._compute_plif_tanimoto()` 调用 `interaction_recovery_per_complex()`，避免 shell 脚本和 multiprocessing 聚合。 |
| Strain energy | 越低越好 | `flowr/flowr/util/metrics.py::evaluate_strain()`；`evaluate_metrics.py` 也通过 `calc_gb3_metrics_parallel()` / `evaluate_gb3_validity()` 聚合 GenBench3D strain | `flowr.rl.structure_rewards._compute_strain_energy()` 调用 `evaluate_strain(..., return_list=True)`。 |
| Vina score | 越低越好 | `flowr/flowr/util/metrics.py::evaluate_gbsb3()` 中 setup `VinaProtein`/`SBGenBench3D`/`setup_vina()`，并在 GenBench3D `VinaScore` 里计算 | `flowr.rl.structure_rewards._compute_vina_score()` 调用 `evaluate_gbsb3(..., return_dict=True)` 并提取 `Vina score` 或 `Minimized Vina score`。 |
| RDKit validity | true/false | `flowr/flowr/util/rdkit.py::mol_is_valid()`；聚合函数在 `flowr/flowr/util/metrics.py::evaluate_validity()` | `flowr.rl.structure_rewards._compute_validity()` 调用 `mol_is_valid(..., connected=True)`。 |
| PoseBusters validity | true/false | `flowr/flowr/util/metrics.py::evaluate_pb_validity()` 调用 `PoseBusters(config="dock")` | `flowr.rl.structure_rewards._compute_posebusters_validity()` 调用 `evaluate_pb_validity(..., return_list=True)`。 |

## C. 新增 reward wrapper 文件

本阶段新增：

```text
flowr/flowr/rl/__init__.py
flowr/flowr/rl/structure_rewards.py
flowr/scripts/test_structure_rewards.py
migration_notes/06_stage1_structure_reward_layer_report.md
```

这些文件均为 default-off：FLOWR 原始 `train.py`、`generate_from_smol.py`、`generate_from_pdb.py`、`evaluate_metrics.py`、`evaluate_interactions.py` 不会自动 import 或调用它们。

## D. `compute_structure_rewards(...)` 输入输出

新增 API：

```python
from flowr.rl.structure_rewards import RewardConfig, compute_structure_rewards

rewards = compute_structure_rewards(
    generated_mols=[...],
    protein_file="pocket_or_protein.pdb",
    reference_ligand=ref_mol,
    config=RewardConfig(objective_mode="plif_strain_vina"),
)
```

也支持：

- `generated_ligand_file=<SDF>`。
- `reference_ligand_file=<SDF>`。
- `flowr_sampling_output=<predictions_multi_*.pt 或 samples_*.pt>`。

输出为 list of reward dict，每个 dict 采用 Stage 1 统一 schema。

## E. Reward dict 格式

每个 sample 输出：

```python
{
  "sample_id": str,
  "valid": bool,
  "posebusters_valid": bool,
  "plif_tanimoto": float | None,
  "strain_energy": float | None,
  "vina_score": float | None,
  "plif_success": bool,
  "strain_success": bool,
  "vina_success": bool,
  "metric_success": bool,
  "feasible": bool,
  "eligible_top": bool,
  "eligible_bottom": bool,
  "main_score": float,
  "error": str | None,
  "warning": str | None,
  "normalized_plif": float | None,
  "normalized_strain_score": float | None,
  "normalized_vina_score": float | None,
  "objective_mode": str,
  "multiobjective_strategy": str,
  "metadata": dict,
}
```

原始 metric 值始终保留：`plif_tanimoto`、`strain_energy`、`vina_score` 不会被 normalized score 替代。

## F. Metric 方向与 normalization 规则

所有内部 `main_score` 都转换为越高越好。

1. PLIF Tanimoto similarity：
   - 原始方向：越高越好。
   - normalization: `normalized_plif = clamp(plif_tanimoto, 0, 1)`。

2. Strain energy：
   - 原始方向：越低越好。
   - normalization: 通过 `strain_good_threshold` / `strain_bad_threshold` 配置线性转换。
   - `strain <= strain_good_threshold` 得 1。
   - `strain >= strain_bad_threshold` 得 0。
   - 中间线性插值。

3. Vina score：
   - 原始方向：越低越好。
   - normalization: 通过 `vina_good_threshold` / `vina_bad_threshold` 配置线性转换。
   - `vina <= vina_good_threshold` 得 1。
   - `vina >= vina_bad_threshold` 得 0。
   - 中间线性插值。

阈值不在 reward wrapper 内写死，必须由实验配置传入；测试脚本只为 smoke test 提供示例默认值。

## G. Fallback 规则

已实现 fallback：

1. `valid == false`：
   - `feasible=false`
   - `eligible_top=false`
   - `eligible_bottom=true`
   - `main_score=0`
   - `error="invalid molecule"`

2. `posebusters_valid == false` 且 `require_posebusters_validity=true`：
   - `feasible=false`
   - `eligible_top=false`
   - `eligible_bottom=true`
   - `main_score=0`

3. 启用 metric 失败：
   - 对应 `*_success=false`
   - `feasible=false`
   - `eligible_top=false`
   - `eligible_bottom=true`
   - `main_score=0`
   - `error` 记录异常类型和消息。

4. 未启用 metric 失败：
   - 默认不计算；若 `compute_non_enabled_metrics=true` 时失败，只记录到 `warning`，不影响 `main_score`。

## H. 多目标非补偿策略

新增配置 `multiobjective_strategy`，支持：

1. `hard_gate_then_weighted_sum`
   - 每个启用目标必须先通过单项 threshold。
   - 全部通过后才 weighted sum。
   - 任一目标不过门槛，`eligible_top=false`。

2. `weighted_min`
   - `main_score = min_i(weight_i * normalized_i)`。
   - 强调短板，防止 PLIF 很高完全补偿 Vina/strain 很差。

3. `constrained_weighted_sum`
   - 默认推荐。
   - 每个目标先过最低门槛；满足后 weighted sum。
   - 不满足门槛则 `main_score=0` 且不能进 top。

## I. 新增配置参数

`RewardConfig` 支持：

| 参数 | 作用 |
|---|---|
| `objective_mode` | `plif`, `strain`, `vina`, `plif_strain`, `plif_vina`, `strain_vina`, `plif_strain_vina`。 |
| `multiobjective_strategy` | `hard_gate_then_weighted_sum`, `weighted_min`, `constrained_weighted_sum`。 |
| `plif_weight` | PLIF normalized score 权重。 |
| `strain_weight` | Strain normalized score 权重。 |
| `vina_weight` | Vina normalized score 权重。 |
| `plif_min_threshold` | PLIF top eligibility 最低门槛。 |
| `strain_max_threshold` | Strain raw value 最大门槛，越低越好。 |
| `vina_max_threshold` | Vina raw value 最大门槛，越低越好。 |
| `strain_good_threshold` | Strain normalization good threshold。 |
| `strain_bad_threshold` | Strain normalization bad threshold。 |
| `vina_good_threshold` | Vina normalization good threshold。 |
| `vina_bad_threshold` | Vina normalization bad threshold。 |
| `posebusters_validity_threshold` | PoseBusters pass 阈值，默认 1.0。 |
| `require_posebusters_validity` | 若 true，PB validity failed 的 sample 不能进 top。 |
| `compute_posebusters_validity` | 是否计算 PB validity。 |
| `compute_non_enabled_metrics` | 是否额外计算非启用 metric 作为辅助统计。 |
| `config_path` | GenBench3D/FLOWR evaluation config path。 |
| `use_minimized_vina_score` | 是否使用 `Minimized Vina score`。 |
| `strain_force_field_name` | `MMFF94s`, `MMFF94`, `UFF`。 |
| `strain_n_steps` | Strain minimization steps。 |

## J. 最小测试命令

Mock normalization / fallback / multi-objective smoke：

```bash
cd flowr
export PYTHONPATH="$PWD"
python scripts/test_structure_rewards.py \
  --mock \
  --objective_mode plif_strain_vina \
  --multiobjective_strategy constrained_weighted_sum \
  --plif_min_threshold 0.3 \
  --strain_good_threshold 0 \
  --strain_bad_threshold 20 \
  --strain_max_threshold 10 \
  --vina_good_threshold -10 \
  --vina_bad_threshold 0 \
  --vina_max_threshold -4
```

真实 FLOWR output 示例：

```bash
cd flowr
export PYTHONPATH="$PWD"
python scripts/test_structure_rewards.py \
  --input /ABS/PATH/TO/predictions_multi_1.pt \
  --objective_mode plif_strain_vina \
  --multiobjective_strategy constrained_weighted_sum \
  --plif_min_threshold 0.3 \
  --strain_good_threshold 0 \
  --strain_bad_threshold 20 \
  --strain_max_threshold 10 \
  --vina_good_threshold -10 \
  --vina_bad_threshold 0 \
  --vina_max_threshold -4
```

SDF + protein + reference ligand 示例：

```bash
cd flowr
export PYTHONPATH="$PWD"
python scripts/test_structure_rewards.py \
  --input /ABS/PATH/TO/generated.sdf \
  --protein /ABS/PATH/TO/pocket_or_protein.pdb \
  --reference_ligand /ABS/PATH/TO/reference_ligand.sdf \
  --objective_mode plif_strain_vina \
  --multiobjective_strategy constrained_weighted_sum
```

## K. 当前能复用的 FLOWR 原始代码

可复用：

- `flowr.util.rdkit.mol_is_valid` for validity。
- `flowr.util.metrics.evaluate_pb_validity` for PoseBusters validity。
- `flowr.util.metrics.interaction_recovery_per_complex` for PLIF Tanimoto。
- `flowr.util.metrics.evaluate_strain` for strain energy。
- `flowr.util.metrics.evaluate_gbsb3` for Vina score。
- `flowr.eval.evaluate_util.gather_predictions` 的 output schema；Stage 1 wrapper 直接读取 `predictions_multi_*.pt` 中的 `gen_ligs`, `ref_ligs`, `ref_pdbs`，并优先用 `ref_ligs_with_hs` / `ref_pdbs_with_hs` 计算 PLIF，以匹配 `evaluate_interactions.py` 的原始做法。

不直接复用 shell：

- `eval_spindr.sh` 是 coarse-grained batch evaluation，不适合训练中高频调用。
- `compute_interaction_recovery_parallel` 和 `calc_gb3_metrics_parallel` 是 multiprocessing aggregation，不适合 future online reward step 逐 batch 高频调用。

## L. 慢 metric、计算代价和失败风险

| Metric | 代价 | 失败风险 | Stage 1 fallback |
|---|---|---|---|
| PLIF Tanimoto | 中到高；需要 protein-ligand complex、ProLIF fingerprint、可能优化 H | 复合物处理失败、H/charge/bond perception 失败、pocket mol 失败 | 该 sample `main_score=0`, `eligible_top=false`, `eligible_bottom=true`, `error` 记录异常。 |
| Strain energy | 中；RDKit force-field minimization | MMFF/UFF 参数缺失、构象无效、不收敛、energy NaN | 该 sample `main_score=0`, 不进 top。 |
| Vina score | 高；需要 receptor preparation/Vina maps/scoring | ADFR path 错误、Vina 安装缺失、PDBQT 转换失败、box 设置失败、速度慢 | 该 sample `main_score=0`, 不进 top。 |
| PoseBusters validity | 中到高；需要 dock config 和 protein context | protein/ligand parse 失败、PB checks 失败 | 若 `require_posebusters_validity=true`，sample 不进 top。 |

## M. 后续训练阶段缓存建议

后续接入 RL / reward-guided fine-tuning 时建议：

1. 以 `(target_id, canonical_smiles, conformer_hash, metric_name, config_hash)` 做缓存 key。
2. PLIF 缓存 native/reference PLIF fingerprint；`_get_plif_recovery()` 已有 per-complex native PLIF pkl 思路，可扩展为训练缓存。
3. Vina 缓存 receptor preparation、Vina maps、PDBQT ligand string 和 final score。
4. Strain 缓存 by conformer hash；同一 ligand graph 但不同 3D conformer 不应共用 strain。
5. 将 PLIF/Vina 作为 low-frequency 或 offline scoring 优先，不建议每个 training step 对大量样本同步计算。
6. 所有 cache 命中/失败都写入 reward dict metadata，便于后续 top/bottom selection debug。
