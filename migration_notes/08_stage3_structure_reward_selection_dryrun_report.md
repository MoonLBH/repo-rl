# Stage 3：FLOWR structure reward → top/middle/bottom dry-run 报告

本阶段新增一个不训练、不 backward 的 dry-run 链路：

```text
FLOWR 原始采样输出
→ existing evaluation output 或 Python metric wrapper
→ structure reward dict
→ top/middle/bottom selection
→ CSV / JSON summary / optional SDF / failed log
```

新增脚本保持默认关闭，不修改 FLOWR 原始 sampling、training、evaluation 入口。

## 1. Dry-run 脚本路径

```text
flowr/scripts/structure_reward_selection_dryrun.py
```

核心命令形式：

```bash
PYTHONPATH="$PWD" python scripts/structure_reward_selection_dryrun.py \
  --generated_dir <flowr_sampling_output_dir_or_predictions_pt> \
  --eval_output <existing_reward_json_or_eval_dir> \
  --metric_source existing_eval_output \
  --output_csv <path.csv> \
  --output_json <summary.json> \
  --objective_mode strain_vina \
  --multiobjective_strategy constrained_weighted_sum \
  --strain_max_threshold 10 \
  --vina_max_threshold -6 \
  --top_ratio 0.1 \
  --bottom_ratio 0.1
```

## 2. 支持的输入格式

### 2.1 `--generated_dir`

支持：

1. FLOWR sampling 输出目录，脚本会查找：
   - `predictions_multi_*.pt`；
   - 如果没有 multi 文件，则查找 `predictions.pt`。
2. 单个 `.pt` 文件，例如：
   - `predictions_multi_1.pt`；
   - `predictions.pt`。

### 2.2 `--eval_output`

`--metric_source existing_eval_output` 时支持：

1. per-sample reward JSON list；
2. 包含 `rewards` / `reward_dicts` / `records` / `per_sample` 字段的 JSON dict；
3. JSONL，每行一个 reward dict；
4. 一个目录，脚本会依次查找：
   - `reward_<objective_mode>.json`；
   - `reward_record/reward_<objective_mode>.json`；
   - `structure_rewards.json`；
   - `rewards.json`。

注意：原始 `eval_spindr.sh` 的 `metrics.pt` 多数是 aggregated metrics，不足以直接恢复每个 sample 的 strain / Vina reward；因此如果要避免重复算 heavy metrics，推荐先用 Stage 1 reward wrapper 生成 per-sample `reward_*.json`，再用本 dry-run 脚本做 selection 和导出。

## 3. 支持的 metric source

### 3.1 `existing_eval_output`

适合 metric 已经算好的情况：

```bash
--metric_source existing_eval_output \
--eval_output /path/to/reward_strain_vina.json
```

该模式不会重复计算 PLIF / strain / Vina / PoseBusters，只读取已有 per-sample reward dict。

### 3.2 `compute`

适合小规模测试或没有现成 reward JSON 的情况：

```bash
--metric_source compute \
--generated_dir /path/to/eval_dir_or_predictions_multi_1.pt
```

该模式调用 `flowr.rl.structure_rewards.compute_structure_rewards(...)`，会实际计算启用目标所需 metric；PLIF、Vina、PoseBusters 可能很慢。

## 4. `eval_spindr.sh` 输出字段映射

`eval_spindr.sh` 调用两个原始 evaluation 脚本：

| 原始脚本 | 输出文件 | 字段 | 是否 per-sample |
|---|---|---|---|
| `flowr.eval.evaluate_metrics` | `metrics.pt` | validity、molecular metrics、GenBench3D SBDD summary、Vina summary、PoseBusters summary、strain summary 等 | 多数为 aggregated，不足以直接按 sample selection |
| `flowr.eval.evaluate_interactions --return_interaction_list` | `interaction_recovery_list.pt` | `PLIF recovery rate`, `PLIF Tanimoto similarity` | 按 target 聚合 list，通常可按目标内生成顺序对应，但仍需谨慎校验 |

关键字段概念映射：

| dry-run / reward 字段 | 来源 |
|---|---|
| `valid` | Stage 1 wrapper 调用 FLOWR RDKit validity；原始 `metrics.pt` 只提供聚合 rate |
| `posebusters_valid` | Stage 1 wrapper 调用 `evaluate_pb_validity(..., return_list=True)`；原始 `metrics.pt` 只提供聚合值 |
| `plif_tanimoto` | Stage 1 wrapper 调用 PLIF API；原始 `interaction_recovery_list.pt` 有 target-level list |
| `strain_energy` | Stage 1 wrapper 调用 `evaluate_strain(..., return_list=True)`；原始 `metrics.pt` 聚合不够 per-sample |
| `vina_score` | Stage 1 wrapper 调用 `evaluate_gbsb3(..., return_dict=True)`；原始 `metrics.pt` 聚合不够 per-sample |

因此 Stage 3 脚本的 `existing_eval_output` 最推荐读取 Stage 1 生成的 per-sample `reward_*.json`，而不是直接读取原始 `metrics.pt`。

## 5. Reward dict 生成流程

### compute 模式

1. 读取 `predictions_multi_*.pt` 或 `predictions.pt`；
2. 对每个文件调用 `compute_structure_rewards(...)`；
3. 每个 molecule 生成一个 structure reward dict；
4. 写入 cache（如果提供 `--cache_jsonl`）；
5. 进入 selection。

### existing_eval_output 模式

1. 读取已有 per-sample reward JSON / JSONL；
2. 不重复计算 heavy metrics；
3. 如果 `generated_dir` 可解析，也会尝试附加 RDKit mol，用于 optional SDF 输出；
4. 进入 selection。

## 6. Selection 流程

Dry-run 调用 Stage 2：

```python
select_top_middle_bottom(
    reward_dicts,
    objective_mode=args.objective_mode,
    multiobjective_strategy=args.multiobjective_strategy,
    top_ratio=args.top_ratio,
    bottom_ratio=args.bottom_ratio,
    top_k=args.top_k,
    bottom_k=args.bottom_k,
    feasible_only_for_top=args.feasible_only_for_top,
    include_failed_in_bottom=args.include_failed_in_bottom,
    plif_min_threshold=args.plif_min_threshold,
    strain_max_threshold=args.strain_max_threshold,
    vina_max_threshold=args.vina_max_threshold,
)
```

Top 仍然必须满足：valid、PoseBusters valid、enabled metric success、feasible、eligible_top、threshold pass。Bottom 默认优先接收 invalid / PoseBusters failed / metric failed / infeasible / threshold failed 样本，然后用低分样本补齐。

## 7. Cache 设计

新增参数：

```bash
--cache_jsonl /path/to/structure_reward_cache.jsonl
```

当前 cache 设计：

- 文件格式：JSONL；
- 每行结构：`{"cache_key": ..., "reward": {...}}`；
- key 组成：
  - `sample_id`；
  - predictions 文件路径、大小、mtime 的 fingerprint；
  - `objective_mode`；
  - `multiobjective_strategy`；
  - PLIF / strain / Vina selection thresholds。

设计原则：

1. 成功 reward 和失败 reward 都可缓存；
2. 失败结果也保存，避免重复失败；
3. cache 用于 dry-run 和后续训练低频 reward 计算；
4. 当前 wrapper 仍以文件/批为单位计算，若某文件有部分 miss，可能需要重新计算该文件；后续可进一步改成 per-sample lazy compute。

## 8. CSV 输出格式

`--output_csv` 至少包含以下列：

```text
sample_id
protein_id
pocket_id
ligand_file
smiles
valid
posebusters_valid
plif_tanimoto
strain_energy
vina_score
plif_success
strain_success
vina_success
metric_success
feasible
eligible_top
eligible_bottom
normalized_plif
normalized_strain
normalized_vina
main_score
selection_label
failure_reason
error
```

其中：

- `selection_label` 为 `top` / `middle` / `bottom`；
- `failure_reason` 来自 selection summary，例如：
  - `invalid`；
  - `posebusters_failed`；
  - `plif_failed`；
  - `strain_failed`；
  - `vina_failed`；
  - `strain_above_threshold`；
  - `vina_above_threshold`；
  - `low_composite_score`。

## 9. JSON 输出格式

`--output_json` 写 summary JSON，至少包含：

```text
num_total
num_valid
num_posebusters_valid
num_metric_success
num_feasible
num_top
num_middle
num_bottom
validity_rate
posebusters_validity_rate
metric_success_rate
plif_mean
strain_mean
vina_mean
main_score_mean
main_score_max
objective_mode
multiobjective_strategy
weights
thresholds
generated_dir
eval_output
metric_source
timestamp
selection
failure_reason_counts
warnings
```

`selection` 中保留 Stage 2 的 indices/masks/labels/summary，便于复查 top/bottom 的具体样本。

## 10. 最小运行命令

### 10.1 读取已有 per-sample reward JSON

```bash
cd /data/bhli/Project/repo-rl/flowr
export PYTHONPATH="$PWD"

python scripts/structure_reward_selection_dryrun.py \
  --generated_dir /data/bhli/Project/repo-rl/checkpoints/flowr_spindr_h/eval_10-lig-per-target_sampled-mol-sizes_100-steps_linear-sampling-strategy \
  --eval_output /data/bhli/Project/repo-rl/checkpoints/flowr_spindr_h/eval_10-lig-per-target_sampled-mol-sizes_100-steps_linear-sampling-strategy/reward_record/reward_strain_vina.json \
  --metric_source existing_eval_output \
  --output_csv /data/bhli/Project/repo-rl/checkpoints/flowr_spindr_h/eval_10-lig-per-target_sampled-mol-sizes_100-steps_linear-sampling-strategy/reward_record/dryrun_strain_vina.csv \
  --output_json /data/bhli/Project/repo-rl/checkpoints/flowr_spindr_h/eval_10-lig-per-target_sampled-mol-sizes_100-steps_linear-sampling-strategy/reward_record/dryrun_strain_vina_summary.json \
  --objective_mode strain_vina \
  --multiobjective_strategy constrained_weighted_sum \
  --strain_max_threshold 10 \
  --vina_max_threshold -6 \
  --top_ratio 0.1 \
  --bottom_ratio 0.1
```

### 10.2 小规模即时 compute

```bash
cd /data/bhli/Project/repo-rl/flowr
export PYTHONPATH="$PWD"

python scripts/structure_reward_selection_dryrun.py \
  --generated_dir /data/bhli/Project/repo-rl/checkpoints/flowr_spindr_h/eval_10-lig-per-target_sampled-mol-sizes_100-steps_linear-sampling-strategy/predictions_multi_1.pt \
  --metric_source compute \
  --output_csv /tmp/dryrun_compute.csv \
  --output_json /tmp/dryrun_compute_summary.json \
  --cache_jsonl /tmp/flowr_structure_reward_cache.jsonl \
  --objective_mode vina \
  --multiobjective_strategy constrained_weighted_sum \
  --vina_good_threshold -10 \
  --vina_bad_threshold 0 \
  --vina_max_threshold -6 \
  --top_ratio 0.1 \
  --bottom_ratio 0.1 \
  --max_prediction_files 1
```

## 11. 已测试内容

已用一个轻量 per-sample reward JSON 测试：

1. `existing_eval_output` 模式；
2. `strain_vina` selection；
3. CSV 写出；
4. JSON summary 写出；
5. invalid / PoseBusters failed / strain threshold failed / Vina threshold failed 的 failure reason 汇总；
6. `py_compile` 和 `git diff --check`。

## 12. 当前无法测试内容

当前环境中未运行真实 FLOWR heavy metric compute，因此未在本阶段实际验证：

1. PLIF 真实计算耗时；
2. Vina 真实计算耗时；
3. PoseBusters 真实 docking validity；
4. top/bottom SDF 中真实 RDKit mol 写出；
5. 真实 `eval_spindr.sh` 输出目录中所有 predictions 文件的大规模 dry-run。

这些需要用户本地已有 FLOWR 环境、ADFR/Vina、PoseBusters、GenBench3D 数据和真实 sampling 输出。

## 13. PLIF / Vina / strain 计算代价评估

- Strain：主要依赖 RDKit force field，通常比 PLIF/Vina 快，但构象或 force field 参数失败时会 fallback；
- PLIF：依赖 protein-ligand complex、ProLIF/MDAnalysis、H 处理和 pocket 解析，可能较慢且失败模式多；
- Vina：依赖 receptor preparation、PDBQT、Meeko、Vina search box 和外部二进制，通常最慢，也最容易受环境影响；
- PoseBusters validity：对 docking-like complex 做多项检查，可能成为计算瓶颈。

## 14. 后续训练中降低计算开销的建议

1. 训练中不要每 step 对大量样本重新算 PLIF/Vina；
2. 使用 `--cache_jsonl` 或后续 SQLite cache 存成功和失败结果；
3. 优先低频更新 reward，例如每 N step / 每 epoch 采样一次；
4. 先用 cheaper reward 过滤，再对候选 top/bottom 计算 Vina；
5. 对同一 pocket / protein 复用 receptor preparation 和 Vina maps；
6. 对 PLIF 保留 `save_dir` 或 pickle cache，避免重复生成 reference PLIF；
7. 对失败类型做统计，若某类 pocket 系统性失败，应先修复数据或环境而不是继续训练；
8. 在 RL fine-tuning 初期可只用 `strain` 或 `strain_vina` 小规模验证梯度链路，再扩展到 PLIF/Vina 三目标。
