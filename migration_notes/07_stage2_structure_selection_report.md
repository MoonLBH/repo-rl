# Stage 2：FLOWR structure reward top/middle/bottom selection 层报告

本阶段在 `flowr/` 中新增默认关闭的 selection 层。该层只消费 Stage 1 的 structure reward dict，不接入 FLOWR 训练 loop，不修改原始训练、采样或 evaluation 入口。

## 1. Selection 模块路径

新增文件：

```text
flowr/flowr/rl/structure_selection.py
flowr/scripts/test_structure_selection.py
```

并在 `flowr/flowr/rl/__init__.py` 中导出：

```python
SelectionConfig
select_top_middle_bottom
select_top_middle_bottom_with_config
```

## 2. 输入输出格式

### 2.1 输入

核心 API：

```python
from flowr.rl.structure_selection import select_top_middle_bottom

selection = select_top_middle_bottom(
    reward_dicts,
    objective_mode="plif_strain_vina",
    multiobjective_strategy="constrained_weighted_sum",
    top_ratio=0.2,
    bottom_ratio=0.2,
    top_k=None,
    bottom_k=None,
    feasible_only_for_top=True,
    include_failed_in_bottom=True,
    min_top_samples=0,
    min_bottom_samples=0,
    main_score_key="main_score",
    plif_min_threshold=0.3,
    strain_max_threshold=10.0,
    vina_max_threshold=-4.0,
    plif_weight=1.0,
    strain_weight=1.0,
    vina_weight=1.0,
)
```

`reward_dicts` 应为 Stage 1 `compute_structure_rewards(...)` 的输出，每个元素至少包含：

```json
{
  "sample_id": "target_0_lig_0",
  "valid": true,
  "posebusters_valid": true,
  "plif_tanimoto": 0.7,
  "strain_energy": 4.0,
  "vina_score": -8.0,
  "plif_success": true,
  "strain_success": true,
  "vina_success": true,
  "feasible": true,
  "eligible_top": true,
  "eligible_bottom": false,
  "main_score": 0.75,
  "normalized_plif": 0.7,
  "normalized_strain_score": 0.8,
  "normalized_vina_score": 0.8
}
```

### 2.2 输出

API 返回：

```json
{
  "top_indices": [0, 1],
  "middle_indices": [10, 11],
  "bottom_indices": [2, 3],
  "top_mask": [true, true, false, false],
  "middle_mask": [false, false, true, true],
  "bottom_mask": [false, false, true, true],
  "selection_labels": ["top", "top", "bottom", "bottom"],
  "summary": {
    "num_samples": 12,
    "num_top": 2,
    "num_middle": 8,
    "num_bottom": 2,
    "requested_top": 3,
    "requested_bottom": 3,
    "bottom_reasons": {
      "2": ["invalid", "not_feasible", "not_eligible_top", "low_composite_score"]
    }
  }
}
```

Masks 目前返回 `List[bool]`，后续接训练时可在 collate 或 Lightning step 中转换为 torch tensor。

## 3. Top selection 规则

Top 候选必须同时满足：

1. `valid == true`；
2. `posebusters_valid == true`；
3. objective 启用的 metric 全部 success，例如 `plif_success/strain_success/vina_success`；
4. 默认 `feasible == true`，除非显式设置 `feasible_only_for_top=False`；
5. `eligible_top == true`；
6. selection 阈值不失败：
   - PLIF：`plif_tanimoto >= plif_min_threshold`；
   - strain：`strain_energy <= strain_max_threshold`；
   - Vina：`vina_score <= vina_max_threshold`；
7. top 按 weighted normalized score 或 `main_score` 从高到低排序；
8. 如果满足条件样本不足，top 数量允许小于 `top_ratio/top_k`；
9. invalid / PoseBusters failed / metric failed / threshold failed 样本不会被强行补进 top。

## 4. Bottom selection 规则

Bottom 默认优先考虑：

1. `valid == false`；
2. `posebusters_valid == false`；
3. 启用 metric 计算失败；
4. `feasible == false`；
5. `eligible_bottom == true`；
6. `main_score` 或 weighted normalized score 低；
7. 多目标中任一目标未过阈值。

排序优先级：

1. invalid；
2. PoseBusters failed；
3. metric failed；
4. 单目标阈值失败；
5. 其他 infeasible；
6. score 为 0；
7. 低分 feasible 样本。

如果 bottom 数量不足，会从未进入 top 的低分样本中补齐；metric failed 样本保留 `main_score=0` 的 fallback 语义，可以进入 bottom。

## 5. 多目标非补偿逻辑

Selection 继承 Stage 1 的非补偿思想，并在 top gate 中再次检查原始 metric 阈值：

- `plif_strain`：PLIF 必须过 `plif_min_threshold`，strain 必须不高于 `strain_max_threshold`；
- `plif_vina`：PLIF 必须过 `plif_min_threshold`，Vina 必须不高于 `vina_max_threshold`；
- `strain_vina`：strain / Vina 都必须过各自阈值；
- `plif_strain_vina`：三项都必须过阈值。

因此不会发生 “PLIF 很高但 strain 很差仍进入 top” 或 “Vina 很好但 PLIF 很低仍进入 top” 的明显补偿现象。权重只影响通过硬门槛后的排序，不允许突破门槛。

## 6. Fallback 样本如何处理

Stage 1 的 reward fallback 一般会设置：

```json
{
  "feasible": false,
  "eligible_top": false,
  "eligible_bottom": true,
  "main_score": 0.0
}
```

Stage 2 会将这类样本识别为 bottom 候选，并记录原因，例如：

```text
not_feasible
not_eligible_top
low_composite_score
```

## 7. Failed metric 样本如何处理

如果 objective 启用了某个 metric，而对应 success flag 为 false：

- `plif_success == false` → `plif_failed`；
- `strain_success == false` → `strain_failed`；
- `vina_success == false` → `vina_failed`。

这些样本：

- 不能进 top；
- 默认可进 bottom；
- 在 `summary.bottom_reasons` 和 `summary.top_reject_reasons` 中记录失败原因。

## 8. Invalid / PoseBusters failed 如何处理

- `valid == false`：不能进 top，优先进入 bottom，原因 `invalid`；
- `posebusters_valid == false`：不能进 top，优先进入 bottom，原因 `posebusters_failed`；
- 如果它们同时存在其他失败，例如 metric failed 或 low score，也会一起记录。

这与 Stage 1 “validity 或 PoseBusters validity failed 的分子 reward 设为 0，不能作为 top 样本” 的规则一致。

## 9. 新增配置参数

`SelectionConfig` 支持：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `objective_mode` | `plif_strain_vina` | 启用目标组合。 |
| `multiobjective_strategy` | `constrained_weighted_sum` | 记录并用于 weighted ranking；top gate 仍强制阈值。 |
| `top_ratio` | `0.2` | 默认 top 比例。 |
| `bottom_ratio` | `0.2` | 默认 bottom 比例。 |
| `top_k` | `None` | 显式 top 数量，优先于 ratio。 |
| `bottom_k` | `None` | 显式 bottom 数量，优先于 ratio。 |
| `feasible_only_for_top` | `True` | top 必须 feasible。 |
| `include_failed_in_bottom` | `True` | failed / fallback 样本优先 bottom。 |
| `min_top_samples` | `0` | smoke/验收用最小 top 期望，不强行补 top。 |
| `min_bottom_samples` | `0` | smoke/验收用最小 bottom 期望。 |
| `main_score_key` | `main_score` | 备用排序分数字段。 |
| `plif_min_threshold` | `0.0` | PLIF top 门槛。 |
| `strain_max_threshold` | `inf` | strain top 最大值。 |
| `vina_max_threshold` | `inf` | Vina top 最大值。 |
| `plif_weight` | `1.0` | 通过门槛后的排序权重。 |
| `strain_weight` | `1.0` | 通过门槛后的排序权重。 |
| `vina_weight` | `1.0` | 通过门槛后的排序权重。 |

## 10. Mock 测试命令

在 `flowr/` 目录下运行：

```bash
PYTHONPATH="$PWD" python scripts/test_structure_selection.py \
  --mock \
  --objective_mode plif_strain_vina \
  --multiobjective_strategy constrained_weighted_sum \
  --top_ratio 0.25 \
  --bottom_ratio 0.35 \
  --plif_min_threshold 0.3 \
  --strain_max_threshold 10 \
  --vina_max_threshold -4
```

对真实 Stage 1 reward JSON 运行：

```bash
PYTHONPATH="$PWD" python scripts/test_structure_selection.py \
  --input /path/to/reward_plif_strain_vina.json \
  --objective_mode plif_strain_vina \
  --multiobjective_strategy constrained_weighted_sum \
  --top_ratio 0.2 \
  --bottom_ratio 0.2 \
  --plif_min_threshold 0.3 \
  --strain_max_threshold 10 \
  --vina_max_threshold -4 \
  --output /path/to/selection_plif_strain_vina.json
```

## 11. 与 sotmol-rl 原始 top/bottom selection 的对应关系

对应关系：

| sotmol-rl 语义 | FLOWR Stage 2 selection |
|---|---|
| reward/objective score | Stage 1 reward dict 的 `main_score` 与 normalized metrics |
| top samples | valid + PoseBusters valid + enabled metric success + feasible + threshold pass + high score |
| bottom samples | invalid / failed / infeasible / threshold failed / low score 样本 |
| middle samples | 未进入 top 或 bottom 的剩余样本 |
| top/bottom masks | `top_mask`, `bottom_mask`, `middle_mask` |
| 多目标 MPO 不补偿 | threshold gate + non-compensatory strategy + failure reasons |

与 sotmol-rl 不同的是，本阶段 selection 针对 structure-based generation，必须额外处理 protein-ligand 结构 metric 的失败、PoseBusters gating、PLIF/Vina/strain 的方向和量纲差异。

## 12. 当前限制

1. 当前只做离线 selection，不接训练 loop；
2. masks 返回 list，后续训练集成时再转 tensor；
3. selection 假设 Stage 1 reward dict 已经按目标和阈值正确计算 `main_score/eligible_top/eligible_bottom`，本阶段只做二次 gate 和排序；
4. 多目标排序优先使用 normalized metric 和权重，如果缺少 normalized 字段，则 fallback 到 `main_score`；
5. 暂不实现跨 batch 的历史 top/bottom buffer；后续接 RL fine-tuning 时可参考 sotmol-rl 的历史分位数/partition 逻辑扩展；
6. 当前 bottom 是失败优先 + 低分补齐，尚未加入 diversity/scaffold 去重约束。
