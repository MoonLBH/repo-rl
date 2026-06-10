# Stage 4 follow-up: LFPO-F-v2 interpolant, metric, mask, and chunking fixes

## 1. 修改文件列表

本阶段只在现有 FLOWR LFPO-F-v2 版本上做定点修复，没有重写算法：

- `flowr/flowr/rl/training.py`
- `flowr/flowr/models/fm_pocket.py`
- `flowr/flowr/train.py`
- `migration_notes/11_stage4_lfpof_fix_interpolant_metrics_chunking_report.md`

保留的目标算法仍是：

```text
z0 ~ pi_ref full FLOWR generation
-> reward / selection
-> selected z0 rebuild pseudo s_t
-> current/ref dual forward
-> top-reward imitation + bottom reference-based negative repulsion
-> original_FLOWR_loss + rl_loss_weight * rl_surrogate_loss
```

## 2. 修复 `rl_num_stratified_timesteps > 1` mask bug 的位置和方式

问题：旧实现中 selected candidate 会 repeat `K=rl_num_stratified_timesteps` 次，但 `top_mask` / `bottom_mask` 可能仍按 candidate-level label 构造，导致 pseudo batch size 为 `selected_count * K` 时 mask 长度不一致。

修复位置：`flowr/flowr/rl/training.py`。

新增逻辑：

1. `expanded_selection_plan(selected_indices, selected_labels, k_steps)` 生成 expanded plan；
2. `build_pseudo_training_batch_from_plan(...)` 从 expanded plan 构造 pseudo batch；
3. pseudo batch 返回 `pseudo["labels"] = expanded_labels`；
4. `lfpof_v2_surrogate_loss(...)` 和 chunk helper 都只从 expanded labels 构造 mask；
5. 对 mask 长度加入断言：

```python
assert top_mask.shape[0] == current_pred["coords"].shape[0]
assert bottom_mask.shape[0] == current_pred["coords"].shape[0]
```

## 3. `expanded_labels` / expanded masks 的实现

Expanded labels 的语义是：

```python
expanded_labels = []
for label in selected_labels:
    for _ in range(rl_num_stratified_timesteps):
        expanded_labels.append(label)
```

实际实现通过 `(candidate_index, label)` 的 expanded plan 完成。这样 chunk-wise path 和 non-chunk path 可以共用同一套 sample-level labels。

新增/补充日志：

- `train-rl-num-top-candidates`
- `train-rl-num-bottom-candidates`
- `train-rl-num-top-expanded`
- `train-rl-num-bottom-expanded`
- `train-rl-expanded-selected-count`
- `train-rl-pseudo-batch-size`
- `train-rl-num-stratified-timesteps`

当 `rl_num_stratified_timesteps=1` 时，expanded count 等于 candidate count；当 `K>1` 时，expanded count 为 candidate count 乘以 `K`。

## 4. `corrupt_ligand_like_flowr` 原问题

旧 `corrupt_ligand_like_flowr(...)` 是 smoke-test 级别近似：

- coordinate：`noise * (1 - t) + target * t`；
- atom/bond/charge：按 `t` 随机保留 target 或替换随机类别。

它没有复用 FLOWR 原始 interpolant 中的 prior sampler、categorical interpolation strategy、OT matching、coordinate noise std、pocket interpolation 和原始 target collation，因此不适合作为默认 LFPO-F-v2 training corruption。

## 5. 当前如何复用 FLOWR 原始 interpolant / noising

默认新增并启用：

```text
rl_use_original_interpolant=true
rl_allow_simple_corruption_fallback=false
```

训练脚本在构建 model 后将 datamodule 的训练 interpolant 挂到模型上：

```python
model.rl_train_interpolant = dm.train_interpolant
```

RL pseudo batch 默认调用 `build_pseudo_with_original_interpolant(...)`，其关键步骤是：

1. 将 selected generated tensor target 转成 `GeometricMol`；
2. 使用 FLOWR 原始 `train_interpolant.prior_sampler.sample_molecule(...)` 采样 prior ligand；
3. 使用 FLOWR 原始 `train_interpolant._match_mols(...)` 做 molecule matching；
4. 使用 FLOWR 原始 `train_interpolant._interpolate_mol(...)` 构造 ligand `s_t`；
5. 使用 FLOWR 原始 `train_interpolant._interpolate_pocket(...)` 构造 pocket interpolation；
6. 通过 `PocketComplexBatch` 转回 FLOWR training dict；
7. 再用 `model.builder.extract_ligand_from_complex(...)` / `extract_pocket_from_complex(...)` 构造与原始 `training_step` 相同的 `target`、`interp`、`pocket` dict。

这样默认路径不再手写 categorical/coordinate corruption，而是调用 FLOWR 原始 interpolant 方法。

## 6. fallback 条件

`corrupt_ligand_like_flowr(...)` 仍保留为 fallback，但默认不允许 fallback。

- 默认：`rl_allow_simple_corruption_fallback=false`，若原始 interpolant 构造失败，会抛出清晰错误并跳过 RL loss；
- 只有显式设置 `--rl_allow_simple_corruption_fallback` 后，才会进入旧 simple fallback；
- fallback 时日志：`train-rl-interpolant-mode=2`；
- 原始 interpolant 成功时日志：`train-rl-interpolant-mode=1`。

当前显式 inactive 的特殊逻辑：

- interaction inpainting；
- scaffold inpainting；
- functional group inpainting；
- linker inpainting；
- mixed unconditional inpainting。

这些逻辑不是本阶段 RL surrogate 的默认路径；如果原始 model 启用了 interaction flow/inpainting，仍需要单独验证 pseudo batch 字段兼容性。

## 7. top imitation loss 定义

Top branch 没有改变算法：target 仍来自 `pi_ref` 完整 generation 后被选中的 high-reward generated ligand `z0`，不是 native ligand。

```text
L_top_atom   = CE(X0, atom_logits_cur)
L_top_bond   = CE(E0, bond_logits_cur)
L_top_charge = CE(C0, charge_logits_cur)
L_top_coord  = masked_MSE(coord_pred_cur, coord_target_from_generated_z0)
```

最终：

```text
L_top =
  rl_top_atom_weight   * L_top_atom
+ rl_top_bond_weight   * L_top_bond
+ rl_top_charge_weight * L_top_charge
+ rl_top_coord_weight  * L_top_coord
```

## 8. bottom repulsion loss 定义

Bottom branch 仍是 reference-based negative repulsion。

离散 head：

```text
logp_cur = log_softmax(logits_cur)
logp_ref = log_softmax(logits_ref)
delta = stop_gradient(logp_cur - logp_ref)
logp_minus = logp_ref - beta * delta
p_minus = softmax(logp_minus).detach()
L_bottom = soft_CE(p_minus, logits_cur)
```

Coordinate head：

```text
delta_r = stop_gradient(pred_cur_r - pred_ref_r)
target_minus_r = pred_ref_r - gamma * delta_r
L_bottom_coord = masked_MSE(pred_cur_r, target_minus_r.detach())
```

没有恢复旧 `relu(1 - coord_mse)` placeholder。

## 9. metric 策略修改

当前策略改为：

```text
all-enabled-metrics-after-feasibility-gating
```

也就是每次 RL update 中：

1. 对 generated candidate 先做 validity；
2. 对 valid candidate 做 PoseBusters validity；
3. invalid / PoseBusters failed 不再计算 expensive metric，reward=0，不能 top，可 bottom；
4. valid + PoseBusters valid 的 candidate 必须计算当前 `rl_objective_mode` 启用的全部 metrics；
5. 任一启用 metric failed，则不能 top，可 bottom。

保留历史 CLI 参数以兼容旧命令：

- `rl_compute_plif_every_n_steps`
- `rl_compute_vina_every_n_steps`
- `rl_compute_strain_every_n_steps`
- `rl_skip_expensive_metrics_in_warmup`
- `rl_warmup_steps`

但当前默认 RL training 逻辑不使用这些参数来跳过启用指标。它们只作为 future option 保留。

## 10. validity / PoseBusters failed 后不计算 expensive metrics 的逻辑

Stage 1 reward wrapper 已经按 cheap-first 顺序执行：

1. `_compute_validity(...)`；
2. `_compute_posebusters_validity(...)`；
3. 若失败直接返回；
4. 只有通过后才计算 PLIF / strain / Vina。

本阶段训练端不再添加 every-n-steps / warmup skip，因此 reward wrapper 会对 feasible candidate 计算所有 enabled metrics。

如果一个 batch 中存在 feasible candidates 但所有 enabled metric 都失败，训练端新增 skip：

```text
train-rl-skip-reason = all_enabled_metrics_failed
```

只返回 original FLOWR loss。

## 11. cache 策略

Cache 保留：

- `rl_metric_source=cached` 时优先读取 JSONL cache；
- cache miss 时计算当前 objective mode 启用的全部 metrics；
- 成功和失败结果都写入 cache；
- `train-rl-cache-hit-rate` 记录命中率。

本阶段还修复了 cache hit/miss 混合时 reward 顺序可能和 candidate 顺序不一致的问题：现在 `rewards` 以 candidate index 填充，返回时保持原始 candidate order，避免 top/bottom selection index 与 candidates 错位。

## 12. chunk-wise RL backward 实现方式

新增参数：

```text
rl_surrogate_chunk_size: int = 0
```

含义：

- `0`：不主动 chunk，沿用 automatic optimization path；
- `>0`：启用 manual optimization + true chunk-wise RL backward。

当 `enable_rl_finetune=true`、`rl_loss_weight>0`、`rl_surrogate_chunk_size>0` 时，模型初始化时设置：

```python
self.automatic_optimization = False
```

训练 step 中执行：

1. 计算 original FLOWR loss；
2. `optimizer.zero_grad()`；
3. `manual_backward(original_flowr_loss)`；
4. 采样/reference/reward/selection；
5. 生成 expanded plan；
6. 按 `rl_surrogate_chunk_size` 切分 expanded samples；
7. 每个 chunk 单独构造 pseudo batch、current/ref forward、计算 chunk RL loss；
8. 立即 `manual_backward(rl_loss_weight * chunk_loss)`；
9. 删除 chunk 图；
10. 所有 chunk 完成后 optimizer step / scheduler step。

这样不会把所有 chunk 的 computation graph 累加到最后统一 backward。

Chunk loss 使用 `len(chunk) / N_selected_expanded` 做 sample-level scale，避免每个 chunk 被等权处理造成梯度尺度漂移。

## 13. manual optimization 启用条件

只在以下条件全部满足时启用 manual optimization：

```text
enable_rl_finetune=true
rl_loss_weight > 0
rl_surrogate_chunk_size > 0
```

否则：

- RL disabled：完全保持原始 FLOWR automatic optimization；
- RL enabled 且 `rl_surrogate_chunk_size=0`：保留 non-chunk automatic path，但可能在大 batch / K>1 下 OOM。

## 14. RL disabled path 是否保持原始 baseline 不变

保持不变。

当 `enable_rl_finetune=false` 或 `rl_loss_weight<=0`：

- `maybe_apply_rl_finetune_loss(...)` 在入口直接返回原 loss 和空 logs；
- 不初始化 `pi_ref`；
- 不调用 `_generate(...)`；
- 不计算 reward / metrics；
- 不构造 pseudo batch；
- 不进入 manual optimization。

## 15. 新增 / 修改配置参数

新增：

- `--rl_surrogate_chunk_size`，默认 `0`；
- `--rl_use_original_interpolant`，默认 true；
- `--rl_allow_simple_corruption_fallback`，默认 false。

保留但当前不用于默认 metric skip：

- `--rl_compute_plif_every_n_steps`
- `--rl_compute_vina_every_n_steps`
- `--rl_compute_strain_every_n_steps`
- `--rl_skip_expensive_metrics_in_warmup`
- `--rl_warmup_steps`

## 16. 新增 logging keys

新增或补充：

- `train-rl-num-top-candidates`
- `train-rl-num-bottom-candidates`
- `train-rl-num-top-expanded`
- `train-rl-num-bottom-expanded`
- `train-rl-expanded-selected-count`
- `train-rl-interpolant-mode`
- `train-rl-num-stratified-timesteps`
- `train-rl-pseudo-batch-size`
- `train-rl-surrogate-chunk-size`
- `train-rl-num-rl-chunks`
- `train-rl-manual-optimization-enabled`
- `train-rl-bottom-branch-used`

原有 `train-rl-delta-*`、reward summary、metric failure、skip reason 日志继续保留。

## 17. smoke test 命令

### 17.1 strain-only non-chunk smoke

```bash
cd /data/bhli/Project/repo-rl/flowr
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0

python -m flowr.train \
  --gpus 1 \
  --dataset spindr \
  --data_path /path/to/spindr \
  --save_dir /path/to/lfpof_stage11_strain_k1 \
  --exp_name lfpof_stage11_strain_k1 \
  --arch pocket \
  --pocket_noise fix \
  --load_ckpt /path/to/pretrained.ckpt \
  --trial_run \
  --enable_rl_finetune \
  --rl_loss_weight 0.01 \
  --rl_objective_mode strain \
  --rl_metric_source cached \
  --rl_metric_cache_path /path/to/rl_metric_cache.jsonl \
  --rl_sampling_steps 10 \
  --rl_num_candidates_per_step 1 \
  --rl_num_stratified_timesteps 1 \
  --rl_surrogate_chunk_size 0 \
  --rl_update_frequency 1
```

### 17.2 K=2 mask/chunk smoke

```bash
python -m flowr.train \
  --gpus 1 \
  --dataset spindr \
  --data_path /path/to/spindr \
  --save_dir /path/to/lfpof_stage11_strain_k2_chunk \
  --exp_name lfpof_stage11_strain_k2_chunk \
  --arch pocket \
  --pocket_noise fix \
  --load_ckpt /path/to/pretrained.ckpt \
  --trial_run \
  --enable_rl_finetune \
  --rl_loss_weight 0.01 \
  --rl_objective_mode strain \
  --rl_metric_source cached \
  --rl_metric_cache_path /path/to/rl_metric_cache.jsonl \
  --rl_sampling_steps 10 \
  --rl_num_candidates_per_step 1 \
  --rl_num_stratified_timesteps 2 \
  --rl_surrogate_chunk_size 1 \
  --rl_update_frequency 1
```

### 17.3 K=4 expanded-label diagnostic

```bash
python -m flowr.train \
  --gpus 1 \
  --dataset spindr \
  --data_path /path/to/spindr \
  --save_dir /path/to/lfpof_stage11_strain_k4_chunk \
  --exp_name lfpof_stage11_strain_k4_chunk \
  --arch pocket \
  --pocket_noise fix \
  --load_ckpt /path/to/pretrained.ckpt \
  --trial_run \
  --enable_rl_finetune \
  --rl_loss_weight 0.01 \
  --rl_objective_mode strain \
  --rl_metric_source cached \
  --rl_metric_cache_path /path/to/rl_metric_cache.jsonl \
  --rl_sampling_steps 10 \
  --rl_num_candidates_per_step 1 \
  --rl_num_stratified_timesteps 4 \
  --rl_surrogate_chunk_size 2 \
  --rl_update_frequency 1
```

验收时重点看：

- `train-rl-num-top-candidates` vs `train-rl-num-top-expanded`；
- `train-rl-num-bottom-candidates` vs `train-rl-num-bottom-expanded`；
- `train-rl-pseudo-batch-size`；
- `train-rl-num-stratified-timesteps`；
- `train-rl-num-rl-chunks`；
- `train-rl-interpolant-mode=1`。

## 18. 当前已验证内容

本阶段已做静态验证：

1. `python -m py_compile` 通过；
2. static text checks 通过：
   - disabled path 在 reference / sampling / reward 前返回；
   - training helper 不再读取 every-n-steps / warmup skip 参数；
   - expanded label plan 存在；
   - 默认使用 original interpolant path；
3. `git diff --check` 通过。

## 19. 当前未验证内容

当前环境未实际运行：

- GPU 上真实 FLOWR training step；
- 真实 `pi_ref._generate(..., rl_sampling_steps=10/100)` 训练内速度；
- PLIF / Vina / PoseBusters / strain heavy metrics 的训练内稳定性；
- manual optimization 与当前 Lightning scheduler / gradient clipping 在完整训练中的长期兼容性；
- interaction-flow 或 inpainting 配置下 pseudo batch 字段完整性；
- science-level reward improvement。

## 20. 仍可能存在的风险

1. 原始 FLOWR `ComplexInterpolant` 是 datamodule 层对象，当前通过 `model.rl_train_interpolant = dm.train_interpolant` 挂载；若从非 `flowr.train` 入口构建 model，需要同样挂载。
2. `rl_surrogate_chunk_size=0` 的 automatic path 仍会一次构建所有 selected expanded samples，可能 OOM；大 batch 建议使用 chunk path。
3. 原始 interpolant 的 private methods（例如 `_interpolate_mol` / `_match_mols`）被复用，未来 FLOWR upstream 改名会影响 RL helper。
4. 当前对 inpainting/interaction-flow 不是主要目标路径；若后续使用，需要额外测试。
5. Manual optimization path 手动 step scheduler；如果使用 epoch-level scheduler，需确认与实验预期一致。
