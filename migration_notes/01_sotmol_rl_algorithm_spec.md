# 01. sotmol-rl / LIFT 算法规格说明

> 范围：本文件只分析 `sotmol-rl/` 中当前 LIFT / LFPO-F 风格 reward-guided fine-tuning 的核心计算逻辑，用于后续迁移到 `flowr/`。本阶段未修改 `sotmol-rl/` 或 `flowr/` 代码。

## A. 总体训练 workflow

`sotmol-rl` 的第一阶段迁移对象主要是 `train_lift_history_only*.py` 系列脚本所启动的 `MolGen_LFPOModel -> LIFT_Lightning` 流程。它不是标准 supervised flow matching 训练，而是一个“reference policy 采样 + objective 打分 + top/bottom 选择 + current model 对 reference 生成样本做正负 rectification”的 fine-tuning 循环。

1. **入口脚本组装配置**
   - 读取 `rl.json` 覆盖全局 `GP` 参数，例如 `D_MODEL`、`LR`、`CUDA_VISIBLE_DEVICES`。
   - 根据 `--objective_name` 构造 `objective_config`，支持 QED、Celecoxib similarity、Perindopril similarity + aromatic ring、xTB force reward、PSA minimization、QED+SA、Ranolazine/Osimertinib MPO 等。
   - 构造 `partition_config`，默认 `top_ratio=0.25`、`bottom_ratio=0.25`、`history_diversity_mode=scaffold`、`history_max_size=4096`。
   - 构造 `lfpo_hparams`，控制时间采样数、正负分支系数、charge head、reference EMA、current-policy evaluation 等。

2. **模型封装与 LightningModule 创建**
   - `MolGen_LFPOModel` 继承通用 `MolGen_Model`，负责创建 SEMLA-like denoising network、vocabulary、datamodule、logger、checkpoint callback。
   - `MolGen_LFPOModel.create_lightning_module()` 将基础模型包装成 `LIFT_Lightning`，并可从 `prior.ckpt` 载入预训练无条件 3D flow matching 模型。

3. **训练启动**
   - `Train()` 使用 `MGDataModule` 加载 `.smol` 数据；这些数据主要在 LIFT 中提供 noise/template batch 的形状、mask、初始噪声、原始 batch 字段，而 RL 目标来自 reference/current 生成样本。
   - Lightning `Trainer.fit()` 运行 `LIFT_Lightning.FM_training_step()`；该类设置 `automatic_optimization=False`，在时间 chunk 内手动 `manual_backward()`，降低显存峰值。

4. **每个 training step 的核心闭环**
   - 从 batch 构造 initial noise/template。
   - 使用 frozen/EMA reference model 采样一批分子。
   - 将生成张量重建成 RDKit mol，调用 objective 计算 score/components/raw properties/validity/connectivity。
   - 通过 history-aware partition selector 选出 top、bottom，剩余为 middle。
   - 把 reference 生成样本转换成 pseudo-target batch。
   - 对每个样本采样 `K=lfpo_num_time_samples` 个 stratified time，current model 和 reference model 都在同一 `x_t` 上前向。
   - top 样本走 imitation / positive branch：current model 拟合 reference 生成出的 pseudo-target。
   - bottom 样本走 repulsion / negative branch：根据 current-vs-reference prediction delta 构造反向隐式目标。
   - middle 默认不参与主损失（`lfpo_middle_weight=0.0`）。
   - 主损失按 top/bottom selected 样本平均；aux FM 与 anchor 按所有 `B*K` 样本平均；optimizer step 后更新 reference EMA。

## B. 关键文件、类、函数路径

| 关注点 | 文件 / 类 / 函数 |
|---|---|
| 主入口脚本 | `sotmol-rl/train_lift_history_only.py`; `train_lift_history_only_celecoxib_ready.py`; `train_lift_history_only_perindopril_ready.py`; `train_lift_history_only_xtb01_ready.py` |
| 模型外层封装 | `sotmol-rl/sot_mol/models/rl_lfpo_interface.py::MolGen_LFPOModel` |
| 网络创建 | `sotmol-rl/sot_mol/models/interface.py::MolGen_Model.__build_network_arch` |
| LIFT LightningModule | `sotmol-rl/sot_mol/models/lift_v2.py::LIFT_Lightning` |
| reference/current 前向 | `lift_v2.py::_forward_with_model`（继承自 `RL_Lightning`）; `diff.py::SC_Lightning.forward` |
| 采样 | `lift_v2.py::_generate_with_model`; `diff.py::_integrate_step`; `diff.py::_uniform_sample_step` |
| pseudo-target 构造 | `rl_diff.py::_build_noise_batch`; `rl_diff.py::_build_generated_target_batch` |
| objective 构建 | `rl_objectives/mpo_tasks.py::build_objective` |
| objective 返回结构 | `rl_objectives/mpo_tasks.py::ScoringResult` |
| top/bottom/history selector | `rl_objectives/partition_history_only.py::PartitionSelector`; `PartitionResult` |
| soft CE / implicit rectification | `lift_v2.py::_lfpof_discrete_loss_per_sample`; `_lfpof_coord_loss_per_sample` |
| positive FM per-sample loss | `rl_diff.py::_loss_per_sample`; `diff.py::_type_loss`; `_bond_loss`; `_charge_loss` |
| RDKit 重建 | `models/molbuilder.py::MolBuilder.mols_from_tensors`; `util/rdkit.py::mol_from_atoms` |
| 质量指标 | `rl_diff.py::_compute_generation_quality_from_mols`; `util/metrics.py` |
| checkpoint / logger | `rl_lfpo_interface.py::MolGen_LFPOModel.Train` |

## C. 一个 training step 的完整数据流

以下按 `LIFT_Lightning.FM_training_step(batch)` 描述。

1. **输入 batch**
   - 主要字段：`noise_coords`, `noise_atomics`, `noise_bonds`, `real_coords`, `real_atomics`, `real_bonds`, `real_charges`, `masks`, `natoms`, `flag_3Ds`。
   - LIFT 先使用 `_build_noise_batch(batch)` 取出 noise/template：`coords=noise_coords`、`atomics=noise_atomics`、`bonds=noise_bonds`、`masks`、`flag_3Ds`。

2. **reference policy 采样**
   - `generated_ref = _generate_with_model(self.ref_gen, noise, inference_steps=max_steps, coord_noise_std=..., cat_noise_level=...)`。
   - 采样过程是 Euler/flow-style integration：时间从 0 到 1，连续坐标按 velocity 更新，离散 atom/bond 通过 categorical transition 采样。
   - 返回 `generated_ref`：`coords` 已乘回 `coord_scale`，`atomics/bonds/charges` 是概率分布，包含 `masks` 和 `flag_3Ds`。

3. **RDKit mol 重建 + objective 打分**
   - `_generate_mols(generated_ref, sanitise=True)` 调 `MolBuilder.mols_from_tensors()`。
   - `MolBuilder` 对每个样本按 mask 截断，atom/bond/charge 取 `argmax`，再调用 `mol_from_atoms()` 生成 RDKit mol。
   - `objective.score_mols(mols)` 输出 `ScoringResult`，包含 scalar `score`、component scores、raw properties、feasible/valid/connected/severe masks、canonical smiles、scaffolds、fingerprints 等。

4. **top/bottom/middle 划分**
   - `partition_ref = partition_selector.select(scoring_ref)`。
   - 输出 `top_mask`, `bottom_mask`, `selected_mask = top | bottom`, `top_weights`，以及 diagnostics。
   - `middle = ~(top | bottom)`；默认 middle 主损失权重为 0。

5. **构造训练 pseudo-target**
   - `train_batch = _build_generated_target_batch(batch, generated_ref)`。
   - `real_coords = generated_ref["coords"] / coord_scale`，`real_atomics/bonds/charges = generated_ref` 中对应概率分布，noise/mask/natoms/flag 从原 batch 继承。

6. **可选 current-policy 低频评估**
   - 当 `lfpo_log_current_reward=True` 且 `global_step % lfpo_eval_current_every == 0`，用 current `self.gen` 额外采样 `lfpo_eval_current_samples` 个分子并打分，仅用于日志和 checkpoint monitor，不参与反向传播。

7. **时间采样与 chunk 训练**
   - `t_bk = _sample_stratified_timesteps(B, K)`；第 `k` 个时间位于 `[k/K, (k+1)/K)`。
   - 每个 `k` 和 batch chunk：
     - `interp_data = interpolate(train_batch, t_chunk, flag_3Ds)`。
     - 可选 self-conditioning：50% 概率先用 current model no-grad 前向生成 `cond_batch`。
     - current model 前向得到 `predicted = {coords, atomics logits, bonds logits, charges logits}`。
     - reference model no-grad 前向得到 `ref_predicted`。
     - 计算 positive / negative / aux / anchor per-sample losses。
     - `chunk_total.backward()`，累积日志值。

8. **优化与 reference 更新**
   - 所有 chunk backward 后 `opt.step(); opt.zero_grad()`；scheduler step。
   - `on_train_batch_end()` 调 `_maybe_update_reference_ema()`：`ref = decay * ref + (1-decay) * current`，buffer 直接复制，reference 保持 eval/frozen。

## D. reward 的输入、输出和计算位置

1. **输入**
   - reward/objective 不直接读模型 logits，而是读 RDKit mol list。
   - mol list 来自 `generated_ref` 或低频 `generated_cur` 的 tensor-to-RDKit 重建结果。

2. **输出结构**
   - `ScoringResult.score`: 主 scalar reward，shape `[B]`。
   - `component_scores`: 已归一化到 `[0,1]` 或近似 `[0,1]` 的 component reward，例如 `qed`, `SA`, `sim_celecoxib`, `sim_perindopril`, `aromatic_ring_reward`, `xtb_reward`, `sim_ranolazine_AP`, `logP`, `TPSA`, `num_F`。
   - `raw_properties`: 原始性质，用于日志和诊断，例如 QED、TPSA、logP、heavy atom count、xTB force RMS、xTB success 等。
   - `feasible`, `severe_violation`, `valid`, `connected`: 用于 partition 与日志。
   - `smiles/canonical_smiles/scaffolds/fps/mols`: 用于 history novelty/diversity 和 oracle log。

3. **主要 objective**
   - `QEDObjective`: valid 且 connected 的 mol 使用 `RDKit QED.qed`，否则 0。
   - `TargetSimilarityObjective`: 对 Celecoxib 或任意 target 计算 Tanimoto similarity；可选 threshold cap；额外记录 QED/TPSA/logP/HAC raw properties。
   - `PerindoprilSimilarityAromaticObjective`: similarity component 与 aromatic ring reward 组合，支持 Pareto component names。
   - `XTBForceObjective`: 调 xTB/ASE 计算 force RMS，并映射为 bounded inverse / exp / linear cutoff / raw negative force reward；可设置失败惩罚、timeout、并行 worker。
   - `PSA3DMinObjective`: RDKit FreeSASA 计算 polar surface area，低 PSA 高 reward。
   - `QEDSAObjective`: QED + SA desirability，多 component 聚合。
   - `MPOObjective`: Ranolazine / Osimertinib 这类 GuacaMol/MPO 风格任务，component 通过 desirability 函数组合。

4. **计算位置**
   - reference training reward：`LIFT_Lightning.FM_training_step()` 中 `objective.score_mols(generated_mols_ref)`。
   - current eval reward：`_evaluate_model_scoring()` 中生成 current mols 后 `objective.score_mols()`。
   - legacy scalar QED reward helper 仍在 `RL_Lightning._compute_rewards_from_mols()`，但当前 LIFT 主流程使用 `mpo_tasks.py` objective abstraction。

## E. top / middle / bottom 划分规则

当前脚本使用 `partition_history_only.py` 的 history-aware feasible top selector，而不是旧的简单 reward top-k。

1. **top 候选限制**
   - top 只能来自 `scoring.feasible == True` 的样本。
   - `top_n = ceil(B * top_ratio)`，默认 `top_ratio=0.25`。

2. **top_score 排序**
   - 默认 `top_selection_score_mode="score"`，直接使用 `scoring.score`。
   - `component_balanced` / `score_plus_min_component`: 在 base score 上加权加入指定 component，再可加 min-component bonus。
   - `tchebycheff`: 用 `tchebycheff_score(component_scores, weights)` 排序。

3. **history novelty 过滤**
   - `history_diversity_mode="scaffold"`: 若候选 scaffold 已在历史 top set 中出现，跳过。
   - `history_diversity_mode="fingerprint"`: 若候选 fingerprint 与历史 top fps 的最大 Tanimoto `>= history_fingerprint_threshold`（默认 0.70），跳过。
   - `history_diversity_mode="none"`: 不做历史过滤。
   - 被选为 top 的样本会更新 canonical smiles、scaffold 或 fingerprint history；`history_max_size` 控制 fps/history_scores 的粗略容量。

4. **bottom 选择**
   - `bottom_n = ceil(B * bottom_ratio)`，默认 `bottom_ratio=0.25`。
   - 从非 top 样本中按优先级选择：
     1. `invalid`: `~feasible | ~valid | ~connected`；
     2. `severe`: `scoring.severe_violation`；
     3. `low_score`: 所有剩余非 top 样本中按 score 从低到高。
   - bottom 不使用 history novelty。

5. **middle**
   - `middle = 1 - clamp(top + bottom, max=1)`。
   - 默认 `lfpo_middle_weight=0.0`，因此 middle 不参与主 LFPO loss；它只可能影响 aux FM / anchor 这类按所有样本计算的项。

## F. positive / negative 分支或 imitation / repulsion 逻辑

LIFT 的主思想可以概括为：

- **Top = explicit imitation**：高 reward reference 生成样本成为 pseudo-target，current model 在随机时间 `x_t` 上学习把它作为 flow matching 终点/目标。
- **Bottom = implicit repulsion**：低 reward 或 invalid/severe 样本不直接作为负标签，而是比较 current 与 reference 的预测差异，并构造“远离 current 相对 reference 偏移方向”的隐式负目标。
- **Middle = 可选混合**：默认关闭；若启用，用 reward-normalized pull weight 在 plus/minus loss 之间插值。

具体为：

1. **Positive branch**
   - `aux_losses = _loss_per_sample(aux_target, predicted)`。
   - `aux_target` 的 coords 是 reference 生成样本坐标；atom/bond/charge 是 reference 生成分布。
   - top 主损失使用：`top_mask * top_weight * aux_loss`。
   - 默认 `top_weight = 1`；若 `lfpo_top_weight_mode="rwr"`，用 `_reward_to_pull_weights(score)` 归一到 top 内 mean=1。

2. **Negative branch for discrete variables**
   - 对 logits 计算 `delta = log_softmax(cur_logits) - log_softmax(ref_logits)`。
   - 构造：
     - `logp_plus = logp_ref + beta * delta`
     - `logp_minus = logp_ref - beta * delta`
   - `p_plus/p_minus = softmax(logp_plus/logp_minus)`，默认 detach target。
   - bottom 主损失使用 `CE(p_minus, cur_logits)`，即让 current 不朝当前相对 reference 的偏移方向继续走。

3. **Negative branch for coordinates**
   - `delta_r = pred_cur_r - pred_ref_r`。
   - `target_plus = pred_ref + beta_coord * delta_r`。
   - `target_minus = pred_ref - gamma_coord * delta_r`。
   - bottom 主损失使用 `MSE(pred_cur, target_minus)`，并按 atom mask、`flag_3Ds` 加权。

4. **主损失组合**
   - `type_rect = top * top_w * pos_type + bottom_weight * bottom * type_minus + middle * mid_type`。
   - bond、coord、charge 同理。
   - `main = lambda_coord * coord_rect + lambda_types * type_rect + lambda_bonds * bond_rect + lambda_charges * charge_rect`。
   - `chunk_total = mean_selected(main) + lfpo_aux_fm_weight * mean_all(aux_fm) + anchor_weight * mean_all(anchor)`。

## G. flow matching surrogate loss 对应的代码位置

这里有两层 surrogate / FM loss：

1. **Positive imitation 的 FM surrogate**
   - `LIFT_Lightning.FM_training_step()` 将 reference 生成样本变成 `train_batch`，再调用 `interpolate()` 产生 `x_t`。
   - 对 top 样本，`_loss_per_sample(aux_target, predicted)` 就是常规 FM 监督损失；它把 reference 生成样本当成 high-reward pseudo data。
   - 这部分是 top imitation 的核心 surrogate：不直接对 reward 求梯度，而对高 reward sampled endpoint 做 flow-matching imitation。

2. **Negative implicit rectification surrogate**
   - `_lfpof_discrete_loss_per_sample()` 和 `_lfpof_coord_loss_per_sample()` 根据 current/ref prediction delta 构造 plus/minus pseudo-target。
   - bottom 使用 minus loss，形成 repulsion surrogate；它同样不对 reward 求梯度。

3. **Auxiliary FM loss**
   - `lfpo_aux_fm_weight` 控制把所有样本上的 `aux_losses` 加到总损失中。
   - 当前训练脚本通常设置 `lfpo_aux_fm_weight=0.0`，即 aux FM 默认关闭；但代码保留了这个正则项。

## H. discrete atom/bond/charge 与 continuous coordinate loss 的表示和组合方式

1. **张量表示**
   - `coords`: `[B, N, 3]` 连续坐标，训练内部常用标准化坐标；生成结束时乘回 `coord_scale`。
   - `atomics`: `[B, N, V]` one-hot/probability/logits；`V=len(GP.TOKENS)`，tokens 包含 `<PAD>`, `<MASK>`, H/C/N/O/F/P/S/Cl/...。
   - `bonds`: `[B, N, N, E]` one-hot/probability/logits；`E=N_BOND_TYPES`，0 通常为 no bond，1-4 映射 single/double/triple/aromatic。
   - `charges`: `[B, N, C]` one-hot/probability/logits；`GP.IDX_CHARGE_MAP = {0:0, 1:1, 2:2, 3:3, 4:-1, 5:-2, 6:-3}`。
   - `masks`: `[B, N]` 有效 atom mask。
   - `flag_3Ds`: `[B]`，用于关闭/开启 coordinate loss 和 coordinate update。

2. **interpolation**
   - 坐标：`x_t = (1-t) * noise_coords + t * real_coords + GaussianNoise`，再乘 `flag_3Ds`。
   - atom/bond：按 Bernoulli mask 在 noise category 与 real category 间采样，得到 one-hot `x_t`。
   - charge 没有显式放入 `interpolate()` 的输入 state；charge 是 model head 的监督目标。

3. **标准 per-sample loss**
   - coordinate: masked MSE over `[N,3]`，乘 `flag_3Ds`。
   - atom type: node-wise CE，按 atom mask 平均，再乘 `loss_weight["types"]`（默认 0.2）。
   - bond type: edge-wise CE，按 `adj_from_node_mask(mask, self_connect=True)` 平均，再乘 `loss_weight["bonds"]`（默认 1.0）。
   - charge: node-wise CE，按 atom mask 平均，再乘 `loss_weight["charges"]`（默认 1.0）。

4. **LIFT 主组合**
   - positive top 使用上述 per-sample loss。
   - negative bottom 对 discrete 用 soft cross entropy，对 coord 用 MSE 到 `target_minus`。
   - 四类变量最终按 `lfpo_lambda_coord_rect`, `lfpo_lambda_types_rect`, `lfpo_lambda_bonds_rect`, `lfpo_lambda_charges_rect` 组合。
   - 训练脚本的 history-only 配置中常见：`lambda_coord_rect=0.0`，`lambda_types/bonds/charges=1.0`，即主要优化离散结构，坐标 rectification 可被关闭；但坐标 positive/aux/anchor 逻辑仍存在。

## I. reference / current model 的使用方式

1. **current model**
   - `self.gen` 是被 optimizer 更新的策略模型。
   - 在 chunk loss 中，current model 对 `interp_data, t` 前向产生 logits/coords，并接收梯度。
   - 低频 current eval 使用 `self.gen` 采样，仅记录 reward/quality，不参与训练梯度。

2. **reference model**
   - `self.ref_gen` 是 frozen/no-grad 参考策略。
   - `on_fit_start()` 如果 `ref_gen` 为空，会 `deepcopy(self.gen)`，并设置 eval / `requires_grad=False`。
   - 每个 train step 的采样数据来自 `ref_gen`，不是 current，也不是 base EMA。
   - chunk loss 中 `ref_gen` 也对同一个 `interp_data,t` 前向，用于构造 plus/minus implicit targets 和 KL anchor。

3. **reference EMA 更新**
   - `on_train_batch_end()` 调 `_maybe_update_reference_ema()`。
   - 参数更新：`ref_param = ref_ema_decay * ref_param + (1-ref_ema_decay) * cur_param`。
   - buffer 直接复制 current buffer。
   - 训练脚本设置 `ref_ema_decay=0.9`；`LIFT_Lightning` 默认是 0.999。

4. **base EMA**
   - `LIFT_Lightning` 默认 `lfpo_disable_base_ema=True`，会禁用 `SC_Lightning` 的额外 `ema_gen` 以节省显存。
   - 因此 LIFT 明确使用 `_generate_with_model(model=...)`，避免 `SC_Lightning.forward(training=False)` 自动切换到 `ema_gen` 的歧义。

## J. 日志指标和 checkpoint 规则

1. **checkpoint**
   - `MolGen_LFPOModel.Train()` 创建 TensorBoard logger 到 `./TensorBoard/<project_name>/version_*`。
   - checkpoint 目录为 logger 下的 `checkpoints/`。
   - `ModelCheckpoint(save_top_k=5, every_n_train_steps=100, monitor="train-mpo-current-score-mean_step", mode="max", save_last=True)`。
   - 这意味着默认按 current-policy 低频评估的 scalar score mean 做 top-k checkpoint；如果关闭 current eval 或更改日志 key，需要同步调整 monitor。

2. **主训练 loss 日志**
   - `train-lfpof-pos-imitation-loss`
   - `train-lfpof-neg-repulsion-loss`
   - `train-lfpof-type-rect-loss`
   - `train-lfpof-bond-rect-loss`
   - `train-lfpof-charge-rect-loss`（有 charge head 时）
   - debug 或可选：`train-lfpof-main-loss`, `train-lfpof-coord-rect-loss`, `train-lfpof-aux-fm-loss`, `train-lfpof-anchor-loss`, `train-lfpof-total-loss`
   - delta diagnostics：`train-lfpof-delta-type-abs-mean`, `train-lfpof-delta-bond-abs-mean`, `train-lfpof-delta-coord-abs-mean`, `train-lfpof-delta-charge-abs-mean`

3. **reference generation / objective 日志**
   - quality: `train-gen-ref-validity`, `train-gen-ref-uniqueness`, `train-gen-ref-connected-validity`。
   - objective: `train-mpo-ref-score-mean`, `train-mpo-ref-score-top10-mean`。
   - partition: `train-partition-ref-feasible-frac`, `train-partition-ref-severe-frac`, `train-partition-ref-top-frac`, `train-partition-ref-bottom-frac`，以及 diagnostics 中的 `top-count`, `history-size-scaffold`, `history-excluded-*`, `bottom-reason-*` 等转换为 `train-partition-ref-*`。
   - components/raw properties: `train-mpo-ref-comp-{k}-mean/max`, `train-mpo-ref-raw-{k}-mean/min/max`。

4. **current-policy eval 日志**
   - reward: `train-lfpof-reward-current-mean/sem/max/top10-mean/top-mean/bottom-mean`。
   - quality: `train-gen-current-validity`, `train-gen-current-uniqueness`, `train-gen-current-connected-validity`。
   - objective checkpoint keys: `train-mpo-current-score-mean`, `train-mpo-current-score-sem`, `train-mpo-current-score-top10-mean`, `train-mpo-current-score-top1`。
   - components/raw properties: `train-mpo-current-comp-{k}-mean/max`, `train-mpo-current-raw-{k}-mean/min/max`。

5. **oracle CSV logging**
   - 如果 `metric_config.enabled=True`（通常由 `--oracle_log_path` 打开），`OracleLogger` 会把 batch scoring 和 partition 结果写到 CSV，用于 offline top-k/AUC/novelty/scaffold 统计。

## K. 迁移时必须保留的最小算法逻辑

迁移到 `flowr/` 时，必须保留以下算法语义，而不必逐行复制实现：

1. **reference-policy on-policy-ish sampling**
   - 每个 RL step 用 reference model 从当前 flow prior/noise/template 采样一批分子。
   - reference 是 current 的 EMA/frozen 版本，不是静态 dataset。

2. **RDKit/objective-based reward interface**
   - 生成结果必须能转 RDKit mol。
   - objective 必须输出 scalar score、component scores、valid/connected/feasible/severe、canonical smiles/scaffold/fingerprint。

3. **history-aware feasible top selection**
   - top 限制在 feasible 样本。
   - top 按 score 或 component-balanced score 排序。
   - history scaffold/fingerprint novelty 过滤必须保留，避免反复 imitation 同一 scaffold/fingerprint basin。

4. **bottom repulsion selection**
   - bottom 包含 invalid/severe/low-score 样本，且与 top 互斥。
   - bottom 用于 negative branch，而不是简单丢弃。

5. **top imitation + bottom implicit repulsion loss**
   - top：对 reference sampled pseudo-target 做 flow matching imitation。
   - bottom：用 current-vs-reference prediction delta 构造 minus target；discrete 用 soft CE，continuous 用 MSE。
   - middle 默认不参与主损失。

6. **四类变量分开建模和加权**
   - atom type、bond type、charge、coordinate 的 loss 必须可分别计算、记录、加权、关闭/开启。

7. **time sampling / surrogate FM training**
   - 对每个 reference sample 采样一个或多个 random/stratified time，在 `x_t` 上训练 surrogate loss，而不是只在 endpoint 训练。

8. **reference EMA 更新和 optional anchor**
   - reference 在每个 step 后以 EMA 追踪 current。
   - 可选 KL/MSE anchor 到 reference，用于限制 drift。

9. **默认关闭 / 不破坏原 flowr**
   - 迁移到 `flowr/` 时，所有 LIFT 功能必须通过配置显式打开，默认不影响原训练、采样、评估。

## L. 可以根据 flowr 架构重写的工程细节

以下属于工程适配层，可以重写：

1. **Lightning/manual optimization 形式**
   - 不必保持 Lightning；可按 flowr 的 trainer/loop 改写。
   - 关键是 chunk-wise backward 或等价显存控制，而不是具体 API。

2. **`MGDataModule` / `.smol` 数据格式**
   - 只需要 flowr 提供 noise/template、mask、atom count、初始 ligand state 的等价 batch。
   - 如果 flowr 是 structure-based，需要把 pocket/condition 保留到 current/reference 前向和采样流程中。

3. **网络类与 forward 签名**
   - 不必迁移 `DenoisingNet` / `EquiInvDynamics`。
   - 只要能在任意 time `t` 输入 noisy/interpolated molecule state，输出 coordinate prediction、atom logits、bond logits、charge logits或 flowr 的等价变量。

4. **采样 integrator**
   - 可复用 flowr 原有 ODE/SDE/discrete sampler。
   - 需要能显式选择 current vs reference model，且输出可重建的 molecule representation。

5. **RDKit builder 实现**
   - 可使用 flowr 自带 molecule reconstruction / sanitization / validity utils。
   - 只需保证 objective 接口拿到 RDKit mol，并且能取 canonical smiles、scaffold、fingerprint。

6. **日志后端**
   - TensorBoard key 可以映射到 flowr 的 logger；checkpoint monitor 也可改成 flowr 的 callback 配置。
   - 但建议保留核心 key 的语义，方便对照迁移前后曲线。

7. **objective 工程实现**
   - xTB timeout/parallel worker、GuacaMol official scorer、SA scorer fallback 等可按 flowr 依赖环境重写。
   - 需要保留同名 score/component/raw/feasibility 语义。

## M. 后续迁移到 flowr 时必须寻找的接口清单

迁移前应在 `flowr/` 中定位下列接口，并形成映射文档：

1. **训练循环接口**
   - 如何自定义一个训练 step？
   - 是否支持 manual optimization / gradient accumulation / chunk backward？
   - checkpoint monitor 如何配置？

2. **模型 forward 接口**
   - 输入 noisy ligand state、time `t`、pocket/structure condition 的方式。
   - 输出是否包含 atom logits、bond logits、charge logits、coordinate velocity/endpoint。
   - 如何显式调用 current model 与 frozen reference model。

3. **采样接口**
   - 如何从 prior/noise/template 采样 ligand？
   - 如何指定 inference steps、coordinate noise、categorical noise。
   - 如何禁用/启用 EMA，避免 current/reference/EMA 混淆。

4. **数据 batch 表示**
   - ligand coords、atom types、bond types、charges、masks、atom counts 的 tensor 名称和 shape。
   - pocket/condition tensor 如何随 batch slicing 和 generated target batch 保留。
   - 是否存在 `flag_3Ds` 或等价的 coordinate-valid mask。

5. **interpolation / noising 接口**
   - flowr 是否已有 flow matching interpolation。
   - 离散变量如何 noising/interpolate：categorical bridge、mask token、uniform sample，还是 diffusion schedule。
   - continuous coordinate target 是 endpoint 还是 velocity。

6. **loss 接口**
   - 原始 flowr 的 atom/bond/charge/coord loss 如何计算和加权。
   - 是否已有 per-sample loss；若没有，需要新增不影响默认训练的 per-sample variant。
   - bond mask 是否包含 self connection，需与 flowr 表示对齐。

7. **molecule reconstruction / validity**
   - tensor -> RDKit mol 的函数在哪里。
   - bond order、formal charge、aromatic bond、explicit H 的映射。
   - sanitization、connectedness、canonical smiles、scaffold、fingerprint 工具。

8. **reward/objective 插件点**
   - 训练中能否调用 Python/RDKit/xTB objective。
   - 多进程/多线程 xTB 在 flowr dataloader/trainer 环境中是否安全。
   - objective 结果如何 broadcast / sync in DDP。

9. **reference model 生命周期**
   - 在 flowr 中如何 deepcopy/load reference。
   - 如何冻结 reference、设置 eval、同步 device/dtype。
   - EMA update 应放在 optimizer step 后的哪个 hook。

10. **logging / metrics / oracle history**
    - flowr logger key 命名和 checkpoint callback。
    - 是否已有 generated mol metrics；若有，如何复用。
    - CSV oracle log 的存放路径、rank-zero-only 写入、DDP 聚合规则。

11. **默认关闭配置**
    - 需要一个总开关，例如 `lift.enabled=false`。
    - 所有 objective/partition/reference/current eval/anchor 参数都必须在 flowr config 中显式配置，不应影响原 flowr baseline。

## 附：主要 CLI 参数和配置项摘要

1. **脚本 CLI**
   - `--config`: 默认 `rl.json`。
   - `--objective_name`: 选择 reward/objective。
   - similarity 任务：`--target_smiles`, `--similarity_fp`, `--similarity_radius`, `--similarity_n_bits`, `--similarity_threshold`。
   - 训练：`--epochs`, `--batchsize`, `--project_name`。
   - partition：`--partition_mode`, `--history_diversity_mode`, `--history_fingerprint_threshold`, `--history_max_size`, `--top_selection_score_mode`, `--use_min_component_bonus`, `--disable_component_floor`。
   - logging：`--oracle_log_path`。
   - xTB / expensive reward 脚本还包含 `--xtb_method`, `--xtb_force_norm`, `--xtb_max_workers`, `--xtb_timeout`, `--disable_lfpo_current_eval`, `--lfpo_eval_current_every`, `--lfpo_eval_current_samples` 等。
   - Perindopril ready 脚本还包含 template atom-count filtering 参数，用于控制 noise/template 分子大小范围。

2. **常用 LIFT hyperparameters**
   - time/reward：`lfpo_num_time_samples`, `lfpo_reward_temperature`, `lfpo_time_chunk_size`。
   - discrete/continuous beta：`lfpo_beta_types`, `lfpo_beta_bonds`, `lfpo_beta_charges`, `lfpo_beta_coord`, `lfpo_gamma_coord`。
   - variable lambdas：`lfpo_lambda_coord_rect`, `lfpo_lambda_types_rect`, `lfpo_lambda_bonds_rect`, `lfpo_lambda_charges_rect`。
   - branch weights：`lfpo_top_ratio`, `lfpo_bottom_ratio`, `lfpo_bottom_repulsion_weight`, `lfpo_middle_weight`, `lfpo_top_weight_mode`, `lfpo_use_top_bottom`。
   - reference/anchor：`ref_ema_decay`, `anchor_weight`, `use_reference_anchor`, `regularization_type`。
   - eval/log：`lfpo_log_current_reward`, `lfpo_eval_current_every`, `lfpo_eval_current_samples`, `lfpo_eval_current_batch_size`, `log_debug_metrics`。

## 最小 smoke test / 验证建议

本阶段只生成分析文档，不运行训练。后续迁移实现后建议至少保留以下 smoke checks：

1. **objective-only smoke**：给 2-3 个 RDKit mol 调用目标 objective，确认 `ScoringResult` 字段、shape、device、dtype 正确。
2. **partition smoke**：构造伪 `ScoringResult`，验证 feasible top、history scaffold/fingerprint 排除、invalid/severe/low-score bottom 优先级。
3. **loss smoke**：构造小 batch，current/ref logits 与 coords 随机，验证 top imitation、bottom repulsion、middle zero、四类 loss shape 为 `[B]` 且总 loss 可 backward。
4. **sampling/reconstruction smoke**：从 flowr sampler 生成少量 mol，能转 RDKit，validity/connectedness 不报错。
5. **reference EMA smoke**：一次 optimizer step 后 reference 参数按 EMA 更新且 `requires_grad=False`。
