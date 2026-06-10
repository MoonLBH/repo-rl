# 03. sotmol-rl → flowr 接口映射设计

> 范围：本文基于 `migration_notes/01_sotmol_rl_algorithm_spec.md`、`02a_flowr_original_usage_guide.md`、`02b_flowr_architecture_map.md` 设计 LIFT / LFPO-F 风格 reward-guided fine-tuning 迁移到 FLOWR 的接口映射。本文只做迁移设计，不修改 `sotmol-rl/` 或 `flowr/` 代码。
>
> 核心原则：迁移的目标不是复制 `sotmol-rl`，而是在 FLOWR 的 pocket-conditioned ligand generation 表示、采样器、训练 loop 与 evaluation 链路上复现“reference sampling → reward/objective → top/bottom selection → flow-matching surrogate update”的算法语义。

## A. 总体迁移判断

### A1. 可以直接抽象迁移的部分

| 逻辑 | 迁移判断 | 原因 |
|---|---|---|
| reward/objective 抽象 | 可迁移，但实现要放到 flowr 数据流里 | `sotmol-rl` 的 `ScoringResult`/objective abstraction 与 FLOWR RDKit mol evaluation 工具兼容；reward 本质不依赖无条件模型。 |
| top/middle/bottom 选择策略 | 可迁移 | 该逻辑只依赖 score、component scores、valid/connected/feasible、canonical SMILES/scaffold/fingerprint；可用于 pocket-conditioned generated ligands。 |
| history scaffold/fingerprint novelty memory | 可迁移 | 对 generated ligand 的 scaffold/fingerprint 做历史过滤，与是否 pocket-conditioned 无关。 |
| reference/current model 框架 | 可迁移但需重写生命周期 | LIFT 的 reference EMA 语义通用；FLOWR 需要显式区分 Lightning EMA callback、current generator 与新 reference generator。 |
| 不可微 reward 不直接反传、使用 surrogate loss | 可迁移 | FLOWR 同样可对 sampled endpoints 做 FM surrogate，不需要 reward gradient。 |
| top imitation + bottom repulsion 思想 | 可迁移 | 适合把高 reward pocket-conditioned ligands 作为 pseudo-target，把低 reward/invalid ligands 作为 repulsion 样本。 |
| logging key 语义 | 可迁移 | 可保留 `train-rl-*`, `train-lift-*`, `train-reward-*`, `train-partition-*` 等语义，具体 logger 用 FLOWR Lightning logger。 |

### A2. 需要适配 FLOWR 后才能迁移的部分

| 逻辑 | 需要适配原因 | FLOWR 适配方向 |
|---|---|---|
| sampling in training step | `sotmol-rl` 是无条件 noise batch；FLOWR 是 pocket-conditioned complex batch | 从 FLOWR batch 的 `prior/data/interpolated/times` 中提取 ligand prior 与 pocket_data，调用 explicit reference generator conditioned on same pocket。 |
| generated target batch | `sotmol-rl` 只需 molecule target；FLOWR 必须保留 pocket condition、lig/pocket masks、COM alignment、fragment masks/interactions | 新增 pocket-aware pseudo-target 构造函数，只替换 ligand target，不破坏 pocket data。 |
| model forward helper | `sotmol-rl` 可 `_forward_with_model(model, data, t)`；FLOWR `forward()` 当前绑定 `self.gen` | 新增 `_forward_ligand_with_gen(gen, lig_batch, pocket_batch, times, ...)` 或 subclass 中显式 model 参数 helper。 |
| per-sample loss | FLOWR 原 `_loss()` 返回 batch-mean scalar | 新增 per-sample `coord/type/bond/charge/interaction` loss helper，用于 top/bottom mask 加权。 |
| reference EMA | FLOWR 原 `--use_ema` 是 callback，不是 LIFT reference model | 新增 `self.ref_gen`，独立于 baseline EMA callback；RL enabled 时可选择关闭 callback EMA 或明确只用于 validation。 |
| coordinate/COM handling | FLOWR complex 数据有 COM 移动与 undo；sotmol 无 pocket COM | reward 前必须使用 `_generate_ligs()` 或同等流程 undo COM，loss 内部使用训练坐标空间。 |
| interactions/inpainting | FLOWR 可 flow/predict interactions、fragment/scaffold/linker inpainting | MVP 可以不优化 interaction head，但必须保留字段和条件；后续再加入 interaction reward/loss。 |
| DDP behavior | `sotmol-rl` history selector 未必直接适配 FLOWR 多 GPU sampling | MVP 先单 GPU；多 GPU 需 rank-local history 或 all-gather scoring/selection。 |

### A3. 不适合迁移或暂时不迁移的部分

| 逻辑 | 暂不迁移原因 | 建议 |
|---|---|---|
| 逐字复制 `sotmol-rl` LightningModule | FLOWR batch、condition、forward、sampling、EMA 都不同 | 只迁移算法语义；新增 FLOWR-native subclass/wrapper。 |
| `sotmol-rl` 无条件 `MGDataModule` / `.smol` batch 构造 | 与 FLOWR `PocketComplexBatch` 不兼容 | 使用 FLOWR 原 `GeometricInterpolantDM`。 |
| 训练中频繁 docking / PoseBusters / GenBench3D | 外部依赖复杂、速度慢、不可作为高频 online reward | 初期只作为 offline/low-frequency evaluation；reward 层预留接口。 |
| xTB force residual 作为默认 reward | 速度慢且失败/timeout 处理复杂 | 作为后续扩展，不进入 MVP。 |
| 修改 FLOWR 原始 scripts 默认参数 | 会污染 baseline fair comparison | 新增 RL scripts 或 CLI flags，默认关闭。 |
| 改写原 `LigandPocketCFM._loss()` 默认行为 | 会改变 baseline training | 新增 per-sample helper；baseline `_loss()` 不变。 |

## B. sotmol-rl 到 flowr 的模块映射表

| sotmol-rl 模块或逻辑 | sotmol-rl 文件/函数 | flowr 最适合接入位置 | 直接迁移？ | 需要重写？ | 输入输出差异 | 风险点 |
|---|---|---|---|---|---|---|
| LIFT 训练入口脚本 | `train_lift_history_only*.py` | 新增 `flowr/scripts/train_spindr_lift.sh`；`flowr.train` 新增 default-off CLI | 否 | 是 | sotmol CLI 面向无条件 model；FLOWR CLI 需保留 dataset/pocket/ckpt/eval params | 改动原脚本会污染 baseline。 |
| Model wrapper | `MolGen_LFPOModel` | `flowr.train::build_model()` 中按 `--lift_enabled` 选择新 class | 否 | 是 | sotmol wrapper 构造 DenoisingNet；FLOWR 已有 PocketEncoder/LigandGenerator | checkpoint hparams 兼容性。 |
| LightningModule | `LIFT_Lightning` | 新增 `flowr/flowr/models/lift_pocket.py::LigandPocketLIFT` 继承 `LigandPocketCFM` | 否 | 是 | sotmol batch 是 molecule-only；FLOWR batch 是 complex tuple | 直接 override 原 class 会破坏 baseline。 |
| Reference model 初始化 | `LIFT_Lightning.on_fit_start()` | `LigandPocketLIFT.on_fit_start()` | 思想可迁移 | 是 | sotmol `self.gen` 无 pocket encoder 缓存；FLOWR `self.gen` 包含 ligand generator + pocket path | Lightning EMA callback 与 ref EMA 混淆。 |
| Reference EMA 更新 | `_maybe_update_reference_ema()` | `LigandPocketLIFT.on_train_batch_end()` 或 optimizer-step hook | 思想可迁移 | 小改 | 需更新 `ref_gen` params/buffers；不应更新 `PocketEncoder` 是否冻结需明确 | EMA 时机必须在 optimizer step 后。 |
| Reference sampling | `_generate_with_model(self.ref_gen, noise, ...)` | 新增 `_generate_with_gen(gen, lig_prior, pocket_data, times, ...)` | 否 | 是 | FLOWR sampling 需 pocket_data、lig/pocket times、inpainting、COM | 采样慢；训练 step 内 OOM。 |
| Current low-frequency eval | `_evaluate_model_scoring()` | 新增 `evaluate_current_policy_reward()` | 思想可迁移 | 是 | FLOWR 按 pocket batch 采样若干 ligands | 多 target/multi GPU 聚合复杂。 |
| RDKit reconstruction | `_generate_mols()` + `MolBuilder` | 复用 `LigandPocketCFM._generate_mols()` / `_generate_ligs()` | 部分 | 小改 | FLOWR 必须 undo COM 并按 ligand mask 提取 | invalid mol 比例影响 reward/selection。 |
| Objective classes | `rl_objectives/mpo_tasks.py` | 新增 `flowr/flowr/rl/objectives.py` 或 `flowr/flowr/util/reward_objectives.py` | 部分 | 是 | 输入 RDKit Mol list + optional pocket/ref info；输出 FLOWR-native RewardResult | Heavy objectives 不能高频。 |
| `ScoringResult` | `mpo_tasks.py::ScoringResult` | 新增 `RewardResult` dataclass | 是（结构） | 小改 | 加入 `target_ids`, `pocket_paths`, `ligand_indices`, optional structure fields | Device/dtype 与 CPU RDKit list 混合。 |
| Partition selector | `partition_history_only.py::PartitionSelector` | 新增 `flowr/flowr/rl/partition.py` | 思想可迁移 | 小改 | 需要 target-aware history 可选：global vs per-pocket | 多 pocket 多目标下 history 粒度。 |
| Positive top imitation | `_loss_per_sample()` on pseudo-target | `LigandPocketLIFT._loss_per_sample()` on generated ligand target | 否 | 是 | FLOWR loss 需 ligand-only target + pocket condition | 原 `_loss()` scalar，必须新增 per-sample。 |
| Negative discrete repulsion | `_lfpof_discrete_loss_per_sample()` | `LigandPocketLIFT._lift_discrete_loss_per_sample()` | 是（公式） | 小改 | atom/bond/charge logits shape 类似；bond mask 使用 FLOWR mask | logits/probs 是否来自 same time 与 same cond。 |
| Negative coordinate repulsion | `_lfpof_coord_loss_per_sample()` | `LigandPocketLIFT._lift_coord_loss_per_sample()` | 是（公式） | 小改 | FLOWR coordinate in COM-normalized training frame；reward mol in real frame | coord lambda 默认可能先置 0。 |
| Anchor KL/MSE | `_anchor_from_reference_per_sample()` | `LigandPocketLIFT._anchor_from_reference_per_sample()` | 部分 | 是 | FLOWR optional interaction head、pocket condition | Anchor 太强可能抵消 RL。 |
| Time sampling | `_sample_stratified_timesteps()` | `LigandPocketLIFT._sample_lift_timesteps()` | 是 | 小改 | FLOWR expects list `[lig_cont, lig_disc, pocket, interaction]` | pocket time 应固定或与 baseline一致。 |
| Interpolation | `SC_Lightning.interpolate()` | FLOWR `ComplexInterpolant` 或 helper 从 generated target 构造 interpolated ligand | 否 | 是 | FLOWR interpolation happens in datamodule; training step内需可重用或手写 ligand-only interpolation | 重复实现要匹配 baseline categorical strategy。 |
| Logging | `train-lfpof-*`, `train-mpo-*` | Lightning `self.log()` in new class | 是（语义） | 小改 | Add `train-rl-*`, `train-lift-*`, `train-reward-*` | checkpoint monitor key 要 default-off。 |
| Oracle CSV | `OracleLogger` | 新增 optional CSV/JSON logger | 部分 | 是 | FLOWR generated result includes target/pocket metadata | DDP rank-zero writing。 |
| Checkpoint monitor | `train-mpo-current-score-mean_step` | RL script-specific monitor e.g. `train-rl-current-score-mean` | 否 | 是 | FLOWR baseline uses `val-fc-validity` | 不应改 baseline monitor。 |
| Flowr evaluation | N/A | Reuse `flowr.eval.evaluate_metrics`, `evaluate_interactions` | N/A | 不改 | RL outputs must match original sampling output format | 若输出格式不同，evaluation 不能复用。 |

## C. flowr 原始 baseline 保留策略

### C1. 原始训练命令保留

- 保留 `flowr/scripts/train_spindr.sh` 和 `python -m flowr.train` baseline 参数不变。
- `flowr.train` 默认仍构建 `LigandPocketCFM`，不启用 reward、selection、reference sampling 或 RL loss。
- 如果新增 CLI：`--lift_enabled` 默认 `False`；没有该 flag 时训练路径应与原始代码一致。

### C2. 原始采样命令保留

- 保留 `flowr/scripts/gen_spindr.sl`、`flowr/scripts/gen_pdb.sl`、`python -m flowr.gen.generate_from_smol`、`python -m flowr.gen.generate_from_pdb`。
- RL 训练后 checkpoint 也应可被原始 sampling entry 加载，或者新增 RL sampling wrapper 输出与原始 `predictions_multi_*.pt` 相同结构。
- 不改变 `generate_from_smol.py` / `generate_from_pdb.py` 的默认输出结构。

### C3. 原始 evaluation 命令保留

- 保留 `flowr/scripts/eval_spindr.sh`、`evaluate_metrics.py`、`evaluate_interactions.py`。
- RL-flowr 采样输出必须能直接被原 evaluation scripts 读取。
- 不改变 metrics key、`metrics.pt`、`interaction_recovery*.pt` 命名语义。

### C4. 新增 RL 逻辑默认关闭

推荐 CLI/config：

```bash
--lift_enabled                 # action='store_true', default False
--lift_objective_name qed      # 只有 lift_enabled 时生效
--lift_top_ratio 0.25
--lift_bottom_ratio 0.25
--lift_ref_ema_decay 0.99
--lift_aux_fm_weight 0.0
--lift_current_eval_every 0
```

默认关闭策略：

- parser 默认 `lift_enabled=False`。
- `build_model()` 中：`if not args.lift_enabled: return LigandPocketCFM(...)`。
- 新增 imports 尽量放在 branch 内，避免 baseline import side effects。
- baseline scripts 不添加任何 RL flag。

### C5. 避免污染 baseline

- 不修改 `LigandPocketCFM.training_step()` 原逻辑；新增 subclass override。
- 不修改原 `_loss()` 返回值和权重行为；新增 `_loss_per_sample()`。
- 不修改原 sampling/evaluation script 默认行为。
- 不改变 `genbench3d/config/default.yaml`、`environment.yml`、原 bash scripts。
- 所有 RL logs/checkpoints 进入单独 `save_dir` / `exp_name`。

### C6. fair comparison 方案

1. 使用同一 SPINDR `test.smol` split。
2. 使用同一 pretrained FLOWR checkpoint 作为 init，或同一 from-scratch baseline training protocol。
3. 使用同一 sampling command 参数：`integration_steps`, `ode_sampling_strategy`, `sample_n_molecules_per_target`, `sample_mol_sizes`, `filter_valid_unique`, `seed`。
4. 使用同一 evaluation commands：`evaluate_metrics.py` + `evaluate_interactions.py`。
5. 保存 original 与 RL 的 checkpoint path、sample dir、metrics.pt、interaction_recovery.pt、reward CSV、selection CSV。

## D. reward/objective 接口设计

### D1. reward 输入

MVP 推荐 reward 输入为 **RDKit Mol list + optional context**：

```python
score_ligands(
    mols: list[Chem.Mol | None],
    context: RewardContext | None = None,
    device=None,
    dtype=torch.float32,
) -> RewardResult
```

理由：

- QED、SA、similarity、validity、scaffold、fingerprint 都以 RDKit Mol 为自然输入。
- FLOWR 已有 generated tensor → RDKit Mol 的链路。
- Structure-based reward 可通过 `RewardContext` 补充 pocket PDB、reference ligand、target id、protein path、SDF path。

可选 context：

```python
@dataclass
class RewardContext:
    target_ids: list[str]
    pocket_pdbs: list[str | None] = None
    pocket_pdbs_with_hs: list[str | None] = None
    reference_ligands: list[Chem.Mol | None] = None
    reference_ligands_with_hs: list[Chem.Mol | None] = None
    raw_complex_objects: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### D2. flowr 生成结果到 reward 输入的转换链路

Training 内链路：

```text
FLOWR batch (prior, data, interpolated, times)
  -> extract ligand prior + pocket_data
  -> reference generator sampling conditioned on pocket
  -> generated ligand tensor in FLOWR coordinate frame
  -> _generate_ligs(generated, lig_mask, scale=coord_scale)
  -> RDKit Mol list after undo COM
  -> objective.score_ligands(mols, RewardContext)
  -> RewardResult
```

Offline/dry-run 链路：

```text
predictions_multi_*.pt or samples_<target>.pt
  -> read out_dict['gen_ligs']
  -> flatten or keep per-target list
  -> objective.score_ligands(...)
  -> selection
  -> CSV/JSON/SDF reports
```

### D3. MVP reward

MVP 支持轻量 reward：

- validity / connected validity。
- QED。
- SA。
- QED-SA weighted/geometric aggregate。
- target/reference ligand similarity（Morgan/AP/RDKit fingerprint Tanimoto）。
- optional novelty/scaffold diversity against history。

### D4. 预留 structure-based reward

暂不作为高频默认训练 reward，但接口预留：

- docking score / Vina score。
- protein-ligand interaction recovery / PLIF score。
- pharmacophore score。
- PoseBusters pass rate / violation penalty。
- PoseCheck clash/strain/interactions。
- strain energy。
- xTB force residual / force RMS。
- pocket-specific constraints，如距离 pocket centroid、clash penalty。

### D5. reward dict 推荐格式

```python
@dataclass
class RewardResult:
    score: torch.Tensor                         # [N], main scalar reward, high is better
    component_scores: dict[str, torch.Tensor]   # [N] each, normalized when possible
    raw_properties: dict[str, torch.Tensor]      # [N] each, raw units allowed
    feasible: torch.BoolTensor                  # [N]
    severe_violation: torch.BoolTensor          # [N]
    valid: torch.BoolTensor                     # [N]
    connected: torch.BoolTensor                 # [N]
    smiles: list[str | None]
    canonical_smiles: list[str | None]
    scaffolds: list[str | None]
    fps: list[Any]
    mols: list[Chem.Mol | None]
    target_ids: list[str | None]
    metadata: dict[str, Any]
```

Invalid mol handling：

- `mol is None` or RDKit sanitize fails → `valid=False`, `connected=False`, `feasible=False`。
- default `score=0.0` for bounded rewards。
- `severe_violation=True` for invalid/disconnected or external scoring failure when configured。
- canonical smiles/scaffold/fp 为 `None`。
- raw failure flags，如 `raw_properties['xtb_success']=0`。

### D6. reward 结果保存

- Dry-run/offline：保存 `reward_results.csv`, `selection_results.csv`, `selected_top.sdf`, `selected_bottom.sdf`, `reward_summary.json`。
- Training：可选 rank-zero CSV `train_reward_oracle.csv`，包含 `global_step`, `target_id`, `sample_idx`, `score`, components, raw properties, valid/connected, top/bottom/middle label。
- Heavy reward：保存 cache，例如 `reward_cache.sqlite` 或 CSV，以避免重复 docking/xTB。

## E. sampling-reward-selection 链路设计

### E1. 调用 FLOWR 原始 sampling

Dry-run 推荐优先复用原始 sampling 输出，而非一开始接入训练：

```bash
python -m flowr.gen.generate_from_smol ... --save_dir /path/to/samples
```

然后新增 default-off/独立命令：

```bash
python -m flowr.rl.score_and_select \
  --predictions_dir /path/to/samples \
  --objective_name qed_sa \
  --top_ratio 0.25 \
  --bottom_ratio 0.25 \
  --history_diversity_mode scaffold \
  --out_dir /path/to/reward_selection
```

### E2. 读取生成结果

- For SPINDR sampling：读取 `predictions_multi_*.pt` 或 `predictions_multi_valid_unique_*.pt`。
- For PDB sampling：读取 `samples_<target>.pt`。
- 输出里的 `gen_ligs` 可能是 list[list[Mol]] 或 list[Mol]；normalize 成 records：

```python
record = {
    'target_id': str,
    'target_index': int,
    'sample_index': int,
    'mol': Chem.Mol | None,
    'ref_lig': Chem.Mol | None,
    'ref_pdb': str | None,
}
```

### E3. 重建 ligand

- Offline 不需要从 tensor 重建，直接用 `.pt` 中 RDKit Mol。
- Training 中使用 `LigandPocketCFM._generate_ligs()` 或等价 helper，从 generated tensor 重建 ligand RDKit Mol。
- 如果要输出 SDF，使用 `flowr.util.rdkit.write_sdf_file()`。

### E4. reward 计算

- 调 objective：`RewardResult = objective.score_ligands(mols, context)`。
- 轻量 reward 可一次 batch 计算。
- heavy reward 按 target/mol 分块，并带 timeout/cache。

### E5. top/middle/bottom selection

- 输入 `RewardResult`。
- 输出 `PartitionResult`：`top_mask`, `bottom_mask`, `selected_mask`, `top_weights`, `bottom_weights`, `diagnostics`, `reasons`。
- 默认 top：feasible + high score + history novelty。
- 默认 bottom：invalid/severe 优先，然后 low score。
- middle：非 top/bottom。
- 可选 selection granularity：
  - `global`: 所有 generated ligands 一起选。
  - `per_target`: 每个 pocket/target 内选 top/bottom（更适合 structure-based fair selection）。
  - MVP 推荐 `per_target` for dry-run reporting；training step 内自然是当前 batch/pocket集合。

### E6. 输出 JSON/CSV/SDF

```text
<out_dir>/
  reward_results.csv
  selection_results.csv
  reward_summary.json
  top_ligands.sdf
  bottom_ligands.sdf
  middle_ligands.sdf               # optional
  selected_records.pt              # optional, for training replay/debug
```

CSV columns：`target_id`, `sample_index`, `smiles`, `score`, component columns, raw columns, `valid`, `connected`, `feasible`, `partition`, `reason`, `rank`, `scaffold`。

### E7. 用于后续 RL fine-tuning

Dry-run selection 的作用：

1. 验证 objective 与 partition 对 FLOWR generated mol 有合理输出。
2. 估计 invalid rate、reward distribution、top/bottom 数量。
3. 决定 RL 训练的 reward scaling、top/bottom ratio、history mode。
4. 可作为 replay/debug 数据，但 MVP 的 online RL 应仍从 reference model 当前采样，而不是只重放固定 offline samples。

## F. RL / reward-guided loss 接入设计

### F1. sotmol-rl surrogate 是否能直接迁移

不能逐行直接迁移，但可以迁移最小等价语义。

不能直接迁移原因：

- `sotmol-rl` 无条件 molecule-only；FLOWR 是 pocket-conditioned complex。
- FLOWR `forward()` 绑定 `self.gen`，需要 explicit model helper。
- FLOWR `_loss()` scalar mean，不支持 per-sample top/bottom。
- FLOWR interpolation 由 datamodule `ComplexInterpolant` 完成，training step 内构造 generated pseudo-target 需重写。
- FLOWR coordinate COM/scale 与 ligand/pocket masks 复杂。

### F2. FLOWR 最小等价 surrogate

MVP surrogate：

1. 使用 reference `ref_gen` 对当前 batch pocket condition 采样 generated ligands。
2. 计算 reward 与 partition。
3. 构造 generated ligand pseudo-target。
4. 在 `K=1` 或 `K=2` 个 ligand time 上构造 interpolated ligand state。
5. current `self.gen` 与 `ref_gen` 对相同 `lig_interp, pocket_data, times` 前向。
6. top 使用 per-sample original FLOWR ligand losses：coord/type/bond/charge CE/MSE imitation。
7. bottom 使用 LIFT implicit minus targets：
   - atom/bond/charge logits: `logp_minus = logp_ref - beta * (logp_cur - logp_ref)`。
   - coords: `target_minus = pred_ref - gamma * (pred_cur - pred_ref)`。
8. middle 默认权重 0。
9. 总损失：

```text
loss = lift_main_loss
     + lift_aux_fm_weight * original_fm_loss_on_generated_targets
     + lift_anchor_weight * reference_anchor_loss
     + optional baseline_fm_weight * original_flowr_loss_on_real_batch
```

推荐 MVP defaults：

- `lift_enabled=False`。
- When enabled：`lift_baseline_fm_weight=0.0` initially for pure LIFT smoke, or `0.1` if stability needed；需实验确认。
- `lift_lambda_coord=0.0` initially optional, because coordinate reward signal and COM alignment may be fragile；atom/bond/charge first。
- `lift_num_time_samples=1` for smoke，后续 `2`。

### F3. loss 接入位置

新增 class：`LigandPocketLIFT(LigandPocketCFM)`。

- Override `training_step()`。
- Keep `validation_step()` inherited unless RL-specific validation needed。
- Reuse `configure_optimizers()` inherited。
- Add helpers:
  - `_build_lift_noise_from_batch()`。
  - `_generate_with_gen()`。
  - `_build_generated_ligand_target()`。
  - `_interpolate_generated_ligand_target()`。
  - `_forward_with_gen()`。
  - `_loss_per_sample()`。
  - `_lift_discrete_loss_per_sample()`。
  - `_lift_coord_loss_per_sample()`。
  - `_anchor_from_reference_per_sample()`。

### F4. reference model 与冻结策略

- Need reference model: yes。
- `ref_gen = deepcopy(self.gen)` on fit start。
- `ref_gen.eval()` and `requires_grad=False`。
- EMA update after optimizer step：`ref = decay * ref + (1-decay) * current`。
- Pocket encoder是否冻结：
  - MVP: reference copy includes full `self.gen` including pocket encoder; current full model trainable。
  - Optional stability: freeze pocket encoder during RL (`--lift_freeze_pocket_encoder=True`) because reward targets ligand quality and pocket condition encoder already pretrained。
  - Do not freeze by default unless experiments show instability；but expose flag default `False`。

### F5. 原始 FLOWR loss 与 RL loss 组合

Recommended flags：

- `--lift_main_weight 1.0`
- `--lift_aux_fm_weight 0.0`
- `--lift_anchor_weight 0.1`
- `--lift_baseline_fm_weight 0.0`
- `--lift_lambda_types 1.0`
- `--lift_lambda_bonds 1.0`
- `--lift_lambda_charges 1.0`
- `--lift_lambda_coords 0.0` initially

If `lift_baseline_fm_weight > 0`：

```text
baseline_fm_loss = original _loss(data, interpolated, predicted_on_real_batch)
total = lift_main + aux + anchor + baseline_fm_weight * baseline_fm_loss
```

### F6. 避免不可微 reward 直接反传

- Reward computation must run under `torch.no_grad()` and on RDKit mols。
- Reward tensors used only for mask/weight selection, not gradient path。
- `top_mask`, `bottom_mask`, `top_weights` detached。
- Current/reference generated samples for scoring are sampled no-grad。
- Surrogate gradients only flow through current model predictions at sampled time states。

### F7. train-rl 日志

Recommended logs：

- Reward：`train-rl-reward-ref-mean`, `max`, `top10-mean`, `top-mean`, `bottom-mean`。
- Selection：`train-rl-partition-top-frac`, `bottom-frac`, `feasible-frac`, `invalid-frac`, `history-excluded-frac`。
- Loss：`train-rl-main-loss`, `train-rl-pos-imitation-loss`, `train-rl-neg-repulsion-loss`, `train-rl-type-loss`, `train-rl-bond-loss`, `train-rl-charge-loss`, `train-rl-coord-loss`, `train-rl-anchor-loss`, `train-rl-aux-fm-loss`, `train-rl-baseline-fm-loss`, `train-rl-total-loss`。
- Current eval：`train-rl-current-score-mean`, `train-rl-current-score-top10-mean` only if low-frequency eval enabled。
- Components：`train-rl-comp-{name}-mean`, `train-rl-raw-{name}-mean`。

## G. 数据和实验对比设计

### G1. 使用 FLOWR 原始数据集时保持不变

- Same SPINDR `train.smol`, `val.smol`, `test.smol`。
- Same `processed/` statistics files。
- Same `train_mols.pkl` for novelty。
- Same `--remove_hs` setting。
- Same `--pocket_noise`, `--arch`, model size, categorical strategy。
- Same pretrained checkpoint or same training seed and baseline protocol。
- Same `genbench3d_data`, ADFR/Vina config for evaluation。

### G2. baseline 与 RL-flowr 采样

Baseline：

```bash
python -m flowr.gen.generate_from_smol ... --ckpt_path baseline.ckpt --save_dir baseline_samples
```

RL-flowr：

```bash
python -m flowr.gen.generate_from_smol ... --ckpt_path rl_flowr.ckpt --save_dir rl_samples
```

所有 sampling 参数一致：

- `dataset_split=test`
- `sample_n_molecules_per_target`
- `integration_steps`
- `ode_sampling_strategy`
- `sample_mol_sizes`
- `coord_noise_std`
- `filter_valid_unique`
- `seed`

### G3. 相同输入

- Same target pocket list from `test.smol`。
- Same PDB/CIF + ligand files for single-target experiments。
- Same pocket cutoff for PDB sampling。
- Same reference ligand for similarity/interaction comparisons。

### G4. 需要记录的评价指标

Minimum：

- validity / fc-validity。
- uniqueness。
- QED、SA、LogP、TPSA、Lipinski、rings。
- novelty。
- strain energy / opt-RMSD。
- GenBench3D validity metrics。
- PoseBusters validity。
- PoseCheck metrics。
- PLIF recovery / Tanimoto similarity。
- Vina/SBDD metrics when external dependencies are configured。
- RL reward distribution and top/bottom fractions。

### G5. 需要保存的中间文件

For each experiment：

```text
experiment_root/
  config_or_command.txt
  checkpoint.ckpt
  samples/
    predictions_multi_*.pt
    optional SDFs
    ref_pdbs/
  evaluation/
    metrics.pt
    interaction_recovery*.pt
  rl_debug/                       # only RL-flowr
    reward_results.csv
    selection_results.csv
    top_ligands.sdf
    bottom_ligands.sdf
    reward_summary.json
    train_reward_oracle.csv
```

## H. 新增文件、函数、配置参数清单

### H1. 建议新增文件

| 文件 | 目的 |
|---|---|
| `flowr/flowr/rl/__init__.py` | RL utilities package。 |
| `flowr/flowr/rl/objectives.py` | Reward objective classes and `RewardResult`。 |
| `flowr/flowr/rl/partition.py` | Top/middle/bottom selection and history memory。 |
| `flowr/flowr/rl/logging.py` | Optional CSV/JSON reward logger。 |
| `flowr/flowr/rl/score_and_select.py` | Offline sampling → reward → selection dry-run CLI。 |
| `flowr/flowr/models/lift_pocket.py` | `LigandPocketLIFT` subclass for default-off RL fine-tuning。 |
| `flowr/scripts/train_spindr_lift.sh` | Optional RL training script; baseline script unchanged。 |
| `flowr/scripts/score_select_spindr.sh` | Optional dry-run script。 |
| `migration_notes/05_smoke_test_results.md` | Future run log for commands/results。 |

### H2. 建议修改文件

| 文件 | 修改内容 | 默认影响 |
|---|---|---|
| `flowr/flowr/train.py` | Add default-off RL CLI args; branch in `build_model()` to instantiate `LigandPocketLIFT` only if enabled | None when disabled。 |
| `flowr/scripts/train_spindr_lift.sh` | New script only | None。 |
| Possibly `flowr/flowr/gen/generate_from_smol.py` | Not required for MVP; only if RL checkpoint hparams need compatibility fix | Avoid unless needed。 |
| Possibly `flowr/flowr/eval/*` | Not required; use existing evaluation | Avoid。 |

### H3. 新增函数

- `build_objective(name, cfg) -> BaseObjective`
- `BaseObjective.score_ligands(mols, context, device, dtype) -> RewardResult`
- `canonical_smiles(mol)`, `murcko_scaffold_smiles(mol)`, `morgan_fp(mol)`
- `PartitionSelector.select(reward_result, group_ids=None) -> PartitionResult`
- `load_prediction_records(predictions_dir, valid_unique=False) -> list[GeneratedRecord]`
- `write_reward_selection_outputs(records, reward_result, partition_result, out_dir)`
- `LigandPocketLIFT._generate_with_gen(gen, lig_prior, pocket_data, times, ...)`
- `LigandPocketLIFT._forward_with_gen(gen, lig_batch, pocket_batch, times, ...)`
- `LigandPocketLIFT._loss_per_sample(data, interpolated, predicted, times=None)`
- `LigandPocketLIFT._lift_discrete_loss_per_sample(cur_logits, ref_logits, mask, beta, is_edge)`
- `LigandPocketLIFT._lift_coord_loss_per_sample(cur_coords, ref_coords, mask)`
- `LigandPocketLIFT._build_generated_ligand_target(batch, generated)`
- `LigandPocketLIFT._maybe_update_reference_ema()`

### H4. 新增类

- `RewardContext`
- `RewardResult`
- `BaseObjective`
- `QEDObjective`
- `QEDSAObjective`
- `SimilarityObjective`
- `ValidityObjective`
- `CompositeObjective`
- `PartitionResult`
- `PartitionSelector`
- `RewardCSVLogger`
- `LigandPocketLIFT`

### H5. 新增 CLI/config 参数

| 参数 | 默认值 | 说明 | 默认关闭方式 |
|---|---:|---|---|
| `--lift_enabled` | `False` | 总开关 | 未传即关闭。 |
| `--lift_objective_name` | `qed` | reward objective | 仅 enabled 生效。 |
| `--lift_objective_config` | `None` | JSON string/path optional | 仅 enabled 生效。 |
| `--lift_top_ratio` | `0.25` | top fraction | 仅 enabled 生效。 |
| `--lift_bottom_ratio` | `0.25` | bottom fraction | 仅 enabled 生效。 |
| `--lift_history_diversity_mode` | `scaffold` | `none/scaffold/fingerprint` | 仅 enabled 生效。 |
| `--lift_history_fingerprint_threshold` | `0.70` | history FP threshold | 仅 enabled 生效。 |
| `--lift_history_max_size` | `4096` | memory size | 仅 enabled 生效。 |
| `--lift_num_time_samples` | `1` | MVP time samples | 仅 enabled 生效。 |
| `--lift_time_chunk_size` | `0` | 0 = no chunk | 仅 enabled 生效。 |
| `--lift_ref_ema_decay` | `0.99` | reference EMA | 仅 enabled 生效。 |
| `--lift_beta_types` | `1.5` | atom implicit beta | 仅 enabled 生效。 |
| `--lift_beta_bonds` | `1.5` | bond implicit beta | 仅 enabled 生效。 |
| `--lift_beta_charges` | `1.5` | charge implicit beta | 仅 enabled 生效。 |
| `--lift_beta_coords` | `1.5` | coord plus beta | 仅 enabled 生效。 |
| `--lift_gamma_coords` | `1.0` | coord minus gamma | 仅 enabled 生效。 |
| `--lift_lambda_types` | `1.0` | type loss weight | 仅 enabled 生效。 |
| `--lift_lambda_bonds` | `1.0` | bond loss weight | 仅 enabled 生效。 |
| `--lift_lambda_charges` | `1.0` | charge loss weight | 仅 enabled 生效。 |
| `--lift_lambda_coords` | `0.0` | coord rect loss; MVP off | 仅 enabled 生效。 |
| `--lift_bottom_repulsion_weight` | `0.5` | bottom branch weight | 仅 enabled 生效。 |
| `--lift_middle_weight` | `0.0` | middle branch default off | 仅 enabled 生效。 |
| `--lift_aux_fm_weight` | `0.0` | auxiliary generated FM | 仅 enabled 生效。 |
| `--lift_anchor_weight` | `0.1` | reference anchor | 仅 enabled 生效。 |
| `--lift_baseline_fm_weight` | `0.0` | original real-batch FM mix-in | 仅 enabled 生效。 |
| `--lift_freeze_pocket_encoder` | `False` | optional stability | 仅 enabled 生效。 |
| `--lift_log_reward_csv` | `None` | CSV path | 未设置不写。 |
| `--lift_current_eval_every` | `0` | 0 disables current eval | 默认关闭。 |
| `--lift_current_eval_samples` | `64` | current eval sample count | 仅 eval enabled 生效。 |

## I. 风险点

1. **数据格式不匹配**
   - `sotmol-rl` molecule batch 与 FLOWR complex batch 完全不同；必须用 FLOWR batch helpers。

2. **生成结果无法稳定转 RDKit Mol**
   - FLOWR generated atom/bond/charge 组合可能 invalid；reward/selection 必须 robustly handle `None`。

3. **FLOWR 采样过程不适合训练中频繁调用**
   - `_generate()` 多步 integration 很慢；MVP 需小 batch、小 steps、low-frequency current eval，训练 reward 采样使用 reference 且控制 `max_steps`。

4. **reward 计算速度过慢**
   - QED/SA/similarity 可在线；docking/PoseBusters/xTB 不适合高频。

5. **docking 或外部评分依赖复杂**
   - ADFR/Vina/Schrödinger/genbench3d_data 路径必须人工确认；失败时不能阻塞 MVP。

6. **原始 baseline 被无意改变**
   - 改原 `_loss()`、原 scripts、默认 parser 都可能影响 baseline；必须 default-off and subclass。

7. **RL loss 与原始 FLOWR loss 冲突**
   - RL top imitation 可能偏离 data distribution；baseline_fm/anchor 权重需小心。

8. **坐标归一化或单位不一致**
   - Reward mol 需要 real coordinates；loss 在 training COM frame；不要混用。

9. **pocket condition 与无条件 sotmol-rl 逻辑不一致**
   - Pseudo-target 只能替换 ligand，不可丢 pocket condition；reference sampling 必须 conditioned on same pocket。

10. **Interaction/inpainting fields 被破坏**
    - If using inpainting flags, generated pseudo-target and interpolated data must preserve `fragment_mask` and `interactions`。

11. **DDP history selection 不一致**
    - 多 GPU 下每个 rank 独立 top/bottom 会导致 selection bias；MVP 先单 GPU或实现 all-gather。

12. **Checkpoint hparams compatibility**
    - New `LigandPocketLIFT` checkpoints must still be loadable for original sampling/evaluation or provide compatible hparams。
