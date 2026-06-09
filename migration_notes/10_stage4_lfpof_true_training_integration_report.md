# Stage 4 修正：FLOWR LFPO-F-v2 true training integration report

## 1. modified files

本次在现有 Stage 4 代码上继续修改：

- `flowr/flowr/rl/training.py`
- `flowr/flowr/models/fm_pocket.py`
- `flowr/flowr/train.py`
- `migration_notes/10_stage4_lfpof_true_training_integration_report.md`

## 2. old simplified Stage 4 behavior

旧 Stage 4 行为是最小 proof-of-concept：

1. 在 `LigandPocketCFM.training_step` 当前 forward 的 `predicted` tensor 上直接重建候选；
2. top loss 是 `predicted["coords"]` 对 dataloader native ligand `lig_data["coords"]` 的坐标 MSE；
3. bottom loss 是 `relu(1 - coord_mse)` margin repulsion；
4. 没有显式 `pi_ref`；
5. 没有完整 ODE sampling；
6. 没有 selected generated ligand 的 pseudo target batch；
7. 没有 current/reference 双前向；
8. 没有 atom/bond/charge 的 top imitation 与 bottom repulsion。

## 3. new LFPO-F-v2 behavior

新的默认关闭 LFPO-F-v2 helper 按如下顺序执行：

1. FLOWR 原始 `training_step` 先计算原始 `_loss` 并求和；
2. 若 `enable_rl_finetune=false` 或 `rl_loss_weight<=0`，直接返回原 loss 和空 RL logs；
3. 若启用且命中 `rl_update_frequency`：
   - 初始化或读取显式 reference model `pi_ref`；
   - 用 `pi_ref._generate(...)` 跑完整 FLOWR ODE sampling 得到 candidate ligands；
   - 对完整 sampled candidates 计算/读取 PLIF、strain、Vina reward；
   - 复用 Stage 2 selection 选 top/middle/bottom；
   - 将 selected generated ligands 转成 pseudo target batch；
   - 构造 stratified noisy states `s_ik`；
   - 对 current model 与 reference model 各 forward 一次；
   - top branch 做 atom/bond/charge/coord imitation；
   - bottom branch 用 current-reference displacement 构造 detached negative targets；
   - 最终 `total_loss = original_flowr_loss + rl_loss_weight * (main + aux + anchor)`。

## 4. where `pi_ref` is initialized and EMA-updated

`flowr.rl.training.ensure_reference_model(...)` 在第一次启用 RL step 时执行：

- `deepcopy(current model)`；
- `eval()`；
- `requires_grad=False`；
- 放到 current model device；
- 存到 `model._rl_ref_model`。

`LigandPocketCFM.on_train_batch_end(...)` 调用 `on_train_batch_end_update_reference(self)`，对 `model._rl_ref_model` 做 EMA：

```text
ref_param <- rl_ref_ema_decay * ref_param + (1 - rl_ref_ema_decay) * current_param
```

默认 `rl_ref_ema_decay=0.999`。

## 5. where full generation / ODE sampling is called

`flowr.rl.training.sample_reference_candidates(...)` 调用：

```python
sampler_model._generate(
    lig_prior,
    pocket_data,
    steps=rl_sampling_steps,
    times=zero_generation_times(...),
    strategy=model.sampling_strategy,
    corr_iters=model.corrector_iters,
)
```

这里的 `sampler_model` 默认是 `pi_ref`，由 `rl_sample_from_reference=true` 控制。`rl_sampling_steps` 默认 100，smoke test 可以设小，例如 10 或 20。

## 6. how generated `z0` is converted into pseudo-target batch

ODE sampling 输出的 generated batch 包含：

- `coords`
- `atomics`
- `bonds`
- `charges`
- `mask`

helper 将 `_generate(...)` 输出从 physical coordinate convention 转回训练 convention：

1. 减去对应 pocket/system COM；
2. 除以 `coord_scale`；
3. detach；
4. 保留 atom/bond/charge distributions 作为 generated pseudo target。

同时 `_generate_mols(...)` 生成 RDKit Mol，用于 reward wrapper。

## 7. how stratified timesteps `s_ik` are constructed

`build_pseudo_training_batch(...)` 对 selected candidate 重复 `K=rl_num_stratified_timesteps` 次。

`stratified_times(...)` 按 strata 采样 `t_ik`。

`corrupt_ligand_like_flowr(...)` 构造 noisy state：

- coordinate：Gaussian prior 与 generated `R0` 线性插值；
- atom/bond/charge：按 `t` 在随机 one-hot token 与 generated target token 之间切换；
- pocket conditioning：按 candidate 的 batch index gather 原 batch pocket context。

注意：当前实现没有访问 datamodule 内部 `ComplexInterpolant` 实例，因此实现的是 FLOWR linear/unmask 训练机制的最小一致版本；后续若要完全复用 datamodule interpolant，可把 train interpolant 暴露给 model 或 RL helper。

## 8. top imitation loss definition

Top branch 只作用于 top candidates，target 是 generated high-reward ligand `z0`，不是 native ligand。

```text
L_top_atom   = CE(argmax X0, atom_logits_cur)
L_top_bond   = CE(argmax E0, bond_logits_cur)
L_top_charge = CE(argmax C0, charge_logits_cur)
L_top_coord  = masked MSE(coords_cur, R0)
```

组合：

```text
L_top =
  rl_top_atom_weight   * L_top_atom
+ rl_top_bond_weight   * L_top_bond
+ rl_top_charge_weight * L_top_charge
+ rl_top_coord_weight  * L_top_coord
```

## 9. bottom repulsion loss definition

Bottom branch 对 atom/bond/charge 使用 current/reference logits 构造 detached negative target：

```text
logp_cur = log_softmax(logits_cur)
logp_ref = log_softmax(logits_ref)
Delta = stop_gradient(logp_cur - logp_ref)
logp_minus = logp_ref - beta * Delta
p_minus = softmax(logp_minus).detach()
L_bottom_h = soft_CE(p_minus, logits_cur)
```

Coordinate branch：

```text
Delta_r = stop_gradient(coords_cur - coords_ref)
u_minus = coords_ref - gamma * Delta_r
L_bottom_coord = masked MSE(coords_cur, u_minus.detach())
```

组合：

```text
L_bottom =
  rl_bottom_atom_weight   * L_bottom_atom
+ rl_bottom_bond_weight   * L_bottom_bond
+ rl_bottom_charge_weight * L_bottom_charge
+ rl_bottom_coord_weight  * L_bottom_coord
```

不再使用旧的 `relu(1 - coord_mse)` placeholder。

## 10. how PLIF / strain / Vina rewards are computed

Reward 仍复用 Stage 1 `compute_structure_rewards_from_records(...)`：

- `compute`：对 sampled candidates 直接计算；
- `cached`：先查 JSONL cache，miss 后计算并写回；
- `existing_eval_output`：读取已有 reward JSON，仅用于 smoke / debug。

cache key 包含 ligand MolBlock hash、system id 和 objective mode。

## 11. validity / PoseBusters gating

训练 reward config 固定：

- `require_posebusters_validity=True`；
- `compute_posebusters_validity=True`；
- enabled metric failure 的 sample 不能进入 top；
- failed / invalid / PoseBusters failed samples 可以进入 bottom。

## 12. top/bottom/middle selection rules

训练中复用 `select_top_middle_bottom(...)`：

- top 必须 valid、PoseBusters valid、enabled metric success、feasible、eligible_top；
- 多目标启用 hard threshold，任一目标不达标不能 top；
- bottom 优先 failed / invalid / metric failed，然后低分 feasible；
- middle 只 logging，不参与 main loss；
- `rl_top_ratio` / `rl_bottom_ratio` 默认 0.25；
- 支持 `rl_top_k` / `rl_bottom_k`。

## 13. fallback and skip conditions

跳过 RL loss 的情况：

1. RL disabled；
2. `rl_update_frequency` 未命中；
3. 无 candidates；
4. 无 rewards；
5. 无 top；
6. reward/metric/surrogate 过程异常；
7. RL loss NaN / non-finite。

bottom 不足不跳过，bottom loss 为 0，只保留 top branch。

## 14. logging keys

保留并扩展 `train-rl-*`：

- `train-rl-enabled`
- `train-rl-loss`
- `train-rl-main-loss`
- `train-rl-top-loss`
- `train-rl-bottom-loss`
- `train-rl-aux-fm-loss`
- `train-rl-anchor-loss`
- `train-rl-loss-weight`
- `train-rl-objective-mode`
- `train-rl-num-candidates`
- `train-rl-num-valid`
- `train-rl-num-posebusters-valid`
- `train-rl-num-metric-success`
- `train-rl-num-top`
- `train-rl-num-bottom`
- `train-rl-num-middle`
- `train-rl-selected-frac`
- `train-rl-plif-mean`
- `train-rl-strain-mean`
- `train-rl-vina-mean`
- `train-rl-main-score-mean`
- `train-rl-main-score-max`
- `train-rl-invalid-count`
- `train-rl-posebusters-failed-count`
- `train-rl-plif-failed-count`
- `train-rl-strain-failed-count`
- `train-rl-vina-failed-count`
- `train-rl-skip-count`
- `train-rl-skip-reason`
- `train-rl-cache-hit-rate`
- `train-rl-delta-atom-abs-mean`
- `train-rl-delta-bond-abs-mean`
- `train-rl-delta-charge-abs-mean`
- `train-rl-delta-coord-abs-mean`
- `train-rl-reward-top-mean`
- `train-rl-reward-bottom-mean`
- `train-rl-reward-top10-mean`

## 15. default-off guarantee

当 `enable_rl_finetune=false` 或 `rl_loss_weight=0`：

- 不初始化 reference model；
- 不调用 `_generate`；
- 不计算 reward；
- 不写 train RL cache；
- 不改变 original FLOWR loss；
- 原始 sampling/evaluation 入口不变。

## 16. minimal commands

### 16.1 strain-only smoke test

```bash
cd /data/bhli/Project/repo-rl/flowr
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0

python -m flowr.train \
  --gpus 1 \
  --dataset spindr \
  --data_path /path/to/spindr \
  --save_dir /path/to/lfpof_v2_strain_smoke \
  --exp_name lfpof_v2_strain_smoke \
  --arch pocket \
  --pocket_noise fix \
  --load_ckpt /path/to/pretrained.ckpt \
  --trial_run \
  --enable_rl_finetune \
  --rl_loss_weight 0.01 \
  --rl_objective_mode strain \
  --rl_metric_source cached \
  --rl_metric_cache_path /path/to/lfpof_v2_strain_cache.jsonl \
  --rl_sampling_steps 10 \
  --rl_num_candidates_per_step 1 \
  --rl_num_stratified_timesteps 1 \
  --rl_update_frequency 1 \
  --rl_top_ratio 0.25 \
  --rl_bottom_ratio 0.25
```

### 16.2 cached dry-run style training smoke

Same as above, but reuse an existing cache path:

```bash
--rl_metric_source cached \
--rl_metric_cache_path /path/to/existing_or_new_cache.jsonl
```

### 16.3 full objective small-scale

```bash
--rl_objective_mode plif_strain_vina \
--rl_metric_source cached \
--rl_sampling_steps 20 \
--rl_num_candidates_per_step 1 \
--rl_update_frequency 10 \
--rl_plif_min_threshold 0.3 \
--rl_strain_max_threshold 10 \
--rl_vina_max_threshold -6
```

## 17. current verified items

Verified in this PR:

- Python syntax / py_compile for modified files;
- disabled path is syntactically gated before reference initialization and reward computation;
- top/bottom surrogate no longer uses native ligand coordinate MSE or ReLU margin placeholder.

## 18. current unverified items

Not verified in this environment:

- real 100-step train-time ODE sampling runtime;
- reference model deepcopy compatibility with full Lightning trainer state;
- true memory footprint for large batch × candidates × timesteps;
- PLIF/Vina/PoseBusters speed in train loop;
- whether generated tensor-to-Mol reconstruction succeeds often enough during early fine-tuning;
- scientific improvement over FLOWR baseline.

## 19. differences from sotmol-rl implementation

- FLOWR is pocket-conditioned; pseudo batches must gather/repeat pocket context;
- coordinates are converted between FLOWR scaled/COM-centered training convention and generated physical coordinates;
- current pseudo corruption mirrors FLOWR linear/unmask behavior, but does not yet directly call the datamodule-held `ComplexInterpolant` instance;
- reference model is explicit deep copy + EMA, not the Lightning EMA callback;
- PLIF/strain/Vina are structure metrics and may be far slower than sotmol property rewards.

## 20. impact on original FLOWR baseline when RL disabled

No intended impact:

- default `enable_rl_finetune=false`;
- default `rl_loss_weight=0.0`;
- helper returns original loss and no logs before reference/reward/sampling;
- original training/sampling/evaluation scripts are not changed except extra CLI options on `flowr.train`.
