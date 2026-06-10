# 04. sotmol-rl → flowr 分阶段实施计划

> 范围：本文给出把 `sotmol-rl` 的 reward-guided fine-tuning / LIFT 语义迁移到 `flowr` 的实施路线、允许/禁止修改范围、验收标准、smoke tests、回滚策略与 MVP 定义。本文只做计划，不修改 `sotmol-rl/` 或 `flowr/` 代码。
>
> 总原则：先保留并验证 FLOWR 原始 baseline，再逐层新增 reward、selection、dry-run，最后接入 default-off RL fine-tuning。任何阶段都不得破坏原始训练、采样、评估入口。

## A. 分阶段实施路线

### 阶段 1：保留并验证 flowr 原始 baseline

目标：确认 FLOWR 原始 training/sampling/evaluation 链路可复现，形成后续对比基线。

任务：

1. 下载/确认 SPINDR `.smol` 数据、processed statistics、`train_mols.pkl`。
2. 下载/确认 pretrained FLOWR checkpoint，或用最小训练命令得到 smoke checkpoint。
3. 修正本地脚本路径，但不改变 repo 中 baseline scripts 默认内容。
4. 跑最小 `generate_from_smol` sampling。
5. 跑最小 `evaluate_metrics` / `evaluate_interactions`（若外部资源不足，记录缺失项）。
6. 记录 baseline command、data path、checkpoint path、sample dir、metrics。

### 阶段 2：新增 reward/objective 层

目标：在不进入训练 loop 的情况下，对 RDKit Mol list 计算统一 reward dict。

任务：

1. 新增 `flowr/flowr/rl/objectives.py`。
2. 定义 `RewardContext`, `RewardResult`, `BaseObjective`。
3. 实现轻量 objectives：validity、QED、SA、QED-SA、similarity。
4. 预留 structure-based objectives 的接口，但不在 MVP 中高频调用。
5. 新增 reward-only smoke test，可直接构造 RDKit mol 或读取 sampling `.pt`。

### 阶段 3：新增 top/middle/bottom selection 层

目标：实现 history-aware feasible top selection 和 bottom repulsion candidate selection。

任务：

1. 新增 `flowr/flowr/rl/partition.py`。
2. 定义 `PartitionResult`, `PartitionSelector`。
3. 支持 `top_ratio`, `bottom_ratio`, top score mode, history scaffold/fingerprint, per-target/global selection。
4. 对 invalid/severe/low-score bottom 优先级写 smoke test。
5. 保证 selection 输入只依赖 `RewardResult`，不依赖训练模型。

### 阶段 4：打通 flowr sampling → reward → selection 的 dry-run 链路

目标：先不训练，只用 FLOWR 原始采样结果跑 reward 和 selection，输出可审计文件。

任务：

1. 新增 `flowr/flowr/rl/score_and_select.py` CLI。
2. 读取 `predictions_multi_*.pt` 或 `samples_<target>.pt`。
3. 将 generated ligands normalize 成 records。
4. 调 reward objective。
5. 调 partition selector。
6. 输出 CSV/JSON/SDF：`reward_results.csv`, `selection_results.csv`, `reward_summary.json`, `top_ligands.sdf`, `bottom_ligands.sdf`。
7. 用原始 sampling smoke output 跑 dry-run。

### 阶段 5：接入 RL / reward-guided fine-tuning，但默认关闭

目标：新增 FLOWR-native LIFT LightningModule，不改变 baseline 默认训练。

任务：

1. 新增 `flowr/flowr/models/lift_pocket.py::LigandPocketLIFT`，继承 `LigandPocketCFM`。
2. `flowr.train` 新增 default-off CLI 参数；`--lift_enabled` 时才 instantiate new class。
3. 在 `LigandPocketLIFT` 中实现 reference model、reference sampling、reward、selection、per-sample surrogate loss、logging。
4. 先单 GPU、small batch、small steps smoke。
5. 保证 `--lift_enabled` 未传时，原 `LigandPocketCFM` 路径完全不变。

### 阶段 6：做最小 smoke test

目标：验证新增 RL 链路能小规模生成、reward、selection、loss、backward、checkpoint。

任务：

1. Run baseline smoke command。
2. Run reward smoke。
3. Run selection smoke。
4. Run dry-run smoke。
5. Run RL training smoke with `--lift_enabled --epochs 1 --batch_cost small`。
6. Check gradients/non-NaN logs/checkpoint。
7. Run original sampling/evaluation on RL checkpoint if compatible。

### 阶段 7：做 original flowr 与 RL-flowr 的最小对比实验

目标：用同一数据、同一 pocket、同一 sampling/evaluation scripts 对比 original vs RL。

任务：

1. Baseline checkpoint：pretrained or original smoke/fine-tuned baseline。
2. RL checkpoint：from same init，small reward-guided fine-tune。
3. Same sampling command except `--ckpt_path` and `--save_dir`。
4. Same evaluation command。
5. Compare `metrics.pt`, `interaction_recovery*.pt`, reward CSV。
6. 记录所有 command 和 artifact。

## B. 每阶段允许修改的文件

| 阶段 | 允许修改/新增文件 |
|---|---|
| 1 baseline 验证 | 不需要修改代码；可新增 `migration_notes/05_baseline_smoke_results.md` 记录结果；本地不提交的 path config 可临时编辑脚本副本。 |
| 2 reward 层 | 新增 `flowr/flowr/rl/__init__.py`, `flowr/flowr/rl/objectives.py`; 可新增 `tests` 或 `migration_notes` smoke 记录。 |
| 3 selection 层 | 新增 `flowr/flowr/rl/partition.py`; 可新增 selection smoke script/test。 |
| 4 dry-run | 新增 `flowr/flowr/rl/score_and_select.py`, `flowr/flowr/rl/io.py` 或 `logging.py`; 可新增 `flowr/scripts/score_select_spindr.sh`。 |
| 5 RL fine-tuning | 新增 `flowr/flowr/models/lift_pocket.py`; 修改 `flowr/flowr/train.py` 只加 default-off CLI/branch; 可新增 `flowr/scripts/train_spindr_lift.sh`。 |
| 6 smoke | 不需新增核心代码；可新增 smoke result 文档。 |
| 7 对比实验 | 不需新增核心代码；可新增 experiment report 文档。 |

## C. 每阶段禁止修改的内容

所有阶段共同禁止：

1. 不要破坏原始训练入口：`python -m flowr.train` 和 `scripts/train_spindr.sh` baseline 行为不变。
2. 不要破坏原始采样入口：`generate_from_smol.py`, `generate_from_pdb.py` 默认输入输出不变。
3. 不要破坏原始 evaluation 入口：`evaluate_metrics.py`, `evaluate_interactions.py` 默认行为不变。
4. 不要删除或重命名原始配置、脚本、模型类、metric key。
5. 不要改变原始默认参数，尤其：`--arch`, `--pocket_noise`, `--use_ema`, loss weights, checkpoint monitor `val-fc-validity`。
6. 不要把 heavy docking/PoseBusters/xTB reward 作为默认 online reward。
7. 不要让新增 RL imports 在 baseline 路径产生外部依赖或 side effects。
8. 不要修改 `sotmol-rl/` 源码。

阶段特定禁止：

- 阶段 2/3/4：不得修改训练 loop。
- 阶段 5：不得改 `LigandPocketCFM._loss()` baseline 语义；不得把 `LigandPocketLIFT` 设为默认。
- 阶段 6/7：不得为了通过 smoke 临时降低 evaluation 标准而改 evaluation 脚本；外部依赖缺失应记录为环境限制。

## D. 每阶段验收标准

| 阶段 | 验收标准 |
|---|---|
| 1 baseline | 原始 FLOWR 数据路径确认；最小 sampling 输出 `predictions_multi_1.pt`；若可用，evaluation 输出 `metrics.pt`；命令记录完整。 |
| 2 reward | 给定 RDKit mol list，输出 `RewardResult` shape 正确；invalid mol score/flags 正确；QED/SA/similarity component 合理；无训练代码改动。 |
| 3 selection | top/bottom/middle mask 数量符合 ratio；top 只从 feasible 中选；history scaffold/fingerprint 能排除重复；bottom invalid/severe/low-score 优先级正确。 |
| 4 dry-run | 能读取 FLOWR sampling `.pt`；输出 CSV/JSON/SDF；top/bottom 样本可人工检查；不需要 checkpoint/training。 |
| 5 RL | `--lift_enabled` 关闭时 baseline instantiate 不变；开启时 `LigandPocketLIFT` 能完成一个 train step；loss finite；至少部分 current model params 有梯度；reference frozen 且 EMA 更新。 |
| 6 smoke | 原始 baseline smoke 仍可运行；reward/selection/dry-run/RL smoke 全部通过或明确记录环境限制；checkpoint 可保存。 |
| 7 对比 | original 与 RL 使用同一 sampling/evaluation protocol；输出两个 sample dirs 和 metrics；对比表包含 key metrics 和 reward distribution。 |

更细验收：

- reward dict 输出合理：`score.shape == [N]`，component/raw fields 同长度。
- selection mask 数量正确：`top + bottom + middle == N`，top/bottom 互斥。
- dry-run CSV 至少包含 `target_id`, `sample_index`, `score`, `valid`, `connected`, `partition`。
- RL loss 有梯度：`sum(p.grad is not None for p in model.gen.parameters()) > 0`。
- 默认关闭时结果与原始流程一致：同一 command 不出现新增 logs/import errors/changed checkpoint monitor。

## E. smoke test 命令

> 说明：以下命令是实施后的建议。当前第三阶段只生成文档，不运行这些命令。所有命令默认在 `repo-rl/flowr` 目录下执行，除非另有说明。

### E1. 原始 flowr baseline smoke test

```bash
export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES=0 python -m flowr.gen.generate_from_smol \
  --mp_index 1 \
  --gpus 1 \
  --batch_cost 4 \
  --arch pocket \
  --pocket_noise fix \
  --dataset_split test \
  --ckpt_path /path/to/flowr.ckpt \
  --data_path /path/to/spindr \
  --dataset spindr \
  --save_dir /path/to/baseline_sample_smoke \
  --max_sample_iter 1 \
  --sample_n_molecules_per_target 1 \
  --integration_steps 5 \
  --ode_sampling_strategy linear \
  --categorical_strategy uniform-sample \
  --sample_mol_sizes
```

### E2. reward 层 smoke test

```bash
export PYTHONPATH="$PWD"
python -m flowr.rl.objectives \
  --objective_name qed_sa \
  --smiles "CCO,c1ccccc1,invalid_smiles" \
  --out /tmp/flowr_reward_smoke.json
```

若不实现 module CLI，可用 Python one-liner：

```bash
export PYTHONPATH="$PWD"
python - <<'PY'
from rdkit import Chem
from flowr.rl.objectives import build_objective
mols = [Chem.MolFromSmiles(s) for s in ['CCO', 'c1ccccc1', 'bad']]
obj = build_objective('qed_sa', {})
res = obj.score_ligands(mols)
print(res.score, res.valid, res.component_scores.keys())
PY
```

### E3. selection 层 smoke test

```bash
export PYTHONPATH="$PWD"
python - <<'PY'
import torch
from flowr.rl.objectives import RewardResult
from flowr.rl.partition import PartitionSelector
res = RewardResult(
    score=torch.tensor([0.9, 0.8, 0.2, 0.1]),
    component_scores={'score': torch.tensor([0.9, 0.8, 0.2, 0.1])},
    raw_properties={},
    feasible=torch.tensor([1, 1, 1, 0], dtype=torch.bool),
    severe_violation=torch.tensor([0, 0, 0, 1], dtype=torch.bool),
    valid=torch.tensor([1, 1, 1, 0], dtype=torch.bool),
    connected=torch.tensor([1, 1, 1, 0], dtype=torch.bool),
    smiles=['A','B','C',None], canonical_smiles=['A','B','C',None], scaffolds=['sa','sb','sc',None], fps=[None]*4, mols=[None]*4, target_ids=['t']*4, metadata={},
)
part = PartitionSelector({'top_ratio':0.25, 'bottom_ratio':0.25, 'history_diversity_mode':'none'}).select(res)
print(part.top_mask, part.bottom_mask, part.selected_mask)
PY
```

### E4. sampling-reward-selection dry-run smoke test

```bash
export PYTHONPATH="$PWD"
python -m flowr.rl.score_and_select \
  --predictions_dir /path/to/baseline_sample_smoke \
  --predictions_pattern "predictions_multi_*.pt" \
  --objective_name qed_sa \
  --top_ratio 0.25 \
  --bottom_ratio 0.25 \
  --history_diversity_mode scaffold \
  --out_dir /path/to/reward_selection_smoke
```

Expected outputs：

```text
/path/to/reward_selection_smoke/reward_results.csv
/path/to/reward_selection_smoke/selection_results.csv
/path/to/reward_selection_smoke/reward_summary.json
/path/to/reward_selection_smoke/top_ligands.sdf
/path/to/reward_selection_smoke/bottom_ligands.sdf
```

### E5. RL training smoke test

```bash
export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES=0 python -m flowr.train \
  --gpus 1 \
  --epochs 1 \
  --batch_cost 8 \
  --val_batch_cost 2 \
  --d_model 128 \
  --d_edge 64 \
  --n_coord_sets 32 \
  --pocket_d_model 128 \
  --pocket_n_layers 2 \
  --n_layers 2 \
  --arch pocket \
  --pocket_noise fix \
  --dataset spindr \
  --exp_name lift_smoke \
  --data_path /path/to/spindr \
  --save_dir /path/to/lift_smoke_ckpts \
  --self_condition \
  --use_ema \
  --remove_hs \
  --categorical_strategy uniform-sample \
  --load_ckpt /path/to/flowr.ckpt \
  --lift_enabled \
  --lift_objective_name qed_sa \
  --lift_num_time_samples 1 \
  --lift_lambda_coords 0.0 \
  --lift_current_eval_every 0
```

Expected：

- No crash in one train step/epoch。
- Logs contain `train-rl-*`。
- Checkpoint saved in `/path/to/lift_smoke_ckpts`。
- Reference model remains frozen。

### E6. evaluation smoke test

```bash
export PYTHONPATH="$PWD"
python -m flowr.eval.evaluate_metrics \
  --num_workers 2 \
  --dataset spindr \
  --data_path /path/to/spindr \
  --save_dir /path/to/baseline_sample_smoke \
  --multiple_files \
  --remove_hs
```

如果 ADFR/genbench3d_data 缺失，记录为环境限制；不要修改 evaluation 脚本绕过，除非后续另设 lightweight evaluation script 并明确不替代 baseline。

## F. 回滚策略

### F1. 分阶段小提交

每个阶段单独 commit：

1. `feat(flowr-rl): add reward objective layer`
2. `feat(flowr-rl): add partition selector`
3. `feat(flowr-rl): add score and select dry run cli`
4. `feat(flowr-rl): add default-off lift training module`
5. `docs: record smoke test results`

这样任一阶段失败可 `git revert <commit>` 只回滚该层。

### F2. 文件级隔离

- Reward/partition/dry-run 都在 `flowr/flowr/rl/`，可整体删除或 revert，不影响 baseline。
- RL training class 在 `flowr/flowr/models/lift_pocket.py`，只有 `flowr.train --lift_enabled` branch 引用。
- `flowr.train` 修改必须小而集中：parser args + branch import。

### F3. 功能级禁用

如果 RL training 失败但 reward/dry-run 可用：

- 保留 `flowr/flowr/rl/objectives.py`、`partition.py`、`score_and_select.py`。
- Revert only `lift_pocket.py` and `flowr.train` branch。
- Baseline scripts continue unaffected。

如果 heavy reward 失败：

- Disable that objective via config。
- Keep lightweight reward objectives。
- Mark external dependency missing。

### F4. checkpoint / artifact rollback

- Baseline checkpoint/sample/evaluation dirs must be separate from RL dirs。
- Never overwrite baseline `save_dir`。
- RL failed runs can delete only `lift_*` directories。

## G. 最小可运行版本定义

MVP 不要求：

- 完整 docking reward。
- 完整 PoseBusters/GenBench3D online reward。
- 大规模 SPINDR training。
- 多 GPU/DDP history synchronization。
- 所有 inpainting modes 同时支持。
- xTB force residual。

MVP 必须完成：

1. FLOWR original sampling output can be scored by new reward layer。
2. Reward layer returns valid `RewardResult` for valid/invalid RDKit mols。
3. Selection layer returns correct top/middle/bottom masks and diagnostics。
4. Dry-run can produce CSV/JSON/SDF from `predictions_multi_*.pt`。
5. `--lift_enabled` default off；off 时 original FLOWR train path unchanged。
6. With `--lift_enabled`, one small train step can:
   - sample ligands from reference conditioned on pocket；
   - reconstruct RDKit mols；
   - compute reward；
   - select top/bottom；
   - compute surrogate loss；
   - run backward with finite gradients；
   - update optimizer and reference EMA。
7. RL checkpoint or model state can be sampled/evaluated using original evaluation protocol or an explicitly compatible sampling path。

MVP success criterion：

```text
small FLOWR batch + pretrained checkpoint
  -> RL train smoke 1 epoch / few steps
  -> generate 1-2 ligands per target
  -> reward/selection CSV
  -> original evaluation smoke or documented external-dependency warning
```

## H. 后续扩展接口

### H1. 多目标 MPO reward

- Extend `CompositeObjective` with weighted geometric mean, linear sum, Tchebycheff, min-component bonus。
- Add component-wise top selection and Pareto-ish selection if needed。
- Preserve `component_scores` and `raw_properties` for logging。

### H2. docking reward

- Add `DockingObjective` consuming `RewardContext.pocket_pdbs` and generated mols。
- Use cache keyed by `(target_id, canonical_smiles, conformer_hash)`。
- Run low-frequency or offline first。
- Handle ADFR/Vina path failures as `severe_violation` or missing metric depending config。

### H3. protein-ligand interaction reward

- Reuse `interaction_recovery_per_complex()` and PLIF tools。
- Require reference ligand and pocket PDB with Hs。
- Reward can be PLIF recovery, Tanimoto similarity, or target interaction class recall。

### H4. pharmacophore reward

- Add objective based on pharmacophore feature matching to reference ligand/pocket。
- Can operate on RDKit features and/or FLOWR interaction profile。

### H5. PoseBusters

- Add `PoseBustersObjective` for offline or low-frequency pass/fail reward。
- Because PoseBusters can be slow, keep disabled by default and cache results。

### H6. strain energy

- Reuse `evaluate_strain()` / RDKit MMFF energy difference。
- Normalize to bounded reward, e.g. `1 / (1 + strain / scale)`。

### H7. xTB force residual

- Add xTB objective similar to `sotmol-rl` xTB force reward but adapted to FLOWR generated ligand / optional complex。
- Must include timeout, max workers, failure score, cache。
- Default off。

### H8. scaffold diversity

- Extend `PartitionSelector` history memory per target/global。
- Add scaffold novelty reward component or selection constraint。

### H9. novelty

- Use `train_mols.pkl` or train canonical SMILES file。
- Add novelty component and/or hard filter for top selection。

### H10. reference-ligand similarity

- Use `RewardContext.reference_ligands`。
- Support Morgan, atom-pair, RDKit fingerprints。
- Useful for scaffold hopping or target-specific optimization。
