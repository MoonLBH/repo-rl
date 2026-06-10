# 02b. flowr 代码架构分析与迁移接口地图

> 范围：本文件分析 `flowr/` 的训练、采样、评估、模型 forward、loss、batch 表示、molecule reconstruction、metric/scoring 与配置系统，为后续迁移 `sotmol-rl/` 的 RL / reward-guided fine-tuning 方法提供接口地图。本阶段未修改 `flowr/` 或 `sotmol-rl/` 代码。

## A. flowr 的总体 workflow

FLOWR 的原始 SBDD workflow 可以抽象为：

1. **数据准备**
   - `.smol` 文件存储 `PocketComplexBatch`，每个 complex 包含 ligand、holo pocket、可选 apo pocket、metadata、interaction profile、fragment mask 等。
   - `GeometricDataset.load()` 读取 `.smol`，`scriptutil.complex_transform()` 将 ligand/pocket 转为模型 vocab、one-hot bond/charge 等张量。
   - `GeometricInterpolantDM` 在 dataloader 中对 raw complex 生成 `(prior, data, interpolated, times)` 四元组。

2. **训练**
   - `flowr.train` 解析 CLI，构建 vocab/statistics/datamodule/model/trainer。
   - `--arch pocket` 时构建 `PocketEncoder` + `LigandGenerator`，包装为 `LigandPocketCFM` LightningModule。
   - `LigandPocketCFM.training_step()` 从 complex 中拆出 pocket condition、ligand target、ligand interpolated state，调用 model forward，计算 coordinate/type/bond/charge/interaction loss。
   - Lightning trainer 记录 train/val metrics，并按 `val-fc-validity` 保存 checkpoint。

3. **采样**
   - `generate_from_smol.py` 或 `generate_from_pdb.py` 重建与 checkpoint hparams 一致的 `LigandPocketCFM`。
   - 从 `.smol` split 或 PDB/CIF+ligand 构造 pocket-conditioned test dataloader。
   - `scriptutil.generate_ligands_per_target()` 调 `model._generate()`；integrator 逐步更新 ligand state。
   - 生成张量通过 `MolBuilder` 转 RDKit mol，并保存为 `.pt`、SDF、PDB/trajectory 等。

4. **评估**
   - `evaluate_metrics.py` 聚合 `predictions_multi_*.pt`，计算 molecule metrics、GenBench3D、PoseBusters、PoseCheck、interaction recovery 等，输出 `metrics.pt`。
   - `evaluate_interactions.py` 计算更详细的 PLIF recovery / Tanimoto similarity，输出 `interaction_recovery*.pt`。

## B. 训练、采样、评估的入口命令和文件路径

| 流程 | Bash / Python 入口 | 主要文件 |
|---|---|---|
| 训练 | `bash scripts/train_spindr.sh` -> `python -m flowr.train` | `flowr/train.py` |
| SPINDR 采样 | `sbatch scripts/gen_spindr.sl` -> `python -m flowr.gen.generate_from_smol` | `flowr/gen/generate_from_smol.py` |
| PDB/CIF 采样 | `sbatch scripts/gen_pdb.sl` -> `python -m flowr.gen.generate_from_pdb` | `flowr/gen/generate_from_pdb.py` |
| 主评估 | `bash scripts/eval_spindr.sh` -> `evaluate_metrics` + `evaluate_interactions` | `flowr/eval/evaluate_metrics.py`, `flowr/eval/evaluate_interactions.py` |
| 数据预处理 | `python -m flowr.data.preprocess`, `accumulate`, `get_statistics`, `preprocess_pdbs` | `flowr/data/*.py` |

建议后续迁移时不要替换这些入口；新增 RL/fine-tuning 应通过新 CLI flag 或新 script 显式启用。

## C. 核心模型类和 forward 输入输出

### C1. 模型构建链

`flowr.train::build_model()`：

1. 根据 vocab 与 categorical strategy 确定 `n_atom_feats`、`n_bond_types`、`n_interaction_types`。
2. `--arch pocket`：
   - `PocketEncoder(...)` 编码 protein pocket。
   - `LigandGenerator(...)` 接收 ligand state + pocket encoding/condition，输出 ligand predictions。
3. 构建 `Integrator(...)`。
4. 构建 `LigandPocketCFM(...)` LightningModule。

### C2. `LigandPocketCFM.forward()` 输入

签名：

```python
def forward(self, batch, pocket_batch, t, training=False, cond_batch=None, pocket_equis=None, pocket_invs=None)
```

- `batch`: ligand-only dict，通常来自 `builder.extract_ligand_from_complex()`。
  - `coords`: `[B, N_lig, 3]`
  - `atomics`: `[B, N_lig, V]`
  - `bonds`: `[B, N_lig, N_lig, E]`
  - `charges`: `[B, N_lig, C]`（forward 本身不直接读取 ligand charges，但 loss 读取）
  - `mask`: `[B, N_lig]`
  - `fragment_mask`: `[B, N_lig]`，用于 inpainting。
  - `interactions`: optional `[B, N_lig, N_pocket, I]` 或相关排列，取决于 collate。
- `pocket_batch`: pocket-only dict。
  - `coords`: `[B, N_pocket, 3]`
  - `atomics`: pocket atom one-hot（用于 builder/重建）
  - `atom_names`: pocket atom-name vocab indices
  - `res_names`: residue-name vocab indices
  - `bonds`: `[B, N_pocket, N_pocket, E]`
  - `charges`: `[B, N_pocket, C]`
  - `mask`: `[B, N_pocket]`
- `t`: list of times。
  - training 中为 `[lig_cont_time, lig_disc_time, pocket_time, interaction_time]`。
  - forward 使用 `t[0]` 给 coordinate/continuous ligand time，`t[1]` 给 discrete ligand time，`t[-1]` 给 interactions。
- `cond_batch`: self-conditioning dict，包含 `coords`, `atomics`, `bonds`。
- `pocket_equis/pocket_invs`: inference 时可预先计算 pocket encoding，避免每一步重复编码。

### C3. `LigandPocketCFM.forward()` 输出

`self.gen(...)` 返回 tuple，训练/采样代码按下列位置解释：

- `out[0]`: predicted ligand coordinates / coordinate endpoint or velocity-like output（由 integrator/formulation 使用）。
- `out[1]`: atom type logits `[B, N_lig, V]`。
- `out[2]`: bond type logits `[B, N_lig, N_lig, E]`。
- `out[3]`: charge logits `[B, N_lig, C]`。
- `out[4]`: interaction logits（当 `predict_interactions` 或 `flow_interactions`）。
- `out[-1]`: ligand mask，在采样中写入 `predicted["mask"]`。

### C4. structure-based condition 如何输入模型

- Pocket condition 通过 `pocket_coords`, `pocket_atom_names`, `pocket_atom_charges`, `pocket_bond_types`, `pocket_res_types`, `pocket_atom_mask` 输入 `LigandGenerator`。
- `use_lig_pocket_rbf` 可启用 ligand-pocket RBF features。
- inference 时 `self.gen.get_pocket_encoding(...)` 在 `_generate()` 开始处对 rigid pocket 编码一次，然后每个 integration step 复用 `pocket_equis/pocket_invs`。
- `pocket_noise=fix` 表示 pocket rigid/fixed；`random` 会移动 holo+lig 到 holo-lig COM；`apo` 使用 apo/holo flow matching（需要 apo pocket）。

## D. batch 数据结构说明

`GeometricInterpolantDM` collate 后，训练 batch 是四元组：

```python
prior, data, interpolated, times = batch
```

其中 `prior/data/interpolated` 都是 dict，complex 数据常见字段：

| key | 含义 | shape / 类型 |
|---|---|---|
| `coords` | complex atom coordinates（ligand + pocket padded） | `[B, N_complex, 3]` |
| `atomics` | atom token one-hot | `[B, N_complex, V]` |
| `bonds` | bond type one-hot adjacency | `[B, N_complex, N_complex, E]` |
| `interactions` | pocket-ligand interaction one-hot/long tensor | usually `[B, N_lig, N_pocket, I]` after batch conversion |
| `charges` | charge one-hot | `[B, N_complex, C]` |
| `atom_names` | pocket atom-name vocab indices; ligand may use LIG token | `[B, N_complex]` |
| `res_names` | residue-name vocab indices; ligand may use LIG token | `[B, N_complex]` |
| `lig_mask` | marks ligand atoms in complex | `[B, N_complex]` |
| `pocket_mask` | marks pocket atoms in complex | `[B, N_complex]` |
| `fragment_mask` | fixed/inpainted ligand fragment atoms | `[B, N_complex]` or ligand-aligned after extraction |
| `mask` | valid atom mask | `[B, N_complex]` |
| `complex` | list of `PocketComplex` Python objects | length `B` |

`LigandPocketCFM.training_step()` immediately converts complex dict into:

- `pocket_data = builder.extract_pocket_from_complex(data)`。
- `lig_interp = builder.extract_ligand_from_complex(interpolated)`。
- `lig_data = builder.extract_ligand_from_complex(data)`。

`extract_ligand_from_complex()` and `extract_pocket_from_complex()` slice by `lig_mask` / `pocket_mask`, repack coordinates, atomics, bonds, charges and masks into ligand-only or pocket-only padded tensors.

## E. ligand 和 pocket 的表示方式

### E1. Ligand

- Atom vocabulary: `_build_vocab()` uses `['<PAD>', '<MASK>', 'H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'I', 'Hg', 'Bi']`。
- Atom types: one-hot over ligand vocab。
- Bond types: one-hot adjacency; number of bond classes is `len(BOND_IDX_MAP)+1` plus optional mask class when categorical strategy is `mask`。
- Charges: one-hot over charge classes from `flowr.util.rdkit.CHARGE_IDX_MAP` / `IDX_CHARGE_MAP`。
- Coordinates: Å-like coordinates, generally not scaled for complex data (`complex_transform` asserts `coord_std == 1.0`)。
- Mask: `mask` marks valid ligand atoms after extraction。

### E2. Pocket

- Pocket atom-name vocabulary: `_build_vocab_pocket_atoms()` includes `<PAD>`, `LIG`, and PLINDER pocket atom names。
- Pocket residue vocabulary: `_build_vocab_pocket_res()` includes `<PAD>`, `LIG`, and PLINDER residue names。
- Pocket atom types, bonds, charges are still available as tensors, but model conditioning primarily uses atom names, residue names, charges, bond types and coordinates。
- `pocket_mask` separates pocket atoms from ligand atoms in complex tensors。

### E3. Complex

- `PocketComplex` contains ligand + holo/apo pocket and metadata。
- `PocketComplexBatch` pads systems to batch tensors and exposes `coords(state)`, `atomics(state)`, `adjacency(state)`, `charges(state)`, `atom_names(state)`, `res_names(state)`, `lig_mask(state)`, `pocket_mask(state)`, `interactions(state)`。

## F. coordinate normalization / denormalization 逻辑

1. For non-complex molecule datasets (`qm9`, `geom-drugs`), `mol_transform()` can scale coordinates by `1 / coord_std` and zero COM。
2. For complex SBDD datasets, `complex_transform()` currently asserts `coord_std == 1.0`，因此 SPINDR pocket-conditioned training does not use coordinate scaling in the same way as molecule-only flow matching。
3. `--scale_coords` in `flowr.train` would set dataset-specific coord scales, but complex transform assertion means current SBDD path effectively requires no scaling unless code is changed。
4. `PocketComplex` alignment utilities move complexes to COM depending on `pocket_noise`：
   - `fix`: move holo+ligand to holo COM。
   - `random`: move holo+ligand to holo-lig COM。
   - `apo`: move apo+holo+ligand to apo COM。
5. During generation, `_generate()` multiplies output coords by `self.coord_scale` and then `undo_zero_com_batch()` adds back each system COM before RDKit reconstruction。
6. `retrieve_pdbs()` / `retrieve_ligs_with_hs()` similarly undo COM alignment before writing reference pocket/ligand files。

## G. 原始 loss 的组成

`LigandPocketCFM.training_step()` computes:

```python
losses = self._loss(lig_data, lig_interp, predicted, times=ligand_times)
loss = sum(losses.values())
```

`_loss()` returns:

1. `coord-loss`
   - `F.mse_loss(pred_coords, coords, reduction="none")`
   - mask by ligand atom mask
   - average per sample by atom count, then mean over batch
   - **注意：在当前 code path 中没有乘 `self.coord_loss_weight`**；虽然 constructor stores `coord_loss_weight`，`_loss()` 里未使用它。这是迁移时需要确认的 baseline 行为。
2. `type-loss`
   - CE against `argmax(data["atomics"])`，或 MSE if `type_strategy == "mse"`。
   - mask strategy 时只在 `<MASK>` atom positions 上算。
   - mean over atoms/batch, multiplied by `self.type_loss_weight`。
3. `bond-loss`
   - CE over `[N_lig, N_lig]` bond classes。
   - masked by `adj_from_node_mask(mask, self_connect=True)`。
   - mask strategy 时只在 masked bonds 上算。
   - multiplied by `self.bond_loss_weight`。
4. `charge-loss`
   - node-wise CE over charge classes。
   - masked/averaged over ligand atoms。
   - multiplied by `self.charge_loss_weight`。
5. `interaction-loss` if `predict_interactions` or `flow_interactions`
   - focal CE over pocket-ligand interaction classes。
   - masked by pocket × ligand mask。
   - multiplied by `self.interaction_loss_weight`。

Training logs each as `train-{name}` plus `train-loss`。

## H. 生成分子重建与合法性检查流程

### H1. Generation tensor -> RDKit Mol

`LigandPocketCFM._generate_mols()`:

1. Reads generated ligand `coords`, `atomics`, `bonds`, `charges`, `mask`。
2. Calls `MolBuilder.mols_from_tensors()`。
3. `MolBuilder` extracts each molecule by mask, takes argmax for atom/bond/charge distributions。
4. Calls `flowr.util.rdkit.mol_from_atoms(coords, tokens, bonds, charges, sanitise=...)`。

`LigandPocketCFM._generate_ligs()` does the same but first `undo_zero_com_batch()` and extracts ligand atoms from complex by `lig_mask`。

### H2. RDKit construction details

`mol_from_atoms()`:

- validates coordinate/bond/charge shapes。
- maps atom symbols to atomic numbers。
- creates RDKit atoms with formal charges。
- adds a conformer from 3D coordinates。
- if bonds are provided, maps bond index to RDKit bond type and adds non-self bonds。
- updates property cache。
- optional `RemoveHs`, `Kekulize`, `SanitizeMol`。
- returns `None` if atom/bond/sanitize fails。

### H3. Validity / uniqueness

- `mol_is_valid(mol, connected=True|False)` checks RDKit sanitization and optionally full connectivity。
- `sanitize_list()` filters invalid mols, optionally sanitizes and removes duplicate canonical SMILES。
- Sampling scripts can pass `--filter_valid_unique` to repeatedly sample until enough valid/unique ligands or `max_sample_iter` is reached。

### H4. Output formats

- `generate_from_smol.py`: `predictions_multi_*.pt` with RDKit Mol objects and reference PDB paths。
- `generate_from_pdb.py`: `.pt` plus SDF (`samples_<target>.sdf`) and optional protonated SDF; pocket/complex PDB trajectories may also be written。
- `write_sdf_file()` writes RDKit mol list to SDF with `Name` properties。

## I. 已有 metric / reward / scoring 相关代码

FLOWR contains extensive **evaluation/scoring** utilities but no native online RL objective loop.

Existing metric/scoring functions include:

1. `flowr.util.metrics.evaluate_validity()`
   - validity and fully-connected validity。
2. `evaluate_uniqueness()`
   - canonical SMILES uniqueness。
3. `evaluate_mol_metrics()`
   - novelty vs train SMILES, QED, SA, LogP, HDonors/HAcceptors, MolWt, Lipinski, ring counts, aromatic rings, rotatable bonds, TPSA, energy validity, energy, strain energy, opt-RMSD。
4. `evaluate_pb_validity()`
   - PoseBusters dock config validity。
5. `evaluate_posecheck()`
   - PoseCheck clashes/strain/interactions style metrics。
6. `evaluate_gb3_validity()` and `evaluate_gbsb3()`
   - GenBench3D validity and SBDD benchmark, including Vina setup through `VinaProtein`。
7. `evaluate_strain()`, `evaluate_clashes()`
   - separate strain/clash utilities。
8. `interaction_recovery_per_complex()` / `evaluate_interaction_recovery()`
   - PLIF recovery rate and PLIF Tanimoto similarity。
9. Single-purpose eval scripts for PoseBusters, PoseCheck, strain, mol properties, energy, xTB, clashes, dihedrals。

Migration implication: objective/reward code can reuse RDKit reconstruction and some metric functions, but most existing evaluation functions are too heavy for every training step (PoseBusters/Vina/GenBench3D/xTB). For online reward, use lightweight QED/SA/similarity/validity first, and keep docking/PoseBusters as low-frequency/offline evaluation unless explicitly needed.

## J. 配置系统和新增参数的合适位置

FLOWR uses argparse CLI, not a central training YAML. Suitable extension points:

1. `flowr/train.py` parser:
   - Add a default-off group, e.g. `--lift_enabled` / `--rl_finetune`。
   - Add objective/partition/reference/current-eval hyperparameters only under that flag。
2. `build_model(args, ...)`:
   - Choose original `LigandPocketCFM` when disabled。
   - Choose new `LigandPocketLIFT` subclass/wrapper when enabled。
3. `scripts/train_spindr.sh`:
   - Keep unchanged for baseline。
   - Add a new script, e.g. `scripts/train_spindr_lift.sh`, rather than editing baseline script。
4. `flowr/eval`:
   - Keep original evaluation scripts unchanged so baseline and RL model outputs are evaluated identically。
5. Logging/checkpoint:
   - Original monitor `val-fc-validity` should remain baseline default。
   - RL checkpoint monitor should be explicit and not alter baseline unless `lift_enabled`。

## K. 最适合接入 sotmol-rl 方法的代码位置

### K1. 最小侵入方案

Create a new LightningModule subclass or wrapper next to `fm_pocket.py`:

```text
flowr/flowr/models/lift_pocket.py          # 新增，默认不导入/不启用
```

Potential class:

```python
class LigandPocketLIFT(LigandPocketCFM):
    # override training_step or add lift_training_step
```

Why this location:

- `LigandPocketCFM` already has all necessary primitives:
  - pocket-conditioned `forward()`。
  - `builder.extract_ligand_from_complex()` / `extract_pocket_from_complex()`。
  - `_generate()` for explicit model sampling needs adaptation because current implementation always uses `self` and EMA behavior may be implicit via callback rather than model attr。
  - `_generate_mols()` / `_generate_ligs()` for RDKit reconstruction。
  - `_loss()` components can be adapted to per-sample losses。
- New class can preserve original `LigandPocketCFM` baseline unchanged。

### K2. Needed additions for LIFT migration

1. **Explicit model choice for generation/forward**
   - Current `_generate()` calls `self(...)` and uses `self.gen` internally。
   - LIFT needs current vs reference model explicit selection. Add helper analogous to sotmol `_forward_with_model(model, ...)` or pass generator module into forward helper。

2. **Reference model lifecycle**
   - Deepcopy `self.gen` into `self.ref_gen` on fit start。
   - Freeze/eval reference。
   - EMA update after optimizer step / train batch end。

3. **Per-sample losses**
   - Current `_loss()` returns batch-mean scalar losses。
   - LIFT needs per-sample coord/type/bond/charge losses `[B]` for top/bottom masking。
   - Add new helper methods; do not change original `_loss()` behavior unless carefully guarded by `lift_enabled`。

4. **Reward/objective interface**
   - Use `_generate_mols()` / `_generate_ligs()` to convert generated ligand tensors to RDKit mol list。
   - Add objective classes in a separate module, or reuse/adapt `sotmol-rl` logic in flowr terms。
   - Keep heavy metrics offline/low-frequency。

5. **Partition selector**
   - Add a history-aware selector module outside core baseline code。
   - It should consume score, components, raw properties, valid/connected, canonical smiles/scaffold/fps。

6. **Pseudo-target batch construction for pocket-conditioned generation**
   - Unlike `sotmol-rl`, generated target is ligand-only but condition must keep pocket_data from original `data`。
   - Need to construct generated ligand target while preserving `complex`, `pocket_mask`, `lig_mask`, interactions/fragment masks as needed。

7. **Sampling from reference in training step**
   - Use original `prior` and `pocket_data` from batch。
   - Generate ligand samples with `ref_gen` conditioned on same pocket。
   - Reconstruct generated ligands and score.

### K3. Alternative lighter path

If adding a full subclass is too large, an intermediate migration phase can implement a **read-only offline scorer** for generated `predictions_multi_*.pt` to reproduce reward/partition logic before integrating into training. This would validate objective and partition mapping without touching baseline training.

## L. 迁移风险点

1. **Current vs EMA vs reference confusion**
   - Original FLOWR uses EMA callback, not necessarily an in-module `ema_gen` like sotmol。
   - LIFT needs explicit frozen reference model; do not rely on Lightning EMA callback for reference sampling unless semantics are clear。

2. **Per-sample loss missing**
   - Original `_loss()` returns mean scalar losses; LIFT masking requires `[B]` losses。

3. **Complex/pocket condition preservation**
   - Generated pseudo-target must keep pocket condition fixed and maintain lig/pocket masks, COM alignment, fragment masks, interaction tensors。

4. **Coordinate scaling and COM alignment**
   - Complex transform currently asserts no coord scaling; generation still multiplies by `coord_scale` and undoes COM。
   - Reward RDKit mols must be in real coordinates after undoing COM.

5. **Interaction tensor shape**
   - Interactions appear as pocket-ligand matrices with permutations in batch/loss. RL reward may ignore interactions initially, but inpainting/interaction-conditioned generation must preserve shapes。

6. **Heavy evaluation dependencies**
   - GenBench3D/Vina/PoseBusters/PoseCheck/xTB can be slow or require external binaries. Avoid online use unless explicitly requested。

7. **Hardcoded paths**
   - `genbench3d/config/default.yaml` has local ADFR/Schrödinger/genbench3d_data paths。
   - Data preprocessing sets PLINDER env paths under home directory。

8. **Script bugs / variable mismatch**
   - `gen_pdb.sl` has likely variable name mismatch (`conditional_generation` vs `conditional_gen`, missing `sample_strategy`)。
   - Direct Python commands are safer for reproducible baseline.

9. **DDP / distributed scoring**
   - Objective scoring with RDKit and history selector in DDP needs rank synchronization or rank-local semantics. Original baseline training does not solve this because it does not score generated rewards online。

10. **Baseline preservation**
    - Changing `LigandPocketCFM.training_step()`, `_loss()`, `_generate()`, or `flowr.train` defaults can silently alter baseline. Prefer subclass + default-off flag.

## M. 最小 smoke test 命令建议

These are suggested after implementation or when validating original baseline resources.

1. **Data/statistics presence check**

```bash
# 在 repo-rl/flowr 目录下运行
python - <<'PY'
from pathlib import Path
p = Path('/path/to/spindr')
for f in ['train.smol', 'val.smol', 'test.smol']:
    print(f, (p/f).exists())
for f in ['processed/train_n_noh.pickle', 'processed/train_atom_types_noh.npy', 'processed/train_bond_types_noh.npy', 'processed/train_charges_noh.npy']:
    print(f, (p/f).exists())
PY
```

2. **Small train smoke（推断）**

```bash
# 在 repo-rl/flowr 目录下运行
export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES=0 python -m flowr.train \
  --gpus 1 --epochs 1 --batch_cost 64 --val_batch_cost 4 \
  --d_model 128 --d_edge 64 --n_coord_sets 32 --pocket_d_model 128 --pocket_n_layers 2 --n_layers 2 \
  --arch pocket --pocket_noise fix --dataset spindr --exp_name smoke \
  --data_path /path/to/spindr --save_dir /path/to/smoke_ckpts \
  --self_condition --use_ema --remove_hs --categorical_strategy uniform-sample
```

3. **Small sampling smoke**

```bash
# 在 repo-rl/flowr 目录下运行
export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES=0 python -m flowr.gen.generate_from_smol \
  --mp_index 1 --gpus 1 --batch_cost 4 --arch pocket --pocket_noise fix \
  --dataset_split test --ckpt_path /path/to/flowr.ckpt \
  --data_path /path/to/spindr --dataset spindr --save_dir /path/to/sample_smoke \
  --max_sample_iter 1 --sample_n_molecules_per_target 1 --integration_steps 5 \
  --ode_sampling_strategy linear --categorical_strategy uniform-sample --sample_mol_sizes
```

4. **Light interaction/molecule output sanity**

```bash
# 在 repo-rl/flowr 目录下运行
python - <<'PY'
from pathlib import Path
import torch
p = Path('/path/to/sample_smoke/predictions_multi_1.pt')
obj = torch.load(p, map_location='cpu')
print(obj.keys())
print(len(obj['gen_ligs']))
print(type(obj['gen_ligs'][0][0]) if obj['gen_ligs'] and obj['gen_ligs'][0] else None)
PY
```

5. **Full evaluation smoke**

```bash
# 在 repo-rl/flowr 目录下运行；需要 genbench3d_data 与 ADFR 路径可用
export PYTHONPATH="$PWD"
python -m flowr.eval.evaluate_metrics \
  --num_workers 2 --dataset spindr --data_path /path/to/spindr \
  --save_dir /path/to/sample_smoke --multiple_files --remove_hs
```

## N. 原始 flowr baseline 应该如何保留，方便后续与 RL 版本对比

1. **不要修改 baseline scripts**
   - 保留 `scripts/train_spindr.sh`, `scripts/gen_spindr.sl`, `scripts/gen_pdb.sl`, `scripts/eval_spindr.sh`。
   - 新增 RL 脚本时使用不同名称，例如 `train_spindr_lift.sh`, `gen_spindr_lift.sl`。

2. **不要改变 baseline defaults**
   - `flowr.train` 默认仍应构建 `LigandPocketCFM`。
   - `--lift_enabled` / `--rl_finetune` 默认必须 false。
   - 原 `val-fc-validity` checkpoint monitor 保持不变。

3. **保存 baseline artifacts**
   - baseline checkpoint 目录：`<main_path>/<baseline_exp_name>/`。
   - baseline sample dir：`<baseline_ckpt_path>/eval_.../`。
   - baseline metrics：`metrics.pt`, `interaction_recovery*.pt`。

4. **对比协议建议**
   - 同一 `test.smol` split。
   - 同一 checkpoint init（pretrained FLOWR 或 baseline trained checkpoint）。
   - 同一 `integration_steps`, `ode_sampling_strategy`, `sample_n_molecules_per_target`, `sample_mol_sizes`, `filter_valid_unique`。
   - 同一 evaluation scripts 和 external config。
   - 记录 random seed、CUDA device、commit hash、data path、checkpoint path。

5. **最小迁移验收标准**
   - 当 RL flag 关闭时，训练/采样/评估输出应与原 baseline 命令兼容。
   - RL 版本输出 `predictions_multi_*.pt` / SDF 格式应能被原 `evaluate_metrics.py` 和 `evaluate_interactions.py` 直接消费。
   - 若新增 reward logs，不应删除或重命名 baseline logs。
