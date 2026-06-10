# 02a. flowr 原始使用流程复现说明

> 范围：本文件只分析 `flowr/` 原始项目如何训练、采样、评估 structure-based / pocket-conditioned 3D ligand generation baseline，供后续与迁移后的 RL / reward-guided fine-tuning 版本做对比。本阶段未修改 `flowr/` 或 `sotmol-rl/` 代码。
>
> 重要说明：`flowr/README.md` 明确该仓库主要用于复现实验，且推荐后续开发转向 FLOWR.root；因此本文同时记录 README/script 中的原始流程，以及根据 Python 调用链推断出的最小运行方式。凡 README 或脚本未完整说明的地方均标注为“推断”。

## A. flowr 原始任务定义

FLOWR 的原始任务是 **structure-aware / protein pocket conditioned de novo ligand generation**：给定 protein pocket（以及可选 reference ligand / interaction / scaffold / fragment / linker 条件），训练一个连续 + 离散 flow matching 模型生成 pocket-conditioned 3D ligand。

核心特点：

1. **训练数据**：SPINDR pocket-ligand complex 数据，主要用 `.smol` 文件表示 train/val/test split。
2. **条件输入**：protein pocket 以 atom coordinates、atom names、residue names、bond types、formal charges、mask 等张量输入模型；ligand 是生成对象。
3. **模型架构**：原始 SBDD baseline 默认使用 `--arch pocket`，即 `LigandPocketCFM` + `LigandGenerator` + `PocketEncoder`。
4. **生成方式**：从 ligand prior / noised ligand state 出发，在固定 pocket condition 下通过 integrator 迭代更新 ligand coordinates、atom types、bond types、charges；可选 interaction/scaffold/functional group/linker/substructure inpainting。
5. **评估方式**：对 generated RDKit mol 计算 validity、uniqueness、QED、SA、LogP、TPSA、strain、GenBench3D、PoseBusters、PoseCheck、Vina / docking-like SBDD metrics、PLIF interaction recovery 等。

原始 baseline 对比建议：后续迁移 RL 方法时，必须保留 `flowr` 原始训练命令、采样命令、评估命令和 checkpoint 目录，使用同一数据 split、同一 pretrained/from-scratch checkpoint、同一采样 seed/steps/target 数量进行对比。

## B. 环境安装说明

README 给出的安装方式：

```bash
# 在 flowr/ 目录下运行
mamba env create -f environment.yml
conda activate flowr
export PYTHONPATH="$PWD"
```

关键依赖来自 `flowr/environment.yml`：

- Python 3.11
- PyTorch 2.5.1 + CUDA toolkit 11.8
- Lightning 2.5.0
- RDKit 2023.9.6
- OpenBabel / openbabel-wheel
- xTB, Vina, OpenMM, OpenFF toolkit, openmmforcefields
- ProDy, Biopython, MDAnalysis, pdbfixer, hydride, biotite
- prolif, posecheck/posebusters local package code
- tensorboard, wandb, mlflow
- `plinder==0.2.5`
- `genbench3d`/PoseBusters相关本地代码

README 还要求安装 ADFR suite，并把 `genbench3d/config/default.yaml` 里的 `prepare_receptor_bin_path` 改成本机 ADFR 路径。

建议环境变量：

```bash
# 在 repo-rl/flowr 目录下
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0        # 单卡示例；多卡训练按脚本使用 --gpus 8
# 可选：MLflow logger 使用环境变量
export MLFLOW_TRACKING_URI=...
export MLFLOW_RUN_ID=...
```

注意：`generate_from_smol.py` 和 `generate_from_pdb.py` 内部直接 `model.to("cuda")`，所以采样脚本实际需要 CUDA；CPU-only 最小 smoke 需要改代码才可行，不属于本阶段。

## C. 数据下载与预处理

### C1. README 中的数据来源

README 提供 3 个 Zenodo 资源：

1. **FLOWR checkpoints**：`https://zenodo.org/records/15737419`
   - README 说有两个 checkpoint：一个不带 explicit hydrogens，一个带 explicit hydrogens。
2. **SPINDR dataset and generated samples**：`https://zenodo.org/records/15257565`
   - 包含 `.smol` 和 `.cif` 格式数据，以及可能的 generated samples。
3. **Raw SPINDR pocket/ligand data**：`https://zenodo.org/records/15991057`
   - 用于从 raw 数据自行预处理。

README 还要求下载 `genbench3d_data.tar` 并解压到 repo 内，供 GenBench3D / SBDD evaluation 使用。

### C2. 训练数据目录结构

训练入口 `flowr.train` 直接从 `--data_path` 读取：

```text
<data_path>/
  train.smol
  val.smol
  test.smol
  processed/
    train_noh.pt / train_h.pt                         # 可能未实际读取，但 get_statistics 预留
    train_n_noh.pickle / train_n_h.pickle
    train_atom_types_noh.npy / train_atom_types_h.npy
    train_bond_types_noh.npy / train_bond_types_h.npy
    train_charges_noh.npy / train_charges_h.npy
    ... val_* ...
    ... test_* ...
  train_mols.pkl                                      # evaluation 的 calc_metrics 需要；README 未明确说明，推断需由数据包提供或另行生成
```

`train_spindr.sh` 中：

```bash
main_path="your_main_directory"
dataset="spindr"
data_path="$main_path/$dataset"
save_dir="$main_path/$exp_name"
```

因此如果在本机放置数据，最小建议结构为：

```text
/workspace/data/flowr/
  spindr/
    train.smol
    val.smol
    test.smol
    processed/
      train_n_noh.pickle
      train_atom_types_noh.npy
      train_bond_types_noh.npy
      train_charges_noh.npy
      ...
    train_mols.pkl
```

### C3. 预处理入口

仓库包含 3 类数据处理脚本：

1. **从 PLINDER/raw SPINDR 生成 intermediate `.smol`**

```bash
# 在 flowr/ 目录下运行；group_index 是 0,a,b,... 分组编号映射的整数索引
python -m flowr.data.preprocess \
  --data_path /path/to/raw_plinder_or_spindr \
  --save_path /path/to/output_spindr \
  --group_index 0 \
  --n_workers 8
```

输出：`<save_path>/intermediate/<group_code>.smol`。

该脚本内部硬编码/设置 PLINDER 环境：`PLINDER_RELEASE=2024-06`、`PLINDER_ITERATION=v2`、`PLINDER_REPO=~/plinder-org/plinder`、`PLINDER_LOCAL_DIR=~/.local/share/plinder`、`GCLOUD_PROJECT=plinder`。如果使用 raw PLINDER 预处理，需要手动确认这些路径和权限。

2. **合并 intermediate 并写 train/val/test `.smol`**

```bash
# 在 flowr/ 目录下运行
python -m flowr.data.accumulate \
  --save_path /path/to/output_spindr
```

输入：`<save_path>/intermediate/*.smol`。
输出：`<save_path>/processed/train.smol`、`val.smol`、`test.smol`。

注意：训练脚本期望 `<data_path>/train.smol` 在根目录，而 `accumulate.py` 写到 `<save_path>/processed/train.smol`。因此如果从 raw 预处理，可能需要手动移动/复制到训练 `data_path` 根目录，或把 `--data_path` 指向含 `train.smol` 的目录。README 推荐直接下载 `smol_data.zip`，这通常更简单。

3. **从 PDB/SDF/TXT 文件预处理小数据集**

```bash
# 在 flowr/ 目录下运行；推断用于自定义 PDB/SDF 数据，不是 README 主训练路径
python -m flowr.data.preprocess_pdbs \
  --data_path /path/to/pdb_sdf_txt_dir \
  --save_path /path/to/smol_out \
  --split train \
  --cut_pocket \
  --pocket_cutoff 6.0 \
  --compute_interactions
```

输出：`<save_path>/<split>.smol`。

4. **统计文件生成**

训练会读取 `<data_path>/processed/{split}_...` 统计文件。如果下载数据包未包含这些文件，可用：

```bash
# 在 flowr/ 目录下运行；分别跑 train/val/test
python -m flowr.data.get_statistics --data_dir /path/to/spindr --state train --remove_hs
python -m flowr.data.get_statistics --data_dir /path/to/spindr --state val --remove_hs
python -m flowr.data.get_statistics --data_dir /path/to/spindr --state test --remove_hs
```

`--remove_hs` 与训练脚本中的 `--remove_hs` 必须一致；否则训练读取的 `*_noh.*` / `*_h.*` 统计文件会不匹配。

## D. 训练流程

### D1. 原始训练入口

原始训练入口是：

```bash
python -m flowr.train ...
```

README 指向 bash 脚本：

```bash
# 在 flowr/ 目录下运行
bash scripts/train_spindr.sh
```

脚本 `scripts/train_spindr.sh` 调用 `python -m flowr.train`，并传入 SPINDR pocket-conditioned baseline 参数。

### D2. 原始训练脚本关键参数

`train_spindr.sh` 中核心参数：

- `--gpus 8`
- `--use_bucket_sampler`
- `--bucket_cost_scale linear`
- `--batch_cost 3000`
- `--val_batch_cost 10`
- model size：`--d_model 512`, `--d_edge 256`, `--n_coord_sets 256`, `--pocket_d_model 384`, `--pocket_n_layers 6`
- `--arch pocket`
- `--pocket_noise fix`
- `--epochs 400`
- `--self_condition`
- `--max_atoms_pocket 600`
- `--dataset spindr`
- `--data_path "$main_path/spindr"`
- `--save_dir "$main_path/$exp_name"`
- `--use_ema --ema_decay 0.999`
- `--lr 5.0e-4 --lr_schedule exponential --lr_gamma 0.998`
- `--use_lig_pocket_rbf`
- `--remove_hs`
- `--optimal_transport equivariant`
- `--categorical_strategy uniform-sample`
- inpainting training flags：`--mixed_uncond_inpaint`, `--interaction_inpainting`, `--scaffold_inpainting`, `--func_group_inpainting`

### D3. checkpoint 与 logger

训练使用 Lightning `Trainer`：

- logger：`MLFlowLogger(experiment_name=args.dataset + "_" + args.exp_name)`；如果 `--wandb`，还会启用 offline WandB。
- checkpoint dir：`--save_dir`。
- checkpoint callback：
  - `--use_ema` 时使用 `EMAModelCheckpoint`；否则使用 Lightning `ModelCheckpoint`。
  - `save_top_k=3`
  - `monitor="val-fc-validity"`
  - `mode="max"`
  - `save_last=True`

因此训练后预期：

```text
<save_dir>/
  last.ckpt
  epoch=...-step=...ckpt     # top-k checkpoint，具体文件名由 Lightning callback 决定
  ... MLflow/WandB artifacts  # logger 相关输出取决于环境
```

### D4. 从 checkpoint 继续训练

`flowr.train` 支持 `--load_ckpt`。若 checkpoint optimizer lr 与 `--lr` 不同，代码会临时保存一个 `retraining_with_lr{lr}.ckpt` 并从该 ckpt 恢复。

## E. 采样流程

FLOWR 有两种 README 主采样路径：

1. 从 SPINDR `.smol` test split 中的 pocket/complex 采样：`flowr.gen.generate_from_smol`。
2. 从用户提供的 PDB/CIF protein/pocket + 可选 ligand 文件采样：`flowr.gen.generate_from_pdb`。

### E1. 从 `.smol` test split 采样

脚本：`scripts/gen_spindr.sl`（SLURM array script）。

调用：

```bash
python -m flowr.gen.generate_from_smol \
  --mp_index "${SLURM_ARRAY_TASK_ID}" \
  --gpus "$num_gpus" \
  --batch_cost "$batch_cost" \
  --arch pocket \
  --pocket_noise fix \
  --dataset_split test \
  --ckpt_path "$ckpt" \
  --data_path "$data_path" \
  --dataset spindr \
  --save_dir "$save_dir" \
  --max_sample_iter 20 \
  --coord_noise_std 0.0 \
  --sample_n_molecules_per_target 100 \
  --integration_steps 100 \
  --ode_sampling_strategy linear \
  --sample_mol_sizes
```

输入：

- `--ckpt_path`: trained/provided FLOWR checkpoint。
- `--data_path`: 包含 `test.smol` 的 SPINDR 数据目录。
- `--dataset_split`: 默认 `test`。
- `--mp_index` 与 `--gpus`: 用于把 systems split 成多个 chunks；注意代码使用 `split_list(systems, args.gpus)[args.mp_index - 1]`，所以 `mp_index` 实际应从 1 开始。
- `--sample_n_molecules_per_target`: 每个 target 生成数量。
- `--sample_mol_sizes`: 从数据分布采样 ligand size，而非固定 reference ligand size。
- 可选 inpainting flags：`--interaction_inpainting`, `--scaffold_inpainting`, `--func_group_inpainting`, `--linker_inpainting`。

输出：

```text
<save_dir>/
  predictions_multi_<mp_index>.pt
  # 或 predictions_multi_valid_unique_<mp_index>.pt，如果 --filter_valid_unique
  ref_pdbs/...
```

`.pt` 是 `torch.save(out_dict)`，主要包含：

- `gen_ligs`: list[list[RDKit Mol]]，每个 target 一组 generated ligands。
- `ref_ligs`: reference ligand RDKit mol。
- `ref_ligs_with_hs`
- `ref_pdbs`
- `ref_pdbs_with_hs`
- `time_per_complex`, `time_per_pocket`

### E2. 从 PDB/CIF + ligand 文件采样

脚本：`scripts/gen_pdb.sl`。

调用：

```bash
python -m flowr.gen.generate_from_pdb \
  --pdb_file "$pdb_file" \
  --ligand_file "$lig_file" \
  --compute_interactions \
  --compute_interaction_recovery \
  --protonate_generated_ligands \
  --cut_pocket \
  --pocket_cutoff 6 \
  --gpus 1 \
  --batch_cost 100 \
  --arch pocket \
  --pocket_type holo \
  --ckpt_path "$ckpt" \
  --save_dir "$save_dir" \
  --max_sample_iter 20 \
  --coord_noise_std 0.0 \
  --sample_n_molecules_per_target 1000 \
  --categorical_strategy uniform-sample \
  --filter_valid_unique \
  --sample_mol_sizes
```

输入要求：

- `--pdb_file`: protein/pocket PDB 或 CIF。
- `--ligand_file`: SDF 或 PDB ligand；如果提供完整 protein 而非 pocket，且要 `--cut_pocket`，必须提供 ligand 用于切 pocket。
- `--num_heavy_atoms`: 若不提供 ligand_file 做 unconditional generation，需要提供 heavy atom 数；若同时 `--sample_mol_sizes`，size 在该值附近变化（README 注释说约 ±10%）。
- `--pocket_type holo|apo` 必填。
- `--ckpt_path`, `--save_dir` 必填。

输出：

```text
<save_dir>/
  samples_<target_name>.pt
  samples_<target_name>.sdf
  samples_<target_name>_protonated.sdf        # 如果 --protonate_generated_ligands
  ref_pdbs/...
  trajectories_*                              # generate_from_pdb 默认 save_traj=True 时可能产生
```

`.pt` 包含 `gen_ligs`, `ref_lig`, `ref_lig_with_hs`, `ref_pdb`, `ref_pdb_with_hs`, `run_time`，以及可选 `gen_ligs_hs`, `interaction_recovery`, `tanimoto_sims`。

## F. 评估流程

README 主评估脚本是：

```bash
# 在 flowr/ 目录下运行
bash scripts/eval_spindr.sh
```

该脚本串行调用两个 Python module：

1. `python -m flowr.eval.evaluate_metrics`
2. `python -m flowr.eval.evaluate_interactions`

### F1. evaluate_metrics

调用参数：

```bash
python -m flowr.eval.evaluate_metrics \
  --num_workers 100 \
  --dataset spindr \
  --data_path "$data_path" \
  --save_dir "$save_dir" \
  --multiple_files \
  --remove_hs
```

输入：

- `<save_dir>/predictions_multi_*.pt`，如果 `--multiple_files`。
- `<data_path>/processed/*` 统计文件。
- `<data_path>/train_mols.pkl`，用于 novelty。
- `./genbench3d/config/default.yaml` 与 `./genbench3d_data/...`，用于 GenBench3D / SBDD metrics。
- ADFR `prepare_receptor` 路径，用于 VinaProtein。

计算指标：

- validity / fully-connected validity。
- uniqueness。
- molecular metrics：novelty、QED、SA、LogP、HDonors、HAcceptors、MolWt、Lipinski、rings、aromatic rings、rotatable bonds、TPSA、energy validity、energy、strain energy、opt-RMSD。
- dataset statistics：atom/bond/charge/valency 等分布距离（由 `evaluate_statistics` 计算）。
- GenBench3D validity：Validity3D、Uniqueness2D/3D、Diversity2D/3D、StrainEnergy 等。
- SBDD metrics：PoseBusters validity、GenBench3D SBDD / Vina-related metrics、PoseCheck metrics。
- interaction recovery：非 crossdocked 时调用 `compute_interaction_recovery_parallel`。

输出：

```text
<save_dir>/metrics.pt
# 或 metrics_valid_unique.pt，如果 --valid_unique
```

### F2. evaluate_interactions

调用参数：

```bash
python -m flowr.eval.evaluate_interactions \
  --num_workers 100 \
  --dataset spindr \
  --data_path "$data_path" \
  --save_dir "$save_dir" \
  --return_interaction_list \
  --multiple_files \
  --remove_hs
```

计算：PLIF recovery rate 与 PLIF Tanimoto similarity，可输出按 target/list 形式的详细结果。

输出：

```text
<save_dir>/interaction_recovery_list.pt
# 或 interaction_recovery_valid_unique_list.pt，如果 --valid_unique
```

### F3. 其他评估入口

仓库还包含单项评估脚本：

- `flowr.eval.evaluate_posebusters`
- `flowr.eval.evaluate_posecheck`
- `flowr.eval.evaluate_strain`
- `flowr.eval.evaluate_mol_properties`
- `flowr.eval.evaluate_energy`
- `flowr.eval.evaluate_xtb`
- `flowr.eval.evaluate_clashes`
- `flowr.eval.evaluate_dihedrals`

README 主流程使用 `evaluate_metrics` + `evaluate_interactions`；其他脚本可作为后续诊断/扩展。

## G. bash 脚本到 Python 文件的调用关系

| bash / SLURM script | 目的 | Python module |
|---|---|---|
| `scripts/train_spindr.sh` | SPINDR 原始 baseline 训练 | `python -m flowr.train` |
| `scripts/gen_spindr.sl` | 从 SPINDR `.smol` split 批量采样 | `python -m flowr.gen.generate_from_smol` |
| `scripts/gen_pdb.sl` | 从 PDB/CIF + ligand 单 target 采样 | `python -m flowr.gen.generate_from_pdb` |
| `scripts/eval_spindr.sh` | 主评估：分子/SBDD metrics + interaction recovery | `python -m flowr.eval.evaluate_metrics`; `python -m flowr.eval.evaluate_interactions` |

## H. 关键配置文件和关键参数

### H1. `environment.yml`

环境依赖配置。详见 B 节。

### H2. `genbench3d/config/default.yaml`

评估配置，关键路径：

- `benchmark_dirpath: ./genbench3d_data/`
- `results_dir: ./genbench3d_data/results/`
- `test_set_dir: ./genbench3d_data/test_set/`
- `bin.prepare_receptor_bin_path: ./ADFRsuite/bin/prepare_receptor`
- `bin.glide_path: /usr/local/shared/schrodinger/current/glide`
- `bin.structconvert_path: /usr/local/shared/schrodinger/current/utilities/structconvert`
- `data.ligboundconf_path: ./genbench3d_data/S2_LigBoundConf_minimized.sdf`
- `data.csd_drug_subset_path: ./genbench3d_data/CSD_Drug_Subset.gcd`
- `vina.*`: scoring function、box border、CPU、seed。

必须本地确认并改掉硬编码路径：ADFR、Schrödinger、genbench3d_data。

### H3. CLI 作为主要配置系统

FLOWR 没有 Hydra/YAML 训练配置；训练/采样主要依赖 argparse CLI。关键参数：

- dataset/data：`--dataset`, `--data_path`, `--remove_hs`。
- model：`--arch`, `--d_model`, `--d_edge`, `--n_coord_sets`, `--pocket_d_model`, `--pocket_n_layers`, `--max_atoms`, `--max_atoms_pocket`。
- pocket：`--pocket_noise fix|random|apo`, `--ligand_only`, `--use_lig_pocket_rbf`。
- loss：`--coord_loss_weight`, `--type_loss_weight`, `--bond_loss_weight`, `--charge_loss_weight`, `--interaction_loss_weight`, `--use_t_loss_weights`。
- flow/noise：`--categorical_strategy`, `--coord_noise_std_dev`, `--pocket_coord_noise_std_dev`, `--time_alpha`, `--time_beta`, `--optimal_transport`。
- training：`--epochs`, `--lr`, `--lr_schedule`, `--lr_gamma`, `--batch_cost`, `--val_batch_cost`, `--use_bucket_sampler`, `--use_ema`, `--ema_decay`, `--self_condition`。
- inpainting：`--mixed_uncond_inpaint`, `--interaction_inpainting`, `--scaffold_inpainting`, `--func_group_inpainting`, `--linker_inpainting`, `--substructure_inpainting`, `--fragment_inpainting`。
- sampling：`--ckpt_path`, `--save_dir`, `--sample_n_molecules_per_target`, `--sample_mol_sizes`, `--integration_steps`, `--ode_sampling_strategy`, `--filter_valid_unique`。

## I. 输入输出文件格式

| 阶段 | 输入 | 输出 |
|---|---|---|
| 训练 | `.smol` train/val/test；`processed/*.npy/*.pickle`; optional `load_ckpt` | Lightning `.ckpt` in `save_dir`; MLflow/WandB logs |
| SPINDR 采样 | checkpoint `.ckpt`; `<data_path>/<split>.smol` | `predictions_multi_*.pt`; ref PDB files |
| PDB 采样 | checkpoint `.ckpt`; PDB/CIF protein/pocket; optional SDF/PDB ligand; optional residue TXT | `samples_<target>.pt`; `samples_<target>.sdf`; optional protonated SDF; ref/generated PDBs/trajectories |
| 主评估 | `predictions_multi_*.pt`; data statistics; train_mols.pkl; genbench3d_data; ADFR path | `metrics.pt`; `interaction_recovery*.pt` |
| molecule tensors | coords, atomics, bonds, charges, masks, lig/pocket masks | RDKit `Chem.Mol`, SDF, PDB, torch `.pt` |

## J. checkpoint 使用方式

1. **下载 pretrained checkpoint**
   - README: `https://zenodo.org/records/15737419`。
   - 需要选择 with-H 或 no-H checkpoint，并与 `--remove_hs` 采样/评估设置一致。

2. **训练得到 checkpoint**
   - `flowr.train` 将 checkpoint 保存到 `--save_dir`。
   - 默认 monitor 是 `val-fc-validity`，使用 EMA 权重评估/保存。

3. **采样加载 checkpoint**
   - `generate_from_smol.py` / `generate_from_pdb.py` 先 `torch.load(args.ckpt_path)` 读取 `hyper_parameters`，再重建 `PocketEncoder`、`LigandGenerator`、`Integrator`，最后调用 `LigandPocketCFM.load_from_checkpoint()`。
   - 因此 checkpoint 内的 hparams 与采样 CLI 必须兼容，例如 `arch=pocket`、`pocket_noise`、`remove_hs`、`flow_interactions`。

4. **继续训练**
   - `flowr.train --load_ckpt /path/to.ckpt`。

## K. 最小训练命令

### K1. README 原始多卡命令

```bash
# 在 repo-rl/flowr 目录下运行
export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_spindr.sh
```

前提：先把 `scripts/train_spindr.sh` 中 `your_main_directory`、`your_experiment_name` 改成本地路径。

### K2. 最小单卡/小成本 smoke train（推断）

```bash
# 在 repo-rl/flowr 目录下运行
export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES=0 python -m flowr.train \
  --gpus 1 \
  --batch_cost 64 \
  --val_batch_cost 4 \
  --d_model 128 \
  --d_edge 64 \
  --n_coord_sets 32 \
  --pocket_d_model 128 \
  --pocket_n_layers 2 \
  --n_layers 2 \
  --arch pocket \
  --pocket_noise fix \
  --epochs 1 \
  --val_check_epochs 1 \
  --self_condition \
  --max_atoms_pocket 600 \
  --dataset spindr \
  --exp_name smoke_flowr_baseline \
  --data_path /path/to/spindr \
  --save_dir /path/to/flowr_smoke_ckpts \
  --use_ema \
  --ema_decay 0.999 \
  --lr 5.0e-4 \
  --lr_schedule exponential \
  --lr_gamma 0.998 \
  --use_lig_pocket_rbf \
  --remove_hs \
  --optimal_transport equivariant \
  --categorical_strategy uniform-sample
```

说明：这是根据 CLI 推断的 smoke 命令，是否能在小 GPU 上跑取决于数据大小、bucket、max pocket atoms 和显存。若数据使用 `--remove_hs`，必须已有 `processed/*_noh.*` 统计文件。

## L. 最小采样命令

### L1. 从 `.smol` test split 采样（推断的非 SLURM 单进程形式）

```bash
# 在 repo-rl/flowr 目录下运行
export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES=0 python -m flowr.gen.generate_from_smol \
  --mp_index 1 \
  --gpus 1 \
  --batch_cost 8 \
  --arch pocket \
  --pocket_noise fix \
  --dataset_split test \
  --ckpt_path /path/to/flowr.ckpt \
  --data_path /path/to/spindr \
  --dataset spindr \
  --save_dir /path/to/flowr_samples \
  --max_sample_iter 2 \
  --coord_noise_std 0.0 \
  --sample_n_molecules_per_target 2 \
  --integration_steps 10 \
  --ode_sampling_strategy linear \
  --categorical_strategy uniform-sample \
  --sample_mol_sizes
```

### L2. 从 PDB/CIF + ligand 采样

```bash
# 在 repo-rl/flowr 目录下运行
export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES=0 python -m flowr.gen.generate_from_pdb \
  --pdb_file /path/to/protein_or_pocket.pdb \
  --ligand_file /path/to/reference_ligand.sdf \
  --cut_pocket \
  --pocket_cutoff 6 \
  --gpus 1 \
  --batch_cost 8 \
  --arch pocket \
  --pocket_type holo \
  --ckpt_path /path/to/flowr.ckpt \
  --save_dir /path/to/flowr_pdb_samples \
  --max_sample_iter 2 \
  --coord_noise_std 0.0 \
  --sample_n_molecules_per_target 2 \
  --integration_steps 10 \
  --categorical_strategy uniform-sample \
  --filter_valid_unique \
  --sample_mol_sizes
```

如需 interaction recovery，则额外加 `--compute_interactions --compute_interaction_recovery --protonate_generated_ligands`，但会增加依赖和运行时间。

## M. 最小评估命令

### M1. 主 metrics

```bash
# 在 repo-rl/flowr 目录下运行
export PYTHONPATH="$PWD"
python -m flowr.eval.evaluate_metrics \
  --num_workers 4 \
  --dataset spindr \
  --data_path /path/to/spindr \
  --save_dir /path/to/flowr_samples \
  --multiple_files \
  --remove_hs
```

### M2. interaction recovery

```bash
# 在 repo-rl/flowr 目录下运行
export PYTHONPATH="$PWD"
python -m flowr.eval.evaluate_interactions \
  --num_workers 4 \
  --dataset spindr \
  --data_path /path/to/spindr \
  --save_dir /path/to/flowr_samples \
  --return_interaction_list \
  --multiple_files \
  --remove_hs
```

注意：严格 SBDD metrics 需要 `genbench3d_data`、ADFR、可能的 Vina/Schrödinger 配置；如果只做快速 molecule metrics，需要后续考虑拆分/使用单项评估脚本，或接受部分 evaluation 因外部资源缺失而失败。

## N. 需要我手动确认或下载的资源

1. SPINDR `.smol` 数据：Zenodo `15257565`。
2. FLOWR pretrained checkpoints：Zenodo `15737419`。
3. raw SPINDR pocket/ligand 数据（如果要自行预处理）：Zenodo `15991057`。
4. `genbench3d_data.tar` 并解压到 `flowr/genbench3d_data/` 或更新 `genbench3d/config/default.yaml`。
5. ADFR suite，并更新 `prepare_receptor_bin_path`。
6. 如果使用 Schrödinger/Glide metrics，确认 `glide_path` 与 `structconvert_path`。
7. `train_mols.pkl` 是否在数据包中；如果没有，需要确定如何从 `train.smol` 生成，用于 novelty。
8. 本机 GPU 资源：原始脚本建议训练/采样用至少 40GB VRAM，训练脚本默认 8 GPUs。
9. `scripts/*.sh/*.sl` 中所有 `your_*` 路径和 SLURM partition/output/error 路径。

## O. 当前不确定的信息

1. README 未直接给出 `smol_data.zip` 解压后的完整目录树；本文根据 `flowr.train` 和 evaluation 代码推断需要 `train.smol/val.smol/test.smol` 在 `data_path` 根目录、statistics 在 `data_path/processed/`。
2. `accumulate.py` 将 split `.smol` 写到 `<save_path>/processed/`，而训练读取 `<data_path>/train.smol`；这可能表示官方数据包已经整理好根目录文件，或者需要人工复制。需要下载数据后确认。
3. `train_mols.pkl` 的生成入口没有在 README 中说明；evaluation `calc_metrics` 会强制读取它。需要确认 Zenodo 数据是否包含，或另写/寻找生成脚本。
4. `generate_from_smol.py::load_util()` 中只有当 `hparams["coord_scale"] == 1.0` 时才设置 `coord_std=1.0`；如果 checkpoint 使用非 1.0 coord scale，变量可能未定义。原始 SPINDR complex transform 当前 assert `coord_std==1.0`，推断官方 checkpoint 使用 `coord_scale=1.0`。
5. `gen_pdb.sl` 的 `save_dir` 使用变量名 `conditional_gen` / `sample_strategy`，但脚本中定义的是 `conditional_generation` 且未定义 `sample_strategy`；需要手动修正脚本或直接运行 Python 命令。
6. `generate_from_pdb.py` 在 `--compute_interaction_recovery` 分支使用 `all_gen_ligs_hs`，如果未启用 `--protonate_generated_ligands` 可能未定义；README 脚本同时启用了 protonation，因此原流程可避免该问题。
7. 评估里 `genbench3d/config/default.yaml` 用相对路径 `./genbench3d_data/...` 和 `./ADFRsuite/...`，因此命令最好在 `flowr/` 目录运行；否则相对路径会失效。
8. 仓库看起来没有原生 RL / reward-guided fine-tuning 训练入口；只有 generation/evaluation scoring utilities。后续 RL 迁移应新增默认关闭的训练分支，而不是覆盖 baseline。
