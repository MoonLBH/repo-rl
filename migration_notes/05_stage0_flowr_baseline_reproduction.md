# 05. Stage 0 — FLOWR 原始 baseline 复现命令记录

> 范围：本文件只整理 FLOWR 原始 training / sampling / evaluation 的可复现命令与当前环境验证结果，用于后续与 RL-FLOWR 做 fair comparison。本文不接入 `sotmol-rl` 的 RL 逻辑，不新增 reward-guided fine-tuning，也不修改 `flowr/` 原始训练、采样、评估逻辑。
>
> 运行目录约定：除特别说明外，命令应在 `repo-rl/flowr/` 目录下运行，并设置 `export PYTHONPATH="$PWD"`。脚本中的 `your_main_directory`、`your_flowr_directory`、`your_ckpt.ckpt` 等占位符必须替换为本地真实路径。

## A. Stage 0 目标与当前结论

Stage 0 的目标是确认 FLOWR 原始 baseline 如何运行，而不是实现或测试 RL 迁移。

当前结论：

1. FLOWR 原始训练入口是 `flowr/scripts/train_spindr.sh`，调用 `python -m flowr.train`。
2. FLOWR 原始 SPINDR test split 采样入口是 `flowr/scripts/gen_spindr.sl`，调用 `python -m flowr.gen.generate_from_smol`。
3. FLOWR 原始 PDB/CIF pocket 采样入口是 `flowr/scripts/gen_pdb.sl`，调用 `python -m flowr.gen.generate_from_pdb`。
4. FLOWR 原始 evaluation 入口是 `flowr/scripts/eval_spindr.sh`，调用 `python -m flowr.eval.evaluate_metrics` 和 `python -m flowr.eval.evaluate_interactions`。
5. 当前容器中没有 FLOWR conda 环境依赖、没有 SPINDR `.smol` 数据、没有 `processed/` statistics、没有 `train_mols.pkl`、没有 pretrained checkpoint，因此不能直接跑训练、采样或 evaluation；只能完成脚本/代码静态核对和 import/help 阻塞点验证。
6. 后续 RL-FLOWR 对比必须固定：同一 dataset split、同一 pocket/target list、同一 checkpoint 初始化、同一采样步数、同一采样数量、同一 filtering 设置、同一 evaluation 脚本和同一外部工具配置。

## B. 原始训练流程

### B1. 原始训练 bash 脚本

- 脚本：`flowr/scripts/train_spindr.sh`
- 运行目录：`repo-rl/flowr/`
- 调用 Python：`python -m flowr.train`
- 任务：在 SPINDR `.smol` 数据上训练 pocket-conditioned FLOWR 模型。

原始脚本的关键变量：

```bash
main_path="your_main_directory"
exp_name="your_experiment_name"
dataset="spindr"
data_path="$main_path/$dataset"
save_dir="$main_path/$exp_name"
```

### B2. 训练 Python 入口

- 文件：`flowr/flowr/train.py`
- parser 入口：`if __name__ == "__main__": parser = argparse.ArgumentParser()`
- 模型构建：`build_model(args, dm, dataset_info, vocab, ...)`
- 数据统计：`build_data_statistic(args)` 从 `$data_path/processed` 读取 train/val/test statistics。
- 数据集：`build_dm(args, ...)` 读取 `$data_path/train.smol` 和 `$data_path/val.smol`。
- checkpoint callback：`build_trainer(args, model)` 使用 `ModelCheckpoint` 或 `EMAModelCheckpoint`，`dirpath=args.save_dir`，monitor 为 `val-fc-validity`，`mode="max"`，`save_last=True`。

### B3. 关键训练配置 / CLI 参数

FLOWR 没有单独的训练 YAML；训练配置主要来自 `flowr/scripts/train_spindr.sh` 和 `flowr.train` CLI。

原始 SPINDR 脚本关键参数：

| 参数 | 原始脚本值 | 作用 |
|---|---:|---|
| `--gpus` | `8` | 多 GPU 训练。 |
| `--use_bucket_sampler` | enabled | 使用 bucket sampler。 |
| `--bucket_cost_scale` | `linear` | bucket cost 缩放。 |
| `--batch_cost` | `3000` | 训练 batch cost。 |
| `--val_batch_cost` | `10` | 验证 batch cost。 |
| `--d_model` | `512` | ligand/model hidden dim。 |
| `--d_edge` | `256` | edge hidden dim。 |
| `--n_coord_sets` | `256` | coordinate heads/sets。 |
| `--pocket_d_model` | `384` | pocket hidden dim。 |
| `--pocket_n_layers` | `6` | pocket encoder layers。 |
| `--arch` | `pocket` | 使用 pocket-conditioned model。 |
| `--pocket_noise` | `fix` | SPINDR 必需，训练中 pocket noise mode。 |
| `--epochs` | `400` | 训练 epoch 数。 |
| `--val_check_epochs` | `1` | 每 epoch 验证。 |
| `--self_condition` | enabled | self-conditioning。 |
| `--max_atoms_pocket` | `600` | pocket atom 上限。 |
| `--dataset` | `spindr` | 数据集名。 |
| `--data_path` | `$main_path/spindr` | 数据目录。 |
| `--save_dir` | `$main_path/$exp_name` | checkpoint/log 保存目录。 |
| `--use_ema` | enabled | 使用 EMA callback 和 EMA checkpoint。 |
| `--ema_decay` | `0.999` | EMA decay。 |
| `--lr` | `5.0e-4` | 学习率。 |
| `--lr_schedule` | `exponential` | LR schedule。 |
| `--lr_gamma` | `0.998` | exponential gamma。 |
| `--use_lig_pocket_rbf` | enabled | ligand-pocket RBF features。 |
| `--remove_hs` | enabled | 去除显式 H。 |
| `--optimal_transport` | `equivariant` | OT strategy。 |
| `--categorical_strategy` | `uniform-sample` | 离散变量 categorical strategy。 |
| `--mixed_uncond_inpaint` | enabled | 混合 unconditional/inpainting。 |
| `--interaction_inpainting` | enabled | interaction inpainting training。 |
| `--scaffold_inpainting` | enabled | scaffold inpainting training。 |
| `--func_group_inpainting` | enabled | functional-group inpainting training。 |

### B4. 训练数据路径和目录结构

训练脚本期望：

```text
$main_path/
  spindr/
    train.smol
    val.smol
    test.smol
    train_mols.pkl              # evaluation 需要；训练不一定直接读
    processed/
      train_*.pkl / train_*.npy  # statistics，具体文件名由 Statistics.get_statistics 约定
      val_*.pkl / val_*.npy
      test_*.pkl / test_*.npy
```

`flowr.train` 明确读取：

- `$data_path/train.smol`
- `$data_path/val.smol`
- `$data_path/processed` 下 train/val/test statistics

README 指向的资源：

- SPINDR `.smol` / `.cif` / generated samples: <https://zenodo.org/records/15257565>
- raw SPINDR pocket/ligand data: <https://zenodo.org/records/15991057>
- FLOWR checkpoints: <https://zenodo.org/records/15737419>
- `genbench3d_data.tar`: README 要求下载并解压到 repo 中；README 未在当前文件内给出单独 URL，需人工从随附数据/Zenodo 页面确认。

### B5. checkpoint 保存路径

训练 checkpoint 保存到：

```text
$save_dir/
  last.ckpt
  epoch=...-step=...ckpt   # top-k by val-fc-validity，具体文件名由 Lightning ModelCheckpoint 决定
```

如果使用 `--use_ema`，使用 `EMAModelCheckpoint`，保存 EMA 权重相关 checkpoint；否则使用标准 `ModelCheckpoint`。两者都监控 `val-fc-validity`。

### B6. 原始训练命令

在 `repo-rl/flowr/` 下运行：

```bash
export PYTHONPATH="$PWD"
export MLFLOW_TRACKING_URI="file:$PWD/../artifacts/mlruns"  # 可选；避免 MLflow 指向未知服务

bash scripts/train_spindr.sh
```

### B7. 最小训练命令（推断，适合 smoke test）

> 推断：该命令保留原始 `flowr.train` 入口，仅把 GPU、epoch、batch cost、维度等降到 smoke 规模；仍需要完整 SPINDR `.smol` 和 processed statistics。若没有 GPU，可把 `--gpus 0` 作为 import/CPU smoke 尝试，但真实训练仍建议 GPU。

在 `repo-rl/flowr/` 下运行：

```bash
export PYTHONPATH="$PWD"
export MLFLOW_TRACKING_URI="file:$PWD/../artifacts/mlruns"
export CUDA_VISIBLE_DEVICES=0

python -m flowr.train \
  --gpus 1 \
  --trial_run \
  --batch_cost 64 \
  --val_batch_cost 16 \
  --d_model 64 \
  --d_edge 32 \
  --n_coord_sets 16 \
  --pocket_d_model 64 \
  --pocket_n_layers 2 \
  --arch pocket \
  --pocket_noise fix \
  --max_atoms_pocket 600 \
  --dataset spindr \
  --exp_name stage0_smoke \
  --data_path /ABS/PATH/TO/spindr \
  --save_dir /ABS/PATH/TO/flowr_stage0_smoke \
  --use_ema \
  --ema_decay 0.999 \
  --lr 5.0e-4 \
  --lr_schedule exponential \
  --lr_gamma 0.998 \
  --use_lig_pocket_rbf \
  --remove_hs \
  --optimal_transport equivariant \
  --categorical_strategy uniform-sample \
  --mixed_uncond_inpaint \
  --interaction_inpainting \
  --scaffold_inpainting \
  --func_group_inpainting
```

### B8. 当前训练流程无法直接运行的阻塞点

当前容器验证结果：

1. `python -m flowr.train --help` 在 import 阶段失败：`ModuleNotFoundError: No module named 'lightning'`。
2. 当前 repo 下未发现 `.smol` 数据、`processed/` statistics、`train_mols.pkl` 或 `.ckpt` checkpoint。
3. 训练需要安装 `environment.yml` 中的依赖，包括 PyTorch、Lightning、RDKit、PyG/torch-geometric 相关包、MLflow、torch_ema 等。
4. 原始训练脚本默认 `--gpus 8`、batch cost 大，不适合当前容器或小资源环境直接运行。

## C. 原始采样流程

FLOWR 有两条原始采样路径：从 SPINDR `.smol` split 采样，以及从单个 PDB/CIF + ligand 文件采样。

### C1. SPINDR test split 采样 bash/SLURM 脚本

- 脚本：`flowr/scripts/gen_spindr.sl`
- 运行方式：原始脚本是 SLURM array job：`sbatch scripts/gen_spindr.sl`
- 调用 Python：`python -m flowr.gen.generate_from_smol`
- 任务：读取 SPINDR test split complexes，基于 checkpoint 为每个 target 生成多个 ligand。

关键变量：

```bash
dataset="spindr"
main_path="your_main_directory"
data_path="$main_path/$dataset"
ckpt_path="$main_path/your_experiment_directory"
ckpt="$ckpt_path/your_ckpt.ckpt"
steps=100
sampling_strategy="linear"
coord_noise_std=0.0
n_molecules_per_target=100
batch_cost=100
save_dir="$ckpt_path/eval_${n_molecules_per_target}-lig-per-target_sampled-mol-sizes_${steps}-steps_linear-sampling-strategy"
```

### C2. SPINDR sampling Python 入口和关键参数

- 文件：`flowr/flowr/gen/generate_from_smol.py`
- parser：`get_args()`
- 必需/关键参数：
  - `--arch pocket`
  - `--pocket_noise fix|random|apo`
  - `--ckpt_path /path/to/flowr.ckpt`
  - `--data_path /path/to/spindr`
  - `--dataset spindr`
  - `--dataset_split test`
  - `--save_dir /path/to/output`
  - `--sample_n_molecules_per_target N`
  - `--integration_steps STEPS`
  - `--ode_sampling_strategy linear|log`
  - `--sample_mol_sizes` for sampled ligand sizes
  - optional `--filter_valid_unique`

The script loads checkpoint hparams, updates runtime hparams such as `save_dir`, `coord_noise_std`, `categorical_strategy`, `integration_steps`, `sample_mol_sizes`, and then writes prediction `.pt` files.

### C3. SPINDR sampling 输入

需要：

1. `--ckpt_path`：训练得到或 Zenodo 提供的 FLOWR checkpoint。
2. `--data_path`：包含 `test.smol` 和 `processed/` statistics 的 SPINDR 数据目录。
3. `--dataset_split test`：从 test split 读取 pocket/ligand complex。
4. 可选 inpainting/condition flags：`--interaction_inpainting`、`--scaffold_inpainting`、`--func_group_inpainting`、`--linker_inpainting`。

### C4. SPINDR sampling 输出目录和格式

`generate_from_smol.py` 写入：

```text
$save_dir/
  predictions_multi_<mp_index>.pt
  predictions_multi_valid_unique_<mp_index>.pt    # 仅当 --filter_valid_unique
  ref_pdbs/
    ...                                           # reference pocket PDBs / with-H PDBs
```

`predictions_multi_*.pt` 是 `torch.save(out_dict)` 的 PyTorch pickle 文件，包含至少：

- `gen_ligs`: generated RDKit Mol lists
- `ref_ligs`
- `ref_ligs_with_hs`
- `ref_pdbs`
- `ref_pdbs_with_hs`
- `time_per_pocket`
- `time_per_complex`

### C5. 原始采样命令 / SPINDR 最小采样命令（非 SLURM，推断）

> 推断：把 SLURM array 中的一项改为单进程命令。仍需要 checkpoint、SPINDR test `.smol`、processed statistics、GPU/Lightning/RDKit 环境。

在 `repo-rl/flowr/` 下运行：

```bash
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0

python -m flowr.gen.generate_from_smol \
  --mp_index 1 \
  --gpus 1 \
  --batch_cost 16 \
  --arch pocket \
  --pocket_noise fix \
  --dataset_split test \
  --ckpt_path /ABS/PATH/TO/flowr.ckpt \
  --data_path /ABS/PATH/TO/spindr \
  --dataset spindr \
  --save_dir /ABS/PATH/TO/stage0_flowr_samples \
  --max_sample_iter 1 \
  --coord_noise_std 0.0 \
  --sample_n_molecules_per_target 1 \
  --integration_steps 5 \
  --ode_sampling_strategy linear \
  --sample_mol_sizes
```

### C6. PDB/CIF + ligand sampling 脚本

- 脚本：`flowr/scripts/gen_pdb.sl`
- 调用 Python：`python -m flowr.gen.generate_from_pdb`
- 用途：给定 protein/pocket PDB/CIF 和可选 reference ligand，生成 ligand；如果输入是完整 protein，需要 reference ligand 来 cut pocket；conditional generation 也需要 reference ligand。

关键输入：

```bash
pdb_file="$data_path/$pdb_id.pdb"
lig_file="$data_path/your_ligand.sdf"
ckpt="$ckpt_path/your_ckpt.ckpt"
save_dir="$ckpt_path/eval_..."
```

关键 flags：

- `--pdb_file`: protein/pocket `.pdb` 或 `.cif`
- `--ligand_file`: optional `.sdf` 或 `.pdb`; cut pocket / conditional generation 时需要
- `--cut_pocket --pocket_cutoff 6`
- `--compute_interactions --compute_interaction_recovery`
- `--protonate_generated_ligands`
- `--protonate_pocket` optional
- `--num_heavy_atoms`：无 ligand 且不从 reference/sample mol sizes 时需要
- `--filter_valid_unique`

### C7. PDB/CIF sampling 输出格式

`generate_from_pdb.py` 写入：

```text
$save_dir/
  samples_<target_name>.pt
  samples_unfiltered_<target_name>.pt      # 未 filter 时
  samples_<target_name>.sdf
  samples_<target_name>_protonated.sdf     # if --protonate_generated_ligands
  ref_pdbs/
  gen_complexes_protonated/                # if protonated complexes are written
```

`.pt` 文件同样是 PyTorch pickle；`.sdf` 是 RDKit SDWriter 输出，适合人工查看和后续 docking/PoseBusters 评估。

### C8. PDB/CIF 最小采样命令（推断）

在 `repo-rl/flowr/` 下运行：

```bash
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0

python -m flowr.gen.generate_from_pdb \
  --pdb_file /ABS/PATH/TO/complex_or_pocket.pdb \
  --ligand_file /ABS/PATH/TO/reference_ligand.sdf \
  --cut_pocket \
  --pocket_cutoff 6 \
  --gpus 1 \
  --batch_cost 16 \
  --arch pocket \
  --pocket_type holo \
  --ckpt_path /ABS/PATH/TO/flowr.ckpt \
  --save_dir /ABS/PATH/TO/stage0_flowr_pdb_samples \
  --max_sample_iter 1 \
  --coord_noise_std 0.0 \
  --sample_n_molecules_per_target 1 \
  --categorical_strategy uniform-sample \
  --filter_valid_unique \
  --sample_mol_sizes
```

如不提供 ligand 且不切 pocket，需要改为指定 `--num_heavy_atoms 20`，并确认输入已经是 pocket 或不需要 reference ligand。

### C9. 当前采样流程无法直接运行的阻塞点

1. `python -m flowr.gen.generate_from_smol --help` 和 `python -m flowr.gen.generate_from_pdb --help` 在 import 阶段失败：`ModuleNotFoundError: No module named 'lightning'`。
2. 当前容器没有 pretrained checkpoint。
3. 当前容器没有 SPINDR `.smol` test split。
4. PDB/CIF sampling 需要用户提供本地 protein/pocket file 与 reference ligand file，当前 repo 未提供。
5. 原始 `gen_spindr.sl` 是 SLURM 脚本，需要本地 HPC 路径、partition、mamba/conda 路径和 `SLURM_ARRAY_TASK_ID`。

## D. 原始评估流程

### D1. 原始 evaluation bash 脚本

- 脚本：`flowr/scripts/eval_spindr.sh`
- 运行目录：`repo-rl/flowr/`
- 调用 Python：
  1. `python -m flowr.eval.evaluate_metrics`
  2. `python -m flowr.eval.evaluate_interactions`

原始脚本使用与 sampling 相同的 `save_dir`，并开启 `--multiple_files`，因此默认读取 `predictions_multi_*.pt`。

### D2. evaluate_metrics 输入/输出

入口：`flowr/flowr/eval/evaluate_metrics.py`

输入：

- `--save_dir`: sampling 输出目录。
- `--multiple_files`: 读取 `$save_dir/predictions_multi_*.pt`。
- `--valid_unique`: 读取 `predictions_multi_valid_unique_*.pt` 并输出 valid_unique metrics。
- `--data_path`: SPINDR 数据目录；需要 `train_mols.pkl` 和 `processed/` statistics。
- `--dataset spindr`。
- `--remove_hs` optional。

输出：

```text
$save_dir/
  metrics.pt
  metrics_valid_unique.pt   # if --valid_unique
```

主要指标：

- validity: `Validity (mean/std)`, `Fc-validity (mean/std)`
- uniqueness: `Uniqueness (mean/std)`
- molecular properties: `QED`, `SA`, `LogP`, donors/acceptors, MolWt, Lipinski, rings, TPSA
- novelty: relative to `train_mols.pkl`
- energy/strain/Opt-RMSD: RDKit/OpenBabel-like energy utilities
- distribution/statistics metrics from `evaluate_statistics`
- GenBench3D validity/diversity/strain-related metrics
- SBDD metrics: GenBench3D SBDD, PoseBusters validity, PoseCheck
- interaction recovery is also computed inside `evaluate_metrics` for non-`crossdocked` datasets

External/extra dependencies:

- `genbench3d/config/default.yaml`
- `genbench3d_data.tar` unpacked in repo as required by README
- PoseBusters bundled code plus dependencies
- PoseCheck/prolif/MDAnalysis/OpenBabel/RDKit ecosystem
- ADFR prepare receptor path in `genbench3d/config/default.yaml` for docking-like preparation flows

### D3. evaluate_interactions 输入/输出

入口：`flowr/flowr/eval/evaluate_interactions.py`

输入：同 `evaluate_metrics` 的 prediction files、reference ligands、reference PDBs。

输出：

```text
$save_dir/
  interaction_recovery.pt
  interaction_recovery_list.pt              # if --return_interaction_list
  interaction_recovery_valid_unique.pt      # if --valid_unique
  interaction_recovery_valid_unique_list.pt # if both flags
```

指标：

- `PLIF recovery rate (mean/std)`
- `PLIF Tanimoto similarity (mean/std)`
- number of molecules / tested molecules / failed molecules
- with `--return_interaction_list`, 保存 per-target/per-molecule recovery list。

### D4. 原始 evaluation 命令

在 `repo-rl/flowr/` 下运行：

```bash
export PYTHONPATH="$PWD"

bash scripts/eval_spindr.sh
```

### D5. 最小 evaluation 命令（推断）

> 推断：要求 `save_dir` 已有 `predictions_multi_*.pt`，且 prediction 中记录的 `ref_pdbs` / `ref_pdbs_with_hs` 文件存在。若只是想做接口 smoke，可先用 `--num_workers 1` 降低并发。

在 `repo-rl/flowr/` 下运行：

```bash
export PYTHONPATH="$PWD"

python -m flowr.eval.evaluate_metrics \
  --num_workers 1 \
  --dataset spindr \
  --data_path /ABS/PATH/TO/spindr \
  --save_dir /ABS/PATH/TO/stage0_flowr_samples \
  --multiple_files \
  --remove_hs

python -m flowr.eval.evaluate_interactions \
  --num_workers 1 \
  --dataset spindr \
  --data_path /ABS/PATH/TO/spindr \
  --save_dir /ABS/PATH/TO/stage0_flowr_samples \
  --return_interaction_list \
  --multiple_files \
  --remove_hs
```

### D6. 当前 evaluation 无法直接运行的阻塞点

1. `python -m flowr.eval.evaluate_metrics --help` 和 `python -m flowr.eval.evaluate_interactions --help` 在 import 阶段失败：`ModuleNotFoundError: No module named 'numpy'`。
2. 当前容器没有 sampling 输出 `predictions_multi_*.pt`。
3. 当前容器没有 `train_mols.pkl`，`evaluate_metrics.calc_metrics()` 会直接报 `FileNotFoundError("Training mols not found.")`。
4. 当前容器没有 `genbench3d_data.tar` 解压结果，也没有外部 ADFR 路径配置。
5. SBDD/PoseBusters/PoseCheck/interactions 可能需要完整 RDKit/prolif/MDAnalysis/OpenBabel/ADFR/prepare receptor 依赖；缺失时不应修改源码绕过，应记录为环境阻塞。

## E. 环境安装与环境变量

### E1. 安装命令

在 `repo-rl/flowr/` 下：

```bash
mamba env create -f environment.yml
conda activate flowr
export PYTHONPATH="$PWD"
```

### E2. 推荐环境变量

| 环境变量 | 示例 | 作用 |
|---|---|---|
| `PYTHONPATH` | `export PYTHONPATH="$PWD"` | 让 `python -m flowr.*` 找到本地 package。 |
| `CUDA_VISIBLE_DEVICES` | `export CUDA_VISIBLE_DEVICES=0` | 控制训练/采样 GPU。 |
| `MLFLOW_TRACKING_URI` | `export MLFLOW_TRACKING_URI="file:$PWD/../artifacts/mlruns"` | 训练 logger 使用 MLflow；本地 file URI 避免未知远程服务。 |
| `MLFLOW_RUN_ID` | optional | 仅当续接已有 MLflow run 时使用。 |
| ADFR/prepare receptor path | 在 `genbench3d/config/default.yaml` 中设置 | README 要求调整 `prepare_receptor_bin_path`。 |

### E3. 当前已验证的 help/import 结果

在 `repo-rl/flowr/` 下执行：

```bash
export PYTHONPATH="$PWD"
python -m flowr.train --help
python -m flowr.gen.generate_from_smol --help
python -m flowr.gen.generate_from_pdb --help
python -m flowr.eval.evaluate_metrics --help
python -m flowr.eval.evaluate_interactions --help
```

当前结果：

- `flowr.train`: import 阶段失败，缺少 `lightning`。
- `flowr.gen.generate_from_smol`: import 阶段失败，缺少 `lightning`；另有 Python `SyntaxWarning`，不影响本阶段判断。
- `flowr.gen.generate_from_pdb`: import 阶段失败，缺少 `lightning`。
- `flowr.eval.evaluate_metrics`: import 阶段失败，缺少 `numpy`。
- `flowr.eval.evaluate_interactions`: import 阶段失败，缺少 `numpy`。

这说明当前容器没有安装 FLOWR runtime environment，不能做实际训练、采样、evaluation smoke。

## F. 关键配置文件与硬编码路径

### F1. 关键配置文件

| 文件 | 用途 |
|---|---|
| `flowr/environment.yml` | conda/mamba 环境定义。 |
| `flowr/scripts/train_spindr.sh` | 原始 SPINDR training 命令模板。 |
| `flowr/scripts/gen_spindr.sl` | 原始 SPINDR SLURM sampling 命令模板。 |
| `flowr/scripts/gen_pdb.sl` | 原始 PDB/CIF sampling 命令模板。 |
| `flowr/scripts/eval_spindr.sh` | 原始 SPINDR evaluation 命令模板。 |
| `flowr/genbench3d/config/default.yaml` | GenBench3D / PoseBusters / receptor preparation 相关配置。 |

### F2. 当前必须替换的硬编码/占位路径

| 位置 | 占位路径/变量 | 需要替换为 |
|---|---|---|
| `scripts/train_spindr.sh` | `your_main_directory`, `your_experiment_name` | 本地数据根目录和实验名。 |
| `scripts/gen_spindr.sl` | `your_flowr_directory`, `your_mamba_path`, `your_conda_path`, `your_partition`, `your_output_path`, `your_main_directory`, `your_experiment_directory`, `your_ckpt.ckpt` | 本地 FLOWR repo、conda、HPC、checkpoint、输出路径。 |
| `scripts/gen_pdb.sl` | 同上 + `your_pdb_id`, `your_ligand.sdf` | 本地 PDB/CIF 和 reference ligand。 |
| `scripts/eval_spindr.sh` | `your_main_directory`, `your_experiment_directory` | 本地数据、checkpoint/sample 输出目录。 |
| `genbench3d/config/default.yaml` | `prepare_receptor_bin_path` 等 | 本地 ADFR installation path。 |

不建议直接修改原始脚本作为 baseline 对比的一部分；更安全做法是在本地实验目录复制脚本或记录 wrapper 命令，并保持 upstream baseline scripts 原样。

## G. 需要的数据、checkpoint 与下载清单

| 资源 | 是否当前 repo 中存在 | 下载/说明 |
|---|---:|---|
| SPINDR `.smol` train/val/test | 否 | README 指向 <https://zenodo.org/records/15257565>。 |
| SPINDR `.cif` train/val/test | 否 | README 指向 <https://zenodo.org/records/15257565>。 |
| raw SPINDR pocket/ligand data | 否 | README 指向 <https://zenodo.org/records/15991057>。 |
| FLOWR pretrained checkpoints | 否 | README 指向 <https://zenodo.org/records/15737419>，有 no-H 和 with-H 两个 checkpoint。 |
| generated samples | 否 | README 称同 SPINDR 数据 Zenodo 提供。 |
| `genbench3d_data.tar` | 否 | README 要求下载并解压到 repo；具体 URL 需要人工在 Zenodo/项目材料中确认。 |
| ADFR suite | 否 | README 指向 <https://ccsb.scripps.edu/adfr/downloads/>。 |

## H. 输出目录与输出文件格式汇总

| 流程 | 输出目录 | 输出格式 |
|---|---|---|
| training | `--save_dir`, 原脚本 `$main_path/$exp_name` | Lightning `.ckpt`, MLflow logs, optional WandB offline logs。 |
| SPINDR sampling | `--save_dir`, 原脚本 `$ckpt_path/eval_...` | `predictions_multi_<idx>.pt`, optional `predictions_multi_valid_unique_<idx>.pt`, `ref_pdbs/*.pdb`。 |
| PDB/CIF sampling | `--save_dir` | `samples_<target>.pt`, `samples_<target>.sdf`, optional protonated SDF and generated complexes。 |
| metrics evaluation | sampling `--save_dir` | `metrics.pt` or `metrics_valid_unique.pt`。 |
| interaction evaluation | sampling `--save_dir` | `interaction_recovery.pt`, `interaction_recovery_list.pt`, valid_unique variants。 |

## I. 当前已验证的内容

本阶段已完成以下静态/轻量验证：

1. 确认训练脚本 `scripts/train_spindr.sh` 调用 `python -m flowr.train`。
2. 确认 SPINDR 采样脚本 `scripts/gen_spindr.sl` 调用 `python -m flowr.gen.generate_from_smol`。
3. 确认 PDB/CIF 采样脚本 `scripts/gen_pdb.sl` 调用 `python -m flowr.gen.generate_from_pdb`。
4. 确认 evaluation 脚本 `scripts/eval_spindr.sh` 调用 `evaluate_metrics` 与 `evaluate_interactions`。
5. 确认训练读取 `$data_path/train.smol`, `$data_path/val.smol`, `$data_path/processed`。
6. 确认 `evaluate_metrics` 读取 `train_mols.pkl`，并写 `metrics.pt` / `metrics_valid_unique.pt`。
7. 确认当前环境缺少 `lightning` 和 `numpy`，无法执行 help/runtime 命令。
8. 确认当前 repo 下未发现 `.smol`、`.ckpt`、`train_mols.pkl` 或 predictions 文件。

## J. 当前无法验证的内容

由于缺少环境和数据，本阶段未实际验证：

1. 单 step / trial training 是否能完整跑通。
2. checkpoint loading 是否与 Zenodo checkpoint 完全兼容当前代码。
3. SPINDR test split sampling 是否能生成 valid RDKit Mol。
4. PDB/CIF + ligand sampling 是否能稳定切 pocket 并写 SDF。
5. `evaluate_metrics` 的 GenBench3D / PoseBusters / PoseCheck 指标是否能在本机外部依赖下运行。
6. `evaluate_interactions` 的 PLIF recovery 是否能在本机 protein protonation / prolif 环境下运行。
7. `genbench3d/config/default.yaml` 中 ADFR / prepare receptor 路径是否正确。
8. README 中 `genbench3d_data.tar` 的具体下载位置。

这些项目需要用户安装 FLOWR 环境、下载数据/checkpoint 并配置外部工具后人工确认。

## K. 后续与 RL-FLOWR 对比时必须固定的变量

为保证 original FLOWR baseline 与 RL-FLOWR fair comparison，后续实验至少固定：

1. dataset：`spindr`。
2. split：同一 `test.smol` 或同一 PDB/CIF target list。
3. checkpoint 初始化：RL fine-tuning 从同一个 original FLOWR checkpoint 出发。
4. explicit-H/no-H setting：对应同一个 checkpoint 和 `--remove_hs` 设置。
5. pocket noise：例如 `--pocket_noise fix`。
6. sampling target 数：同一 `mp_index` / SLURM array 范围 / target subset。
7. `--sample_n_molecules_per_target`。
8. `--integration_steps`。
9. `--ode_sampling_strategy`。
10. `--coord_noise_std`。
11. `--sample_mol_sizes` vs fixed mol size setting。
12. inpainting/conditional flags：interaction/scaffold/func_group/linker/substructure 等必须一致。
13. `--filter_valid_unique` 是否开启。
14. evaluation scripts：同一 `evaluate_metrics` / `evaluate_interactions` 版本。
15. external scoring configuration：同一 `genbench3d/config/default.yaml`、ADFR path、PoseBusters/PoseCheck/prolif versions。
16. output artifact layout：baseline 与 RL 输出目录分开，不能覆盖 baseline samples/checkpoints/metrics。

## L. 推荐的 Stage 0 baseline artifact 命名

建议后续实际运行时采用明确目录，避免 RL 实验覆盖 baseline：

```text
/ABS/PATH/TO/experiments/
  flowr_original_ckpt/
    flowr.ckpt
  flowr_stage0_baseline_train_smoke/
    last.ckpt
    metrics/logs...
  flowr_stage0_baseline_samples/
    predictions_multi_1.pt
    ref_pdbs/
  flowr_stage0_baseline_eval/
    # 如果 evaluation 必须写在 sampling save_dir，则把 sampling dir 作为 eval dir
```

如果使用原始 `evaluate_metrics` / `evaluate_interactions`，它们会把 `metrics.pt` 和 `interaction_recovery*.pt` 写回 sampling `save_dir`；因此 sampling output dir 应被视为 evaluation artifact dir。
