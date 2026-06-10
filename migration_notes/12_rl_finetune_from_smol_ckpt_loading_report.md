# Stage 12 — Checkpoint-driven FLOWR RL fine-tuning entry

## 1. Why the new entry does not rely only on `flowr.train`

The generic `flowr.train` entry remains the correct path for original FLOWR training and from-scratch experiments, but it exposes many architecture and representation flags (`d_model`, `d_edge`, pocket encoder sizes, interpolation options, inpainting options, etc.). For RL fine-tuning we start from a pretrained FLOWR checkpoint, so the safest behavior is to recover those structural choices from `checkpoint["hyper_parameters"]` and expose only fine-tuning/RL controls. This avoids accidentally constructing a model whose architecture differs from the checkpoint.

## 2. New RL fine-tuning entry file

The new entry is:

```bash
python -m flowr.train_rl_from_smol
```

implemented in `flowr/flowr/train_rl_from_smol.py`. Its required core arguments are `--ckpt_path`, `--data_path`, and `--save_dir`.

## 3. Alignment with `generate_from_smol.py` loading logic

The entry reuses `flowr.gen.generate_from_smol.load_model(...)` instead of reimplementing model construction. The wrapper builds the minimal argument namespace required by that loader, including the checkpoint path, integration steps, sampling strategy, categorical strategy inferred from the checkpoint, and inpainting flags restored from checkpoint hparams.

## 4. Checkpoint hparams used for model structure

The script prints a checkpoint hparams summary before model loading and recovers model-structure/data-representation settings from the checkpoint, including:

- architecture;
- ligand generator dimensions (`d_equi`, `d_inv`, `d_edge`, `d_message`, `d_message_ff`, `n_layers`, `n_attn_heads`);
- pocket encoder dimensions (`pocket-d_equi`, `pocket-d_inv`, `pocket-n_layers`);
- `pocket_noise`, `remove_hs`, and interaction/prediction flags;
- inpainting flags such as interaction/scaffold/functional-group/linker/mixed unconditional inpainting;
- categorical strategy and train interpolant defaults when building the datamodule.

The new CLI intentionally does not expose model-architecture flags such as `--d_model`, `--d_edge`, `--n_coord_sets`, or `--pocket_d_model`.

## 5. CLI override scope

The new entry allows overriding fine-tuning and RL settings only, for example:

- data/training: `--data_path`, `--dataset`, `--save_dir`, `--batch_cost`, `--val_batch_cost`, `--bucket_cost_scale`, `--use_bucket_sampler`, `--gpus`, `--epochs`, `--lr`, `--lr_schedule`, `--lr_gamma`, `--gradient_clip_val`, `--acc_batches`;
- generation/RL sampling: `--sample_n_molecules_per_target`, `--integration_steps`, `--rl_sampling_steps`, `--ode_sampling_strategy`;
- RL activation: `--enable_rl_finetune`, `--rl_loss_weight`, `--rl_update_frequency`;
- reward/objectives: `--rl_objective_mode`, `--rl_metric_source`, `--rl_metric_cache_path`, weights, and thresholds;
- LFPO-F-v2 surrogate: top/bottom branch weights, negative-target coefficients, `--rl_num_stratified_timesteps`, `--rl_surrogate_chunk_size`, and reference EMA decay;
- interpolant controls: `--rl_use_original_interpolant` defaults to true; simple corruption fallback is disabled unless explicitly allowed.

## 6. `sample_n_molecules_per_target` semantics

`--sample_n_molecules_per_target` follows the generation script semantics: every target/pocket in the current training batch receives N generated ligand candidates during an RL update. Therefore the expected number of candidates is approximately:

```text
num_candidates = num_targets_in_batch * sample_n_molecules_per_target
```

Generation failures are represented as failed candidates/rewards rather than being silently treated as a different sampling semantics.

## 7. Compatibility with `rl_num_candidates_per_step`

`--rl_num_candidates_per_step` is retained only as a deprecated alias. Internally it is mapped to `sample_n_molecules_per_target` and no longer means “total molecules per optimizer step.” If both arguments are provided, `--sample_n_molecules_per_target` wins and a warning is emitted.

## 8. Reference model initialization

The confirmed LFPO-F-v2 contract is unchanged: the current model is first loaded from the pretrained checkpoint, then the explicit RL reference generator `π_ref` is initialized lazily by deepcopying the loaded current model when RL is enabled and `rl_loss_weight > 0`. The reference model is stored outside the trainable module registry and updated by EMA after train batches.

## 9. RL sampling from reference model

The default `--rl_sample_from_reference` is true. During RL updates, the sampling helper now uses `sample_n_molecules_per_target` to perform that many full ODE generation rounds for the whole batch, which yields one candidate per target per round, sampled from `π_ref` by default.

## 10. Train interpolant construction and attach

RL fine-tuning still builds a training datamodule, not just a generation/evaluation loader. The datamodule uses checkpoint hparams for representation and noising choices and command-line overrides only for data/batching options. The entry then attaches:

```python
model.rl_train_interpolant = dm.train_interpolant
```

so selected generated ligands are rebuilt through the original FLOWR train interpolant instead of simple corruption by default.

## 11. Top imitation / bottom repulsion unchanged

This change does not alter the confirmed LFPO-F-v2 surrogate:

- top branch: atom CE + bond CE + charge CE + coordinate target loss against selected high-reward generated `z0`;
- bottom branch: reference-based negative target repulsion for atom/bond/charge distributions and coordinate/vector predictions;
- middle samples: excluded from the main RL loss by default;
- total loss: `original_FLOWR_loss + rl_loss_weight * rl_surrogate_loss`.

## 12. Metric strategy

The current metric strategy is preserved:

1. check validity;
2. check PoseBusters validity;
3. invalid/PoseBusters-failed candidates do not run PLIF/strain/Vina;
4. valid and PoseBusters-valid candidates compute all metrics enabled by `rl_objective_mode`;
5. no every-n-step or warmup skip is used for enabled metrics;
6. metric-failed candidates cannot enter top but can enter bottom.

## 13. Cache strategy

`rl_metric_source=cached` reads JSONL cache entries keyed by ligand/pocket/objective information. Cache hits return stored success or failure results. Cache misses compute all enabled metrics after validity/PoseBusters gating, then write both successes and failures back to the cache.

## 14. Chunk-wise backward

The previous chunk-wise manual optimization path is preserved. When RL is enabled and `--rl_surrogate_chunk_size > 0`, the model uses manual optimization so each expanded pseudo-sample chunk is backpropagated independently. The default baseline path remains automatic optimization when RL is disabled.

## 15. New shell script

A smoke-test shell wrapper was added at:

```bash
flowr/scripts/train_rl_spindr.sh
```

It follows the style of existing FLOWR scripts and exposes checkpoint path, data path, save dir, GPU selection, `sample_n_molecules_per_target`, integration steps, objective mode, cache path, update frequency, and chunk size.

## 16. Smoke test commands

### Strain-only checkpoint-driven smoke test

```bash
cd flowr
python -m flowr.train_rl_from_smol \
  --ckpt_path "$ckpt" \
  --data_path "$data_path" \
  --dataset spindr \
  --save_dir "$save_dir" \
  --batch_cost 100 \
  --val_batch_cost 10 \
  --bucket_cost_scale quadratic \
  --sample_n_molecules_per_target 1 \
  --integration_steps 10 \
  --ode_sampling_strategy linear \
  --enable_rl_finetune \
  --rl_loss_weight 0.01 \
  --rl_objective_mode strain \
  --rl_metric_source cached \
  --rl_metric_cache_path "$save_dir/rl_cache/strain.jsonl" \
  --rl_num_stratified_timesteps 1 \
  --rl_surrogate_chunk_size 1 \
  --rl_update_frequency 1 \
  --rl_top_ratio 0.25 \
  --rl_bottom_ratio 0.25 \
  --rl_strain_good_threshold 0 \
  --rl_strain_bad_threshold 20 \
  --acc_batches 1
```

### Vina-only small smoke test

```bash
cd flowr
python -m flowr.train_rl_from_smol \
  --ckpt_path "$ckpt" \
  --data_path "$data_path" \
  --dataset spindr \
  --save_dir "$save_dir" \
  --sample_n_molecules_per_target 1 \
  --integration_steps 10 \
  --enable_rl_finetune \
  --rl_loss_weight 0.01 \
  --rl_objective_mode vina \
  --rl_metric_source cached \
  --rl_metric_cache_path "$save_dir/rl_cache/vina.jsonl" \
  --rl_vina_good_threshold -10 \
  --rl_vina_bad_threshold 0 \
  --rl_surrogate_chunk_size 1 \
  --acc_batches 1
```

### Strain + Vina small smoke test

```bash
cd flowr
python -m flowr.train_rl_from_smol \
  --ckpt_path "$ckpt" \
  --data_path "$data_path" \
  --dataset spindr \
  --save_dir "$save_dir" \
  --sample_n_molecules_per_target 1 \
  --integration_steps 10 \
  --enable_rl_finetune \
  --rl_loss_weight 0.01 \
  --rl_objective_mode strain_vina \
  --rl_metric_source cached \
  --rl_metric_cache_path "$save_dir/rl_cache/strain_vina.jsonl" \
  --rl_strain_good_threshold 0 \
  --rl_strain_bad_threshold 20 \
  --rl_strain_max_threshold 10 \
  --rl_vina_good_threshold -10 \
  --rl_vina_bad_threshold 0 \
  --rl_vina_max_threshold -4 \
  --rl_surrogate_chunk_size 1 \
  --acc_batches 1
```

## 17. RL disabled path

The dedicated script can load a pretrained model and build a datamodule, but the actual training hook still skips all RL sampling/reward/reference work unless `enable_rl_finetune=true` and `rl_loss_weight > 0`. The original `flowr.train` entry is left in place for non-RL baseline training.

## 18. Verified items

- Python syntax compilation for the new entry and modified RL training helper.
- `python -m flowr.train_rl_from_smol --help` works without importing heavy model dependencies first.
- The help output exposes `sample_n_molecules_per_target` and RL arguments.
- The help output does not expose structural architecture flags such as `--d_model`, `--d_edge`, `--n_coord_sets`, or `--pocket_d_model`.
- The RL sampling helper logs `train-rl-sample-n-molecules-per-target`, `train-rl-num-targets-in-batch`, and `train-rl-num-candidates`.

## 19. Unverified items

- Full checkpoint loading was not executed here because no real FLOWR checkpoint/data path was provided.
- Full RL training with PLIF/Vina/PoseBusters/strain metrics was not run in this environment.
- Multi-GPU behavior, scheduler resume behavior, and large chunk-wise memory behavior still need validation in the target training environment.

## 20. Risks

- The checkpoint hparams schema may differ between older FLOWR checkpoints; the loader uses robust defaults, but unusual checkpoints may need an explicit compatibility shim.
- `generate_from_smol.load_model` currently mutates selected hparams for sampling/inpainting; because this script feeds checkpoint-derived values into those fields, behavior should align with checkpoint defaults, but future changes to the generation loader should be reviewed for RL fine-tuning compatibility.
- The new CLI deliberately hides architecture flags. If a checkpoint is incompatible with current code, users must fix the checkpoint/code compatibility rather than override architecture flags manually.

## 21. TensorBoard-only logging update for RL entry

The checkpoint-driven RL entry now builds its own trainer instead of calling the generic `flowr.train.build_trainer(...)`. This is intentional because the generic trainer always initializes MLflow and may also initialize WandB. RL smoke tests should not depend on an MLflow backend or WandB setup.

The dedicated function `build_rl_trainer(args, model)` uses only `lightning.pytorch.loggers.TensorBoardLogger` with:

```python
tb_logger = TensorBoardLogger(
    save_dir=str(Path(args.save_dir) / "tensorboard"),
    name=args.exp_name or "rl_finetune",
    default_hp_metric=False,
)
```

The RL entry prints:

```text
[RL fine-tune] Logger: TensorBoard only
[RL fine-tune] TensorBoard log dir: <save_dir>/tensorboard
```

Checkpoint files are saved under `<save_dir>/checkpoints`, with `save_last=True` and epoch checkpoints enabled. The ordinary `flowr.train` logger behavior is unchanged; this TensorBoard-only policy applies only to `flowr.train_rl_from_smol`.

When chunk-wise manual RL is active (`enable_rl_finetune=true`, `rl_loss_weight>0`, and `rl_surrogate_chunk_size>0`), the trainer disables automatic gradient clipping by setting trainer `gradient_clip_val=0.0`; manual optimization continues to use the explicit clipping inside `flowr.rl.training.step_optimizer_and_scheduler(...)`.

The Spindr wrapper now prints the TensorBoard launch command:

```bash
tensorboard --logdir "$save_dir/tensorboard" --port 6006 --host 0.0.0.0
```
