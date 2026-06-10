#!/bin/sh
# Smoke-test LFPO-F-v2 RL fine-tuning from a pretrained FLOWR Spindr checkpoint.
# Edit paths and objective parameters before running in a full FLOWR environment.

main_path="${MAIN_PATH:-/data/bhli/Project/repo-rl}"
dataset="${DATASET:-spindr}"
ckpt="${CKPT_PATH:-$main_path/checkpoints/flowr_spindr.ckpt}"
data_path="${DATA_PATH:-$main_path/data/$dataset}"
save_dir="${SAVE_DIR:-$main_path/checkpoints/flowr_spindr_rl_strain_smoke}"

cuda_devices="${CUDA_VISIBLE_DEVICES:-0}"
sample_n_molecules_per_target="${SAMPLE_N_MOLECULES_PER_TARGET:-1}"
integration_steps="${INTEGRATION_STEPS:-10}"
ode_sampling_strategy="${ODE_SAMPLING_STRATEGY:-linear}"
batch_cost="${BATCH_COST:-100}"
val_batch_cost="${VAL_BATCH_COST:-10}"
objective_mode="${RL_OBJECTIVE_MODE:-strain}"
metric_source="${RL_METRIC_SOURCE:-cached}"
cache_path="${RL_METRIC_CACHE_PATH:-$save_dir/rl_cache/${objective_mode}.jsonl}"
chunk_size="${RL_SURROGATE_CHUNK_SIZE:-1}"
update_frequency="${RL_UPDATE_FREQUENCY:-1}"
rl_loss_weight="${RL_LOSS_WEIGHT:-0.01}"

mkdir -p "$save_dir/rl_cache"

echo "[RL fine-tune] TensorBoard command:"
echo "tensorboard --logdir \"$save_dir/tensorboard\" --port 6006 --host 0.0.0.0"

CUDA_VISIBLE_DEVICES="$cuda_devices" python -m flowr.train_rl_from_smol \
    --ckpt_path "$ckpt" \
    --data_path "$data_path" \
    --dataset "$dataset" \
    --save_dir "$save_dir" \
    --exp_name "rl_${objective_mode}_smoke" \
    --gpus 1 \
    --use_bucket_sampler \
    --bucket_cost_scale quadratic \
    --batch_cost "$batch_cost" \
    --val_batch_cost "$val_batch_cost" \
    --sample_n_molecules_per_target "$sample_n_molecules_per_target" \
    --integration_steps "$integration_steps" \
    --ode_sampling_strategy "$ode_sampling_strategy" \
    --enable_rl_finetune \
    --rl_loss_weight "$rl_loss_weight" \
    --rl_objective_mode "$objective_mode" \
    --rl_metric_source "$metric_source" \
    --rl_metric_cache_path "$cache_path" \
    --rl_num_stratified_timesteps 1 \
    --rl_surrogate_chunk_size "$chunk_size" \
    --rl_update_frequency "$update_frequency" \
    --rl_top_ratio 0.25 \
    --rl_bottom_ratio 0.25 \
    --rl_strain_good_threshold 0 \
    --rl_strain_bad_threshold 20 \
    --rl_vina_good_threshold -10 \
    --rl_vina_bad_threshold 0 \
    --acc_batches 1
