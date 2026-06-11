#!/bin/sh
# Diagnose LFPO-F-v2 RL candidate quality for a pretrained FLOWR Spindr checkpoint.
# This is read-only: it reuses the training path's ref_gen sampling, reward, and
# top/middle/bottom selection logic, then writes JSONL + summary JSON.
#
# Typical usage:
#   CKPT_PATH=/path/to/pretrained.ckpt DATA_PATH=/path/to/spindr SAVE_DIR=/tmp/rl_diag \
#     RL_OBJECTIVE_MODE=strain SAMPLE_N_MOLECULES_PER_TARGET=10 MAX_BATCHES=20 \
#     sh flowr/scripts/diagnose_rl_spindr.sh
#
# Force recompute metrics instead of trusting cache:
#   DIAGNOSTIC_FORCE_RECOMPUTE=1 sh flowr/scripts/diagnose_rl_spindr.sh
#
# Compare sampling quality at 100 ODE steps:
#   INTEGRATION_STEPS=100 RL_SAMPLING_STEPS=100 \
#     RL_METRIC_CACHE_PATH=/tmp/rl_diag/rl_cache/strain_steps100.jsonl \
#     OUTPUT_TAG=strain_steps100 DIAGNOSTIC_FORCE_RECOMPUTE=1 \
#     sh flowr/scripts/diagnose_rl_spindr.sh

set -eu

main_path="${MAIN_PATH:-/data/bhli/Project/repo-rl}"
dataset="${DATASET:-spindr}"
ckpt="${CKPT_PATH:-$main_path/checkpoints/flowr_spindr.ckpt}"
data_path="${DATA_PATH:-$main_path/data/$dataset}"
save_dir="${SAVE_DIR:-$main_path/checkpoints/flowr_spindr_rl_diagnostics}"

cuda_devices="${CUDA_VISIBLE_DEVICES:-0}"
gpus="${GPUS:-1}"
batch_cost="${BATCH_COST:-100}"
val_batch_cost="${VAL_BATCH_COST:-10}"
bucket_cost_scale="${BUCKET_COST_SCALE:-quadratic}"
sample_n_molecules_per_target="${SAMPLE_N_MOLECULES_PER_TARGET:-10}"
integration_steps="${INTEGRATION_STEPS:-20}"
rl_sampling_steps="${RL_SAMPLING_STEPS:-$integration_steps}"
ode_sampling_strategy="${ODE_SAMPLING_STRATEGY:-linear}"
corrector_iters="${CORRECTOR_ITERS:-0}"
coord_noise_std="${COORD_NOISE_STD:-0.0}"
cat_sampling_noise_level="${CAT_SAMPLING_NOISE_LEVEL:-1.0}"

objective_mode="${RL_OBJECTIVE_MODE:-strain}"
metric_source="${RL_METRIC_SOURCE:-cached}"
cache_path="${RL_METRIC_CACHE_PATH:-$save_dir/rl_cache/${objective_mode}.jsonl}"
max_batches="${MAX_BATCHES:-10}"
max_candidates="${MAX_CANDIDATES:-0}"
output_tag="${OUTPUT_TAG:-${objective_mode}_steps${rl_sampling_steps}}"
output_jsonl="${OUTPUT_JSONL:-$save_dir/diagnostics/rl_${output_tag}_candidates.jsonl}"
output_summary_json="${OUTPUT_SUMMARY_JSON:-$save_dir/diagnostics/rl_${output_tag}_summary.json}"
seed="${SEED:-1}"

rl_top_ratio="${RL_TOP_RATIO:-0.25}"
rl_bottom_ratio="${RL_BOTTOM_RATIO:-0.25}"
rl_strain_good_threshold="${RL_STRAIN_GOOD_THRESHOLD:-0}"
rl_strain_bad_threshold="${RL_STRAIN_BAD_THRESHOLD:-20}"
rl_strain_max_threshold="${RL_STRAIN_MAX_THRESHOLD:-inf}"
rl_vina_good_threshold="${RL_VINA_GOOD_THRESHOLD:--10}"
rl_vina_bad_threshold="${RL_VINA_BAD_THRESHOLD:-0}"
rl_vina_max_threshold="${RL_VINA_MAX_THRESHOLD:-inf}"
rl_plif_min_threshold="${RL_PLIF_MIN_THRESHOLD:-0}"
rl_plif_weight="${RL_PLIF_WEIGHT:-1.0}"
rl_strain_weight="${RL_STRAIN_WEIGHT:-1.0}"
rl_vina_weight="${RL_VINA_WEIGHT:-1.0}"
rl_ref_ema_decay="${RL_REF_EMA_DECAY:-0.999}"

mkdir -p "$save_dir/rl_cache" "$save_dir/diagnostics"

force_recompute_arg=""
if [ "${DIAGNOSTIC_FORCE_RECOMPUTE:-0}" = "1" ] || [ "${DIAGNOSTIC_FORCE_RECOMPUTE:-false}" = "true" ]; then
    force_recompute_arg="--diagnostic_force_recompute"
fi

bucket_sampler_arg=""
if [ "${USE_BUCKET_SAMPLER:-1}" = "1" ] || [ "${USE_BUCKET_SAMPLER:-true}" = "true" ]; then
    bucket_sampler_arg="--use_bucket_sampler"
fi

sample_from_reference_arg="--rl_sample_from_reference"
if [ "${RL_SAMPLE_FROM_REFERENCE:-1}" = "0" ] || [ "${RL_SAMPLE_FROM_REFERENCE:-true}" = "false" ]; then
    sample_from_reference_arg="--no-rl_sample_from_reference"
fi

failed_as_bottom_arg="--rl_failed_as_bottom"
if [ "${RL_FAILED_AS_BOTTOM:-1}" = "0" ] || [ "${RL_FAILED_AS_BOTTOM:-true}" = "false" ]; then
    failed_as_bottom_arg="--no-rl_failed_as_bottom"
fi

echo "[diagnose-rl] ckpt=$ckpt"
echo "[diagnose-rl] data_path=$data_path"
echo "[diagnose-rl] save_dir=$save_dir"
echo "[diagnose-rl] objective=$objective_mode metric_source=$metric_source cache=$cache_path"
echo "[diagnose-rl] outputs:"
echo "  jsonl=$output_jsonl"
echo "  summary=$output_summary_json"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$cuda_devices" python -m flowr.diagnose_rl_candidate_selection \
    --ckpt_path "$ckpt" \
    --data_path "$data_path" \
    --dataset "$dataset" \
    --save_dir "$save_dir" \
    --gpus "$gpus" \
    $bucket_sampler_arg \
    --bucket_cost_scale "$bucket_cost_scale" \
    --batch_cost "$batch_cost" \
    --val_batch_cost "$val_batch_cost" \
    --sample_n_molecules_per_target "$sample_n_molecules_per_target" \
    --integration_steps "$integration_steps" \
    --rl_sampling_steps "$rl_sampling_steps" \
    --ode_sampling_strategy "$ode_sampling_strategy" \
    --corrector_iters "$corrector_iters" \
    --coord_noise_std "$coord_noise_std" \
    --cat_sampling_noise_level "$cat_sampling_noise_level" \
    --rl_objective_mode "$objective_mode" \
    --rl_metric_source "$metric_source" \
    --rl_metric_cache_path "$cache_path" \
    --rl_top_ratio "$rl_top_ratio" \
    --rl_bottom_ratio "$rl_bottom_ratio" \
    --rl_strain_good_threshold "$rl_strain_good_threshold" \
    --rl_strain_bad_threshold "$rl_strain_bad_threshold" \
    --rl_strain_max_threshold "$rl_strain_max_threshold" \
    --rl_vina_good_threshold "$rl_vina_good_threshold" \
    --rl_vina_bad_threshold "$rl_vina_bad_threshold" \
    --rl_vina_max_threshold "$rl_vina_max_threshold" \
    --rl_plif_min_threshold "$rl_plif_min_threshold" \
    --rl_plif_weight "$rl_plif_weight" \
    --rl_strain_weight "$rl_strain_weight" \
    --rl_vina_weight "$rl_vina_weight" \
    $sample_from_reference_arg \
    $failed_as_bottom_arg \
    --rl_ref_ema_decay "$rl_ref_ema_decay" \
    --max_batches "$max_batches" \
    --max_candidates "$max_candidates" \
    --output_jsonl "$output_jsonl" \
    --output_summary_json "$output_summary_json" \
    --seed "$seed" \
    $force_recompute_arg
