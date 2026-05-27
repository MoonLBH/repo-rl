export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# python train_lift_history_only_xtb01_ready.py \
#   --objective_name xtb_force \
#   --project_name SOTMOL_LIFT_XTB_FORCE \
#   --history_diversity_mode none \
#   --top_selection_score_mode score \
#   --xtb_method GFN2-xTB \
#   --xtb_force_norm atom_rms \
#   --xtb_reward_transform bounded_inverse \
#   --xtb_force_scale 1.0 \
#   --xtb_fail_score 0.0 \
#   --xtb_max_workers 2 \
#   --disable_lfpo_current_eval \
#   --epochs 10 \
#   --batchsize 50



python train_rl_objective_xtb01_ready.py \
  --objective_name xtb_force \
  --project_name SOTMOL_RL_XTB_FORCE \
  --xtb_method GFN2-xTB \
  --xtb_reward_transform bounded_inverse \
  --xtb_force_scale 1.0 \
  --xtb_fail_score 0.0 \
  --xtb_max_workers 4 \
  --xtb_timeout 60 \
  --epochs 10 \
  --batchsize 50 \
  --mini_batchsize 1 \
  --max_steps 128 \
  --regularization_type Parametric_L2