# 
python train_lift_history_only_celecoxib_ready.py \
  --objective_name celecoxib_similarity \
  --project_name SOTMOL_LIFT_CELECOXIB_SIM \
  --history_diversity_mode none \
  --top_selection_score_mode score \
  --epochs 10 \
  --batchsize 50

# scaffold-diverse
# python train_lift_history_only_celecoxib_ready.py \
#   --objective_name celecoxib_similarity \
#   --project_name SOTMOL_LIFT_CELECOXIB_SIM_SCAF_HISTORY \
#   --history_diversity_mode scaffold \
#   --top_selection_score_mode score \
#   --epochs 10 \
#   --batchsize 50


# -----------------------------------------------------------------------
# RWR
# python train_rl_objective_ready.py \
#   --objective_name celecoxib_similarity \
#   --project_name SOTMOL_RL_CELECOXIB_SIM \
#   --epochs 10 \
#   --batchsize 60 \
#   --mini_batchsize 1 \
#   --max_steps 128 \
#   --regularization_type Parametric_L2