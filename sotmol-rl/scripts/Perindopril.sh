# python train_lift_history_only_perindopril_ready.py \
#   --objective_name perindopril_similarity_aromatic \
#   --project_name SOTMOL_LIFT_PERINDOPRIL \
#   --history_diversity_mode none \
#   --top_selection_score_mode score \
#   --template_atom_count_min 53 \
#   --template_atom_count_max 63 \
#   --template_atom_count_splits all \
#   --template_count_with_hs \
#   --epochs 100 \
#   --batchsize 64



# python train_lift_history_only_perindopril_ready.py \
#   --objective_name perindopril_similarity_aromatic \
#   --project_name SOTMOL_LIFT_PERINDOPRIL_History \
#   --history_diversity_mode scaffold \
#   --top_selection_score_mode score \
#   --template_atom_count_min 53 \
#   --template_atom_count_max 63 \
#   --template_atom_count_splits all \
#   --template_count_with_hs \
#   --epochs 100 \
#   --batchsize 64

# ------------------------------------------------------------------
python train_rl_objective_ready.py \
  --objective_name perindopril_similarity_aromatic \
  --project_name SOTMOL_RL_PERINDOPRIL_SIM_AROM \
  --filtered_train_datafile /data/bhli/Project/Mol-RL/sotmol-rl/filtered_smol/train_atoms_53_63_pid2118425.smol \
  --perindopril_similarity_weight 0.8 \
  --perindopril_aromatic_weight 0.2 \
  --perindopril_near_aromatic_score 0.5 \
  --epochs 100 \
  --batchsize 60 \
  --mini_batchsize 1 \
  --max_steps 128 \
  --regularization_type Parametric_L2