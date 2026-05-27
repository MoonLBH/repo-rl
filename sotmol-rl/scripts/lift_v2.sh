# qed
# python train_lift_v1_2.py \
#   --config rl.json \
#   --objective_name qed \
#   --partition_mode scalar_top_bottom \
#   --top_selection_score_mode score \
#   --disable_component_floor \
#   --oracle_log_path /data/bhli/Project/Mol-RL/sotmol-rl/oracle_logs/qed_scalar.csv

#psa
# python train_lift_v1_2.py \
#     --config rl.json \
#     --objective_name PSA_Min \
#     --partition_mode scalar_top_bottom \
#     --top_selection_score_mode score \
#     --disable_component_floor \
#     --oracle_log_path /data/bhli/Project/Mol-RL/sotmol-rl/oracle_logs/psa_min_scalar.csv


# qed + sa
# python train_lift_v1_2.py \
#   --config rl.json \
#   --objective_name QED_SA \
#   --partition_mode scalar_top_bottom \
#   --top_selection_score_mode score \
#   --disable_component_floor \
#   --oracle_log_path /data/bhli/Project/Mol-RL/sotmol-rl/oracle_logs/qed_sa_scalar.csv

# qed + sa with feasible pareto partition
# python train_lift_v1_2.py \
#   --config rl.json \
#   --objective_name QED_SA \
#   --partition_mode feasible_pareto \
#   --top_selection_score_mode score \
#   --disable_component_floor \
#   --oracle_log_path /data/bhli/Project/Mol-RL/sotmol-rl/oracle_logs/qed_sa_feasible_pareto2.csv

python train_lift_history_only.py \
  --config rl.json \
  --objective_name QED_SA \
  --partition_mode feasible_pareto \
  --history_diversity_mode scaffold \
  --top_selection_score_mode score \
  --disable_component_floor \
  --oracle_log_path /data/bhli/Project/Mol-RL/sotmol-rl/oracle_logs/qed_sa_history_scaffold.csv