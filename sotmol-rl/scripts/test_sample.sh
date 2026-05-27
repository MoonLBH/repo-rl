python profile_ranolazine_prior_simple.py \
  --config rl.json \
  --load_ckpt /data/bhli/Project/Mol-RL/sotmol-rl/TensorBoard/SOTMOL_LIFT_HISTORY_ONLY/version_0/checkpoints/epoch=3-step=16068.ckpt \
  --num_samples 1000 \
  --batch_size 512 \
  --gen_chunk_size 384 \
  --score_device cpu \
  --output_dir prior_profile/samples/qed_sa/our_history \
  --conditions 1,2,3,4 \
  --sim_threshold 0.7 \
  --logp_threshold 7.0 \
  --tpsa_threshold 95.0 \
  --num_f_target 1


# python profile_ranolazine_prior_simple.py \
#   --config rl.json \
#   --load_ckpt /data/bhli/Project/Mol-RL/sotmol-rl/TensorBoard/SOTMOL_RWR_PSA/version_PSA_Min_Parametric_L2/checkpoints/epoch=2-step=11200.ckpt \
#   --num_samples 1000 \
#   --batch_size 256 \
#   --gen_chunk_size 384 \
#   --score_device cpu \
#   --output_dir prior_profile/samples/psa/rwr \
#   --conditions 1,2,3,4 \
#   --sim_threshold 0.7 \
#   --logp_threshold 7.0 \
#   --tpsa_threshold 95.0 \
#   --num_f_target 1 \
#   --filter_start_by_atom_count \
#   --atom_count_with_h \
#   --atom_count_tolerance 10 \
#   --min_start_pool_size 100





  # --load_ckpt /data/bhli/Project/Mol-RL/sotmol-rl/prior.ckpt \