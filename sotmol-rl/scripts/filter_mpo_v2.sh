# python profile_ref_mpo_support_v2.py \
#   --config rl.json \
#   --load_ckpt /data/bhli/Project/Mol-RL/sotmol-rl/prior.ckpt \
#   --samples_per_ref 500 \
#   --max_candidate_refs 200 \
#   --atom_count_tolerance 15 \
#   --output_dir prior_profile/Ref_MPO_support_v2


# python profile_ref_mpo_support_v2.py \
#   --config rl.json \
#   --load_ckpt /data/bhli/Project/Mol-RL/sotmol-rl/prior.ckpt \
#   --fine_from_csv prior_profile/Ref_MPO_support_v2/coarse/reference_profile_summary.csv \
#   --fine_selection rankAB \
#   --fine_top_k_per_group 5 \
#   --fine_samples_per_ref 5000 \
#   --atom_count_tolerance 15 \
#   --output_dir prior_profile/Ref_MPO_support_v2

python profile_ref_mpo_support_v3_fast.py \
  --config rl.json \
  --load_ckpt /data/bhli/Project/Mol-RL/sotmol-rl/prior.ckpt \
  --samples_per_ref 500 \
  --max_candidate_refs 200 \
  --atom_count_tolerance 10 \
  --batch_size 512 \
  --mini_batch_size 8 \
  --gen_chunk_size 384 \
  --cleanup_every 20 \
  --output_dir prior_profile/Ref_MPO_support