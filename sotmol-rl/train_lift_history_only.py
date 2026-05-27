from pathlib import Path
import argparse as arg
import os

import lightning as L
import torch

from sot_mol.comparm import GP, Update_PARAMS
from sot_mol.models.rl_lfpo_interface import MolGen_LFPOModel


parser = arg.ArgumentParser(description="LFPO-F FM RL with history-aware feasible top selection")
parser.add_argument("--config", type=str, default="rl.json")
parser.add_argument("--objective_name", type=str, default="qed")
parser.add_argument("--partition_mode", type=str, default="feasible_pareto")
parser.add_argument("--history_diversity_mode", type=str, default="scaffold", choices=["none", "scaffold", "fingerprint"])
parser.add_argument("--history_fingerprint_threshold", type=float, default=0.70)
parser.add_argument("--history_max_size", type=int, default=4096)
parser.add_argument("--top_selection_score_mode", type=str, default="score", choices=["score", "component_balanced", "score_plus_min_component", "tchebycheff"])
parser.add_argument("--use_min_component_bonus", action="store_true")
parser.add_argument("--disable_component_floor", action="store_true")
parser.add_argument("--oracle_log_path", type=str, default="")
args = parser.parse_args()

script_dir = Path(__file__).resolve().parent
config_path = Path(args.config)
if not config_path.is_absolute():
    config_path = script_dir / config_path

GP = Update_PARAMS(GP, str(config_path))

os.environ["CUDA_VISIBLE_DEVICES"] = GP.CUDA_VISIBLE_DEVICES
torch.set_float32_matmul_precision("high")
L.seed_everything(12345)

import torch._dynamo

torch._dynamo.config.suppress_errors = True

objective_name = args.objective_name
obj_lower = objective_name.lower()

if obj_lower == "qed":
    objective_config = {"aggregate": "geometric"}
elif obj_lower in ["psa_min", "psa3d_min"]:
    objective_config = {
        "psa_center": 75.0,
        "psa_k": 0.05,
        "psa_add_hs": True,
        "psa_include_polar_hs": True,
    }
elif obj_lower in ["qed_sa", "qed_sa_min", "qedsa", "qed_sa_mpo"]:
    objective_config = {
        "aggregate": "geometric",
        "sa_transform": "linear",
        "component_weights": {"qed": 1.0, "SA": 1.0},
        "pareto_component_names": ["qed", "SA"],
        "sa_allow_proxy": False,
        "feasibility": {"valid": True, "connected": True},
    }
else:
    objective_config = {
        "aggregate": "official",
        "use_official_guacamol": True,
        "fallback_aggregate": "geometric",
        "pareto_component_names": ["sim_ranolazine_AP", "logP", "TPSA", "num_F"],
        "feasibility": {"valid": True, "connected": True},
    }

if obj_lower in ["qed_sa", "qed_sa_min", "qedsa", "qed_sa_mpo"]:
    top_selection_component_weights = {"qed": 1.0, "SA": 1.0}
elif obj_lower in ["psa_min", "psa3d_min"]:
    top_selection_component_weights = {"PSA": 1.0}
else:
    top_selection_component_weights = {"sim_ranolazine_AP": 2.0, "num_F": 2.0}

partition_config = {
    "mode": args.partition_mode,  # simplified partition.py treats all modes as feasible_pareto semantics
    "top_ratio": 0.25,
    "bottom_ratio": 0.25,
    "top_selection_score_mode": args.top_selection_score_mode,
    "top_selection_component_weights": top_selection_component_weights,
    "use_min_component_bonus": bool(args.use_min_component_bonus),
    "min_component_weight": 1.0,
    "bottom_component_floor": {} if args.disable_component_floor else {"num_F": 0.60},
    "history_diversity_mode": args.history_diversity_mode,
    "history_fingerprint_threshold": args.history_fingerprint_threshold,
    "history_max_size": args.history_max_size,
}

metric_config = {
    "enabled": bool(args.oracle_log_path),
    "oracle_log_path": args.oracle_log_path if args.oracle_log_path else str(script_dir / "oracle_logs" / f"{objective_name}.csv"),
    "novelty_reference_path": str(script_dir / "train_smiles.txt"),
    "log_ref_train": True,
    "log_current_eval": True,
}

model = MolGen_LFPOModel(
    d_model=GP.D_MODEL,
    atom_tokens=GP.TOKENS,
    n_bond_types=GP.N_BOND_TYPES,
    coord_std=GP.COORDS_STD_DEV,
    scale_ot=GP.SCALE_OT,
    self_cond=True,
    coord_noise_std=0.2,
    formulation="endpoint",
    eval_3D_props=False,
    ot_bond_weight=1,
    reward_name="qed",
    objective_name=objective_name,
    objective_config=objective_config,
    partition_config=partition_config,
    metric_config=metric_config,
    anchor_weight=0.1,
    anchor_loss_weight=1.0,
    use_reference_anchor=True,
    lfpo_hparams={
        "lfpo_num_time_samples": 2,
        "lfpo_reward_temperature": 0.5,
        "lfpo_beta_types": 1.5,
        "lfpo_beta_bonds": 1.5,
        "lfpo_beta_charges": 1.5,
        "lfpo_beta_coord": 1.5,
        "lfpo_gamma_coord": 1.0,
        "lfpo_lambda_coord_rect": 0.0,
        "lfpo_lambda_types_rect": 1.0,
        "lfpo_lambda_bonds_rect": 1.0,
        "lfpo_lambda_charges_rect": 1.0,
        "lfpo_aux_fm_weight": 0.0,
        "anchor_weight": 1.0,
        "lfpo_top_weight_mode": "uniform",
        "lfpo_top_ratio": 0.25,
        "lfpo_bottom_ratio": 0.25,
        "lfpo_bottom_repulsion_weight": 0.5,
        "lfpo_middle_weight": 0.0,
        "lfpo_use_top_bottom": True,
        "ref_ema_decay": 0.9,
        "lfpo_use_charge_head": True,
        "lfpo_detach_targets": True,
        "lfpo_time_chunk_size": 64,
        "lfpo_eval_current_every": 100,
        "lfpo_log_current_reward": True,
        "lfpo_eval_current_samples": 1000,
        "lfpo_eval_current_batch_size": 256,
    },
)

prior_ckpt = script_dir / "prior.ckpt"
datasets_dir = script_dir.parent / "datasets"

model.Train(
    train_datafile=datasets_dir / "train.smol",
    val_datafile=datasets_dir / "val.smol",
    test_datafile=datasets_dir / "test.smol",
    epochs=10,
    save_path=str(script_dir / "models"),
    project_name="SOTMOL_LIFT_HISTORY_ONLY",
    load_ckpt=str(prior_ckpt),
    lr=GP.LR,
    debug=False,
    ngpus=1,
    batchsize=60,
    log_steps=1,
)
