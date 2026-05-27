from pathlib import Path
import argparse as arg
import inspect
import os

import lightning as L
import torch

from sot_mol.comparm import GP, Update_PARAMS
from sot_mol.models.rl_interface import MolGen_RLModel


def build_objective_config(args):
    """Build objective_config for the unified objective system in mpo_tasks.py.

    The actual reward is computed inside RL_Lightning via
    build_objective(objective_name, objective_config).score_mols(...).score.
    This function only supplies task-specific configuration and keeps non-MPO
    tasks from inheriting Ranolazine-specific settings.
    """
    obj_lower = args.objective_name.lower()

    if obj_lower == "qed":
        return {
            "aggregate": "geometric",
        }

    if obj_lower in ["psa_min", "psa3d_min"]:
        return {
            "psa_center": float(args.psa_center),
            "psa_k": float(args.psa_k),
            "psa_add_hs": bool(args.psa_add_hs),
            "psa_include_polar_hs": bool(args.psa_include_polar_hs),
        }

    if obj_lower in ["qed_sa", "qed_sa_min", "qedsa", "qed_sa_mpo"]:
        return {
            "aggregate": "geometric",
            "sa_transform": args.sa_transform,
            "component_weights": {
                "qed": float(args.qed_weight),
                "SA": float(args.sa_weight),
            },
            "pareto_component_names": ["qed", "SA"],
            "sa_allow_proxy": bool(args.sa_allow_proxy),
            "feasibility": {"valid": True, "connected": True},
        }

    # Fallback for existing MPO tasks such as Ranolazine_MPO / Osimertinib_MPO.
    return {
        "aggregate": args.aggregate,
        "use_official_guacamol": bool(args.use_official_guacamol),
        "fallback_aggregate": "geometric",
        "pareto_component_names": ["sim_ranolazine_AP", "logP", "TPSA", "num_F"],
        "feasibility": {"valid": True, "connected": True},
    }


parser = arg.ArgumentParser(description="Reward-Weighted FM RL training with unified objectives")
parser.add_argument("--config", type=str, default="rl.json")
parser.add_argument("--objective_name", type=str, default="qed",
                    help="Objective registered in sot_mol.rl_objectives.mpo_tasks, e.g. qed, PSA_Min, QED_SA")
parser.add_argument("--project_name", type=str, default="",
                    help="TensorBoard/checkpoint project name. Defaults to SOTMOL_RL_<objective>.")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batchsize", type=int, default=48)
parser.add_argument("--mini_batchsize", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=128)
parser.add_argument("--log_steps", type=int, default=1)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--regularization_type", type=str, default="Parametric_L2", choices=["KL", "Parametric_L2"])
parser.add_argument("--exp_tag", type=str, default="")

# RWR reward-to-weight settings.
parser.add_argument("--reward_beta", type=float, default=2.0)
parser.add_argument("--reward_weight_min", type=float, default=0.1)
parser.add_argument("--reward_weight_max", type=float, default=10.0)
parser.add_argument("--anchor_weight", type=float, default=0.1)
parser.add_argument("--anchor_loss_weight", type=float, default=1.0)
parser.add_argument("--disable_reference_anchor", action="store_true")

# PSA_Min settings.
parser.add_argument("--psa_center", type=float, default=75.0)
parser.add_argument("--psa_k", type=float, default=0.05)
parser.add_argument("--psa_add_hs", action=arg.BooleanOptionalAction, default=True)
parser.add_argument("--psa_include_polar_hs", action=arg.BooleanOptionalAction, default=True)

# QED_SA settings.
parser.add_argument("--sa_transform", type=str, default="linear", choices=["linear", "reverse_sigmoid"])
parser.add_argument("--qed_weight", type=float, default=1.0)
parser.add_argument("--sa_weight", type=float, default=1.0)
parser.add_argument("--sa_allow_proxy", action="store_true")

# Existing MPO fallback settings.
parser.add_argument("--aggregate", type=str, default="official", choices=["official", "geometric", "linear", "tchebycheff"])
parser.add_argument("--use_official_guacamol", action=arg.BooleanOptionalAction, default=True)

args = parser.parse_args()

script_dir = Path(__file__).resolve().parent
config_path = Path(args.config)
if not config_path.is_absolute():
    config_path = script_dir / config_path

GP = Update_PARAMS(GP, str(config_path))

# Keep this before any CUDA work. If CUDA_VISIBLE_DEVICES is set externally, respect the external value.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(GP.CUDA_VISIBLE_DEVICES))
torch.set_float32_matmul_precision("high")
L.seed_everything(args.seed)

import torch._dynamo
torch._dynamo.config.suppress_errors = True

objective_config = build_objective_config(args)
project_name = args.project_name or f"SOTMOL_RL_{args.objective_name}"
exp_tag = args.exp_tag or f"version_{args.objective_name}_{args.regularization_type}"

base_model_kwargs = dict(
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
    # Backward compatible: RL_Lightning uses objective_name if provided;
    # otherwise it falls back to reward_name. Passing reward_name keeps old
    # interfaces functional for simple objectives.
    reward_name=args.objective_name,
    reward_beta=args.reward_beta,
    reward_weight_min=args.reward_weight_min,
    reward_weight_max=args.reward_weight_max,
    anchor_weight=args.anchor_weight,
    anchor_loss_weight=args.anchor_loss_weight,
    use_reference_anchor=not args.disable_reference_anchor,
)

# Only pass new objective kwargs if MolGen_RLModel supports them. If your
# rl_interface.py has not yet been updated to forward objective_config into
# RL_Lightning, reward_name still works for objectives with default configs.
sig = inspect.signature(MolGen_RLModel.__init__)
params = sig.parameters
accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
if accepts_var_kwargs or "objective_name" in params:
    base_model_kwargs["objective_name"] = args.objective_name
if accepts_var_kwargs or "objective_config" in params:
    base_model_kwargs["objective_config"] = objective_config

model = MolGen_RLModel(**base_model_kwargs)

prior_ckpt = script_dir / "prior.ckpt"
datasets_dir = script_dir.parent / "datasets"

print(f"[train_rl] objective_name={args.objective_name}")
print(f"[train_rl] objective_config={objective_config}")
print(f"[train_rl] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[train_rl] project_name={project_name}, exp_tag={exp_tag}")

model.Train(
    train_datafile=datasets_dir / "train.smol",
    val_datafile=datasets_dir / "val.smol",
    test_datafile=datasets_dir / "test.smol",
    epochs=args.epochs,
    save_path=str(script_dir / "models"),
    project_name=project_name,
    load_ckpt=str(prior_ckpt),
    lr=GP.LR,
    debug=False,
    ngpus=1,
    batchsize=args.batchsize,
    mini_batchsize=args.mini_batchsize,
    max_steps=args.max_steps,
    log_steps=args.log_steps,
    exp_tag=exp_tag,
    regularization_type=args.regularization_type,
)
