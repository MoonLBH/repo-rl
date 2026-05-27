from pathlib import Path
import argparse as arg
import os
import pickle

import lightning as L
import torch

from sot_mol.comparm import GP, Update_PARAMS
from sot_mol.models.rl_lfpo_interface import MolGen_LFPOModel


CELECOXIB_SMILES = "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"
PERINDOPRIL_SMILES = "CCC[C@@H](C(=O)OCC)N[C@@H](C)C(=O)N1[C@H]2CCCC[C@H]2C[C@H]1C(=O)O"


def _filter_smol_by_atom_count(src_path, min_atoms=None, max_atoms=None, out_dir=None, with_hs=True):
    """Create a temporary .smol file containing only molecules within an atom-count range.

    The generated .smol keeps the original serialized MolGraph bytes; it only filters
    which templates enter MGDataModule. This changes the noise-template/mask/natoms
    distribution without changing model code or objective code.
    """
    if min_atoms is None and max_atoms is None:
        return Path(src_path)

    from sot_mol.data.molgraph import MolGraph
    from rdkit import Chem

    src_path = Path(src_path)
    out_dir = Path(out_dir or src_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    min_tag = "min" if min_atoms is None else str(int(min_atoms))
    max_tag = "max" if max_atoms is None else str(int(max_atoms))
    out_path = out_dir / f"{src_path.stem}_atoms_{min_tag}_{max_tag}_pid{os.getpid()}.smol"

    datas = pickle.loads(src_path.read_bytes())
    kept = []
    failed = 0
    for data in datas:
        try:
            mg = MolGraph.from_bytes(data)
            mol = mg.to_rdkit()
            if mol is None:
                failed += 1
                continue
            if not with_hs:
                mol = Chem.RemoveHs(mol)
            n_atoms = int(mol.GetNumAtoms())
            if (min_atoms is None or n_atoms >= int(min_atoms)) and (max_atoms is None or n_atoms <= int(max_atoms)):
                kept.append(data)
        except Exception:
            failed += 1

    if len(kept) == 0:
        raise RuntimeError(
            f"Atom-count filtering produced 0 molecules for {src_path} with "
            f"min_atoms={min_atoms}, max_atoms={max_atoms}, with_hs={with_hs}. "
            f"Failed conversions: {failed}."
        )

    out_path.write_bytes(pickle.dumps(kept, protocol=pickle.HIGHEST_PROTOCOL))
    print(
        f"[atom-count filter] {src_path.name}: kept {len(kept)}/{len(datas)} molecules "
        f"with atoms in [{min_atoms}, {max_atoms}] (with_hs={with_hs}); failed={failed}; "
        f"output={out_path}"
    )
    return out_path


parser = arg.ArgumentParser(description="LFPO-F FM RL with history-aware feasible top selection")
parser.add_argument("--config", type=str, default="rl.json")
parser.add_argument("--objective_name", type=str, default="qed")
parser.add_argument("--target_smiles", type=str, default="",
                    help="Reference SMILES for target_similarity / celecoxib_similarity objectives.")
parser.add_argument("--similarity_fp", type=str, default="morgan", choices=["morgan", "ecfp", "ecfp4", "ap", "atom_pair", "rdkit"])
parser.add_argument("--similarity_radius", type=int, default=2)
parser.add_argument("--similarity_n_bits", type=int, default=2048)
parser.add_argument("--similarity_threshold", type=float, default=None,
                    help="Optional capped reward min(sim, threshold)/threshold. Leave unset for raw Tanimoto.")
parser.add_argument("--perindopril_similarity_weight", type=float, default=0.8,
                    help="Weight of Perindopril Tanimoto component in final reward.")
parser.add_argument("--perindopril_aromatic_weight", type=float, default=0.2,
                    help="Weight of aromatic-ring-count component in final reward.")
parser.add_argument("--perindopril_near_aromatic_score", type=float, default=0.5,
                    help="Aromatic component score when aromatic ring count is 1 or 3.")
parser.add_argument("--template_atom_count_min", type=int, default=None,
                    help="Optional minimum atom count for noise-template molecules. For Perindopril with explicit Hs, use 53.")
parser.add_argument("--template_atom_count_max", type=int, default=None,
                    help="Optional maximum atom count for noise-template molecules. For Perindopril with explicit Hs, use 63.")
parser.add_argument("--template_atom_count_splits", type=str, default="none", choices=["none", "train", "all"],
                    help="Which .smol splits to filter by atom count before creating the datamodule.")
parser.add_argument("--template_count_with_hs", action=arg.BooleanOptionalAction, default=True,
                    help="Count atoms with explicit hydrogens when filtering templates. Keep True if smol files were loaded with Hs.")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batchsize", type=int, default=60)
parser.add_argument("--project_name", type=str, default="SOTMOL_LIFT_HISTORY_ONLY")
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
elif obj_lower in ["celecoxib_similarity", "celecoxib_sim", "celecoxib_scaffold_hopping"]:
    objective_config = {
        "target_name": "celecoxib",
        "target_smiles": args.target_smiles or CELECOXIB_SMILES,
        "fp_type": args.similarity_fp,
        "radius": int(args.similarity_radius),
        "n_bits": int(args.similarity_n_bits),
        "similarity_threshold": args.similarity_threshold,
        "similarity_key": "sim_celecoxib",
        "feasibility": {"valid": True, "connected": True},
        # Extension hook: later promote QED/TPSA/logP from raw_properties into
        # component_scores/objective score inside TargetSimilarityObjective.
        "future_property_components": ["QED", "TPSA", "logP"],
    }
elif obj_lower in ["perindopril_similarity_aromatic", "perindopril_aromatic", "perindopril_scaffold_hopping"]:
    objective_config = {
        "target_smiles": args.target_smiles or PERINDOPRIL_SMILES,
        "fp_type": args.similarity_fp,
        "radius": int(args.similarity_radius),
        "n_bits": int(args.similarity_n_bits),
        "similarity_threshold": args.similarity_threshold,
        "similarity_key": "sim_perindopril",
        "aromatic_key": "aromatic_ring_reward",
        "raw_aromatic_key": "num_aromatic_rings",
        "target_aromatic_rings": 2,
        "near_aromatic_rings": [1, 3],
        "near_aromatic_score": float(args.perindopril_near_aromatic_score),
        "similarity_weight": float(args.perindopril_similarity_weight),
        "aromatic_weight": float(args.perindopril_aromatic_weight),
        "pareto_component_names": ["sim_perindopril", "aromatic_ring_reward"],
        "feasibility": {"valid": True, "connected": True},
    }
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
elif obj_lower in ["celecoxib_similarity", "celecoxib_sim", "celecoxib_scaffold_hopping"]:
    top_selection_component_weights = {"sim_celecoxib": 1.0}
elif obj_lower in ["perindopril_similarity_aromatic", "perindopril_aromatic", "perindopril_scaffold_hopping"]:
    top_selection_component_weights = {"sim_perindopril": float(args.perindopril_similarity_weight), "aromatic_ring_reward": float(args.perindopril_aromatic_weight)}
else:
    top_selection_component_weights = {"sim_ranolazine_AP": 2.0, "num_F": 2.0}

bottom_component_floor = {}
no_numf_floor_objectives = {
    "celecoxib_similarity", "celecoxib_sim", "celecoxib_scaffold_hopping",
    "perindopril_similarity_aromatic", "perindopril_aromatic", "perindopril_scaffold_hopping",
}
if obj_lower not in no_numf_floor_objectives:
    bottom_component_floor = {} if args.disable_component_floor else {"num_F": 0.60}

partition_config = {
    "mode": args.partition_mode,  # simplified partition.py treats all modes as feasible_pareto semantics
    "top_ratio": 0.25,
    "bottom_ratio": 0.25,
    "top_selection_score_mode": args.top_selection_score_mode,
    "top_selection_component_weights": top_selection_component_weights,
    "use_min_component_bonus": bool(args.use_min_component_bonus),
    "min_component_weight": 1.0,
    "bottom_component_floor": bottom_component_floor,
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

train_datafile = datasets_dir / "train.smol"
val_datafile = datasets_dir / "val.smol"
test_datafile = datasets_dir / "test.smol"

if args.template_atom_count_splits != "none" and (
    args.template_atom_count_min is not None or args.template_atom_count_max is not None
):
    filtered_dir = script_dir / "filtered_smol"
    train_datafile = _filter_smol_by_atom_count(
        train_datafile,
        min_atoms=args.template_atom_count_min,
        max_atoms=args.template_atom_count_max,
        out_dir=filtered_dir,
        with_hs=bool(args.template_count_with_hs),
    )
    if args.template_atom_count_splits == "all":
        val_datafile = _filter_smol_by_atom_count(
            val_datafile,
            min_atoms=args.template_atom_count_min,
            max_atoms=args.template_atom_count_max,
            out_dir=filtered_dir,
            with_hs=bool(args.template_count_with_hs),
        )
        test_datafile = _filter_smol_by_atom_count(
            test_datafile,
            min_atoms=args.template_atom_count_min,
            max_atoms=args.template_atom_count_max,
            out_dir=filtered_dir,
            with_hs=bool(args.template_count_with_hs),
        )

model.Train(
    train_datafile=train_datafile,
    val_datafile=val_datafile,
    test_datafile=test_datafile,
    epochs=args.epochs,
    save_path=str(script_dir / "models"),
    project_name=args.project_name,
    load_ckpt=str(prior_ckpt),
    lr=GP.LR,
    debug=False,
    ngpus=1,
    batchsize=args.batchsize,
    log_steps=1,
)
