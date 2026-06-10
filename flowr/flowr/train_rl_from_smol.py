"""Checkpoint-driven LFPO-F-v2 RL fine-tuning entry for FLOWR pocket models.

This entry intentionally mirrors ``flowr.gen.generate_from_smol.load_model`` for
pretrained checkpoint loading: model architecture and representation hparams are
read from ``checkpoint["hyper_parameters"]`` instead of being re-specified on the
command line.  The command line only overrides data, optimizer, and RL fine-
tuning controls.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any, Mapping


# Keep imports light so ``python -m flowr.train_rl_from_smol --help`` works even
# before heavy FLOWR dependencies (torch/lightning/RDKit) are installed.  The
# actual model/data imports happen inside ``main`` after argument parsing.
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_CATEGORICAL_STRATEGY = "uniform-sample"
DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_ODE_SAMPLING_STRATEGY = "linear"
DEFAULT_ACC_BATCHES = 1
DEFAULT_BATCH_COST = 512
DEFAULT_BUCKET_COST_SCALE = "linear"
DEFAULT_EPOCHS = 200
DEFAULT_GRADIENT_CLIP_VAL = 10.0
DEFAULT_LR = 2e-4
DEFAULT_LR_GAMMA = 0.998
DEFAULT_LR_SCHEDULE = "constant"
DEFAULT_N_VALIDATION_MOLS = 64


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def hp_get(hparams: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in hparams and hparams[key] is not None:
            return hparams[key]
    return default


def hp_bool(hparams: Mapping[str, Any], *keys: str, default: bool = False) -> bool:
    return bool(hp_get(hparams, *keys, default=default))


def infer_categorical_strategy(hparams: Mapping[str, Any]) -> str:
    return hp_get(
        hparams,
        "categorical_strategy",
        "integration-type-strategy",
        "val-ligand-type-interpolation",
        default=DEFAULT_CATEGORICAL_STRATEGY,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RL fine-tune a pretrained FLOWR smol checkpoint with the LFPO-F-v2 "
            "top-imitation/bottom-repulsion surrogate. Model structure is restored "
            "from --ckpt_path hyper_parameters."
        )
    )

    # Checkpoint and data entry points.
    parser.add_argument("--ckpt_path", required=True, help="Pretrained FLOWR checkpoint used for model structure and weights.")
    parser.add_argument("--resume_rl_ckpt", default=None, help="Optional Lightning checkpoint for resuming an interrupted RL fine-tuning run.")
    parser.add_argument("--data_path", required=True, help="Dataset directory containing train.smol/val.smol and processed statistics.")
    parser.add_argument("--dataset", default="spindr", help="Dataset name, e.g. spindr/crossdocked/bindingmoad.")
    parser.add_argument("--save_dir", required=True, help="Output directory for RL fine-tuning checkpoints and logs.")
    parser.add_argument("--exp_name", default="rl_finetune", help="Experiment name used by loggers.")

    # Dataloader / trainer overrides.  Model architecture is intentionally not exposed here.
    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--val_batch_cost", type=int, default=10)
    parser.add_argument("--bucket_cost_scale", default=DEFAULT_BUCKET_COST_SCALE)
    parser.add_argument("--use_bucket_sampler", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--val_check_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--lr_schedule", default=DEFAULT_LR_SCHEDULE)
    parser.add_argument("--lr_gamma", type=float, default=DEFAULT_LR_GAMMA)
    parser.add_argument("--gradient_clip_val", type=float, default=DEFAULT_GRADIENT_CLIP_VAL)
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trial_run", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--use_ema", action="store_true", help="Optional original FLOWR EMA callback; independent from RL pi_ref EMA.")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--n_validation_mols", type=int, default=DEFAULT_N_VALIDATION_MOLS)

    # Generation / RL candidate sampling semantics aligned with generate_from_smol.
    parser.add_argument("--sample_n_molecules_per_target", type=int, default=None, help="Number of RL candidates sampled per pocket/target at each RL update.")
    parser.add_argument("--rl_num_candidates_per_step", type=int, default=None, help="Deprecated alias for --sample_n_molecules_per_target; interpreted per target, not per step.")
    parser.add_argument("--integration_steps", type=int, default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument("--rl_sampling_steps", type=int, default=None, help="RL sampling ODE steps. Defaults to --integration_steps.")
    parser.add_argument("--ode_sampling_strategy", default=DEFAULT_ODE_SAMPLING_STRATEGY)
    parser.add_argument("--corrector_iters", type=int, default=0)
    parser.add_argument("--coord_noise_std", type=float, default=0.0)
    parser.add_argument("--cat_sampling_noise_level", type=float, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)

    # RL enable/objective/selection controls.
    parser.add_argument("--enable_rl_finetune", action="store_true")
    parser.add_argument("--rl_loss_weight", type=float, default=0.0)
    parser.add_argument("--rl_update_frequency", type=int, default=1)
    parser.add_argument("--rl_surrogate_type", default="top_imitation_bottom_repulsion")
    parser.add_argument("--rl_num_stratified_timesteps", type=int, default=1)
    parser.add_argument("--rl_surrogate_chunk_size", type=int, default=0)
    parser.add_argument("--rl_top_ratio", type=float, default=0.25)
    parser.add_argument("--rl_bottom_ratio", type=float, default=0.25)
    parser.add_argument("--rl_top_k", type=int, default=None)
    parser.add_argument("--rl_bottom_k", type=int, default=None)
    parser.add_argument("--rl_middle_weight", type=float, default=0.0)
    parser.add_argument("--rl_bottom_repulsion_weight", type=float, default=1.0)
    parser.add_argument("--rl_use_reference_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl_sample_from_reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl_ref_ema_decay", type=float, default=0.999)
    parser.add_argument("--rl_reference_checkpoint", default=None)
    parser.add_argument("--rl_use_original_interpolant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl_allow_simple_corruption_fallback", action="store_true")

    # Reward metric controls.
    parser.add_argument("--rl_objective_mode", default="strain", choices=["plif", "strain", "vina", "plif_strain", "plif_vina", "strain_vina", "plif_strain_vina"])
    parser.add_argument("--rl_multiobjective_strategy", default="constrained_weighted_sum")
    parser.add_argument("--rl_metric_source", default="compute", choices=["compute", "cached", "existing_eval_output"])
    parser.add_argument("--rl_metric_cache_path", default=None)
    parser.add_argument("--rl_plif_weight", type=float, default=1.0)
    parser.add_argument("--rl_strain_weight", type=float, default=1.0)
    parser.add_argument("--rl_vina_weight", type=float, default=1.0)
    parser.add_argument("--rl_plif_min_threshold", type=float, default=0.0)
    parser.add_argument("--rl_strain_max_threshold", type=float, default=float("inf"))
    parser.add_argument("--rl_vina_max_threshold", type=float, default=float("inf"))
    parser.add_argument("--rl_strain_good_threshold", type=float, default=0.0)
    parser.add_argument("--rl_strain_bad_threshold", type=float, default=20.0)
    parser.add_argument("--rl_vina_good_threshold", type=float, default=-10.0)
    parser.add_argument("--rl_vina_bad_threshold", type=float, default=0.0)
    parser.add_argument("--rl_invalid_reward", type=float, default=0.0)
    parser.add_argument("--rl_metric_failed_reward", type=float, default=0.0)
    parser.add_argument("--rl_failed_as_bottom", action=argparse.BooleanOptionalAction, default=True)

    # LFPO-F-v2 surrogate weights.
    parser.add_argument("--rl_top_atom_weight", type=float, default=1.0)
    parser.add_argument("--rl_top_bond_weight", type=float, default=1.0)
    parser.add_argument("--rl_top_charge_weight", type=float, default=1.0)
    parser.add_argument("--rl_top_coord_weight", type=float, default=1.0)
    parser.add_argument("--rl_bottom_atom_weight", type=float, default=1.0)
    parser.add_argument("--rl_bottom_bond_weight", type=float, default=1.0)
    parser.add_argument("--rl_bottom_charge_weight", type=float, default=1.0)
    parser.add_argument("--rl_bottom_coord_weight", type=float, default=1.0)
    parser.add_argument("--rl_beta_atom", type=float, default=1.0)
    parser.add_argument("--rl_beta_bond", type=float, default=1.0)
    parser.add_argument("--rl_beta_charge", type=float, default=1.0)
    parser.add_argument("--rl_gamma_coord", type=float, default=1.0)
    parser.add_argument("--rl_aux_fm_weight", type=float, default=0.0)
    parser.add_argument("--rl_anchor_weight", type=float, default=0.0)
    parser.add_argument("--rl_detach_targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl_detach_reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl_log_selected_samples", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl_save_selected_samples", action="store_true")
    parser.add_argument("--rl_log_metric_failures", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def checkpoint_hparams(ckpt_path: str) -> dotdict:
    import torch

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    return dotdict(checkpoint["hyper_parameters"])


def print_checkpoint_summary(hparams: Mapping[str, Any], ckpt_path: str) -> None:
    keys = [
        "architecture",
        "d_equi",
        "d_inv",
        "d_edge",
        "d_message",
        "d_message_ff",
        "n_layers",
        "n_attn_heads",
        "pocket-d_equi",
        "pocket-d_inv",
        "pocket-n_layers",
        "pocket_noise",
        "remove_hs",
        "flow_interactions",
        "predict_interactions",
        "interaction_inpainting",
        "scaffold_inpainting",
        "func_group_inpainting",
        "linker_inpainting",
        "mixed_uncond_inpaint",
    ]
    print(f"[RL fine-tune] Loading pretrained FLOWR checkpoint: {ckpt_path}")
    print("[RL fine-tune] Checkpoint hparams summary:")
    for key in keys:
        if key in hparams:
            print(f"  - {key}: {hparams[key]}")


def resolve_sample_count(args: argparse.Namespace) -> int:
    if args.sample_n_molecules_per_target is None:
        if args.rl_num_candidates_per_step is not None:
            warnings.warn(
                "--rl_num_candidates_per_step is deprecated and is now interpreted as "
                "candidates per pocket/target. Use --sample_n_molecules_per_target instead.",
                stacklevel=2,
            )
            return max(1, int(args.rl_num_candidates_per_step))
        return 1
    if args.rl_num_candidates_per_step is not None:
        warnings.warn(
            "Both --sample_n_molecules_per_target and --rl_num_candidates_per_step were set; "
            "using --sample_n_molecules_per_target.",
            stacklevel=2,
        )
    return max(1, int(args.sample_n_molecules_per_target))


def make_checkpoint_model_args(args: argparse.Namespace, hparams: Mapping[str, Any]) -> argparse.Namespace:
    """Build the minimal arg namespace expected by generate_from_smol.load_model."""

    return argparse.Namespace(
        ckpt_path=args.ckpt_path,
        integration_steps=args.integration_steps,
        ode_sampling_strategy=args.ode_sampling_strategy,
        interaction_inpainting=hp_bool(hparams, "interaction_inpainting", default=False),
        scaffold_inpainting=hp_bool(hparams, "scaffold_inpainting", default=False),
        func_group_inpainting=hp_bool(hparams, "func_group_inpainting", default=False),
        linker_inpainting=hp_bool(hparams, "linker_inpainting", default=False),
        corrector_iters=args.corrector_iters,
        categorical_strategy=infer_categorical_strategy(hparams),
        arch=hp_get(hparams, "architecture", default="pocket"),
        coord_noise_std=args.coord_noise_std,
        cat_sampling_noise_level=args.cat_sampling_noise_level,
        data_path=args.data_path,
        save_dir=args.save_dir,
    )


def make_datamodule_args(args: argparse.Namespace, hparams: Mapping[str, Any]) -> argparse.Namespace:
    """Create train.build_dm args using checkpoint representation defaults."""

    return argparse.Namespace(
        dataset=args.dataset,
        data_path=args.data_path,
        save_dir=args.save_dir,
        batch_cost=args.batch_cost,
        val_batch_cost=args.val_batch_cost,
        bucket_cost_scale=args.bucket_cost_scale,
        use_bucket_sampler=args.use_bucket_sampler,
        n_validation_mols=args.n_validation_mols,
        remove_hs=hp_bool(hparams, "remove_hs", default=True),
        scale_coords=hp_bool(hparams, "scale_coords", default=True),
        categorical_strategy=infer_categorical_strategy(hparams),
        pocket_noise=hp_get(hparams, "pocket_noise", default="fix"),
        flow_interactions=hp_bool(hparams, "flow_interactions", default=False),
        predict_interactions=hp_bool(hparams, "predict_interactions", default=False),
        interaction_inpainting=hp_bool(hparams, "interaction_inpainting", default=False),
        scaffold_inpainting=hp_bool(hparams, "scaffold_inpainting", default=False),
        func_group_inpainting=hp_bool(hparams, "func_group_inpainting", default=False),
        linker_inpainting=hp_bool(hparams, "linker_inpainting", default=False),
        fragment_inpainting=hp_bool(hparams, "fragment_inpainting", default=False),
        substructure_inpainting=hp_bool(hparams, "substructure_inpainting", default=False),
        substructure=hp_get(hparams, "substructure", default=None),
        max_fragment_cuts=int(hp_get(hparams, "max_fragment_cuts", default=2) or 2),
        mixed_uncond_inpaint=hp_bool(hparams, "mixed_uncond_inpaint", default=False),
        coord_noise_std_dev=float(hp_get(hparams, "coord_noise_std_dev", "train-ligand-coord-noise-std", default=0.2)),
        pocket_coord_noise_std_dev=float(hp_get(hparams, "pocket_coord_noise_std_dev", "train-pocket-coord-noise-std", default=0.0)),
        type_dist_temp=float(hp_get(hparams, "type_dist_temp", default=1.0)),
        time_alpha=float(hp_get(hparams, "time_alpha", "train-ligand-time-alpha", default=1.0)),
        time_beta=float(hp_get(hparams, "time_beta", "train-ligand-time-beta", default=1.0)),
        mixed_uniform_beta_time=hp_bool(hparams, "mixed_uniform_beta_time", default=False),
        optimal_transport=hp_get(hparams, "optimal_transport", default="none"),
        split_continuous_discrete_time=hp_bool(hparams, "split_continuous_discrete_time", default=False),
        separate_pocket_interpolation=hp_bool(hparams, "separate_pocket_interpolation", default=False),
        separate_interaction_interpolation=hp_bool(hparams, "separate_interaction_interpolation", default=False),
        interaction_fixed_time=hp_get(hparams, "interaction_fixed_time", default=None),
    )


def make_trainer_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        trial_run=args.trial_run,
        epochs=args.epochs,
        dataset=args.dataset,
        exp_name=args.exp_name,
        save_dir=args.save_dir,
        wandb=args.wandb,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        val_check_epochs=args.val_check_epochs,
        gpus=args.gpus,
        acc_batches=args.acc_batches,
        gradient_clip_val=args.gradient_clip_val,
        seed=args.seed,
    )


def set_hparam(model: Any, key: str, value: Any) -> None:
    try:
        model.hparams[key] = value
    except Exception:
        pass
    try:
        setattr(model.hparams, key, value)
    except Exception:
        pass


def apply_finetune_overrides(model: Any, args: argparse.Namespace, sample_count: int) -> None:
    rl_sampling_steps = args.rl_sampling_steps if args.rl_sampling_steps is not None else args.integration_steps
    overrides = {
        "data_path": args.data_path,
        "dataset": args.dataset,
        "save_dir": args.save_dir,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "lr_gamma": args.lr_gamma,
        "gradient_clip_val": args.gradient_clip_val,
        "epochs": args.epochs,
        "enable_rl_finetune": args.enable_rl_finetune,
        "rl_loss_weight": args.rl_loss_weight,
        "sample_n_molecules_per_target": sample_count,
        "rl_num_candidates_per_step": sample_count,
        "rl_sampling_steps": rl_sampling_steps,
        "rl_update_frequency": args.rl_update_frequency,
        "rl_surrogate_type": args.rl_surrogate_type,
        "rl_num_stratified_timesteps": args.rl_num_stratified_timesteps,
        "rl_surrogate_chunk_size": args.rl_surrogate_chunk_size,
        "rl_top_ratio": args.rl_top_ratio,
        "rl_bottom_ratio": args.rl_bottom_ratio,
        "rl_top_k": args.rl_top_k,
        "rl_bottom_k": args.rl_bottom_k,
        "rl_middle_weight": args.rl_middle_weight,
        "rl_bottom_repulsion_weight": args.rl_bottom_repulsion_weight,
        "rl_use_reference_model": args.rl_use_reference_model,
        "rl_sample_from_reference": args.rl_sample_from_reference,
        "rl_ref_ema_decay": args.rl_ref_ema_decay,
        "rl_reference_checkpoint": args.rl_reference_checkpoint,
        "rl_use_original_interpolant": args.rl_use_original_interpolant,
        "rl_allow_simple_corruption_fallback": args.rl_allow_simple_corruption_fallback,
        "rl_objective_mode": args.rl_objective_mode,
        "rl_multiobjective_strategy": args.rl_multiobjective_strategy,
        "rl_metric_source": args.rl_metric_source,
        "rl_metric_cache_path": args.rl_metric_cache_path,
        "rl_plif_weight": args.rl_plif_weight,
        "rl_strain_weight": args.rl_strain_weight,
        "rl_vina_weight": args.rl_vina_weight,
        "rl_plif_min_threshold": args.rl_plif_min_threshold,
        "rl_strain_max_threshold": args.rl_strain_max_threshold,
        "rl_vina_max_threshold": args.rl_vina_max_threshold,
        "rl_strain_good_threshold": args.rl_strain_good_threshold,
        "rl_strain_bad_threshold": args.rl_strain_bad_threshold,
        "rl_vina_good_threshold": args.rl_vina_good_threshold,
        "rl_vina_bad_threshold": args.rl_vina_bad_threshold,
        "rl_invalid_reward": args.rl_invalid_reward,
        "rl_metric_failed_reward": args.rl_metric_failed_reward,
        "rl_failed_as_bottom": args.rl_failed_as_bottom,
        "rl_top_atom_weight": args.rl_top_atom_weight,
        "rl_top_bond_weight": args.rl_top_bond_weight,
        "rl_top_charge_weight": args.rl_top_charge_weight,
        "rl_top_coord_weight": args.rl_top_coord_weight,
        "rl_bottom_atom_weight": args.rl_bottom_atom_weight,
        "rl_bottom_bond_weight": args.rl_bottom_bond_weight,
        "rl_bottom_charge_weight": args.rl_bottom_charge_weight,
        "rl_bottom_coord_weight": args.rl_bottom_coord_weight,
        "rl_beta_atom": args.rl_beta_atom,
        "rl_beta_bond": args.rl_beta_bond,
        "rl_beta_charge": args.rl_beta_charge,
        "rl_gamma_coord": args.rl_gamma_coord,
        "rl_aux_fm_weight": args.rl_aux_fm_weight,
        "rl_anchor_weight": args.rl_anchor_weight,
        "rl_detach_targets": args.rl_detach_targets,
        "rl_detach_reward": args.rl_detach_reward,
        "rl_log_selected_samples": args.rl_log_selected_samples,
        "rl_save_selected_samples": args.rl_save_selected_samples,
        "rl_log_metric_failures": args.rl_log_metric_failures,
    }
    for key, value in overrides.items():
        set_hparam(model, key, value)
    model.lr = args.lr
    model.lr_schedule = args.lr_schedule
    model.lr_gamma = args.lr_gamma
    model.hparams.lr = args.lr
    model.hparams.lr_schedule = args.lr_schedule
    model.hparams.lr_gamma = args.lr_gamma
    model.sampling_strategy = args.ode_sampling_strategy
    model.automatic_optimization = not (
        args.enable_rl_finetune and args.rl_loss_weight > 0.0 and args.rl_surrogate_chunk_size > 0
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    sample_count = resolve_sample_count(args)
    args.sample_n_molecules_per_target = sample_count
    args.rl_num_candidates_per_step = sample_count

    from flowr.data.data_info import GeneralInfos as DataInfos
    from flowr.gen.generate_from_smol import load_model as load_smol_model
    from flowr.train import build_data_statistic, build_dm, build_trainer

    hparams = checkpoint_hparams(args.ckpt_path)
    print_checkpoint_summary(hparams, args.ckpt_path)

    load_args = make_checkpoint_model_args(args, hparams)
    model, ckpt_hparams, vocab, vocab_pocket_atoms, vocab_pocket_res = load_smol_model(load_args)
    apply_finetune_overrides(model, args, sample_count)
    model.train()

    dm_args = make_datamodule_args(args, ckpt_hparams)
    print("[RL fine-tune] Loading training datamodule and train interpolant...")
    statistics = build_data_statistic(dm_args)
    dataset_info = DataInfos(statistics, vocab, dm_args)
    dm = build_dm(
        dm_args,
        vocab,
        vocab_pocket_atoms=vocab_pocket_atoms,
        vocab_pocket_res=vocab_pocket_res,
        atom_types_distribution=dataset_info.atom_types.float(),
        bond_types_distribution=dataset_info.edge_types.float(),
    )
    model.rl_train_interpolant = dm.train_interpolant
    print("[RL fine-tune] Attached dm.train_interpolant to model.rl_train_interpolant")
    print(
        "[RL fine-tune] RL sampling: "
        f"sample_n_molecules_per_target={sample_count}, "
        f"rl_sampling_steps={model.hparams.rl_sampling_steps}, "
        f"sample_from_reference={model.hparams.rl_sample_from_reference}"
    )

    trainer = build_trainer(make_trainer_args(args), model=model)
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_rl_ckpt)


if __name__ == "__main__":
    main()
