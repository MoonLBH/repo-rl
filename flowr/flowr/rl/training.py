"""Default-off LFPO-F-v2 training helpers for FLOWR pocket models.

This module implements a FLOWR-adapted Top-Reward Imitation + Bottom-Reward
Repulsion surrogate.  The code is intentionally opt-in: it never initializes a
reference model, runs ODE sampling, or computes structure rewards unless
``enable_rl_finetune`` is true and ``rl_loss_weight`` is positive.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

import flowr.util.functional as smolF
import flowr.util.rdkit as smolRD
from flowr.util.molrepr import GeometricMol
from flowr.util.pocket import PocketComplex, PocketComplexBatch
from flowr.rl.structure_rewards import OBJECTIVE_METRICS, RewardConfig, compute_structure_rewards_from_records
from flowr.rl.structure_selection import select_top_middle_bottom

OBJECTIVE_MODE_IDS = {
    "plif": 1,
    "strain": 2,
    "vina": 3,
    "plif_strain": 4,
    "plif_vina": 5,
    "strain_vina": 6,
    "plif_strain_vina": 7,
}
SKIP_REASON_IDS = {
    "disabled": 0,
    "no_top": 1,
    "metric_error": 2,
    "nan_loss": 3,
    "missing_rewards": 4,
    "frequency": 5,
    "no_candidates": 6,
    "surrogate_error": 7,
    "all_enabled_metrics_failed": 8,
    "interpolant_error": 9,
    "pseudo_batch_error": 10,
    "reference_generation_failed": 11,
    "no_bottom_use_top_only": 12,
    "reference_init_failed": 13,
}


def rl_enabled(hparams: Any) -> bool:
    return bool(getattr(hparams, "enable_rl_finetune", False)) and float(getattr(hparams, "rl_loss_weight", 0.0)) > 0.0


def rl_debug_raise_exceptions(model: Any) -> bool:
    return bool(getattr(model.hparams, "rl_debug_raise_exceptions", False))


def is_rank0(model: Any) -> bool:
    return int(getattr(model, "global_rank", 0) or 0) == 0


def rank0_debug_print(model: Any, message: str) -> None:
    if is_rank0(model):
        print(message, flush=True)


def handle_rl_exception(model: Any, reason: str, exc: BaseException) -> None:
    rank0_debug_print(model, f"[train-rl] {reason}: {type(exc).__name__}: {exc}")
    if is_rank0(model):
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    if rl_debug_raise_exceptions(model):
        raise exc


def tensor_shape(value: Any) -> Any:
    return tuple(value.shape) if torch.is_tensor(value) else None


def maybe_apply_rl_finetune_loss(
    model: Any,
    loss: torch.Tensor,
    prior: Mapping[str, Any],
    data: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply LFPO-F-v2 surrogate to ``loss`` when the RL path is enabled."""

    if not rl_enabled(model.hparams):
        return loss, {}

    logs = base_logs(model)
    if not should_run_rl_update(model):
        logs.update(skip_logs("frequency"))
        return loss, logs

    try:
        ref_gen = ensure_reference_generator(model)
    except Exception as exc:
        handle_rl_exception(model, "reference_init_failed", exc)
        logs.update(skip_logs("reference_init_failed"))
        return loss, logs

    try:
        candidates = sample_reference_candidates(model, ref_gen, prior, data)
        logs.update(candidate_sampling_logs(model))
    except Exception as exc:
        handle_rl_exception(model, "reference_generation_failed", exc)
        logs.update(skip_logs("reference_generation_failed"))
        return loss, logs

    if not candidates:
        logs.update(skip_logs("no_candidates"))
        return loss, logs

    try:
        rewards, cache_hit_rate = get_candidate_rewards(model, candidates)
    except Exception as exc:
        handle_rl_exception(model, "metric_error", exc)
        logs.update(skip_logs("metric_error"))
        return loss, logs

    if not rewards:
        logs.update(skip_logs("missing_rewards"))
        return loss, logs
    if enabled_metrics_all_failed(rewards, model.hparams):
        logs.update(reward_summary_logs(rewards, empty_selection_summary(rewards), cache_hit_rate))
        logs.update(skip_logs("all_enabled_metrics_failed"))
        return loss, logs

    selection = select_rewards_for_training(model, rewards)
    logs.update(reward_summary_logs(rewards, selection, cache_hit_rate))
    try:
        surrogate, surrogate_logs = lfpof_v2_surrogate_loss(model, ref_gen, prior, data, candidates, rewards, selection)
    except RuntimeError as exc:
        if "interpolant" in str(exc).lower():
            reason = "interpolant_error"
        elif "pseudo" in str(exc).lower():
            reason = "pseudo_batch_error"
        else:
            reason = "surrogate_error"
        handle_rl_exception(model, reason, exc)
        logs.update(skip_logs(reason))
        return loss, logs
    except Exception as exc:
        handle_rl_exception(model, "surrogate_error", exc)
        logs.update(skip_logs("surrogate_error"))
        return loss, logs

    logs.update(surrogate_logs)
    if surrogate is None:
        logs.update(skip_logs("no_top"))
        return loss, logs
    if not torch.isfinite(surrogate):
        logs.update(skip_logs("nan_loss"))
        return loss, logs
    logs["train-rl-loss"] = surrogate.detach()
    return loss + float(model.hparams.rl_loss_weight) * surrogate, logs


def rl_chunkwise_manual_enabled(model: Any) -> bool:
    return rl_enabled(model.hparams) and int(getattr(model.hparams, "rl_surrogate_chunk_size", 0) or 0) > 0


def run_rl_chunkwise_manual_optimization(
    model: Any,
    original_loss: torch.Tensor,
    prior: Mapping[str, Any],
    data: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Manual-optimization path with true per-chunk RL backward.

    This path is only used when RL is enabled and ``rl_surrogate_chunk_size > 0``.
    The original FLOWR loss is backpropagated once, then each RL pseudo-batch
    chunk is built, forwarded, backpropagated, and released independently.
    """

    logs = base_logs(model)
    logs["train-rl-manual-optimization-enabled"] = 1.0
    logs["train-rl-surrogate-chunk-size"] = float(getattr(model.hparams, "rl_surrogate_chunk_size", 0))
    optimizer = model.optimizers()
    optimizer.zero_grad()
    model.manual_backward(original_loss)

    def finish(skip_reason: Optional[str] = None, rl_loss_value: float = 0.0) -> tuple[torch.Tensor, dict[str, Any]]:
        if skip_reason is not None:
            logs.update(skip_logs(skip_reason))
        step_optimizer_and_scheduler(model, optimizer)
        detached_total = original_loss.detach() + original_loss.detach().new_tensor(float(getattr(model.hparams, "rl_loss_weight", 0.0)) * rl_loss_value)
        return detached_total, logs

    if not should_run_rl_update(model):
        return finish("frequency")

    try:
        ref_gen = ensure_reference_generator(model)
    except Exception as exc:
        handle_rl_exception(model, "reference_init_failed", exc)
        return finish("reference_init_failed")

    try:
        candidates = sample_reference_candidates(model, ref_gen, prior, data)
        logs.update(candidate_sampling_logs(model))
    except Exception as exc:
        handle_rl_exception(model, "reference_generation_failed", exc)
        return finish("reference_generation_failed")
    if not candidates:
        return finish("no_candidates")

    try:
        rewards, cache_hit_rate = get_candidate_rewards(model, candidates)
    except Exception as exc:
        handle_rl_exception(model, "metric_error", exc)
        return finish("metric_error")
    if not rewards:
        return finish("missing_rewards")
    selection = empty_selection_summary(rewards) if enabled_metrics_all_failed(rewards, model.hparams) else select_rewards_for_training(model, rewards)
    logs.update(reward_summary_logs(rewards, selection, cache_hit_rate))
    if enabled_metrics_all_failed(rewards, model.hparams):
        return finish("all_enabled_metrics_failed")
    if not selection["top_indices"]:
        return finish("no_top")

    selected_indices = list(selection["top_indices"]) + list(selection["bottom_indices"])
    selected_labels = ["top"] * len(selection["top_indices"]) + ["bottom"] * len(selection["bottom_indices"])
    full_plan = expanded_selection_plan(selected_indices, selected_labels, max(1, int(getattr(model.hparams, "rl_num_stratified_timesteps", 1))))
    if not full_plan:
        return finish("pseudo_batch_error")
    chunk_size = max(1, int(getattr(model.hparams, "rl_surrogate_chunk_size", 1)))
    chunks = [full_plan[i : i + chunk_size] for i in range(0, len(full_plan), chunk_size)]
    logs.update(expanded_count_logs([label for _, label, _ in full_plan], selected_labels))
    logs["train-rl-num-rl-chunks"] = float(len(chunks))
    logs["train-rl-expanded-selected-count"] = float(len(full_plan))

    rl_weight = float(getattr(model.hparams, "rl_loss_weight", 0.0))
    total_rl_value = 0.0
    aggregate = init_chunk_aggregate(model)
    try:
        for chunk in chunks:
            chunk_loss, chunk_logs = lfpof_v2_surrogate_loss_for_plan(model, ref_gen, prior, data, candidates, rewards, selection, chunk, len(full_plan))
            if chunk_loss is None:
                continue
            if not torch.isfinite(chunk_loss):
                return finish("nan_loss", total_rl_value)
            model.manual_backward(rl_weight * chunk_loss)
            total_rl_value += float(chunk_loss.detach().cpu())
            update_chunk_aggregate(aggregate, chunk_logs)
            del chunk_loss
    except RuntimeError as exc:
        if "interpolant" in str(exc).lower():
            reason = "interpolant_error"
        elif "pseudo" in str(exc).lower():
            reason = "pseudo_batch_error"
        else:
            reason = "surrogate_error"
        handle_rl_exception(model, reason, exc)
        return finish(reason, total_rl_value)
    except Exception as exc:
        handle_rl_exception(model, "surrogate_error", exc)
        return finish("surrogate_error", total_rl_value)

    logs.update(finalize_chunk_aggregate(aggregate, max(1, len(chunks))))
    logs.update(reward_selection_logs(rewards, selection))
    logs["train-rl-loss"] = float(total_rl_value)
    return finish(None, total_rl_value)


def step_optimizer_and_scheduler(model: Any, optimizer: Any) -> None:
    grad_clip = float(getattr(model.hparams, "gradient_clip_val", 0.0) or 0.0)
    if grad_clip > 0.0:
        model.clip_gradients(optimizer, gradient_clip_val=grad_clip, gradient_clip_algorithm="norm")
    optimizer.step()
    scheduler = model.lr_schedulers()
    if scheduler is None:
        return
    schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]
    for sched in schedulers:
        try:
            sched.step()
        except TypeError:
            sched.scheduler.step()


def on_train_batch_end_update_reference(model: Any) -> None:
    """EMA-update the explicit RL reference generator after optimizer updates."""

    ref_gen = get_reference_generator(model)
    if not rl_enabled(model.hparams) or ref_gen is None:
        return
    decay = float(getattr(model.hparams, "rl_ref_ema_decay", 0.999))
    with torch.no_grad():
        for ref_param, cur_param in zip(ref_gen.parameters(), model.gen.parameters()):
            ref_param.data.mul_(decay).add_(cur_param.detach().data, alpha=1.0 - decay)
        for ref_buffer, cur_buffer in zip(ref_gen.buffers(), model.gen.buffers()):
            if ref_buffer.dtype.is_floating_point:
                ref_buffer.data.mul_(decay).add_(cur_buffer.detach().data, alpha=1.0 - decay)
            else:
                ref_buffer.data.copy_(cur_buffer.detach().data)
    ref_gen.eval()
    for param in ref_gen.parameters():
        param.requires_grad_(False)


def should_run_rl_update(model: Any) -> bool:
    freq = max(1, int(getattr(model.hparams, "rl_update_frequency", 1)))
    return int(getattr(model, "global_step", 0)) % freq == 0


def get_reference_generator(model: Any) -> Any:
    return model.__dict__.get("_rl_ref_gen")


def set_reference_generator(model: Any, ref_gen: Any) -> None:
    remove_registered_reference_generator(model)
    model.__dict__["_rl_ref_gen"] = ref_gen


def remove_registered_reference_generator(model: Any) -> None:
    modules = getattr(model, "_modules", None)
    if isinstance(modules, dict) and "_rl_ref_gen" in modules:
        modules.pop("_rl_ref_gen")
    model.__dict__.pop("_rl_ref_gen", None)


def ensure_reference_generator(model: Any) -> Any:
    """Create or return the frozen EMA reference generator only.

    This mirrors the LIFT/LFPO-F convention of maintaining
    ``ref_gen = deepcopy(model.gen)`` and avoids copying the full LightningModule.
    """

    ref_gen = get_reference_generator(model)
    if ref_gen is None:
        ref_gen = copy.deepcopy(model.gen)
        ref_gen.eval()
        ref_gen.to(model.device)
        for param in ref_gen.parameters():
            param.requires_grad_(False)
        set_reference_generator(model, ref_gen)
        rank0_debug_print(model, "[train-rl] Initialized ref_gen by deepcopy(model.gen)")
    else:
        ref_gen.eval()
        ref_gen.to(model.device)
        for param in ref_gen.parameters():
            param.requires_grad_(False)
    return ref_gen


def ensure_reference_model(model: Any) -> Any:
    """Backward-compatible alias; returns the ref_gen, not a full model."""

    return ensure_reference_generator(model)


def get_reference_model(model: Any) -> Any:
    return get_reference_generator(model)


def set_reference_model(model: Any, ref_model: Any) -> None:
    set_reference_generator(model, ref_model)


def remove_registered_reference_module(model: Any) -> None:
    remove_registered_reference_generator(model)



def candidate_sampling_logs(model: Any) -> dict[str, float]:
    return {
        "train-rl-sample-n-molecules-per-target": float(
            getattr(model, "_rl_last_sample_n_molecules_per_target", get_sample_n_molecules_per_target(model.hparams))
        ),
        "train-rl-num-targets-in-batch": float(getattr(model, "_rl_last_num_targets_in_batch", 0)),
        "train-rl-num-candidates": float(getattr(model, "_rl_last_num_candidates", 0)),
        "train-rl-reference-type": 1.0,
        "train-rl-ref-ema-decay": float(getattr(model.hparams, "rl_ref_ema_decay", 0.999)),
    }

def base_logs(model: Any) -> dict[str, Any]:
    return {
        "train-rl-enabled": float(rl_enabled(model.hparams)),
        "train-rl-loss": 0.0,
        "train-rl-main-loss": 0.0,
        "train-rl-top-loss": 0.0,
        "train-rl-bottom-loss": 0.0,
        "train-rl-aux-fm-loss": 0.0,
        "train-rl-anchor-loss": 0.0,
        "train-rl-loss-weight": float(getattr(model.hparams, "rl_loss_weight", 0.0)),
        "train-rl-reference-type": 1.0,
        "train-rl-ref-ema-decay": float(getattr(model.hparams, "rl_ref_ema_decay", 0.999)),
        "train-rl-objective-mode": float(OBJECTIVE_MODE_IDS.get(getattr(model.hparams, "rl_objective_mode", "strain"), 0)),
        "train-rl-num-candidates": 0.0,
        "train-rl-sample-n-molecules-per-target": float(get_sample_n_molecules_per_target(model.hparams)),
        "train-rl-num-targets-in-batch": 0.0,
        "train-rl-num-valid": 0.0,
        "train-rl-num-posebusters-valid": 0.0,
        "train-rl-num-metric-success": 0.0,
        "train-rl-num-top": 0.0,
        "train-rl-num-bottom": 0.0,
        "train-rl-num-middle": 0.0,
        "train-rl-selected-frac": 0.0,
        "train-rl-plif-mean": 0.0,
        "train-rl-strain-mean": 0.0,
        "train-rl-vina-mean": 0.0,
        "train-rl-main-score-mean": 0.0,
        "train-rl-main-score-max": 0.0,
        "train-rl-invalid-count": 0.0,
        "train-rl-posebusters-failed-count": 0.0,
        "train-rl-plif-failed-count": 0.0,
        "train-rl-strain-failed-count": 0.0,
        "train-rl-vina-failed-count": 0.0,
        "train-rl-skip-count": 0.0,
        "train-rl-skip-reason": 0.0,
        "train-rl-cache-hit-rate": 0.0,
        "train-rl-num-top-candidates": 0.0,
        "train-rl-num-bottom-candidates": 0.0,
        "train-rl-num-top-expanded": 0.0,
        "train-rl-num-bottom-expanded": 0.0,
        "train-rl-interpolant-mode": 0.0,
        "train-rl-num-stratified-timesteps": float(getattr(model.hparams, "rl_num_stratified_timesteps", 1)),
        "train-rl-pseudo-batch-size": 0.0,
        "train-rl-surrogate-chunk-size": float(getattr(model.hparams, "rl_surrogate_chunk_size", 0)),
        "train-rl-num-rl-chunks": 0.0,
        "train-rl-expanded-selected-count": 0.0,
        "train-rl-manual-optimization-enabled": 0.0,
        "train-rl-bottom-branch-used": 0.0,
        "train-rl-delta-atom-abs-mean": 0.0,
        "train-rl-delta-bond-abs-mean": 0.0,
        "train-rl-delta-charge-abs-mean": 0.0,
        "train-rl-delta-coord-abs-mean": 0.0,
        "train-rl-reward-top-mean": 0.0,
        "train-rl-reward-bottom-mean": 0.0,
        "train-rl-reward-top10-mean": 0.0,
    }


def skip_logs(reason: str) -> dict[str, float]:
    return {"train-rl-skip-count": 1.0, "train-rl-skip-reason": float(SKIP_REASON_IDS.get(reason, -1))}


def reward_config_from_hparams(hparams: Any) -> RewardConfig:
    multi = is_multiobjective(getattr(hparams, "rl_objective_mode", "strain"))
    return RewardConfig(
        objective_mode=getattr(hparams, "rl_objective_mode", "strain"),
        multiobjective_strategy=getattr(hparams, "rl_multiobjective_strategy", "constrained_weighted_sum"),
        plif_weight=float(getattr(hparams, "rl_plif_weight", 1.0)),
        strain_weight=float(getattr(hparams, "rl_strain_weight", 1.0)),
        vina_weight=float(getattr(hparams, "rl_vina_weight", 1.0)),
        plif_min_threshold=float(getattr(hparams, "rl_plif_min_threshold", 0.0)) if multi else 0.0,
        strain_max_threshold=float(getattr(hparams, "rl_strain_max_threshold", math.inf)) if multi else math.inf,
        vina_max_threshold=float(getattr(hparams, "rl_vina_max_threshold", math.inf)) if multi else math.inf,
        strain_good_threshold=float(getattr(hparams, "rl_strain_good_threshold", 0.0)),
        strain_bad_threshold=float(getattr(hparams, "rl_strain_bad_threshold", 20.0)),
        vina_good_threshold=float(getattr(hparams, "rl_vina_good_threshold", -10.0)),
        vina_bad_threshold=float(getattr(hparams, "rl_vina_bad_threshold", 0.0)),
        require_posebusters_validity=True,
        compute_posebusters_validity=True,
        compute_non_enabled_metrics=False,
    )


def is_multiobjective(objective_mode: str) -> bool:
    return len(OBJECTIVE_METRICS[objective_mode]) > 1


@contextmanager
def temporarily_use_generator(model: Any, gen: Any):
    original_gen = model.gen
    try:
        model.gen = gen
        yield
    finally:
        model.gen = original_gen


def generate_with_generator(
    model: Any,
    gen: Any,
    lig_prior: Mapping[str, Any],
    pocket_data: Mapping[str, Any],
    steps: int,
    times: list[torch.Tensor],
    strategy: str,
    corr_iters: int,
) -> Mapping[str, Any]:
    """Run FLOWR generation with an explicit generator while reusing model utilities."""

    gen.eval()
    with torch.no_grad(), temporarily_use_generator(model, gen):
        return model._generate(
            lig_prior,
            pocket_data,
            steps=steps,
            times=times,
            strategy=strategy,
            corr_iters=corr_iters,
        )


def sample_reference_candidates(model: Any, ref_gen: Any, prior: Mapping[str, Any], data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Sample complete ligand candidates with FLOWR ODE generation from pi_ref."""

    num_rounds = get_sample_n_molecules_per_target(model.hparams)
    sampling_steps = max(1, int(getattr(model.hparams, "rl_sampling_steps", 100)))
    sampler_gen = ref_gen if bool(getattr(model.hparams, "rl_sample_from_reference", True)) else model.gen
    candidates: list[dict[str, Any]] = []

    pocket_data = model.builder.extract_pocket_from_complex(data)
    pocket_data["interactions"] = prior.get("interactions", data.get("interactions"))
    pocket_data["complex"] = data.get("complex")
    lig_prior = model.builder.extract_ligand_from_complex(prior)
    lig_prior["fragment_mask"] = prior.get("fragment_mask")
    lig_prior["interactions"] = prior.get("interactions")
    num_targets = int(lig_prior["mask"].size(0))
    model._rl_last_num_targets_in_batch = num_targets
    model._rl_last_sample_n_molecules_per_target = num_rounds
    times = zero_generation_times(model, prior, pocket_data)
    systems = data.get("complex") or []
    protein_files = write_training_pockets(model, systems, int(lig_prior["mask"].size(0)))
    ref_ligs = [system.ligand.orig_mol.to_rdkit() for system in systems]

    was_training = sampler_gen.training
    rank0_debug_print(
        model,
        "[train-rl] reference generation input: "
        f"num_rounds={num_rounds}, sampling_steps={sampling_steps}, "
        f"sampler_gen.training={sampler_gen.training}, "
        f"lig_prior_mask_shape={tensor_shape(lig_prior.get('mask'))}, "
        f"pocket_mask_shape={tensor_shape(pocket_data.get('mask'))}, "
        f"len_systems={len(systems)}, system0_type={type(systems[0]).__name__ if systems else None}",
    )
    sampler_gen.eval()
    with torch.no_grad():
        for round_idx in range(num_rounds):
            generated = generate_with_generator(
                model=model,
                gen=sampler_gen,
                lig_prior=detach_mapping(lig_prior),
                pocket_data=detach_mapping(pocket_data),
                steps=sampling_steps,
                times=[t.clone() for t in times],
                strategy=model.sampling_strategy,
                corr_iters=model.corrector_iters,
            )
            rank0_debug_print(
                model,
                "[train-rl] reference generation output: "
                f"round_idx={round_idx}, generated_keys={sorted(generated.keys())}, "
                f"generated_mask_shape={tensor_shape(generated.get('mask'))}",
            )
            mols = model._generate_mols(generated)
            train_targets = generated_to_training_targets(model, generated, systems)
            rank0_debug_print(
                model,
                "[train-rl] reference mol conversion: "
                f"round_idx={round_idx}, len_mols={len(mols)}, "
                f"train_targets_mask_shape={tensor_shape(train_targets.get('mask'))}",
            )
            batch_size = int(train_targets["mask"].size(0))
            for idx in range(batch_size):
                system = systems[idx] if idx < len(systems) else None
                system_id = system.metadata.get("system_id", f"batch_{idx}") if system is not None else f"batch_{idx}"
                target = slice_ligand_target(train_targets, idx)
                candidates.append(
                    {
                        "sample_id": f"train_step_{model.global_step}_{round_idx}_{system_id}_{idx}",
                        "batch_index": idx,
                        "candidate_round": round_idx,
                        "system_id": system_id,
                        "target": target,
                        "mol": mols[idx] if idx < len(mols) else None,
                        "protein_file": protein_files[idx] if idx < len(protein_files) else None,
                        "reference_ligand": ref_ligs[idx] if idx < len(ref_ligs) else None,
                        "metadata": {"system_id": system_id, "batch_index": idx, "candidate_round": round_idx},
                    }
                )
    if was_training:
        sampler_gen.train()
    model._rl_last_num_candidates = len(candidates)
    rank0_debug_print(model, f"[train-rl] reference generation final: len_candidates={len(candidates)}")
    return candidates


def zero_generation_times(model: Any, prior: Mapping[str, Any], pocket_data: Mapping[str, Any]) -> list[torch.Tensor]:
    lig_bsz = int(prior["coords"].size(0))
    pocket_bsz = int(pocket_data["coords"].size(0))
    return [
        torch.zeros(lig_bsz, device=model.device),
        torch.zeros(lig_bsz, device=model.device),
        torch.zeros(pocket_bsz, device=model.device),
        torch.zeros(lig_bsz, device=model.device),
    ]


def detach_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.detach().clone() if torch.is_tensor(value) else value for key, value in mapping.items()}


def generated_to_training_targets(model: Any, generated: Mapping[str, Any], systems: Sequence[Any]) -> dict[str, torch.Tensor]:
    coords = generated["coords"].clone().to(model.device)
    mask = generated["mask"].clone().to(model.device)
    if systems:
        com = torch.stack([system.com.to(model.device) for system in systems])
        coords = coords - com[:, None, :]
    coords = coords / float(model.hparams.coord_scale)
    return {
        "coords": coords.detach(),
        "atomics": generated["atomics"].clone().to(model.device).detach(),
        "bonds": generated["bonds"].clone().to(model.device).detach(),
        "charges": generated["charges"].clone().to(model.device).detach(),
        "mask": mask.detach(),
    }


def slice_ligand_target(targets: Mapping[str, torch.Tensor], idx: int) -> dict[str, torch.Tensor]:
    return {key: value[idx].detach().clone() for key, value in targets.items()}


def write_training_pockets(model: Any, systems: Sequence[Any], batch_size: int) -> list[Optional[str]]:
    if not systems:
        return [None] * batch_size
    pocket_dir = Path(model.hparams.save_dir) / "train_rl_ref_pdbs"
    pocket_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for system in systems:
        system_id = system.metadata.get("system_id", "system")
        pdb_file = pocket_dir / f"{system_id}_with_hs.pdb"
        if not pdb_file.exists():
            system.holo.orig_pocket.write_pdb(str(pdb_file), include_bonds=True)
        paths.append(str(pdb_file))
    return paths


def get_candidate_rewards(model: Any, candidates: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    """Score candidates with all objective-enabled metrics after validity/PB gating.

    The historical every-n-steps and warmup parameters are intentionally not used
    here: once an RL update runs, every valid and PoseBusters-valid cache miss is
    evaluated for every metric enabled by ``rl_objective_mode``. Cache hits and
    misses are returned in original candidate order so selection indices stay
    aligned with sampled candidates.
    """

    metric_source = getattr(model.hparams, "rl_metric_source", "compute")
    if metric_source == "existing_eval_output":
        return load_existing_train_rewards(model, len(candidates)), 1.0

    records = [candidate_to_reward_record(candidate) for candidate in candidates]
    cache = load_metric_cache(getattr(model.hparams, "rl_metric_cache_path", None)) if metric_source == "cached" else {}
    rewards: list[Optional[dict[str, Any]]] = [None] * len(records)
    hits = 0
    misses: list[tuple[int, str, dict[str, Any]]] = []
    for idx, record in enumerate(records):
        key = train_cache_key(record, model.hparams)
        if key in cache:
            rewards[idx] = dict(cache[key])
            hits += 1
        else:
            misses.append((idx, key, record))
    if misses:
        with torch.no_grad():
            computed = compute_structure_rewards_from_records(
                [record for _, _, record in misses],
                config=reward_config_from_hparams(model.hparams),
            )
        for (idx, key, _), reward in zip(misses, computed):
            reward = dict(reward)
            reward["cache_key"] = key
            cache[key] = reward
            rewards[idx] = reward
        if metric_source == "cached":
            rewrite_metric_cache(getattr(model.hparams, "rl_metric_cache_path", None), cache)
    hit_rate = hits / len(records) if records else 0.0
    return [reward if reward is not None else metric_failed_reward(records[idx], model.hparams) for idx, reward in enumerate(rewards)], hit_rate


def metric_failed_reward(record: Mapping[str, Any], hparams: Any) -> dict[str, Any]:
    return {
        "sample_id": record.get("sample_id", "sample"),
        "valid": False,
        "posebusters_valid": False,
        "plif_success": False,
        "strain_success": False,
        "vina_success": False,
        "metric_success": False,
        "feasible": False,
        "eligible_top": False,
        "eligible_bottom": True,
        "main_score": float(getattr(hparams, "rl_metric_failed_reward", 0.0)),
        "error": "metric_failed_missing_result",
        "metadata": record.get("metadata", {}),
    }

def candidate_to_reward_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": candidate["sample_id"],
        "mol": candidate.get("mol"),
        "protein_file": candidate.get("protein_file"),
        "reference_ligand": candidate.get("reference_ligand"),
        "structure_protein_file": candidate.get("protein_file"),
        "structure_reference_ligand": candidate.get("reference_ligand"),
        "plif_protein_file": candidate.get("protein_file"),
        "plif_reference_ligand": candidate.get("reference_ligand"),
        "metadata": candidate.get("metadata", {}),
    }


def load_existing_train_rewards(model: Any, n_candidates: int) -> list[dict[str, Any]]:
    path = getattr(model.hparams, "rl_metric_cache_path", None)
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        return []
    if not hasattr(model, "_rl_existing_rewards"):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("rewards") or payload.get("reward_dicts") or payload.get("records") or []
        model._rl_existing_rewards = [dict(item) for item in payload]
        model._rl_reward_cursor = 0
    rewards = model._rl_existing_rewards
    if not rewards:
        return []
    start = int(getattr(model, "_rl_reward_cursor", 0))
    selected = [rewards[(start + idx) % len(rewards)] for idx in range(n_candidates)]
    model._rl_reward_cursor = (start + n_candidates) % len(rewards)
    return [dict(item) for item in selected]


def load_metric_cache(path: Optional[str]) -> dict[str, dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if "cache_key" in item and "reward" in item:
            cache[item["cache_key"]] = item["reward"]
    return cache


def rewrite_metric_cache(path: Optional[str], cache: Mapping[str, Mapping[str, Any]]) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for key, reward in sorted(cache.items()):
            handle.write(json.dumps({"cache_key": key, "reward": reward}, sort_keys=True, default=str) + "\n")


def train_cache_key(record: Mapping[str, Any], hparams: Any) -> str:
    from rdkit import Chem

    mol_block = ""
    try:
        mol_block = Chem.MolToMolBlock(record["mol"])
    except Exception:
        mol_block = str(record.get("sample_id"))
    raw = {
        "mol_hash": hashlib.sha1(mol_block.encode()).hexdigest(),
        "system_id": record.get("metadata", {}).get("system_id"),
        "objective_mode": getattr(hparams, "rl_objective_mode", "strain"),
    }
    return hashlib.sha1(json.dumps(raw, sort_keys=True).encode()).hexdigest()


def enabled_metrics_all_failed(rewards: Sequence[Mapping[str, Any]], hparams: Any) -> bool:
    enabled = OBJECTIVE_METRICS[getattr(hparams, "rl_objective_mode", "strain")]
    feasible_seen = False
    success_seen = False
    for reward in rewards:
        if bool(reward.get("valid")) and bool(reward.get("posebusters_valid")):
            feasible_seen = True
            if all(bool(reward.get(f"{metric}_success")) for metric in enabled):
                success_seen = True
                break
    return feasible_seen and not success_seen


def empty_selection_summary(rewards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rewards)
    return {
        "top_indices": [],
        "middle_indices": list(range(n)),
        "bottom_indices": [],
        "summary": {"num_top": 0, "num_bottom": 0, "num_middle": n, "enabled_metrics": []},
    }


def select_rewards_for_training(model: Any, rewards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    multi = is_multiobjective(getattr(model.hparams, "rl_objective_mode", "strain"))
    return select_top_middle_bottom(
        rewards,
        objective_mode=getattr(model.hparams, "rl_objective_mode", "strain"),
        multiobjective_strategy=getattr(model.hparams, "rl_multiobjective_strategy", "constrained_weighted_sum"),
        top_ratio=float(getattr(model.hparams, "rl_top_ratio", 0.25)),
        bottom_ratio=float(getattr(model.hparams, "rl_bottom_ratio", 0.25)),
        top_k=getattr(model.hparams, "rl_top_k", None),
        bottom_k=getattr(model.hparams, "rl_bottom_k", None),
        feasible_only_for_top=True,
        include_failed_in_bottom=bool(getattr(model.hparams, "rl_failed_as_bottom", True)),
        plif_min_threshold=float(getattr(model.hparams, "rl_plif_min_threshold", 0.0)) if multi else 0.0,
        strain_max_threshold=float(getattr(model.hparams, "rl_strain_max_threshold", math.inf)) if multi else math.inf,
        vina_max_threshold=float(getattr(model.hparams, "rl_vina_max_threshold", math.inf)) if multi else math.inf,
        plif_weight=float(getattr(model.hparams, "rl_plif_weight", 1.0)),
        strain_weight=float(getattr(model.hparams, "rl_strain_weight", 1.0)),
        vina_weight=float(getattr(model.hparams, "rl_vina_weight", 1.0)),
    )


def lfpof_v2_surrogate_loss(
    model: Any,
    ref_gen: Any,
    prior: Mapping[str, Any],
    data: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    rewards: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
    if not selection["top_indices"]:
        return None, {}

    selected_indices = list(selection["top_indices"]) + list(selection["bottom_indices"])
    selected_labels = ["top"] * len(selection["top_indices"]) + ["bottom"] * len(selection["bottom_indices"])
    pseudo = build_pseudo_training_batch(model, prior, data, candidates, selected_indices, selected_labels)
    current_pred = forward_ligand_pocket(model, pseudo)
    with torch.no_grad():
        ref_pred = forward_ligand_pocket_with_generator(model, ref_gen, pseudo)

    top_mask, bottom_mask, _ = masks_from_labels(pseudo["labels"], current_pred["coords"].device)
    assert top_mask.shape[0] == current_pred["coords"].shape[0]
    assert bottom_mask.shape[0] == current_pred["coords"].shape[0]
    top_loss, top_logs = top_imitation_loss(model, pseudo["target"], current_pred, top_mask)
    bottom_loss, bottom_logs = bottom_repulsion_loss(model, current_pred, ref_pred, pseudo["target"], bottom_mask)
    aux_fm_loss = torch.zeros((), device=model.device)
    if float(getattr(model.hparams, "rl_aux_fm_weight", 0.0)) > 0.0:
        aux_losses = model._loss(pseudo["target"], pseudo["interp"], current_pred)
        aux_fm_loss = sum(aux_losses.values())
    anchor_loss = anchor_regularization_loss(current_pred, ref_pred, pseudo["target"]) if float(getattr(model.hparams, "rl_anchor_weight", 0.0)) > 0.0 else torch.zeros((), device=model.device)
    main_loss = top_loss + float(getattr(model.hparams, "rl_bottom_repulsion_weight", 1.0)) * bottom_loss
    total = main_loss + float(getattr(model.hparams, "rl_aux_fm_weight", 0.0)) * aux_fm_loss + float(getattr(model.hparams, "rl_anchor_weight", 0.0)) * anchor_loss

    expanded_counts = expanded_count_logs(pseudo["labels"], selected_labels)
    logs: dict[str, Any] = {
        "train-rl-main-loss": main_loss.detach(),
        "train-rl-top-loss": top_loss.detach(),
        "train-rl-bottom-loss": bottom_loss.detach(),
        "train-rl-aux-fm-loss": aux_fm_loss.detach(),
        "train-rl-anchor-loss": anchor_loss.detach(),
        "train-rl-bottom-branch-used": float(bool(bottom_mask.any())),
        **expanded_counts,
        **pseudo.get("logs", {}),
        **top_logs,
        **bottom_logs,
    }
    if not bool(bottom_mask.any()):
        logs.update(skip_logs("no_bottom_use_top_only"))
    logs.update(reward_selection_logs(rewards, selection))
    return total, logs



def lfpof_v2_surrogate_loss_for_plan(
    model: Any,
    ref_gen: Any,
    prior: Mapping[str, Any],
    data: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    rewards: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    expanded_plan: Sequence[tuple[int, str, int]],
    global_expanded_count: int,
) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
    pseudo = build_pseudo_training_batch_from_plan(model, prior, data, candidates, expanded_plan)
    current_pred = forward_ligand_pocket(model, pseudo)
    with torch.no_grad():
        ref_pred = forward_ligand_pocket_with_generator(model, ref_gen, pseudo)
    top_mask, bottom_mask, _ = masks_from_labels(pseudo["labels"], current_pred["coords"].device)
    assert top_mask.shape[0] == current_pred["coords"].shape[0]
    assert bottom_mask.shape[0] == current_pred["coords"].shape[0]
    if not bool(top_mask.any()) and not bool(bottom_mask.any()):
        return None, {}
    top_loss, top_logs = top_imitation_loss(model, pseudo["target"], current_pred, top_mask)
    bottom_loss, bottom_logs = bottom_repulsion_loss(model, current_pred, ref_pred, pseudo["target"], bottom_mask)
    aux_fm_loss = torch.zeros((), device=model.device)
    if float(getattr(model.hparams, "rl_aux_fm_weight", 0.0)) > 0.0:
        aux_losses = model._loss(pseudo["target"], pseudo["interp"], current_pred)
        aux_fm_loss = sum(aux_losses.values())
    anchor_loss = anchor_regularization_loss(current_pred, ref_pred, pseudo["target"]) if float(getattr(model.hparams, "rl_anchor_weight", 0.0)) > 0.0 else torch.zeros((), device=model.device)
    main_loss = top_loss + float(getattr(model.hparams, "rl_bottom_repulsion_weight", 1.0)) * bottom_loss
    total = main_loss + float(getattr(model.hparams, "rl_aux_fm_weight", 0.0)) * aux_fm_loss + float(getattr(model.hparams, "rl_anchor_weight", 0.0)) * anchor_loss
    scale = float(len(expanded_plan)) / float(max(1, global_expanded_count))
    logs = {
        "train-rl-main-loss": main_loss.detach() * scale,
        "train-rl-top-loss": top_loss.detach() * scale,
        "train-rl-bottom-loss": bottom_loss.detach() * scale,
        "train-rl-aux-fm-loss": aux_fm_loss.detach() * scale,
        "train-rl-anchor-loss": anchor_loss.detach() * scale,
        "train-rl-bottom-branch-used": float(bool(bottom_mask.any())),
        **pseudo.get("logs", {}),
        **top_logs,
        **bottom_logs,
    }
    return total * scale, logs


def init_chunk_aggregate(model: Any) -> dict[str, float]:
    keys = [
        "train-rl-main-loss", "train-rl-top-loss", "train-rl-bottom-loss",
        "train-rl-aux-fm-loss", "train-rl-anchor-loss",
        "train-rl-delta-atom-abs-mean", "train-rl-delta-bond-abs-mean",
        "train-rl-delta-charge-abs-mean", "train-rl-delta-coord-abs-mean",
        "train-rl-bottom-branch-used", "train-rl-interpolant-mode",
        "train-rl-pseudo-batch-size",
    ]
    return {key: 0.0 for key in keys}


def update_chunk_aggregate(aggregate: dict[str, float], logs: Mapping[str, Any]) -> None:
    for key in aggregate:
        value = logs.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            value = float(value.detach().cpu())
        aggregate[key] += float(value)


def finalize_chunk_aggregate(aggregate: dict[str, float], n_chunks: int) -> dict[str, float]:
    averaged = {key: value for key, value in aggregate.items()}
    for key in [
        "train-rl-delta-atom-abs-mean", "train-rl-delta-bond-abs-mean",
        "train-rl-delta-charge-abs-mean", "train-rl-delta-coord-abs-mean",
        "train-rl-interpolant-mode",
    ]:
        averaged[key] = aggregate[key] / float(max(1, n_chunks))
    return averaged


def build_pseudo_training_batch(
    model: Any,
    prior: Mapping[str, Any],
    data: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
    selected_labels: Sequence[str],
) -> dict[str, Any]:
    plan = expanded_selection_plan(selected_indices, selected_labels, max(1, int(getattr(model.hparams, "rl_num_stratified_timesteps", 1))))
    return build_pseudo_training_batch_from_plan(model, prior, data, candidates, plan)


def expanded_selection_plan(selected_indices: Sequence[int], selected_labels: Sequence[str], k_steps: int) -> list[tuple[int, str, int]]:
    expanded: list[tuple[int, str, int]] = []
    for idx, label in zip(selected_indices, selected_labels):
        for k_id in range(k_steps):
            expanded.append((int(idx), str(label), int(k_id)))
    return expanded


def build_pseudo_training_batch_from_plan(
    model: Any,
    prior: Mapping[str, Any],
    data: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    expanded_plan: Sequence[tuple[int, str, int]],
) -> dict[str, Any]:
    if not expanded_plan:
        raise RuntimeError("Cannot build RL pseudo batch without selected expanded samples")
    expanded_candidates = [candidates[idx] for idx, _, _ in expanded_plan]
    expanded_labels = [label for _, label, _ in expanded_plan]
    k_ids = [k_id for _, _, k_id in expanded_plan]
    k_steps = max(1, int(getattr(model.hparams, "rl_num_stratified_timesteps", 1)))
    times_cont, times_disc = stratified_times_from_k_ids(k_ids, k_steps, model.device)
    mode = "original_interpolant"
    try:
        pseudo = build_pseudo_with_original_interpolant(model, data, expanded_candidates, expanded_labels, times_cont, times_disc)
    except Exception as exc:
        if bool(getattr(model.hparams, "rl_use_original_interpolant", True)) and not bool(getattr(model.hparams, "rl_allow_simple_corruption_fallback", False)):
            raise RuntimeError(f"FLOWR original interpolant construction failed: {exc}") from exc
        target = stack_targets([candidate["target"] for candidate in expanded_candidates], model.device)
        interp = corrupt_ligand_like_flowr(target, times_cont, times_disc)
        batch_indices = [int(candidate["batch_index"]) for candidate in expanded_candidates]
        pocket = gather_pocket_for_candidates(model, data, prior, batch_indices)
        target["pocket_mask"] = pocket["mask"]
        interp["fragment_mask"] = gather_optional_rows(prior.get("fragment_mask"), batch_indices, model.device)
        interp["interactions"] = gather_optional_rows(prior.get("interactions"), batch_indices, model.device)
        target["interactions"] = interp["interactions"]
        pseudo = {"target": target, "interp": interp, "pocket": pocket}
        mode = "simple_fallback"
    times = [times_cont, times_disc, torch.zeros_like(times_cont), torch.zeros_like(times_cont)]
    pseudo["times"] = times
    pseudo["labels"] = expanded_labels
    pseudo["logs"] = {
        "train-rl-interpolant-mode": 1.0 if mode == "original_interpolant" else 2.0,
        "train-rl-num-stratified-timesteps": float(getattr(model.hparams, "rl_num_stratified_timesteps", 1)),
        "train-rl-pseudo-batch-size": float(len(expanded_labels)),
    }
    assert len(expanded_labels) == int(pseudo["target"]["coords"].shape[0])
    return pseudo


def build_pseudo_with_original_interpolant(
    model: Any,
    data: Mapping[str, Any],
    expanded_candidates: Sequence[Mapping[str, Any]],
    expanded_labels: Sequence[str],
    times_cont: torch.Tensor,
    times_disc: torch.Tensor,
) -> dict[str, Any]:
    if not bool(getattr(model.hparams, "rl_use_original_interpolant", True)):
        raise RuntimeError("rl_use_original_interpolant is false")
    interpolant = getattr(model, "rl_train_interpolant", None)
    if interpolant is None:
        raise RuntimeError("model.rl_train_interpolant is not attached; cannot use original FLOWR interpolant")
    systems = data.get("complex") or []
    target_systems: list[PocketComplex] = []
    interp_systems: list[PocketComplex] = []
    for row, candidate in enumerate(expanded_candidates):
        batch_idx = int(candidate["batch_index"])
        base_system = systems[batch_idx]
        to_ligand = target_to_geometric_mol(candidate["target"], model)
        from_ligand = interpolant.prior_sampler.sample_molecule(to_ligand.seq_length)
        from_ligand = interpolant._match_mols(from_ligand, to_ligand, mol_size=to_ligand.seq_length)
        interp_ligand = interpolant._interpolate_mol(from_ligand, to_ligand, float(times_cont[row].detach().cpu()), float(times_disc[row].detach().cpu()))
        holo = base_system.holo
        apo = holo if getattr(interpolant, "rigid_pocket", False) or base_system.apo is None else base_system.apo
        interp_pocket = interpolant._interpolate_pocket(apo, holo, times_cont[row].detach().cpu())
        target_systems.append(base_system._copy_with(ligand=to_ligand, holo=holo, apo=apo, interactions=inactive_or_existing_interactions(base_system, to_ligand)))
        interp_systems.append(PocketComplex(apo=interp_pocket, ligand=interp_ligand, interactions=inactive_or_existing_interactions(base_system, to_ligand), metadata=base_system.metadata, fragment_mask=None, com=base_system.com))
    target_complex = complex_batch_to_training_dict(PocketComplexBatch.from_list(target_systems), state="holo", systems=target_systems, device=model.device)
    interp_complex = complex_batch_to_training_dict(PocketComplexBatch.from_list(interp_systems), state="apo", systems=interp_systems, device=model.device)
    target = model.builder.extract_ligand_from_complex(target_complex)
    interp = model.builder.extract_ligand_from_complex(interp_complex)
    pocket = model.builder.extract_pocket_from_complex(target_complex)
    interp["fragment_mask"] = target_complex["fragment_mask"]
    interp["interactions"] = interp_complex["interactions"]
    target["interactions"] = target_complex["interactions"]
    target["pocket_mask"] = pocket["mask"]
    return {"target": target, "interp": interp, "pocket": pocket}


def inactive_or_existing_interactions(base_system: Any, ligand: GeometricMol) -> Any:
    # RL fine-tuning does not enable interaction/scaffold/linker inpainting.  If
    # the original system has interaction tensors, keep the tensor shape but crop
    #/pad only through the original PocketComplexBatch collation path.
    return base_system.interactions if getattr(base_system, "interactions", None) is not None else None


def target_to_geometric_mol(target: Mapping[str, torch.Tensor], model: Any) -> GeometricMol:
    mask = target["mask"].bool().detach().cpu()
    coords = target["coords"].detach().cpu()[mask]
    atomics = target["atomics"].detach().cpu()[mask]
    charges = target["charges"].detach().cpu()[mask] if "charges" in target else None
    bonds = target["bonds"].detach().cpu()[mask][:, mask]
    n_atoms = int(mask.sum().item())
    bond_indices = torch.ones((n_atoms, n_atoms), dtype=torch.long).nonzero()
    bond_types = bonds[bond_indices[:, 0], bond_indices[:, 1]]
    return GeometricMol(coords, atomics, bond_indices=bond_indices, bond_types=bond_types, charges=charges, is_mmap=False)


def complex_batch_to_training_dict(batch: PocketComplexBatch, state: str, systems: Sequence[Any], device: torch.device) -> dict[str, Any]:
    out = batch.to_dict(state=state)
    out["bonds"] = out.pop("bonds")
    interactions = batch.interactions(state=state)
    out["interactions"] = interactions if torch.is_tensor(interactions) else interactions
    out["fragment_mask"] = batch.fragment_mask()
    out["complex"] = list(systems)
    if out.get("charges") is not None and torch.is_tensor(out["charges"]) and out["charges"].dim() == 2:
        n_charges = len(smolRD.CHARGE_IDX_MAP.keys())
        out["charges"] = smolF.one_hot_encode_tensor(out["charges"].long(), n_charges)
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in out.items()}


def masks_from_labels(labels: Sequence[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top_mask = torch.tensor([label == "top" for label in labels], device=device, dtype=torch.bool)
    bottom_mask = torch.tensor([label == "bottom" for label in labels], device=device, dtype=torch.bool)
    middle_mask = torch.tensor([label == "middle" for label in labels], device=device, dtype=torch.bool)
    return top_mask, bottom_mask, middle_mask


def expanded_count_logs(expanded_labels: Sequence[str], candidate_labels: Sequence[str]) -> dict[str, float]:
    return {
        "train-rl-num-top-candidates": float(sum(label == "top" for label in candidate_labels)),
        "train-rl-num-bottom-candidates": float(sum(label == "bottom" for label in candidate_labels)),
        "train-rl-num-top-expanded": float(sum(label == "top" for label in expanded_labels)),
        "train-rl-num-bottom-expanded": float(sum(label == "bottom" for label in expanded_labels)),
        "train-rl-expanded-selected-count": float(len(expanded_labels)),
    }

def stack_targets(targets: Sequence[Mapping[str, torch.Tensor]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "coords": torch.stack([target["coords"] for target in targets]).to(device).detach(),
        "atomics": torch.stack([target["atomics"] for target in targets]).to(device).detach(),
        "bonds": torch.stack([target["bonds"] for target in targets]).to(device).detach(),
        "charges": torch.stack([target["charges"] for target in targets]).to(device).detach(),
        "mask": torch.stack([target["mask"] for target in targets]).to(device).detach(),
    }


def stratified_times_from_k_ids(k_ids: Sequence[int], k_steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not k_ids:
        return torch.empty(0, device=device), torch.empty(0, device=device)
    if k_steps <= 1:
        times = torch.rand(len(k_ids), device=device).clamp(1e-3, 0.999)
        return times, times.clone()
    strata = torch.tensor(k_ids, device=device, dtype=torch.float32).clamp(0, k_steps - 1)
    times = ((strata + torch.rand(len(k_ids), device=device)) / float(k_steps)).clamp(1e-3, 0.999)
    return times, times.clone()


def corrupt_ligand_like_flowr(target: Mapping[str, torch.Tensor], times_cont: torch.Tensor, times_disc: torch.Tensor) -> dict[str, torch.Tensor]:
    coords_target = target["coords"]
    mask = target["mask"].float()
    coord_noise = torch.randn_like(coords_target)
    coords = coord_noise * (1.0 - times_cont[:, None, None]) + coords_target * times_cont[:, None, None]
    atomics = mix_discrete_with_random(target["atomics"], times_disc, mask)
    charges = mix_discrete_with_random(target["charges"], times_disc, mask)
    bond_mask = mask[:, :, None] * mask[:, None, :]
    bonds = mix_discrete_with_random(target["bonds"], times_disc, bond_mask)
    return {"coords": coords, "atomics": atomics, "bonds": bonds, "charges": charges, "mask": target["mask"]}


def mix_discrete_with_random(target: torch.Tensor, times: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n_classes = target.size(-1)
    target_idx = torch.argmax(target, dim=-1)
    random_idx = torch.randint(0, n_classes, target_idx.shape, device=target.device)
    choose_target = torch.rand(target_idx.shape, device=target.device) < times.reshape((times.size(0),) + (1,) * (target_idx.dim() - 1))
    mixed_idx = torch.where(choose_target, target_idx, random_idx)
    mixed = F.one_hot(mixed_idx, n_classes).float()
    return mixed * mask.unsqueeze(-1).float()


def gather_pocket_for_candidates(model: Any, data: Mapping[str, Any], prior: Mapping[str, Any], batch_indices: Sequence[int]) -> dict[str, Any]:
    pocket_data = model.builder.extract_pocket_from_complex(data)
    gathered = {key: value[batch_indices].to(model.device) if torch.is_tensor(value) else value for key, value in pocket_data.items()}
    if "interactions" in prior and torch.is_tensor(prior["interactions"]):
        gathered["interactions"] = prior["interactions"][batch_indices].to(model.device)
    return gathered


def gather_optional_rows(value: Any, batch_indices: Sequence[int], device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value[batch_indices].to(device)
    return value


def forward_ligand_pocket(model: Any, pseudo: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    out = model(pseudo["interp"], pseudo["pocket"], pseudo["times"], training=True, cond_batch=None)
    predicted = {"coords": out[0], "atomics": out[1], "bonds": out[2], "charges": out[3], "mask": pseudo["target"]["mask"]}
    if getattr(model, "predict_interactions", False) or getattr(model, "flow_interactions", False):
        predicted["interactions"] = out[4]
    return predicted


def forward_ligand_pocket_with_generator(model: Any, gen: Any, pseudo: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    gen.eval()
    with temporarily_use_generator(model, gen):
        return forward_ligand_pocket(model, pseudo)


def top_imitation_loss(model: Any, target: Mapping[str, torch.Tensor], current: Mapping[str, torch.Tensor], top_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    if not bool(top_mask.any()):
        return torch.zeros((), device=current["coords"].device), {}
    atom_loss = masked_hard_ce(current["atomics"], target["atomics"], target["mask"], top_mask)
    bond_loss = masked_hard_ce(current["bonds"], target["bonds"], target["mask"][:, :, None] * target["mask"][:, None, :], top_mask)
    charge_loss = masked_hard_ce(current["charges"], target["charges"], target["mask"], top_mask)
    coord_loss = masked_coord_mse(current["coords"], target["coords"].detach(), target["mask"], top_mask)
    loss = (
        float(getattr(model.hparams, "rl_top_atom_weight", 1.0)) * atom_loss
        + float(getattr(model.hparams, "rl_top_bond_weight", 1.0)) * bond_loss
        + float(getattr(model.hparams, "rl_top_charge_weight", 1.0)) * charge_loss
        + float(getattr(model.hparams, "rl_top_coord_weight", 1.0)) * coord_loss
    )
    return loss, {}


def bottom_repulsion_loss(
    model: Any,
    current: Mapping[str, torch.Tensor],
    ref: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    bottom_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = current["coords"].device
    if not bool(bottom_mask.any()):
        return torch.zeros((), device=device), {}
    atom_loss, atom_delta = discrete_negative_target_loss(current["atomics"], ref["atomics"], target["mask"], bottom_mask, float(getattr(model.hparams, "rl_beta_atom", 1.0)))
    bond_loss, bond_delta = discrete_negative_target_loss(current["bonds"], ref["bonds"], target["mask"][:, :, None] * target["mask"][:, None, :], bottom_mask, float(getattr(model.hparams, "rl_beta_bond", 1.0)))
    charge_loss, charge_delta = discrete_negative_target_loss(current["charges"], ref["charges"], target["mask"], bottom_mask, float(getattr(model.hparams, "rl_beta_charge", 1.0)))
    coord_loss, coord_delta = coord_negative_target_loss(current["coords"], ref["coords"], target["mask"], bottom_mask, float(getattr(model.hparams, "rl_gamma_coord", 1.0)))
    loss = (
        float(getattr(model.hparams, "rl_bottom_atom_weight", 1.0)) * atom_loss
        + float(getattr(model.hparams, "rl_bottom_bond_weight", 1.0)) * bond_loss
        + float(getattr(model.hparams, "rl_bottom_charge_weight", 1.0)) * charge_loss
        + float(getattr(model.hparams, "rl_bottom_coord_weight", 1.0)) * coord_loss
    )
    logs = {
        "train-rl-delta-atom-abs-mean": atom_delta.detach(),
        "train-rl-delta-bond-abs-mean": bond_delta.detach(),
        "train-rl-delta-charge-abs-mean": charge_delta.detach(),
        "train-rl-delta-coord-abs-mean": coord_delta.detach(),
    }
    return loss, logs


def masked_hard_ce(logits: torch.Tensor, target_dist: torch.Tensor, mask: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    labels = torch.argmax(target_dist.detach(), dim=-1)
    flat_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none").reshape(labels.shape)
    full_mask = mask.float() * sample_mask.reshape((sample_mask.size(0),) + (1,) * (mask.dim() - 1)).float()
    denom = full_mask.sum().clamp(min=1.0)
    return (flat_loss * full_mask).sum() / denom


def masked_coord_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    per = F.mse_loss(pred, target, reduction="none").sum(dim=-1)
    full_mask = mask.float() * sample_mask[:, None].float()
    return (per * full_mask).sum() / full_mask.sum().clamp(min=1.0)


def discrete_negative_target_loss(cur_logits: torch.Tensor, ref_logits: torch.Tensor, mask: torch.Tensor, sample_mask: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    logp_cur = F.log_softmax(cur_logits, dim=-1)
    logp_ref = F.log_softmax(ref_logits.detach(), dim=-1)
    delta = (logp_cur - logp_ref).detach()
    logp_minus = logp_ref - beta * delta
    p_minus = F.softmax(logp_minus, dim=-1).detach()
    soft_ce = -(p_minus * logp_cur).sum(dim=-1)
    full_mask = mask.float() * sample_mask.reshape((sample_mask.size(0),) + (1,) * (mask.dim() - 1)).float()
    loss = (soft_ce * full_mask).sum() / full_mask.sum().clamp(min=1.0)
    delta_mean = (delta.abs().mean(dim=-1) * full_mask).sum() / full_mask.sum().clamp(min=1.0)
    return loss, delta_mean


def coord_negative_target_loss(cur: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, sample_mask: torch.Tensor, gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    delta = (cur - ref.detach()).detach()
    target_minus = (ref.detach() - gamma * delta).detach()
    loss = masked_coord_mse(cur, target_minus, mask, sample_mask)
    full_mask = mask.float() * sample_mask[:, None].float()
    delta_mean = (delta.abs().sum(dim=-1) * full_mask).sum() / full_mask.sum().clamp(min=1.0)
    return loss, delta_mean


def anchor_regularization_loss(current: Mapping[str, torch.Tensor], ref: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = target["mask"].bool()
    all_samples = torch.ones(mask.size(0), dtype=torch.bool, device=mask.device)
    coord = masked_coord_mse(current["coords"], ref["coords"].detach(), target["mask"], all_samples)
    atom = discrete_kl_to_ref(current["atomics"], ref["atomics"], target["mask"], all_samples)
    bond = discrete_kl_to_ref(current["bonds"], ref["bonds"], target["mask"][:, :, None] * target["mask"][:, None, :], all_samples)
    charge = discrete_kl_to_ref(current["charges"], ref["charges"], target["mask"], all_samples)
    return coord + atom + bond + charge


def discrete_kl_to_ref(cur_logits: torch.Tensor, ref_logits: torch.Tensor, mask: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    logp_cur = F.log_softmax(cur_logits, dim=-1)
    p_ref = F.softmax(ref_logits.detach(), dim=-1)
    soft_ce = -(p_ref * logp_cur).sum(dim=-1)
    full_mask = mask.float() * sample_mask.reshape((sample_mask.size(0),) + (1,) * (mask.dim() - 1)).float()
    return (soft_ce * full_mask).sum() / full_mask.sum().clamp(min=1.0)


def reward_summary_logs(rewards: Sequence[Mapping[str, Any]], selection: Mapping[str, Any], cache_hit_rate: float) -> dict[str, Any]:
    n = len(rewards)
    plifs = finite_values(rewards, "plif_tanimoto")
    strains = finite_values(rewards, "strain_energy")
    vinas = finite_values(rewards, "vina_score")
    scores = finite_values(rewards, "main_score")
    enabled_metrics = set(selection.get("summary", {}).get("enabled_metrics", []))
    return {
        "train-rl-num-candidates": float(n),
        "train-rl-num-valid": float(sum(bool(r.get("valid")) for r in rewards)),
        "train-rl-num-posebusters-valid": float(sum(bool(r.get("posebusters_valid")) for r in rewards)),
        "train-rl-num-metric-success": float(sum(bool(r.get("metric_success")) for r in rewards)),
        "train-rl-num-top": float(selection["summary"]["num_top"]),
        "train-rl-num-bottom": float(selection["summary"]["num_bottom"]),
        "train-rl-num-middle": float(selection["summary"].get("num_middle", 0)),
        "train-rl-selected-frac": float((selection["summary"]["num_top"] + selection["summary"]["num_bottom"]) / n) if n else 0.0,
        "train-rl-plif-mean": mean_or_zero(plifs),
        "train-rl-strain-mean": mean_or_zero(strains),
        "train-rl-vina-mean": mean_or_zero(vinas),
        "train-rl-main-score-mean": mean_or_zero(scores),
        "train-rl-main-score-max": max(scores) if scores else 0.0,
        "train-rl-invalid-count": float(sum(not bool(r.get("valid")) for r in rewards)),
        "train-rl-posebusters-failed-count": float(sum(not bool(r.get("posebusters_valid")) for r in rewards)),
        "train-rl-plif-failed-count": float(sum(not bool(r.get("plif_success")) for r in rewards)) if "plif" in enabled_metrics else 0.0,
        "train-rl-strain-failed-count": float(sum(not bool(r.get("strain_success")) for r in rewards)) if "strain" in enabled_metrics else 0.0,
        "train-rl-vina-failed-count": float(sum(not bool(r.get("vina_success")) for r in rewards)) if "vina" in enabled_metrics else 0.0,
        "train-rl-cache-hit-rate": float(cache_hit_rate),
    }


def reward_selection_logs(rewards: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]) -> dict[str, Any]:
    scores = [float(r.get("main_score", 0.0)) for r in rewards]
    top_scores = [scores[idx] for idx in selection.get("top_indices", [])]
    bottom_scores = [scores[idx] for idx in selection.get("bottom_indices", [])]
    sorted_scores = sorted(scores, reverse=True)
    top10 = sorted_scores[: max(1, min(10, len(sorted_scores)))] if sorted_scores else []
    return {
        "train-rl-reward-top-mean": mean_or_zero(top_scores),
        "train-rl-reward-bottom-mean": mean_or_zero(bottom_scores),
        "train-rl-reward-top10-mean": mean_or_zero(top10),
    }


def finite_values(rewards: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for reward in rewards:
        try:
            value = float(reward.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean_or_zero(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
