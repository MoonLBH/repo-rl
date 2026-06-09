"""Default-off LFPO-F-v2 training helpers for FLOWR pocket models.

This module implements a FLOWR-adapted Top-Reward Imitation + Bottom-Reward
Repulsion surrogate.  The code is intentionally opt-in: it never initializes a
reference model, runs ODE sampling, or computes structure rewards unless
``enable_rl_finetune`` is true and ``rl_loss_weight`` is positive.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

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
}


def rl_enabled(hparams: Any) -> bool:
    return bool(getattr(hparams, "enable_rl_finetune", False)) and float(getattr(hparams, "rl_loss_weight", 0.0)) > 0.0


def maybe_apply_rl_finetune_loss(
    model: Any,
    loss: torch.Tensor,
    prior: Mapping[str, Any],
    data: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply LFPO-F-v2 surrogate to ``loss`` when the RL path is enabled.

    Disabled path returns the original FLOWR loss and an empty log dict.  Enabled
    path samples candidates with the explicit reference generator, scores them,
    selects top/bottom, rebuilds noisy training states, and computes a detached
    reward-guided top-imitation/bottom-repulsion surrogate.
    """

    if not rl_enabled(model.hparams):
        return loss, {}

    logs = base_logs(model)
    if not should_run_rl_update(model):
        logs.update(skip_logs("frequency"))
        return loss, logs

    try:
        ref_model = ensure_reference_model(model)
        candidates = sample_reference_candidates(model, ref_model, prior, data)
        if not candidates:
            logs.update(skip_logs("no_candidates"))
            return loss, logs
        rewards, cache_hit_rate = get_candidate_rewards(model, candidates)
        if not rewards:
            logs.update(skip_logs("missing_rewards"))
            return loss, logs
        selection = select_rewards_for_training(model, rewards)
        logs.update(reward_summary_logs(rewards, selection, cache_hit_rate))
        surrogate, surrogate_logs = lfpof_v2_surrogate_loss(model, ref_model, prior, data, candidates, rewards, selection)
        logs.update(surrogate_logs)
        if surrogate is None:
            logs.update(skip_logs("no_top"))
            return loss, logs
        if not torch.isfinite(surrogate):
            logs.update(skip_logs("nan_loss"))
            return loss, logs
        logs["train-rl-loss"] = surrogate.detach()
        return loss + float(model.hparams.rl_loss_weight) * surrogate, logs
    except Exception:
        logs.update(skip_logs("metric_error"))
        return loss, logs


def on_train_batch_end_update_reference(model: Any) -> None:
    """EMA-update the explicit RL reference generator after optimizer updates."""

    if not rl_enabled(model.hparams) or not hasattr(model, "_rl_ref_model"):
        return
    ref_model = model._rl_ref_model
    decay = float(getattr(model.hparams, "rl_ref_ema_decay", 0.999))
    with torch.no_grad():
        for ref_param, cur_param in zip(ref_model.parameters(), model.parameters()):
            ref_param.data.mul_(decay).add_(cur_param.detach().data, alpha=1.0 - decay)
        for ref_buffer, cur_buffer in zip(ref_model.buffers(), model.buffers()):
            if ref_buffer.dtype.is_floating_point:
                ref_buffer.data.mul_(decay).add_(cur_buffer.detach().data, alpha=1.0 - decay)
            else:
                ref_buffer.data.copy_(cur_buffer.detach().data)


def should_run_rl_update(model: Any) -> bool:
    freq = max(1, int(getattr(model.hparams, "rl_update_frequency", 1)))
    return int(getattr(model, "global_step", 0)) % freq == 0


def ensure_reference_model(model: Any) -> Any:
    """Create a frozen explicit reference generator on first enabled RL step."""

    if hasattr(model, "_rl_ref_model"):
        return model._rl_ref_model
    old_ref = getattr(model, "_rl_ref_model", None)
    if hasattr(model, "_rl_ref_model"):
        delattr(model, "_rl_ref_model")
    ref_model = copy.deepcopy(model)
    if old_ref is not None:
        model._rl_ref_model = old_ref
    reference_checkpoint = getattr(model.hparams, "rl_reference_checkpoint", None)
    if reference_checkpoint:
        ckpt = torch.load(reference_checkpoint, map_location=model.device)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, Mapping) else ckpt
        ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.to(model.device)
    for param in ref_model.parameters():
        param.requires_grad_(False)
    ref_model._rl_is_reference_model = True
    model._rl_ref_model = ref_model
    return ref_model


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
        "train-rl-objective-mode": float(OBJECTIVE_MODE_IDS.get(getattr(model.hparams, "rl_objective_mode", "strain"), 0)),
        "train-rl-num-candidates": 0.0,
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


def sample_reference_candidates(model: Any, ref_model: Any, prior: Mapping[str, Any], data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Sample complete ligand candidates with FLOWR ODE generation from pi_ref."""

    num_rounds = max(1, int(getattr(model.hparams, "rl_num_candidates_per_step", 1)))
    sampling_steps = max(1, int(getattr(model.hparams, "rl_sampling_steps", 100)))
    sampler_model = ref_model if bool(getattr(model.hparams, "rl_sample_from_reference", True)) else model
    candidates: list[dict[str, Any]] = []

    pocket_data = model.builder.extract_pocket_from_complex(data)
    pocket_data["interactions"] = prior.get("interactions", data.get("interactions"))
    pocket_data["complex"] = data.get("complex")
    lig_prior = model.builder.extract_ligand_from_complex(prior)
    lig_prior["fragment_mask"] = prior.get("fragment_mask")
    lig_prior["interactions"] = prior.get("interactions")
    times = zero_generation_times(model, prior, pocket_data)
    systems = data.get("complex") or []
    protein_files = write_training_pockets(model, systems, int(lig_prior["mask"].size(0)))
    ref_ligs = [system.ligand.orig_mol.to_rdkit() for system in systems]

    was_training = sampler_model.training
    sampler_model.eval()
    with torch.no_grad():
        for round_idx in range(num_rounds):
            generated = sampler_model._generate(
                detach_mapping(lig_prior),
                detach_mapping(pocket_data),
                steps=sampling_steps,
                times=[t.clone() for t in times],
                strategy=model.sampling_strategy,
                corr_iters=model.corrector_iters,
            )
            mols = sampler_model._generate_mols(generated)
            train_targets = generated_to_training_targets(model, generated, systems)
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
        sampler_model.train()
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
    metric_source = getattr(model.hparams, "rl_metric_source", "compute")
    if metric_source == "existing_eval_output":
        return load_existing_train_rewards(model, len(candidates)), 1.0

    records = [candidate_to_reward_record(candidate) for candidate in candidates]
    cache = load_metric_cache(getattr(model.hparams, "rl_metric_cache_path", None)) if metric_source == "cached" else {}
    rewards: list[dict[str, Any]] = []
    hits = 0
    misses: dict[str, dict[str, Any]] = {}
    for record in records:
        key = train_cache_key(record, model.hparams)
        if key in cache:
            rewards.append(dict(cache[key]))
            hits += 1
        else:
            misses[key] = record
    if misses:
        with torch.no_grad():
            computed = compute_structure_rewards_from_records(misses.values(), config=reward_config_from_hparams(model.hparams))
        for key, reward in zip(misses, computed):
            reward = dict(reward)
            reward["cache_key"] = key
            cache[key] = reward
            rewards.append(reward)
        if metric_source == "cached":
            rewrite_metric_cache(getattr(model.hparams, "rl_metric_cache_path", None), cache)
    hit_rate = hits / len(records) if records else 0.0
    return rewards, hit_rate


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
    ref_model: Any,
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
        ref_pred = forward_ligand_pocket(ref_model, pseudo)

    top_mask = torch.tensor([label == "top" for label in selected_labels], device=model.device)
    bottom_mask = torch.tensor([label == "bottom" for label in selected_labels], device=model.device)
    top_loss, top_logs = top_imitation_loss(model, pseudo["target"], current_pred, top_mask)
    bottom_loss, bottom_logs = bottom_repulsion_loss(model, current_pred, ref_pred, pseudo["target"], bottom_mask)
    aux_fm_loss = torch.zeros((), device=model.device)
    if float(getattr(model.hparams, "rl_aux_fm_weight", 0.0)) > 0.0:
        aux_losses = model._loss(pseudo["target"], pseudo["interp"], current_pred)
        aux_fm_loss = sum(aux_losses.values())
    anchor_loss = anchor_regularization_loss(current_pred, ref_pred, pseudo["target"]) if float(getattr(model.hparams, "rl_anchor_weight", 0.0)) > 0.0 else torch.zeros((), device=model.device)
    main_loss = top_loss + float(getattr(model.hparams, "rl_bottom_repulsion_weight", 1.0)) * bottom_loss
    total = main_loss + float(getattr(model.hparams, "rl_aux_fm_weight", 0.0)) * aux_fm_loss + float(getattr(model.hparams, "rl_anchor_weight", 0.0)) * anchor_loss

    logs: dict[str, Any] = {
        "train-rl-main-loss": main_loss.detach(),
        "train-rl-top-loss": top_loss.detach(),
        "train-rl-bottom-loss": bottom_loss.detach(),
        "train-rl-aux-fm-loss": aux_fm_loss.detach(),
        "train-rl-anchor-loss": anchor_loss.detach(),
        **top_logs,
        **bottom_logs,
    }
    logs.update(reward_selection_logs(rewards, selection))
    return total, logs


def build_pseudo_training_batch(
    model: Any,
    prior: Mapping[str, Any],
    data: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
    selected_labels: Sequence[str],
) -> dict[str, Any]:
    k_steps = max(1, int(getattr(model.hparams, "rl_num_stratified_timesteps", 1)))
    expanded_candidates: list[Mapping[str, Any]] = []
    for idx in selected_indices:
        for _ in range(k_steps):
            expanded_candidates.append(candidates[idx])
    target = stack_targets([candidate["target"] for candidate in expanded_candidates], model.device)
    times_cont, times_disc = stratified_times(len(expanded_candidates), k_steps, model.device)
    interp = corrupt_ligand_like_flowr(target, times_cont, times_disc)
    batch_indices = [int(candidate["batch_index"]) for candidate in expanded_candidates]
    pocket = gather_pocket_for_candidates(model, data, prior, batch_indices)
    times = [times_cont, times_disc, torch.zeros_like(times_cont), torch.zeros_like(times_cont)]
    target["pocket_mask"] = pocket["mask"]
    interp["fragment_mask"] = gather_optional_rows(prior.get("fragment_mask"), batch_indices, model.device)
    interp["interactions"] = gather_optional_rows(prior.get("interactions"), batch_indices, model.device)
    target["interactions"] = interp["interactions"]
    return {"target": target, "interp": interp, "pocket": pocket, "times": times, "labels": selected_labels}


def stack_targets(targets: Sequence[Mapping[str, torch.Tensor]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "coords": torch.stack([target["coords"] for target in targets]).to(device).detach(),
        "atomics": torch.stack([target["atomics"] for target in targets]).to(device).detach(),
        "bonds": torch.stack([target["bonds"] for target in targets]).to(device).detach(),
        "charges": torch.stack([target["charges"] for target in targets]).to(device).detach(),
        "mask": torch.stack([target["mask"] for target in targets]).to(device).detach(),
    }


def stratified_times(n_samples: int, k_steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if k_steps <= 1:
        times = torch.rand(n_samples, device=device).clamp(1e-3, 0.999)
        return times, times.clone()
    repeats = math.ceil(n_samples / k_steps)
    strata = torch.arange(k_steps, device=device).float().repeat(repeats)[:n_samples]
    times = ((strata + torch.rand(n_samples, device=device)) / float(k_steps)).clamp(1e-3, 0.999)
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
