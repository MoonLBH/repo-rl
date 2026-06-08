"""Default-off RL fine-tuning helpers for FLOWR Lightning modules.

The functions in this module are intentionally opt-in.  They never run unless
``enable_rl_finetune`` is true and ``rl_loss_weight`` is positive in the FLOWR
training args/hparams.  Structure rewards are treated as detached, non-
differentiable selection signals; gradients only flow through a small surrogate
loss on FLOWR model outputs.
"""

from __future__ import annotations

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
}


def rl_enabled(hparams: Any) -> bool:
    return bool(getattr(hparams, "enable_rl_finetune", False)) and float(getattr(hparams, "rl_loss_weight", 0.0)) > 0.0


def maybe_apply_rl_finetune_loss(model: Any, loss: torch.Tensor, lig_data: Mapping[str, Any], predicted: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return ``loss + rl_weight * surrogate`` plus train-rl-* logs when enabled."""

    if not rl_enabled(model.hparams):
        return loss, {}

    logs = base_logs(model)
    try:
        rewards, cache_hit_rate = get_train_rewards(model, lig_data, predicted)
        if not rewards:
            logs.update(skip_logs("missing_rewards"))
            return loss, logs
        selection = select_rewards_for_training(model, rewards)
        surrogate, surrogate_logs = rl_surrogate_loss(lig_data, predicted, rewards, selection)
        logs.update(reward_summary_logs(rewards, selection, cache_hit_rate))
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


def base_logs(model: Any) -> dict[str, Any]:
    return {
        "train-rl-enabled": float(rl_enabled(model.hparams)),
        "train-rl-loss": 0.0,
        "train-rl-loss-weight": float(getattr(model.hparams, "rl_loss_weight", 0.0)),
        "train-rl-objective-mode": float(OBJECTIVE_MODE_IDS.get(getattr(model.hparams, "rl_objective_mode", "strain"), 0)),
        "train-rl-num-candidates": 0.0,
        "train-rl-num-valid": 0.0,
        "train-rl-num-posebusters-valid": 0.0,
        "train-rl-num-metric-success": 0.0,
        "train-rl-num-top": 0.0,
        "train-rl-num-bottom": 0.0,
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


def get_train_rewards(model: Any, lig_data: Mapping[str, Any], predicted: Mapping[str, Any]) -> tuple[list[dict[str, Any]], float]:
    metric_source = getattr(model.hparams, "rl_metric_source", "compute")
    if metric_source == "existing_eval_output":
        return load_existing_train_rewards(model, lig_data), 1.0

    records = build_training_reward_records(model, lig_data, predicted)
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


def load_existing_train_rewards(model: Any, lig_data: Mapping[str, Any]) -> list[dict[str, Any]]:
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
    batch_size = int(lig_data["mask"].size(0))
    start = int(getattr(model, "_rl_reward_cursor", 0))
    selected = [rewards[(start + idx) % len(rewards)] for idx in range(batch_size)]
    model._rl_reward_cursor = (start + batch_size) % len(rewards)
    return [dict(item) for item in selected]


def build_training_reward_records(model: Any, lig_data: Mapping[str, Any], predicted: Mapping[str, Any]) -> list[dict[str, Any]]:
    gen_mols = generated_mols_from_prediction(model, lig_data, predicted)
    systems = lig_data.get("complex") or []
    ref_ligs = [system.ligand.orig_mol.to_rdkit() for system in systems]
    protein_files = write_training_pockets(model, lig_data)
    records = []
    for idx, mol in enumerate(gen_mols):
        ref_lig = ref_ligs[idx] if idx < len(ref_ligs) else None
        protein_file = protein_files[idx] if idx < len(protein_files) else None
        system_id = systems[idx].metadata.get("system_id", f"batch_{idx}") if idx < len(systems) else f"batch_{idx}"
        records.append(
            {
                "sample_id": f"train_step_{model.global_step}_{system_id}_{idx}",
                "mol": mol,
                "protein_file": protein_file,
                "reference_ligand": ref_lig,
                "structure_protein_file": protein_file,
                "structure_reference_ligand": ref_lig,
                "plif_protein_file": protein_file,
                "plif_reference_ligand": ref_lig,
                "metadata": {"system_id": system_id},
            }
        )
    return records


def generated_mols_from_prediction(model: Any, lig_data: Mapping[str, Any], predicted: Mapping[str, Any]) -> list[Any]:
    coords = predicted["coords"].detach() * float(model.hparams.coord_scale)
    systems = lig_data.get("complex") or []
    if systems:
        coords = model.builder.undo_zero_com_batch(coords, lig_data["mask"].detach(), [system.com for system in systems])
    return model.builder.mols_from_tensors(
        coords,
        predicted["atomics"].detach(),
        lig_data["mask"].detach(),
        bond_dists=predicted["bonds"].detach(),
        charge_dists=predicted["charges"].detach(),
        sanitise=True,
    )


def write_training_pockets(model: Any, lig_data: Mapping[str, Any]) -> list[Optional[str]]:
    systems = lig_data.get("complex") or []
    if not systems:
        return [None] * int(lig_data["mask"].size(0))
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
        top_ratio=float(getattr(model.hparams, "rl_top_ratio", 0.1)),
        bottom_ratio=float(getattr(model.hparams, "rl_bottom_ratio", 0.1)),
        feasible_only_for_top=True,
        include_failed_in_bottom=True,
        plif_min_threshold=float(getattr(model.hparams, "rl_plif_min_threshold", 0.0)) if multi else 0.0,
        strain_max_threshold=float(getattr(model.hparams, "rl_strain_max_threshold", math.inf)) if multi else math.inf,
        vina_max_threshold=float(getattr(model.hparams, "rl_vina_max_threshold", math.inf)) if multi else math.inf,
        plif_weight=float(getattr(model.hparams, "rl_plif_weight", 1.0)),
        strain_weight=float(getattr(model.hparams, "rl_strain_weight", 1.0)),
        vina_weight=float(getattr(model.hparams, "rl_vina_weight", 1.0)),
    )


def rl_surrogate_loss(lig_data: Mapping[str, Any], predicted: Mapping[str, Any], rewards: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
    device = predicted["coords"].device
    mask = lig_data["mask"].to(device).float()
    per_atom = F.mse_loss(predicted["coords"], lig_data["coords"], reduction="none").sum(dim=-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    per_sample = (per_atom * mask).sum(dim=1) / denom
    top_indices = selection["top_indices"]
    bottom_indices = selection["bottom_indices"]
    logs: dict[str, Any] = {}
    if not top_indices:
        return None, logs
    top_tensor = torch.tensor(top_indices, device=device, dtype=torch.long)
    top_loss = per_sample.index_select(0, top_tensor).mean()
    if bottom_indices:
        bottom_tensor = torch.tensor(bottom_indices, device=device, dtype=torch.long)
        # Bounded repulsion: encourage low-reward examples not to imitate their supervised target too closely.
        bottom_loss = F.relu(1.0 - per_sample.index_select(0, bottom_tensor)).mean()
    else:
        bottom_loss = torch.zeros((), device=device)
    # Rewards are detached/non-differentiable and only influence selection above.
    _ = [float(reward.get("main_score", 0.0)) for reward in rewards]
    logs["train-rl-positive-loss"] = top_loss.detach()
    logs["train-rl-negative-loss"] = bottom_loss.detach()
    return top_loss + bottom_loss, logs


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
