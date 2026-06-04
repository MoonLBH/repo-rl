"""Structure-based reward wrappers for FLOWR generated ligands.

This module is intentionally default-off: it is not imported by FLOWR's training,
sampling, or evaluation entrypoints unless a user calls it explicitly.  It wraps
FLOWR's existing structure metrics into per-sample reward dictionaries suitable
for future reward-guided fine-tuning experiments.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


class ObjectiveMode(str, Enum):
    """Supported structure-based reward objectives."""

    PLIF = "plif"
    STRAIN = "strain"
    VINA = "vina"
    PLIF_STRAIN = "plif_strain"
    PLIF_VINA = "plif_vina"
    STRAIN_VINA = "strain_vina"
    PLIF_STRAIN_VINA = "plif_strain_vina"


class MultiObjectiveStrategy(str, Enum):
    """Non-compensatory multi-objective scoring strategies."""

    HARD_GATE_THEN_WEIGHTED_SUM = "hard_gate_then_weighted_sum"
    WEIGHTED_MIN = "weighted_min"
    CONSTRAINED_WEIGHTED_SUM = "constrained_weighted_sum"


OBJECTIVE_METRICS: dict[str, tuple[str, ...]] = {
    ObjectiveMode.PLIF.value: ("plif",),
    ObjectiveMode.STRAIN.value: ("strain",),
    ObjectiveMode.VINA.value: ("vina",),
    ObjectiveMode.PLIF_STRAIN.value: ("plif", "strain"),
    ObjectiveMode.PLIF_VINA.value: ("plif", "vina"),
    ObjectiveMode.STRAIN_VINA.value: ("strain", "vina"),
    ObjectiveMode.PLIF_STRAIN_VINA.value: ("plif", "strain", "vina"),
}


@dataclass
class RewardConfig:
    """Configuration for structure reward normalization and composition.

    Thresholds are intentionally not hard-coded scientific defaults.  Callers
    should set them from experiment config after inspecting baseline FLOWR
    metric distributions on the exact dataset/split used for comparison.
    """

    objective_mode: str = ObjectiveMode.PLIF_STRAIN_VINA.value
    multiobjective_strategy: str = MultiObjectiveStrategy.CONSTRAINED_WEIGHTED_SUM.value
    plif_weight: float = 1.0
    strain_weight: float = 1.0
    vina_weight: float = 1.0
    plif_min_threshold: float = 0.0
    strain_max_threshold: float = math.inf
    vina_max_threshold: float = math.inf
    strain_good_threshold: Optional[float] = None
    strain_bad_threshold: Optional[float] = None
    vina_good_threshold: Optional[float] = None
    vina_bad_threshold: Optional[float] = None
    posebusters_validity_threshold: float = 1.0
    require_posebusters_validity: bool = True
    compute_posebusters_validity: bool = True
    compute_non_enabled_metrics: bool = False
    config_path: str = "./genbench3d/config/default.yaml"
    use_minimized_vina_score: bool = False
    strain_force_field_name: str = "MMFF94s"
    strain_n_steps: int = 1000

    def enabled_metrics(self) -> tuple[str, ...]:
        try:
            return OBJECTIVE_METRICS[self.objective_mode]
        except KeyError as exc:
            valid = ", ".join(sorted(OBJECTIVE_METRICS))
            raise ValueError(f"Unknown objective_mode={self.objective_mode!r}; choose from {valid}") from exc


@dataclass
class StructureRewardResult:
    """Per-sample structure reward dictionary.

    Raw metric values are preserved while ``main_score`` is always a normalized
    higher-is-better scalar for top/bottom selection.
    """

    sample_id: str
    valid: bool = False
    posebusters_valid: bool = False
    plif_tanimoto: Optional[float] = None
    strain_energy: Optional[float] = None
    vina_score: Optional[float] = None
    plif_success: bool = False
    strain_success: bool = False
    vina_success: bool = False
    metric_success: bool = False
    feasible: bool = False
    eligible_top: bool = False
    eligible_bottom: bool = True
    main_score: float = 0.0
    error: Optional[str] = None
    warning: Optional[str] = None
    normalized_plif: Optional[float] = None
    normalized_strain_score: Optional[float] = None
    normalized_vina_score: Optional[float] = None
    objective_mode: str = ObjectiveMode.PLIF_STRAIN_VINA.value
    multiobjective_strategy: str = MultiObjectiveStrategy.CONSTRAINED_WEIGHTED_SUM.value
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def normalize_plif(plif_tanimoto: Optional[float]) -> Optional[float]:
    value = _as_float(plif_tanimoto)
    return None if value is None else clamp(value, 0.0, 1.0)


def normalize_lower_is_better(
    value: Optional[float], good_threshold: Optional[float], bad_threshold: Optional[float]
) -> Optional[float]:
    value = _as_float(value)
    if value is None:
        return None
    if good_threshold is None or bad_threshold is None:
        return None
    if good_threshold >= bad_threshold:
        raise ValueError("good_threshold must be lower than bad_threshold for lower-is-better metrics")
    if value <= good_threshold:
        return 1.0
    if value >= bad_threshold:
        return 0.0
    return clamp((bad_threshold - value) / (bad_threshold - good_threshold), 0.0, 1.0)


def _weighted_average(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    total_weight = sum(weights[name] for name in scores if weights[name] > 0)
    if total_weight <= 0:
        return 0.0
    return sum(scores[name] * weights[name] for name in scores if weights[name] > 0) / total_weight


def _passes_thresholds(normalized_scores: Mapping[str, float], config: RewardConfig) -> bool:
    if "plif" in normalized_scores and normalized_scores["plif"] < config.plif_min_threshold:
        return False
    if "strain" in normalized_scores:
        if config.strain_max_threshold != math.inf:
            raw_threshold_score = normalize_lower_is_better(
                config.strain_max_threshold, config.strain_good_threshold, config.strain_bad_threshold
            )
            if raw_threshold_score is not None and normalized_scores["strain"] < raw_threshold_score:
                return False
    if "vina" in normalized_scores:
        if config.vina_max_threshold != math.inf:
            raw_threshold_score = normalize_lower_is_better(
                config.vina_max_threshold, config.vina_good_threshold, config.vina_bad_threshold
            )
            if raw_threshold_score is not None and normalized_scores["vina"] < raw_threshold_score:
                return False
    return True


def compute_main_score(
    *,
    plif_tanimoto: Optional[float] = None,
    strain_energy: Optional[float] = None,
    vina_score: Optional[float] = None,
    valid: bool = True,
    posebusters_valid: bool = True,
    config: Optional[RewardConfig] = None,
) -> dict[str, Any]:
    """Compute normalized metrics and non-compensatory higher-is-better score."""

    config = config or RewardConfig()
    enabled = config.enabled_metrics()
    normalized: dict[str, float] = {}
    missing: list[str] = []

    normalized_plif = normalize_plif(plif_tanimoto)
    normalized_strain = normalize_lower_is_better(
        strain_energy, config.strain_good_threshold, config.strain_bad_threshold
    )
    normalized_vina = normalize_lower_is_better(
        vina_score, config.vina_good_threshold, config.vina_bad_threshold
    )

    if "plif" in enabled:
        if normalized_plif is None:
            missing.append("plif")
        else:
            normalized["plif"] = normalized_plif
    if "strain" in enabled:
        if normalized_strain is None:
            missing.append("strain")
        else:
            normalized["strain"] = normalized_strain
    if "vina" in enabled:
        if normalized_vina is None:
            missing.append("vina")
        else:
            normalized["vina"] = normalized_vina

    if not valid:
        return _score_result(0.0, False, False, True, "invalid molecule", normalized_plif, normalized_strain, normalized_vina)
    if config.require_posebusters_validity and not posebusters_valid:
        return _score_result(0.0, False, False, True, "PoseBusters validity failed", normalized_plif, normalized_strain, normalized_vina)
    if missing:
        return _score_result(
            0.0,
            False,
            False,
            True,
            f"enabled metric(s) failed or missing: {', '.join(missing)}",
            normalized_plif,
            normalized_strain,
            normalized_vina,
        )

    weights = {"plif": config.plif_weight, "strain": config.strain_weight, "vina": config.vina_weight}
    passes = _passes_thresholds(normalized, config)
    strategy = config.multiobjective_strategy

    if strategy == MultiObjectiveStrategy.HARD_GATE_THEN_WEIGHTED_SUM.value:
        if not passes:
            return _score_result(0.0, True, False, True, "hard gate threshold failed", normalized_plif, normalized_strain, normalized_vina)
        score = _weighted_average(normalized, weights)
    elif strategy == MultiObjectiveStrategy.WEIGHTED_MIN.value:
        score = min(normalized[name] * max(weights[name], 0.0) for name in normalized)
        passes = passes and score > 0.0
    elif strategy == MultiObjectiveStrategy.CONSTRAINED_WEIGHTED_SUM.value:
        score = _weighted_average(normalized, weights) if passes else 0.0
    else:
        valid_strategies = ", ".join(strategy.value for strategy in MultiObjectiveStrategy)
        raise ValueError(f"Unknown multiobjective_strategy={strategy!r}; choose from {valid_strategies}")

    return _score_result(
        score,
        True,
        bool(passes and score > 0.0),
        not bool(passes and score > 0.0),
        None if passes else "objective threshold failed",
        normalized_plif,
        normalized_strain,
        normalized_vina,
    )


def _score_result(
    main_score: float,
    feasible: bool,
    eligible_top: bool,
    eligible_bottom: bool,
    error: Optional[str],
    normalized_plif: Optional[float],
    normalized_strain_score: Optional[float],
    normalized_vina_score: Optional[float],
) -> dict[str, Any]:
    return {
        "main_score": float(clamp(main_score, 0.0, 1.0)),
        "feasible": feasible,
        "eligible_top": eligible_top,
        "eligible_bottom": eligible_bottom,
        "error": error,
        "normalized_plif": normalized_plif,
        "normalized_strain_score": normalized_strain_score,
        "normalized_vina_score": normalized_vina_score,
    }


def compute_structure_rewards(
    generated_mols: Optional[Sequence[Any]] = None,
    *,
    generated_ligand_file: Optional[str | Path] = None,
    protein_file: Optional[str | Path] = None,
    reference_ligand: Optional[Any] = None,
    reference_ligand_file: Optional[str | Path] = None,
    flowr_sampling_output: Optional[str | Path] = None,
    sample_ids: Optional[Sequence[str]] = None,
    config: Optional[RewardConfig] = None,
) -> list[dict[str, Any]]:
    """Compute per-ligand structure rewards.

    Args:
        generated_mols: RDKit molecules from FLOWR generation.
        generated_ligand_file: SDF file containing generated ligands.
        protein_file: pocket/protein PDB used by PLIF/PoseBusters/Vina.
        reference_ligand: RDKit reference ligand.
        reference_ligand_file: SDF file for native/reference ligand.
        flowr_sampling_output: FLOWR ``predictions_multi_*.pt`` or ``samples_*.pt`` file.
        sample_ids: Optional IDs matching ``generated_mols``.
        config: Reward and normalization config.

    Returns:
        List of reward dictionaries in the schema requested by Stage 1.
    """

    config = config or RewardConfig()
    records = _coerce_records(
        generated_mols=generated_mols,
        generated_ligand_file=generated_ligand_file,
        protein_file=protein_file,
        reference_ligand=reference_ligand,
        reference_ligand_file=reference_ligand_file,
        flowr_sampling_output=flowr_sampling_output,
        sample_ids=sample_ids,
    )
    return [_compute_single_record_reward(record, config).to_dict() for record in records]


def _coerce_records(
    *,
    generated_mols: Optional[Sequence[Any]],
    generated_ligand_file: Optional[str | Path],
    protein_file: Optional[str | Path],
    reference_ligand: Optional[Any],
    reference_ligand_file: Optional[str | Path],
    flowr_sampling_output: Optional[str | Path],
    sample_ids: Optional[Sequence[str]],
) -> list[dict[str, Any]]:
    if flowr_sampling_output is not None:
        return _records_from_flowr_sampling_output(flowr_sampling_output)

    if generated_mols is None and generated_ligand_file is not None:
        generated_mols = _read_sdf_mols(generated_ligand_file)
    if reference_ligand is None and reference_ligand_file is not None:
        ref_mols = _read_sdf_mols(reference_ligand_file)
        reference_ligand = ref_mols[0] if ref_mols else None
    if generated_mols is None:
        raise ValueError("Provide generated_mols, generated_ligand_file, or flowr_sampling_output")

    ids = sample_ids or [f"sample_{idx}" for idx in range(len(generated_mols))]
    protein_file_str = str(protein_file) if protein_file is not None else None
    return [
        {
            "sample_id": ids[idx],
            "mol": mol,
            "protein_file": protein_file_str,
            "reference_ligand": reference_ligand,
            "structure_protein_file": protein_file_str,
            "structure_reference_ligand": reference_ligand,
            "plif_protein_file": protein_file_str,
            "plif_reference_ligand": reference_ligand,
        }
        for idx, mol in enumerate(generated_mols)
    ]


def _read_sdf_mols(sdf_path: str | Path) -> list[Any]:
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    return [mol for mol in supplier if mol is not None]


def _records_from_flowr_sampling_output(path: str | Path) -> list[dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu")
    records: list[dict[str, Any]] = []

    if "gen_ligs" in payload and "ref_ligs" in payload and "ref_pdbs" in payload:
        has_plif_refs = "ref_ligs_with_hs" in payload and "ref_pdbs_with_hs" in payload
        for target_idx, gen_ligs in enumerate(payload["gen_ligs"]):
            ref_lig = payload["ref_ligs"][target_idx]
            protein_file = payload["ref_pdbs"][target_idx]
            structure_ref_lig = payload["ref_ligs_with_hs"][target_idx] if has_plif_refs else ref_lig
            structure_protein_file = payload["ref_pdbs_with_hs"][target_idx] if has_plif_refs else protein_file
            plif_ref_lig = structure_ref_lig
            plif_protein_file = structure_protein_file
            for lig_idx, mol in enumerate(gen_ligs):
                records.append(
                    {
                        "sample_id": f"target_{target_idx}_lig_{lig_idx}",
                        "mol": mol,
                        "protein_file": protein_file,
                        "reference_ligand": ref_lig,
                        "structure_protein_file": structure_protein_file,
                        "structure_reference_ligand": structure_ref_lig,
                        "plif_protein_file": plif_protein_file,
                        "plif_reference_ligand": plif_ref_lig,
                        "metadata": {"has_structure_with_hs_refs": has_plif_refs, "has_plif_with_hs_refs": has_plif_refs},
                    }
                )
        return records

    if "all_gen_ligs" in payload:
        for idx, mol in enumerate(payload["all_gen_ligs"]):
            records.append({"sample_id": f"sample_{idx}", "mol": mol, "protein_file": None, "reference_ligand": None})
        return records

    raise ValueError(f"Unsupported FLOWR sampling output format: {path}")


def _compute_single_record_reward(record: Mapping[str, Any], config: RewardConfig) -> StructureRewardResult:
    sample_id = str(record.get("sample_id", "sample"))
    mol = record.get("mol")
    protein_file = record.get("protein_file")
    reference_ligand = record.get("reference_ligand")
    structure_protein_file = record.get("structure_protein_file", protein_file)
    structure_reference_ligand = record.get("structure_reference_ligand", reference_ligand)
    plif_protein_file = record.get("plif_protein_file", structure_protein_file)
    plif_reference_ligand = record.get("plif_reference_ligand", structure_reference_ligand)
    enabled = config.enabled_metrics()
    result = StructureRewardResult(
        sample_id=sample_id,
        objective_mode=config.objective_mode,
        multiobjective_strategy=config.multiobjective_strategy,
    )
    errors: list[str] = []
    warnings: list[str] = []

    result.valid = _safe_metric_bool(lambda: _compute_validity(mol), errors, "validity")
    if not result.valid:
        result.error = "invalid molecule"
        return result

    if config.compute_posebusters_validity or config.require_posebusters_validity:
        pb_value = _safe_metric_value(
            lambda: _compute_posebusters_validity(mol, structure_protein_file, structure_reference_ligand, config), errors, "posebusters_validity"
        )
        result.posebusters_valid = bool(pb_value) if pb_value is not None else False
    else:
        result.posebusters_valid = True

    if config.require_posebusters_validity and not result.posebusters_valid:
        result.error = "; ".join(errors) if errors else "PoseBusters validity failed"
        return result

    if "plif" in enabled or config.compute_non_enabled_metrics:
        value = _safe_metric_value(lambda: _compute_plif_tanimoto(mol, plif_protein_file, plif_reference_ligand), errors if "plif" in enabled else warnings, "plif")
        result.plif_tanimoto = value
        result.plif_success = value is not None
    if "strain" in enabled or config.compute_non_enabled_metrics:
        value = _safe_metric_value(lambda: _compute_strain_energy(mol, config), errors if "strain" in enabled else warnings, "strain")
        result.strain_energy = value
        result.strain_success = value is not None
    if "vina" in enabled or config.compute_non_enabled_metrics:
        value = _safe_metric_value(lambda: _compute_vina_score(mol, structure_protein_file, structure_reference_ligand, config), errors if "vina" in enabled else warnings, "vina")
        result.vina_score = value
        result.vina_success = value is not None

    score_info = compute_main_score(
        plif_tanimoto=result.plif_tanimoto,
        strain_energy=result.strain_energy,
        vina_score=result.vina_score,
        valid=result.valid,
        posebusters_valid=result.posebusters_valid,
        config=config,
    )
    result.main_score = score_info["main_score"]
    result.feasible = score_info["feasible"]
    result.eligible_top = score_info["eligible_top"]
    result.eligible_bottom = score_info["eligible_bottom"]
    result.normalized_plif = score_info["normalized_plif"]
    result.normalized_strain_score = score_info["normalized_strain_score"]
    result.normalized_vina_score = score_info["normalized_vina_score"]
    result.metric_success = result.feasible
    scoring_error = score_info["error"]
    detailed_errors = "; ".join(errors) if errors else None
    if scoring_error and detailed_errors:
        result.error = f"{scoring_error}; {detailed_errors}"
    else:
        result.error = scoring_error or detailed_errors
    result.warning = "; ".join(warnings) if warnings else None
    result.metadata.update(record.get("metadata", {}))
    return result


def _safe_metric_bool(fn: Any, errors: list[str], name: str) -> bool:
    value = _safe_metric_value(fn, errors, name)
    return bool(value) if value is not None else False


def _safe_metric_value(fn: Any, errors: list[str], name: str) -> Optional[float]:
    try:
        return fn()
    except Exception as exc:
        errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return None


def _compute_validity(mol: Any) -> bool:
    import flowr.util.rdkit as smol_rdkit

    return bool(smol_rdkit.mol_is_valid(mol, connected=True))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _compute_posebusters_validity(mol: Any, protein_file: Optional[str], reference_ligand: Any, config: RewardConfig) -> bool:
    if protein_file is None:
        raise ValueError("protein_file is required for PoseBusters validity")
    from flowr.util.metrics import evaluate_pb_validity

    eval_config = _load_yaml(config.config_path)
    result = evaluate_pb_validity([mol], ref_lig=reference_ligand, pdb_file=str(protein_file), config=eval_config, return_list=True)
    if not result:
        return False
    return bool(float(result[0]) >= config.posebusters_validity_threshold)


def _compute_plif_tanimoto(mol: Any, protein_file: Optional[str], reference_ligand: Any) -> Optional[float]:
    if protein_file is None or reference_ligand is None:
        raise ValueError("protein_file and reference_ligand are required for PLIF Tanimoto")
    from flowr.util.metrics import interaction_recovery_per_complex

    _, tanimoto = interaction_recovery_per_complex(
        [mol],
        reference_ligand,
        str(protein_file),
        add_optimize_gen_lig_hs=True,
        add_optimize_ref_lig_hs=True,
        optimize_pocket_hs=False,
        process_pocket=False,
        optimization_method="prolif_mmff",
        pocket_cutoff=6.0,
        strip_invalid=True,
        return_list=False,
    )
    return _as_float(tanimoto)


def _compute_strain_energy(mol: Any, config: RewardConfig) -> Optional[float]:
    from flowr.util.metrics import evaluate_strain

    values = evaluate_strain(
        [mol],
        n_steps=config.strain_n_steps,
        force_field_name=config.strain_force_field_name,
        return_list=True,
    )
    if not values:
        return None
    return _as_float(values[0])


def _compute_vina_score(mol: Any, protein_file: Optional[str], reference_ligand: Any, config: RewardConfig) -> Optional[float]:
    if protein_file is None or reference_ligand is None:
        raise ValueError("protein_file and reference_ligand are required for Vina score")
    from flowr.util.metrics import evaluate_gbsb3

    eval_config = _load_yaml(config.config_path)
    results = evaluate_gbsb3([mol], ref_lig=reference_ligand, pdb_file=str(protein_file), config=eval_config, return_dict=True)
    key = "Minimized Vina score" if config.use_minimized_vina_score else "Vina score"
    values = results.get(key)
    if isinstance(values, Sequence) and not isinstance(values, str):
        return _as_float(values[0]) if values else None
    return _as_float(values)


def rewards_to_json(rewards: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(list(rewards), indent=2, sort_keys=True))
