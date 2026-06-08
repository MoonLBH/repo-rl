"""Top/middle/bottom selection utilities for structure reward dictionaries.

This module is intentionally default-off: it is not imported by FLOWR's
training, sampling, or evaluation entrypoints unless a user calls it explicitly.
It consumes per-sample reward dictionaries produced by
``flowr.rl.structure_rewards`` and returns masks/indices suitable for future
reward-guided fine-tuning experiments.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from flowr.rl.structure_rewards import OBJECTIVE_METRICS, MultiObjectiveStrategy, ObjectiveMode


@dataclass
class SelectionConfig:
    """Configuration for structure reward top/bottom selection."""

    objective_mode: str = ObjectiveMode.PLIF_STRAIN_VINA.value
    multiobjective_strategy: str = MultiObjectiveStrategy.CONSTRAINED_WEIGHTED_SUM.value
    top_ratio: float = 0.2
    bottom_ratio: float = 0.2
    top_k: Optional[int] = None
    bottom_k: Optional[int] = None
    feasible_only_for_top: bool = True
    include_failed_in_bottom: bool = True
    min_top_samples: int = 0
    min_bottom_samples: int = 0
    main_score_key: str = "main_score"
    plif_min_threshold: float = 0.0
    strain_max_threshold: float = math.inf
    vina_max_threshold: float = math.inf
    plif_weight: float = 1.0
    strain_weight: float = 1.0
    vina_weight: float = 1.0

    def enabled_metrics(self) -> tuple[str, ...]:
        try:
            return OBJECTIVE_METRICS[self.objective_mode]
        except KeyError as exc:
            valid = ", ".join(sorted(OBJECTIVE_METRICS))
            raise ValueError(f"Unknown objective_mode={self.objective_mode!r}; choose from {valid}") from exc

    def validate(self) -> None:
        if self.multiobjective_strategy not in {strategy.value for strategy in MultiObjectiveStrategy}:
            valid = ", ".join(strategy.value for strategy in MultiObjectiveStrategy)
            raise ValueError(f"Unknown multiobjective_strategy={self.multiobjective_strategy!r}; choose from {valid}")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k must be non-negative or None")
        if self.bottom_k is not None and self.bottom_k < 0:
            raise ValueError("bottom_k must be non-negative or None")
        if not 0.0 <= self.top_ratio <= 1.0:
            raise ValueError("top_ratio must be in [0, 1]")
        if not 0.0 <= self.bottom_ratio <= 1.0:
            raise ValueError("bottom_ratio must be in [0, 1]")
        if self.min_top_samples < 0 or self.min_bottom_samples < 0:
            raise ValueError("min_top_samples and min_bottom_samples must be non-negative")


def select_top_middle_bottom(
    reward_dicts: Sequence[Mapping[str, Any]],
    *,
    objective_mode: str = ObjectiveMode.PLIF_STRAIN_VINA.value,
    multiobjective_strategy: str = MultiObjectiveStrategy.CONSTRAINED_WEIGHTED_SUM.value,
    top_ratio: float = 0.2,
    bottom_ratio: float = 0.2,
    top_k: Optional[int] = None,
    bottom_k: Optional[int] = None,
    feasible_only_for_top: bool = True,
    include_failed_in_bottom: bool = True,
    min_top_samples: int = 0,
    min_bottom_samples: int = 0,
    main_score_key: str = "main_score",
    plif_min_threshold: float = 0.0,
    strain_max_threshold: float = math.inf,
    vina_max_threshold: float = math.inf,
    plif_weight: float = 1.0,
    strain_weight: float = 1.0,
    vina_weight: float = 1.0,
) -> dict[str, Any]:
    """Select top/middle/bottom samples from Stage-1 structure reward dicts.

    Top samples are strictly gated by validity, PoseBusters validity, enabled
    metric success, ``feasible`` and ``eligible_top``.  Bottom samples prioritize
    failed/invalid/fallback samples and then fill from low-scoring feasible
    samples if needed.  No selected index can appear in more than one split.
    """

    config = SelectionConfig(
        objective_mode=objective_mode,
        multiobjective_strategy=multiobjective_strategy,
        top_ratio=top_ratio,
        bottom_ratio=bottom_ratio,
        top_k=top_k,
        bottom_k=bottom_k,
        feasible_only_for_top=feasible_only_for_top,
        include_failed_in_bottom=include_failed_in_bottom,
        min_top_samples=min_top_samples,
        min_bottom_samples=min_bottom_samples,
        main_score_key=main_score_key,
        plif_min_threshold=plif_min_threshold,
        strain_max_threshold=strain_max_threshold,
        vina_max_threshold=vina_max_threshold,
        plif_weight=plif_weight,
        strain_weight=strain_weight,
        vina_weight=vina_weight,
    )
    return select_top_middle_bottom_with_config(reward_dicts, config)


def select_top_middle_bottom_with_config(
    reward_dicts: Sequence[Mapping[str, Any]], config: SelectionConfig
) -> dict[str, Any]:
    """Select top/middle/bottom samples using an explicit config object."""

    config.validate()
    rewards = list(reward_dicts)
    n_samples = len(rewards)
    enabled = config.enabled_metrics()
    top_target = _target_count(n_samples, config.top_ratio, config.top_k)
    bottom_target = _target_count(n_samples, config.bottom_ratio, config.bottom_k)

    diagnostics = [_diagnose_reward(reward, enabled, config) for reward in rewards]

    top_candidates = [idx for idx, diag in enumerate(diagnostics) if diag["top_candidate"]]
    top_indices = sorted(
        top_candidates,
        key=lambda idx: (_ranking_score(rewards[idx], enabled, config), -idx),
        reverse=True,
    )[:top_target]
    top_set = set(top_indices)

    failed_bottom_candidates = [
        idx
        for idx, diag in enumerate(diagnostics)
        if idx not in top_set and diag["failed_bottom_candidate"] and config.include_failed_in_bottom
    ]
    failed_bottom_candidates = sorted(
        failed_bottom_candidates,
        key=lambda idx: (
            diagnostics[idx]["bottom_priority"],
            _ranking_score(rewards[idx], enabled, config),
            idx,
        ),
    )

    bottom_indices = failed_bottom_candidates[:bottom_target]
    bottom_set = set(bottom_indices)

    if len(bottom_indices) < bottom_target:
        remaining = [idx for idx in range(n_samples) if idx not in top_set and idx not in bottom_set]
        low_score_candidates = sorted(
            remaining,
            key=lambda idx: (_ranking_score(rewards[idx], enabled, config), idx),
        )
        needed = bottom_target - len(bottom_indices)
        bottom_indices.extend(low_score_candidates[:needed])
        bottom_set = set(bottom_indices)

    middle_indices = [idx for idx in range(n_samples) if idx not in top_set and idx not in bottom_set]
    selection_labels = ["middle"] * n_samples
    for idx in top_indices:
        selection_labels[idx] = "top"
    for idx in bottom_indices:
        selection_labels[idx] = "bottom"

    top_mask = _mask(n_samples, top_indices)
    bottom_mask = _mask(n_samples, bottom_indices)
    middle_mask = _mask(n_samples, middle_indices)
    bottom_reasons = {str(idx): diagnostics[idx]["reasons"] for idx in bottom_indices}
    top_reject_reasons = {str(idx): diagnostics[idx]["reasons"] for idx, diag in enumerate(diagnostics) if not diag["top_candidate"]}
    warning_messages: list[str] = []
    if len(top_indices) < min(config.min_top_samples, top_target):
        warning_messages.append(
            f"top candidates below min_top_samples: selected={len(top_indices)}, min_top_samples={config.min_top_samples}"
        )
    if len(bottom_indices) < min(config.min_bottom_samples, bottom_target):
        warning_messages.append(
            f"bottom candidates below min_bottom_samples: selected={len(bottom_indices)}, min_bottom_samples={config.min_bottom_samples}"
        )

    summary = {
        "num_samples": n_samples,
        "num_top": len(top_indices),
        "num_middle": len(middle_indices),
        "num_bottom": len(bottom_indices),
        "requested_top": top_target,
        "requested_bottom": bottom_target,
        "num_top_candidates": len(top_candidates),
        "num_failed_bottom_candidates": len(failed_bottom_candidates),
        "objective_mode": config.objective_mode,
        "multiobjective_strategy": config.multiobjective_strategy,
        "enabled_metrics": list(enabled),
        "main_score_key": config.main_score_key,
        "config": asdict(config),
        "bottom_reasons": bottom_reasons,
        "top_reject_reasons": top_reject_reasons,
        "warnings": warning_messages,
    }

    return {
        "top_indices": top_indices,
        "middle_indices": middle_indices,
        "bottom_indices": bottom_indices,
        "top_mask": top_mask,
        "middle_mask": middle_mask,
        "bottom_mask": bottom_mask,
        "selection_labels": selection_labels,
        "summary": summary,
    }


def _target_count(n_samples: int, ratio: float, k_value: Optional[int]) -> int:
    if n_samples <= 0:
        return 0
    if k_value is not None:
        return min(k_value, n_samples)
    return min(int(math.ceil(n_samples * ratio)), n_samples)


def _mask(n_samples: int, indices: Sequence[int]) -> list[bool]:
    selected = set(indices)
    return [idx in selected for idx in range(n_samples)]


def _score(reward: Mapping[str, Any], main_score_key: str) -> float:
    value = reward.get(main_score_key, 0.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _as_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _ranking_score(reward: Mapping[str, Any], enabled: Sequence[str], config: SelectionConfig) -> float:
    normalized_keys = {
        "plif": "normalized_plif",
        "strain": "normalized_strain_score",
        "vina": "normalized_vina_score",
    }
    weights = {"plif": config.plif_weight, "strain": config.strain_weight, "vina": config.vina_weight}
    scores: dict[str, float] = {}
    for metric in enabled:
        value = _safe_float(reward.get(normalized_keys[metric]))
        if value is None:
            return _score(reward, config.main_score_key)
        scores[metric] = max(0.0, min(1.0, value))

    if not scores:
        return _score(reward, config.main_score_key)
    if config.multiobjective_strategy == MultiObjectiveStrategy.WEIGHTED_MIN.value:
        return min(scores[metric] * max(weights[metric], 0.0) for metric in scores)

    total_weight = sum(weights[metric] for metric in scores if weights[metric] > 0.0)
    if total_weight <= 0.0:
        return _score(reward, config.main_score_key)
    return sum(scores[metric] * weights[metric] for metric in scores if weights[metric] > 0.0) / total_weight


def _metric_success(reward: Mapping[str, Any], metric: str) -> bool:
    success_key = f"{metric}_success"
    if success_key in reward:
        return _as_bool(reward.get(success_key))
    metric_key = {"plif": "plif_tanimoto", "strain": "strain_energy", "vina": "vina_score"}[metric]
    return reward.get(metric_key) is not None


def _diagnose_reward(reward: Mapping[str, Any], enabled: Sequence[str], config: SelectionConfig) -> dict[str, Any]:
    reasons: list[str] = []
    valid = _as_bool(reward.get("valid"))
    posebusters_valid = _as_bool(reward.get("posebusters_valid"))
    feasible = _as_bool(reward.get("feasible"))
    eligible_top = _as_bool(reward.get("eligible_top"))
    eligible_bottom = _as_bool(reward.get("eligible_bottom"))

    if not valid:
        reasons.append("invalid")
    if not posebusters_valid:
        reasons.append("posebusters_failed")

    for metric in enabled:
        if not _metric_success(reward, metric):
            reasons.append(f"{metric}_failed")

    threshold_failed = False
    use_hard_thresholds = len(enabled) > 1
    if use_hard_thresholds and "plif" in enabled and _metric_success(reward, "plif"):
        plif_value = _safe_float(reward.get("plif_tanimoto"))
        if plif_value is None or plif_value < config.plif_min_threshold:
            reasons.append("plif_below_threshold")
            threshold_failed = True
    if use_hard_thresholds and "strain" in enabled and _metric_success(reward, "strain") and config.strain_max_threshold != math.inf:
        strain_value = _safe_float(reward.get("strain_energy"))
        if strain_value is None or strain_value > config.strain_max_threshold:
            reasons.append("strain_above_threshold")
            threshold_failed = True
    if use_hard_thresholds and "vina" in enabled and _metric_success(reward, "vina") and config.vina_max_threshold != math.inf:
        vina_value = _safe_float(reward.get("vina_score"))
        if vina_value is None or vina_value > config.vina_max_threshold:
            reasons.append("vina_above_threshold")
            threshold_failed = True

    if not feasible:
        reasons.append("not_feasible")
    if not eligible_top:
        reasons.append("not_eligible_top")
    if _score(reward, config.main_score_key) <= 0.0:
        reasons.append("low_composite_score")

    metric_success = all(_metric_success(reward, metric) for metric in enabled)
    top_candidate = valid and posebusters_valid and metric_success and eligible_top and not threshold_failed
    if config.feasible_only_for_top:
        top_candidate = top_candidate and feasible

    explicit_failure = (not valid) or (not posebusters_valid) or (not metric_success) or (not feasible)
    failed_bottom_candidate = eligible_bottom or explicit_failure
    if not failed_bottom_candidate and _score(reward, config.main_score_key) <= 0.0:
        failed_bottom_candidate = True

    bottom_priority = _bottom_priority(reasons, explicit_failure, _score(reward, config.main_score_key))
    return {
        "top_candidate": top_candidate,
        "failed_bottom_candidate": failed_bottom_candidate,
        "reasons": _dedupe(reasons) or ["low_composite_score"],
        "bottom_priority": bottom_priority,
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _bottom_priority(reasons: Sequence[str], explicit_failure: bool, score: float) -> int:
    if "invalid" in reasons:
        return 0
    if "posebusters_failed" in reasons:
        return 1
    if any(reason.endswith("_failed") for reason in reasons):
        return 2
    if any(reason.endswith("_below_threshold") or reason.endswith("_above_threshold") for reason in reasons):
        return 3
    if explicit_failure:
        return 4
    if score <= 0.0:
        return 5
    return 6


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
