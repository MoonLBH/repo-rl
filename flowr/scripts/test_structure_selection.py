#!/usr/bin/env python
"""Smoke test CLI for Stage-2 structure reward top/middle/bottom selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flowr.rl.structure_selection import select_top_middle_bottom


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test FLOWR structure reward selection without touching training.")
    parser.add_argument("--input", type=str, default=None, help="Optional JSON file from test_structure_rewards.py.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path for selection result.")
    parser.add_argument("--objective_mode", type=str, default="plif_strain_vina")
    parser.add_argument("--multiobjective_strategy", type=str, default="constrained_weighted_sum")
    parser.add_argument("--top_ratio", type=float, default=0.2)
    parser.add_argument("--bottom_ratio", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--bottom_k", type=int, default=None)
    parser.add_argument("--min_top_samples", type=int, default=0)
    parser.add_argument("--min_bottom_samples", type=int, default=0)
    parser.add_argument("--main_score_key", type=str, default="main_score")
    parser.add_argument("--plif_min_threshold", type=float, default=0.3)
    parser.add_argument("--strain_max_threshold", type=float, default=10.0)
    parser.add_argument("--vina_max_threshold", type=float, default=-4.0)
    parser.add_argument("--plif_weight", type=float, default=1.0)
    parser.add_argument("--strain_weight", type=float, default=1.0)
    parser.add_argument("--vina_weight", type=float, default=1.0)
    parser.add_argument("--allow_infeasible_top", action="store_true", help="Debug only: do not require feasible for top.")
    parser.add_argument("--exclude_failed_bottom", action="store_true", help="Debug only: do not prioritize failed samples for bottom.")
    parser.add_argument("--mock", action="store_true", help="Use built-in mock reward dictionaries.")
    return parser


def load_rewards(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input is not None and not args.mock:
        return json.loads(Path(args.input).read_text())
    return mock_rewards()


def mock_rewards() -> list[dict[str, Any]]:
    """Construct rewards covering success, failure, threshold, and shortage cases."""

    return [
        _reward("good_all_metrics", True, True, 0.82, 2.0, -8.0, 0.86, True, True),
        _reward("good_lower_score", True, True, 0.50, 6.0, -6.0, 0.55, True, True),
        _reward("invalid", False, True, 0.95, 1.0, -10.0, 0.0, False, False),
        _reward("posebusters_failed", True, False, 0.90, 1.5, -9.0, 0.0, False, False),
        _reward("plif_failed", True, True, None, 2.0, -8.0, 0.0, False, False),
        _reward("strain_failed", True, True, 0.70, None, -8.5, 0.0, False, False),
        _reward("vina_failed", True, True, 0.72, 3.0, None, 0.0, False, False),
        _reward("high_plif_bad_strain", True, True, 0.95, 25.0, -8.0, 0.0, False, False),
        _reward("good_vina_low_plif", True, True, 0.10, 3.0, -11.0, 0.0, False, False),
        _reward("three_goal_one_bad", True, True, 0.65, 3.0, -1.0, 0.0, False, False),
        _reward("barely_passes", True, True, 0.35, 9.5, -4.5, 0.32, True, True),
        _reward("not_eligible_low", True, True, 0.31, 9.8, -4.1, 0.05, True, False),
    ]


def _reward(
    sample_id: str,
    valid: bool,
    posebusters_valid: bool,
    plif: float | None,
    strain: float | None,
    vina: float | None,
    main_score: float,
    feasible: bool,
    eligible_top: bool,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "valid": valid,
        "posebusters_valid": posebusters_valid,
        "plif_tanimoto": plif,
        "strain_energy": strain,
        "vina_score": vina,
        "plif_success": plif is not None,
        "strain_success": strain is not None,
        "vina_success": vina is not None,
        "metric_success": feasible,
        "feasible": feasible,
        "eligible_top": eligible_top,
        "eligible_bottom": not eligible_top or main_score <= 0.0,
        "main_score": main_score,
        "normalized_plif": plif if plif is not None else None,
        "normalized_strain_score": None if strain is None else max(0.0, min(1.0, (20.0 - strain) / 20.0)),
        "normalized_vina_score": None if vina is None else max(0.0, min(1.0, (0.0 - vina) / 10.0)),
        "objective_mode": "plif_strain_vina",
        "multiobjective_strategy": "constrained_weighted_sum",
        "error": None if feasible else "mock failure or threshold fallback",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rewards = load_rewards(args)
    return select_top_middle_bottom(
        rewards,
        objective_mode=args.objective_mode,
        multiobjective_strategy=args.multiobjective_strategy,
        top_ratio=args.top_ratio,
        bottom_ratio=args.bottom_ratio,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        feasible_only_for_top=not args.allow_infeasible_top,
        include_failed_in_bottom=not args.exclude_failed_bottom,
        min_top_samples=args.min_top_samples,
        min_bottom_samples=args.min_bottom_samples,
        main_score_key=args.main_score_key,
        plif_min_threshold=args.plif_min_threshold,
        strain_max_threshold=args.strain_max_threshold,
        vina_max_threshold=args.vina_max_threshold,
        plif_weight=args.plif_weight,
        strain_weight=args.strain_weight,
        vina_weight=args.vina_weight,
    )


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
