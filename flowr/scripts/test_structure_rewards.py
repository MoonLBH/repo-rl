#!/usr/bin/env python
"""Smoke test CLI for FLOWR structure reward normalization and fallback rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowr.rl.structure_rewards import RewardConfig, compute_main_score, compute_structure_rewards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test FLOWR structure reward wrappers without touching training.")
    parser.add_argument("--input", type=str, default=None, help="FLOWR sampling output .pt or generated ligand SDF.")
    parser.add_argument("--protein", type=str, default=None, help="Protein/pocket PDB for PLIF/PoseBusters/Vina.")
    parser.add_argument("--reference_ligand", type=str, default=None, help="Reference ligand SDF.")
    parser.add_argument("--objective_mode", type=str, default="plif_strain_vina")
    parser.add_argument("--multiobjective_strategy", type=str, default="constrained_weighted_sum")
    parser.add_argument("--plif_weight", type=float, default=1.0)
    parser.add_argument("--strain_weight", type=float, default=1.0)
    parser.add_argument("--vina_weight", type=float, default=1.0)
    parser.add_argument("--plif_min_threshold", type=float, default=0.3)
    parser.add_argument("--strain_max_threshold", type=float, default=10.0)
    parser.add_argument("--vina_max_threshold", type=float, default=-4.0)
    parser.add_argument("--strain_good_threshold", type=float, default=0.0)
    parser.add_argument("--strain_bad_threshold", type=float, default=20.0)
    parser.add_argument("--vina_good_threshold", type=float, default=-10.0)
    parser.add_argument("--vina_bad_threshold", type=float, default=0.0)
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--mock", action="store_true", help="Run mock normalization/fallback checks only.")
    return parser


def config_from_args(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        objective_mode=args.objective_mode,
        multiobjective_strategy=args.multiobjective_strategy,
        plif_weight=args.plif_weight,
        strain_weight=args.strain_weight,
        vina_weight=args.vina_weight,
        plif_min_threshold=args.plif_min_threshold,
        strain_max_threshold=args.strain_max_threshold,
        vina_max_threshold=args.vina_max_threshold,
        strain_good_threshold=args.strain_good_threshold,
        strain_bad_threshold=args.strain_bad_threshold,
        vina_good_threshold=args.vina_good_threshold,
        vina_bad_threshold=args.vina_bad_threshold,
    )


def run_mock(args: argparse.Namespace) -> list[dict]:
    config = config_from_args(args)
    cases = [
        {
            "sample_id": "good_all_metrics",
            "valid": True,
            "posebusters_valid": True,
            "plif_tanimoto": 0.75,
            "strain_energy": 4.0,
            "vina_score": -8.0,
        },
        {
            "sample_id": "invalid_bottom",
            "valid": False,
            "posebusters_valid": True,
            "plif_tanimoto": 0.95,
            "strain_energy": 1.0,
            "vina_score": -10.0,
        },
        {
            "sample_id": "missing_enabled_metric",
            "valid": True,
            "posebusters_valid": True,
            "plif_tanimoto": None,
            "strain_energy": 2.0,
            "vina_score": -9.0,
        },
        {
            "sample_id": "fails_constraint",
            "valid": True,
            "posebusters_valid": True,
            "plif_tanimoto": 0.2,
            "strain_energy": 15.0,
            "vina_score": -2.0,
        },
    ]
    outputs = []
    for case in cases:
        score = compute_main_score(
            plif_tanimoto=case["plif_tanimoto"],
            strain_energy=case["strain_energy"],
            vina_score=case["vina_score"],
            valid=case["valid"],
            posebusters_valid=case["posebusters_valid"],
            config=config,
        )
        outputs.append({**case, **score})
    return outputs


def run_real(args: argparse.Namespace) -> list[dict]:
    config = config_from_args(args)
    if args.input is None:
        raise ValueError("--input is required unless --mock is set")
    input_path = Path(args.input)
    kwargs = {"config": config, "protein_file": args.protein, "reference_ligand_file": args.reference_ligand}
    if input_path.suffix == ".pt":
        kwargs["flowr_sampling_output"] = input_path
    else:
        kwargs["generated_ligand_file"] = input_path
    return compute_structure_rewards(**kwargs)


def main() -> None:
    args = build_parser().parse_args()
    results = run_mock(args) if args.mock or args.input is None else run_real(args)
    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
