"""Diagnose FLOWR RL candidate reward/selection quality.

This script intentionally reuses the checkpoint loading, ref_gen initialization,
reference sampling, reward computation, and top/middle/bottom selection helpers
used by ``flowr.train_rl_from_smol`` and ``flowr.rl.training``.  It does not
train; it only writes per-candidate JSONL and aggregate summary JSON diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

# Heavy FLOWR imports are delayed until main() so --help stays lightweight.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose RL reference candidates, rewards, and top/bottom selection.")

    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset", default="spindr")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_bucket_sampler", action="store_true")
    parser.add_argument("--bucket_cost_scale", default="quadratic")
    parser.add_argument("--batch_cost", type=int, default=100)
    parser.add_argument("--val_batch_cost", type=int, default=10)
    parser.add_argument("--n_validation_mols", type=int, default=0)

    parser.add_argument("--sample_n_molecules_per_target", type=int, default=None)
    parser.add_argument("--rl_num_candidates_per_step", type=int, default=None)
    parser.add_argument("--integration_steps", type=int, default=20)
    parser.add_argument("--rl_sampling_steps", type=int, default=None)
    parser.add_argument("--ode_sampling_strategy", default="linear")
    parser.add_argument("--corrector_iters", type=int, default=0)
    parser.add_argument("--coord_noise_std", type=float, default=0.0)
    parser.add_argument("--cat_sampling_noise_level", type=float, default=1.0)

    parser.add_argument("--rl_objective_mode", default="strain", choices=["plif", "strain", "vina", "plif_strain", "plif_vina", "strain_vina", "plif_strain_vina"])
    parser.add_argument("--rl_metric_source", default="compute", choices=["compute", "cached", "existing_eval_output"])
    parser.add_argument("--rl_metric_cache_path", default=None)
    parser.add_argument("--rl_multiobjective_strategy", default="constrained_weighted_sum")
    parser.add_argument("--rl_top_ratio", type=float, default=0.25)
    parser.add_argument("--rl_bottom_ratio", type=float, default=0.25)
    parser.add_argument("--rl_top_k", type=int, default=None)
    parser.add_argument("--rl_bottom_k", type=int, default=None)
    parser.add_argument("--rl_strain_good_threshold", type=float, default=0.0)
    parser.add_argument("--rl_strain_bad_threshold", type=float, default=20.0)
    parser.add_argument("--rl_strain_max_threshold", type=float, default=float("inf"))
    parser.add_argument("--rl_vina_good_threshold", type=float, default=-10.0)
    parser.add_argument("--rl_vina_bad_threshold", type=float, default=0.0)
    parser.add_argument("--rl_vina_max_threshold", type=float, default=float("inf"))
    parser.add_argument("--rl_plif_min_threshold", type=float, default=0.0)
    parser.add_argument("--rl_plif_weight", type=float, default=1.0)
    parser.add_argument("--rl_strain_weight", type=float, default=1.0)
    parser.add_argument("--rl_vina_weight", type=float, default=1.0)
    parser.add_argument("--rl_failed_as_bottom", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl_sample_from_reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl_ref_ema_decay", type=float, default=0.999)

    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--output_summary_json", default=None)
    parser.add_argument("--diagnostic_force_recompute", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    # Defaults required by apply_finetune_overrides; not algorithmically used by diagnostics.
    parser.set_defaults(
        lr=1e-4,
        lr_schedule="constant",
        lr_gamma=1.0,
        gradient_clip_val=0.0,
        epochs=1,
        enable_rl_finetune=True,
        rl_loss_weight=1.0,
        rl_update_frequency=1,
        rl_surrogate_type="top_imitation_bottom_repulsion",
        rl_num_stratified_timesteps=1,
        rl_surrogate_chunk_size=0,
        rl_middle_weight=0.0,
        rl_bottom_repulsion_weight=1.0,
        rl_use_reference_model=True,
        rl_reference_checkpoint=None,
        rl_use_original_interpolant=True,
        rl_allow_simple_corruption_fallback=False,
        rl_debug_raise_exceptions=False,
        rl_debug_reference_generation=False,
        rl_debug_pseudo_shapes=False,
        rl_invalid_reward=0.0,
        rl_metric_failed_reward=0.0,
        rl_top_atom_weight=1.0,
        rl_top_bond_weight=1.0,
        rl_top_charge_weight=1.0,
        rl_top_coord_weight=1.0,
        rl_bottom_atom_weight=1.0,
        rl_bottom_bond_weight=1.0,
        rl_bottom_charge_weight=1.0,
        rl_bottom_coord_weight=1.0,
        rl_beta_atom=1.0,
        rl_beta_bond=1.0,
        rl_beta_charge=1.0,
        rl_gamma_coord=1.0,
        rl_aux_fm_weight=0.0,
        rl_anchor_weight=0.0,
        rl_detach_targets=True,
        rl_detach_reward=True,
        rl_log_selected_samples=True,
        rl_save_selected_samples=False,
        rl_log_metric_failures=True,
    )
    return parser.parse_args()


def recursive_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: recursive_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [recursive_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(recursive_to_device(v, device) for v in x)
    return x


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mean(values: Sequence[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else None


def max_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return max(finite) if finite else None


def min_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return min(finite) if finite else None


def grouped_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("selection_label", "unselected"))].append(row)
    out: dict[str, Any] = {}
    for label in ["top", "middle", "bottom", "unselected"]:
        group = groups.get(label, [])
        n = max(1, len(group))
        out[label] = {
            "count": len(group),
            "valid_frac": sum(bool(r.get("valid")) for r in group) / n,
            "posebusters_valid_frac": sum(bool(r.get("posebusters_valid")) for r in group) / n,
            "metric_success_frac": sum(bool(r.get("metric_success")) for r in group) / n,
            "mean_score": mean([r.get("main_score") for r in group]),
        }
    return out


def finalize_summary(rows: Sequence[Mapping[str, Any]], num_batches: int) -> dict[str, Any]:
    n = max(1, len(rows))
    top_rows = [r for r in rows if r.get("selection_label") == "top"]
    bottom_rows = [r for r in rows if r.get("selection_label") == "bottom"]
    summary = {
        "num_batches": num_batches,
        "num_candidates": len(rows),
        "num_valid": sum(bool(r.get("valid")) for r in rows),
        "num_posebusters_valid": sum(bool(r.get("posebusters_valid")) for r in rows),
        "num_metric_success": sum(bool(r.get("metric_success")) for r in rows),
        "num_feasible": sum(bool(r.get("feasible")) for r in rows),
        "num_eligible_top": sum(bool(r.get("eligible_top")) for r in rows),
        "num_eligible_bottom": sum(bool(r.get("eligible_bottom")) for r in rows),
        "num_top": len(top_rows),
        "num_middle": sum(r.get("selection_label") == "middle" for r in rows),
        "num_bottom": len(bottom_rows),
        "invalid_count": sum(not bool(r.get("valid")) for r in rows),
        "posebusters_failed_count": sum(not bool(r.get("posebusters_valid")) for r in rows),
        "metric_failed_count": sum(not bool(r.get("metric_success")) for r in rows),
        "strain_failed_count": sum(bool(r.get("reward", {}).get("strain_success") is False) for r in rows),
        "main_score_mean": mean([r.get("main_score") for r in rows]),
        "main_score_max": max_or_none([r.get("main_score") for r in rows]),
        "strain_energy_mean": mean([r.get("strain_energy") for r in rows]),
        "strain_energy_min": min_or_none([r.get("strain_energy") for r in rows]),
        "strain_energy_max": max_or_none([r.get("strain_energy") for r in rows]),
        "top_main_score_mean": mean([r.get("main_score") for r in top_rows]),
        "bottom_main_score_mean": mean([r.get("main_score") for r in bottom_rows]),
    }
    for key in ["valid", "posebusters_valid", "metric_success", "feasible", "eligible_top", "eligible_bottom"]:
        summary[f"{key}_frac"] = sum(bool(r.get(key)) for r in rows) / n
    summary["top_frac"] = summary["num_top"] / n
    summary["middle_frac"] = summary["num_middle"] / n
    summary["bottom_frac"] = summary["num_bottom"] / n
    summary["by_selection_label"] = grouped_summary(rows)
    return summary


def print_batch_summary(batch_idx: int, candidates: Sequence[Mapping[str, Any]], rewards: Sequence[Mapping[str, Any]], selection: Mapping[str, Any], cache_hit_rate: float) -> None:
    n = len(rewards)
    valid = sum(bool(r.get("valid")) for r in rewards)
    pb = sum(bool(r.get("posebusters_valid")) for r in rewards)
    metric = sum(bool(r.get("metric_success")) for r in rewards)
    top = len(selection.get("top_indices", []))
    middle = len(selection.get("middle_indices", []))
    bottom = len(selection.get("bottom_indices", []))
    score_mean = mean([safe_float(r.get("main_score")) for r in rewards])
    print(
        f"[diagnose-rl] batch={batch_idx} candidates={len(candidates)} valid={valid} pb={pb} "
        f"metric={metric} top={top} middle={middle} bottom={bottom} "
        f"score_mean={score_mean if score_mean is not None else 'nan'} cache_hit_rate={cache_hit_rate:.3f}",
        flush=True,
    )
    if n and top == 0:
        print("[diagnose-rl] warning: no top candidates in this batch; check validity/PoseBusters/metric gates.", flush=True)


def main() -> None:
    args = parse_args()

    global torch
    import torch

    from flowr.data.data_info import GeneralInfos as DataInfos
    from flowr.gen.generate_from_smol import load_model as load_smol_model
    from flowr.rl.training import (
        ensure_reference_generator,
        get_candidate_rewards,
        sample_reference_candidates,
        select_rewards_for_training,
    )
    from flowr.train import build_data_statistic, build_dm
    from flowr.train_rl_from_smol import (
        apply_finetune_overrides,
        checkpoint_hparams,
        make_checkpoint_model_args,
        make_datamodule_args,
        print_checkpoint_summary,
        resolve_sample_count,
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl is None:
        args.output_jsonl = str(save_dir / "diagnostics" / "rl_candidates.jsonl")
    if args.output_summary_json is None:
        args.output_summary_json = str(save_dir / "diagnostics" / "rl_candidate_summary.json")

    hparams = checkpoint_hparams(args.ckpt_path)
    print_checkpoint_summary(hparams, args.ckpt_path)
    load_args = make_checkpoint_model_args(args, hparams)
    model, ckpt_hparams, vocab, vocab_pocket_atoms, vocab_pocket_res = load_smol_model(load_args)

    sample_count = resolve_sample_count(args)
    args.sample_n_molecules_per_target = sample_count
    args.rl_num_candidates_per_step = sample_count
    args.enable_rl_finetune = True
    args.rl_loss_weight = 1.0
    apply_finetune_overrides(model, args, sample_count)

    device = torch.device("cuda:0" if torch.cuda.is_available() and int(args.gpus or 0) else "cpu")
    model.to(device)
    model.eval()

    dm_args = make_datamodule_args(args, ckpt_hparams)
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
    dm.setup("fit")
    model.rl_train_interpolant = dm.train_interpolant

    ref_gen = ensure_reference_generator(model)
    ref_gen.eval()

    if args.diagnostic_force_recompute:
        model.hparams.rl_metric_source = "compute"
        print("[diagnose-rl] diagnostic_force_recompute enabled; using rl_metric_source=compute", flush=True)

    rows: list[dict[str, Any]] = []
    num_batches = 0
    loader = dm.train_dataloader()
    for batch_idx, batch in enumerate(loader):
        if args.max_batches and batch_idx >= args.max_batches:
            break
        if args.max_candidates and len(rows) >= args.max_candidates:
            break

        prior, data, *_ = batch
        prior = recursive_to_device(prior, device)
        data = recursive_to_device(data, device)

        with torch.no_grad():
            candidates = sample_reference_candidates(model, ref_gen, prior, data)
        rewards, cache_hit_rate = get_candidate_rewards(model, candidates)
        selection = select_rewards_for_training(model, rewards)

        selected_label: dict[int, str] = {}
        for i in selection.get("top_indices", []):
            selected_label[int(i)] = "top"
        for i in selection.get("middle_indices", []):
            selected_label[int(i)] = "middle"
        for i in selection.get("bottom_indices", []):
            selected_label[int(i)] = "bottom"

        for i, reward in enumerate(rewards):
            if args.max_candidates and len(rows) >= args.max_candidates:
                break
            cand = candidates[i]
            rows.append(
                {
                    "batch_idx": batch_idx,
                    "candidate_idx": i,
                    "sample_id": cand.get("sample_id"),
                    "system_id": cand.get("system_id"),
                    "candidate_round": cand.get("candidate_round"),
                    "batch_index": cand.get("batch_index"),
                    "selection_label": selected_label.get(i, "unselected"),
                    "valid": bool(reward.get("valid")),
                    "posebusters_valid": bool(reward.get("posebusters_valid")),
                    "metric_success": bool(reward.get("metric_success")),
                    "feasible": bool(reward.get("feasible")),
                    "eligible_top": bool(reward.get("eligible_top")),
                    "eligible_bottom": bool(reward.get("eligible_bottom")),
                    "main_score": safe_float(reward.get("main_score")),
                    "strain_energy": safe_float(reward.get("strain_energy")),
                    "vina_score": safe_float(reward.get("vina_score")),
                    "plif_tanimoto": safe_float(reward.get("plif_tanimoto")),
                    "error": reward.get("error"),
                    "failure_reason": reward.get("failure_reason") or reward.get("reason"),
                    "reward": reward,
                }
            )

        num_batches += 1
        print_batch_summary(batch_idx, candidates, rewards, selection, cache_hit_rate)

    summary = finalize_summary(rows, num_batches)
    write_jsonl(args.output_jsonl, rows)
    write_json(args.output_summary_json, summary)
    print(f"[diagnose-rl] wrote candidates: {args.output_jsonl}", flush=True)
    print(f"[diagnose-rl] wrote summary: {args.output_summary_json}", flush=True)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
