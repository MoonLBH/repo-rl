#!/usr/bin/env python
"""Dry-run FLOWR structure reward -> selection analysis without training/backward.

The script is default-off and does not modify FLOWR sampling, training, or
existing evaluation entrypoints.  It can either consume existing per-sample
reward JSON files or compute rewards through ``flowr.rl.structure_rewards`` for
small-scale diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from flowr.rl.structure_rewards import RewardConfig, compute_structure_rewards
from flowr.rl.structure_selection import select_top_middle_bottom

CSV_COLUMNS = [
    "sample_id",
    "protein_id",
    "pocket_id",
    "ligand_file",
    "smiles",
    "valid",
    "posebusters_valid",
    "plif_tanimoto",
    "strain_energy",
    "vina_score",
    "plif_success",
    "strain_success",
    "vina_success",
    "metric_success",
    "feasible",
    "eligible_top",
    "eligible_bottom",
    "normalized_plif",
    "normalized_strain",
    "normalized_vina",
    "main_score",
    "selection_label",
    "failure_reason",
    "error",
]


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run FLOWR structure reward and top/bottom selection.")
    parser.add_argument("--generated_dir", type=str, required=True, help="FLOWR sampling output directory or .pt file.")
    parser.add_argument("--eval_output", type=str, default=None, help="Existing eval/reward output path or directory.")
    parser.add_argument("--metric_source", choices=["existing_eval_output", "compute"], default="existing_eval_output")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--cache_jsonl", type=str, default=None, help="Optional JSONL cache for per-sample reward dicts.")
    parser.add_argument("--output_top_sdf", type=str, default=None)
    parser.add_argument("--output_bottom_sdf", type=str, default=None)
    parser.add_argument("--failed_log", type=str, default=None)
    parser.add_argument("--top_summary_csv", type=str, default=None)
    parser.add_argument("--objective_mode", type=str, default="plif_strain_vina")
    parser.add_argument("--multiobjective_strategy", type=str, default="constrained_weighted_sum")
    parser.add_argument("--plif_weight", type=float, default=1.0)
    parser.add_argument("--strain_weight", type=float, default=1.0)
    parser.add_argument("--vina_weight", type=float, default=1.0)
    parser.add_argument("--plif_min_threshold", type=float, default=0.3)
    parser.add_argument("--strain_max_threshold", type=float, default=10.0)
    parser.add_argument("--vina_max_threshold", type=float, default=-6.0)
    parser.add_argument("--strain_good_threshold", type=float, default=0.0)
    parser.add_argument("--strain_bad_threshold", type=float, default=20.0)
    parser.add_argument("--vina_good_threshold", type=float, default=-10.0)
    parser.add_argument("--vina_bad_threshold", type=float, default=0.0)
    parser.add_argument("--top_ratio", type=float, default=0.1)
    parser.add_argument("--bottom_ratio", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--bottom_k", type=int, default=None)
    parser.add_argument("--include_failed_in_bottom", type=str_to_bool, default=True)
    parser.add_argument("--feasible_only_for_top", type=str_to_bool, default=True)
    parser.add_argument("--compute_non_enabled_metrics", type=str_to_bool, default=False)
    parser.add_argument("--require_posebusters_validity", type=str_to_bool, default=True)
    parser.add_argument("--compute_posebusters_validity", type=str_to_bool, default=True)
    parser.add_argument("--config_path", type=str, default="./genbench3d/config/default.yaml")
    parser.add_argument("--max_prediction_files", type=int, default=None, help="Optional limit for quick smoke tests.")
    return parser


def reward_config_from_args(args: argparse.Namespace) -> RewardConfig:
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
        require_posebusters_validity=args.require_posebusters_validity,
        compute_posebusters_validity=args.compute_posebusters_validity,
        compute_non_enabled_metrics=args.compute_non_enabled_metrics,
        config_path=args.config_path,
    )


def prediction_files(generated_dir: str | Path, max_files: Optional[int] = None) -> list[Path]:
    path = Path(generated_dir)
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("predictions_multi_*.pt"))
        if not files and (path / "predictions.pt").exists():
            files = [path / "predictions.pt"]
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No FLOWR predictions file found under {generated_dir}")
    return files


def source_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha1()
    for path in paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(int(stat.st_mtime)).encode())
    return digest.hexdigest()[:16]


def cache_key(reward: Mapping[str, Any], args: argparse.Namespace, source_fp: str) -> str:
    return cache_key_from_sample_id(str(reward.get("sample_id")), args, source_fp)


def cache_key_from_sample_id(sample_id: str, args: argparse.Namespace, source_fp: str) -> str:
    raw = {
        "sample_id": sample_id,
        "source_fp": source_fp,
        "objective_mode": args.objective_mode,
        "multiobjective_strategy": args.multiobjective_strategy,
        "plif_min_threshold": args.plif_min_threshold,
        "strain_max_threshold": args.strain_max_threshold,
        "vina_max_threshold": args.vina_max_threshold,
    }
    return hashlib.sha1(json.dumps(raw, sort_keys=True).encode()).hexdigest()


def load_cache(path: Optional[str | Path]) -> dict[str, dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if "cache_key" in item and "reward" in item:
                cache[item["cache_key"]] = item["reward"]
    return cache


def rewrite_cache(path: Optional[str | Path], cache: Mapping[str, Mapping[str, Any]]) -> None:
    if path is None:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        for key, reward in sorted(cache.items()):
            handle.write(json.dumps({"cache_key": key, "reward": reward}, sort_keys=True) + "\n")


def load_existing_rewards(eval_output: str | Path, objective_mode: str) -> list[dict[str, Any]]:
    path = Path(eval_output)
    if path.is_file():
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            return coerce_reward_payload(payload)
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        raise ValueError(f"Unsupported existing eval output file: {path}")

    candidates = [
        path / f"reward_{objective_mode}.json",
        path / "reward_record" / f"reward_{objective_mode}.json",
        path / "structure_rewards.json",
        path / "rewards.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_existing_rewards(candidate, objective_mode)

    # eval_spindr.sh writes metrics.pt and interaction_recovery_list.pt.  Those
    # are useful summaries but do not provide per-sample strain/Vina values, so
    # they cannot fully construct reward dicts for top/bottom selection.
    existing = [p.name for p in path.glob("*.pt")]
    raise ValueError(
        "Could not find per-sample reward JSON in eval_output. "
        f"Found PT files {existing}; eval_spindr metrics.pt is aggregated and is not enough for per-sample selection. "
        "Use --metric_source compute or pass a Stage-1 reward_*.json file."
    )


def coerce_reward_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        for key in ("rewards", "reward_dicts", "records", "per_sample"):
            if isinstance(payload.get(key), list):
                return [dict(item) for item in payload[key]]
    raise ValueError("Existing reward JSON must be a list or contain rewards/reward_dicts/records/per_sample")


def compute_rewards(args: argparse.Namespace, files: Sequence[Path], source_fp: str) -> list[dict[str, Any]]:
    config = reward_config_from_args(args)
    cache = load_cache(args.cache_jsonl)
    all_rewards: list[dict[str, Any]] = []
    changed = False

    # Cache is per sample.  If every sample in a predictions file is cached, the
    # heavy metric wrapper is skipped completely; if any sample is missing, the
    # current FLOWR wrapper computes the whole file and then updates the cache.
    for file_path in files:
        cached_file_rewards = cached_rewards_for_file(file_path, args, source_fp, cache)
        if cached_file_rewards is not None:
            all_rewards.extend(cached_file_rewards)
            continue

        file_rewards = compute_structure_rewards(flowr_sampling_output=file_path, config=config)
        for reward in file_rewards:
            key = cache_key(reward, args, source_fp)
            if key in cache:
                all_rewards.append(dict(cache[key]))
            else:
                reward = dict(reward)
                reward["cache_key"] = key
                cache[key] = reward
                all_rewards.append(reward)
                changed = True
    if changed:
        rewrite_cache(args.cache_jsonl, cache)
    return all_rewards


def cached_rewards_for_file(
    file_path: Path,
    args: argparse.Namespace,
    source_fp: str,
    cache: Mapping[str, Mapping[str, Any]],
) -> Optional[list[dict[str, Any]]]:
    if not cache:
        return None
    try:
        from flowr.rl.structure_rewards import _records_from_flowr_sampling_output

        records = _records_from_flowr_sampling_output(file_path)
    except Exception:
        return None

    rewards: list[dict[str, Any]] = []
    for record in records:
        key = cache_key_from_sample_id(str(record.get("sample_id")), args, source_fp)
        if key not in cache:
            return None
        rewards.append(dict(cache[key]))
    return rewards


def load_or_compute_rewards(args: argparse.Namespace, files: Sequence[Path], source_fp: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if args.metric_source == "existing_eval_output":
        if args.eval_output is None:
            raise ValueError("--eval_output is required when --metric_source existing_eval_output")
        rewards = load_existing_rewards(args.eval_output, args.objective_mode)
        warnings.append("Loaded existing per-sample reward JSON; heavy metrics were not recomputed.")
        return rewards, warnings
    rewards = compute_rewards(args, files, source_fp)
    warnings.append("Computed rewards through flowr.rl.structure_rewards; PLIF/Vina/strain may be slow.")
    return rewards, warnings


def attach_molecules(rewards: Sequence[Mapping[str, Any]], files: Sequence[Path]) -> list[dict[str, Any]]:
    try:
        from flowr.rl.structure_rewards import _records_from_flowr_sampling_output
    except Exception:
        return [dict(reward) for reward in rewards]

    mols_by_id: dict[str, Any] = {}
    for file_path in files:
        try:
            for record in _records_from_flowr_sampling_output(file_path):
                mols_by_id[str(record.get("sample_id"))] = record.get("mol")
        except Exception:
            continue
    output: list[dict[str, Any]] = []
    for reward in rewards:
        item = dict(reward)
        sample_id = str(item.get("sample_id"))
        if sample_id in mols_by_id:
            item["mol"] = mols_by_id[sample_id]
        output.append(item)
    return output


def enrich_records(rewards: Sequence[Mapping[str, Any]], selection: Mapping[str, Any], generated_dir: str | Path) -> list[dict[str, Any]]:
    labels = selection["selection_labels"]
    bottom_reasons = selection["summary"].get("bottom_reasons", {})
    top_reject_reasons = selection["summary"].get("top_reject_reasons", {})
    rows: list[dict[str, Any]] = []
    for idx, reward in enumerate(rewards):
        row = dict(reward)
        row["selection_label"] = labels[idx]
        reasons = bottom_reasons.get(str(idx)) or top_reject_reasons.get(str(idx)) or []
        row["failure_reason"] = ";".join(reasons)
        row["protein_id"] = infer_protein_id(row)
        row["pocket_id"] = infer_pocket_id(row)
        row.setdefault("ligand_file", "")
        row.setdefault("smiles", mol_to_smiles(row.get("mol")))
        row["normalized_strain"] = row.get("normalized_strain_score")
        row["normalized_vina"] = row.get("normalized_vina_score")
        row["generated_dir"] = str(generated_dir)
        rows.append(row)
    return rows


def infer_protein_id(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    for key in ("protein_id", "pocket_id", "protein_file", "structure_protein_file", "plif_protein_file"):
        value = row.get(key) or metadata.get(key)
        if value:
            return Path(str(value)).stem
    sample_id = str(row.get("sample_id", ""))
    return sample_id.split("_lig_")[0] if "_lig_" in sample_id else ""


def infer_pocket_id(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    for key in ("pocket_id", "protein_id", "protein_file", "structure_protein_file", "plif_protein_file"):
        value = row.get(key) or metadata.get(key)
        if value:
            return Path(str(value)).stem
    return infer_protein_id(row)


def mol_to_smiles(mol: Any) -> str:
    if mol is None:
        return ""
    try:
        from rdkit import Chem

        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(row.get(key)) for key in CSV_COLUMNS})


def serialize_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def summary_json(
    rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    args: argparse.Namespace,
    warnings: Sequence[str],
) -> dict[str, Any]:
    n = len(rows)
    num_valid = sum(bool(row.get("valid")) for row in rows)
    num_pb = sum(bool(row.get("posebusters_valid")) for row in rows)
    num_metric_success = sum(bool(row.get("metric_success")) for row in rows)
    num_feasible = sum(bool(row.get("feasible")) for row in rows)
    main_scores = finite_values(rows, "main_score")
    summary = {
        "num_total": n,
        "num_valid": num_valid,
        "num_posebusters_valid": num_pb,
        "num_metric_success": num_metric_success,
        "num_feasible": num_feasible,
        "num_top": selection["summary"]["num_top"],
        "num_middle": selection["summary"]["num_middle"],
        "num_bottom": selection["summary"]["num_bottom"],
        "validity_rate": num_valid / n if n else None,
        "posebusters_validity_rate": num_pb / n if n else None,
        "metric_success_rate": num_metric_success / n if n else None,
        "plif_mean": mean_or_none(finite_values(rows, "plif_tanimoto")),
        "strain_mean": mean_or_none(finite_values(rows, "strain_energy")),
        "vina_mean": mean_or_none(finite_values(rows, "vina_score")),
        "main_score_mean": mean_or_none(main_scores),
        "main_score_max": max(main_scores) if main_scores else None,
        "objective_mode": args.objective_mode,
        "multiobjective_strategy": args.multiobjective_strategy,
        "weights": {"plif": args.plif_weight, "strain": args.strain_weight, "vina": args.vina_weight},
        "thresholds": {
            "plif_min_threshold": args.plif_min_threshold,
            "strain_max_threshold": args.strain_max_threshold,
            "vina_max_threshold": args.vina_max_threshold,
            "strain_good_threshold": args.strain_good_threshold,
            "strain_bad_threshold": args.strain_bad_threshold,
            "vina_good_threshold": args.vina_good_threshold,
            "vina_bad_threshold": args.vina_bad_threshold,
        },
        "generated_dir": str(args.generated_dir),
        "eval_output": str(args.eval_output) if args.eval_output is not None else None,
        "metric_source": args.metric_source,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection": selection,
        "failure_reason_counts": failure_reason_counts(rows),
        "warnings": list(warnings),
    }
    return summary


def failure_reason_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reasons = str(row.get("failure_reason") or "").split(";")
        for reason in reasons:
            if not reason:
                continue
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def write_failed_log(rows: Sequence[Mapping[str, Any]], path: Optional[str | Path]) -> None:
    if path is None:
        return
    failed = [row for row in rows if row.get("failure_reason") or row.get("error")]
    write_json({"failed": failed}, path)


def write_sdf(rows: Sequence[Mapping[str, Any]], labels: set[str], path: Optional[str | Path]) -> None:
    if path is None:
        return
    from rdkit import Chem

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))
    try:
        for row in rows:
            if row.get("selection_label") not in labels:
                continue
            mol = row.get("mol")
            if mol is None:
                continue
            mol = Chem.Mol(mol)
            for key in CSV_COLUMNS:
                value = row.get(key)
                if value is not None:
                    mol.SetProp(key, str(value))
            writer.write(mol)
    finally:
        writer.close()


def strip_mol_objects(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = dict(row)
        item.pop("mol", None)
        output.append(item)
    return output


def main() -> None:
    args = build_parser().parse_args()
    files = prediction_files(args.generated_dir, max_files=args.max_prediction_files)
    source_fp = source_fingerprint(files)
    rewards, warnings = load_or_compute_rewards(args, files, source_fp)

    rewards = attach_molecules(rewards, files)
    selection = select_top_middle_bottom(
        rewards,
        objective_mode=args.objective_mode,
        multiobjective_strategy=args.multiobjective_strategy,
        top_ratio=args.top_ratio,
        bottom_ratio=args.bottom_ratio,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        feasible_only_for_top=args.feasible_only_for_top,
        include_failed_in_bottom=args.include_failed_in_bottom,
        main_score_key="main_score",
        plif_min_threshold=args.plif_min_threshold,
        strain_max_threshold=args.strain_max_threshold,
        vina_max_threshold=args.vina_max_threshold,
        plif_weight=args.plif_weight,
        strain_weight=args.strain_weight,
        vina_weight=args.vina_weight,
    )
    rows = enrich_records(rewards, selection, args.generated_dir)
    write_csv(rows, args.output_csv)
    write_failed_log(rows, args.failed_log)
    write_sdf(rows, {"top"}, args.output_top_sdf)
    write_sdf(rows, {"bottom"}, args.output_bottom_sdf)
    if args.top_summary_csv is not None:
        write_csv([row for row in rows if row.get("selection_label") == "top"], args.top_summary_csv)
    payload = summary_json(strip_mol_objects(rows), selection, args, warnings)
    write_json(payload, args.output_json)
    print(json.dumps({key: payload[key] for key in ("num_total", "num_top", "num_middle", "num_bottom")}, indent=2))


if __name__ == "__main__":
    main()
