from __future__ import annotations

"""
Support-aware reference screening for Ref-MPO experiments.

Purpose
-------
This script profiles candidate reference molecules from the test set by:
  1) filtering reasonable drug-like reference candidates;
  2) generating size-matched molecules from a pretrained prior;
  3) computing similarity/support statistics and Ref-MPO-A/B rewards;
  4) ranking references suitable for:
       - Experiment 2A: medium/high-support Ref-MPO
       - Experiment 2B: weak-support Ref-MPO

It is adapted from profile_ranolazine_prior_simple.py, but generalizes the
single fixed Ranolazine target into an automatic test-set reference selection
pipeline.
"""

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import lightning as L
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from sot_mol.comparm import GP, Update_PARAMS
from sot_mol.models.rl_lfpo_interface import MolGen_LFPOModel


# -----------------------------
# RDKit helpers
# -----------------------------

def load_sa_scorer():
    """Load RDKit SA scorer from RDKit Contrib."""
    try:
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore
        return sascorer
    except Exception:
        pass

    # Common fallback paths in conda/pip RDKit installs.
    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "share" / "RDKit" / "Contrib" / "SA_Score")
    candidates.append(Path(sys.prefix) / "share" / "RDKit" / "Contrib" / "SA_Score")

    for p in candidates:
        if p.exists():
            sys.path.append(str(p))
            try:
                import sascorer  # type: ignore
                return sascorer
            except Exception:
                continue

    raise ImportError(
        "Could not import RDKit SA_Score/sascorer. Install RDKit Contrib or add "
        "the SA_Score directory to PYTHONPATH."
    )


SA_SCORER = None


def get_sa_score(mol: Chem.Mol) -> float:
    global SA_SCORER
    if SA_SCORER is None:
        SA_SCORER = load_sa_scorer()
    return float(SA_SCORER.calculateScore(mol))


def sanitize_mol(mol: Chem.Mol | None) -> Chem.Mol | None:
    if mol is None:
        return None
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def canonical_smiles(mol: Chem.Mol | None) -> str:
    mol = standardize_for_scoring(mol, remove_hs=True)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def is_connected(mol: Chem.Mol | None) -> bool:
    if mol is None:
        return False
    try:
        return len(Chem.GetMolFrags(mol)) == 1
    except Exception:
        return False


def num_atoms(mol: Chem.Mol, with_h: bool = False) -> int:
    if with_h:
        return int(Chem.AddHs(mol).GetNumAtoms())
    return int(mol.GetNumHeavyAtoms())


def mol_props(mol: Chem.Mol) -> dict[str, float]:
    m = standardize_for_scoring(mol, remove_hs=True)
    if m is None:
        raise ValueError("Invalid molecule for property calculation")
    return {
        "qed": float(QED.qed(m)),
        "sa": float(get_sa_score(m)),
        "logp": float(Crippen.MolLogP(m)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(m)),
        "mw": float(Descriptors.MolWt(m)),
        "heavy_atoms": float(m.GetNumHeavyAtoms()),
    }


def morgan_fp(mol: Chem.Mol | None, radius: int = 2, nbits: int = 2048):
    m = standardize_for_scoring(mol, remove_hs=True)
    if m is None:
        return None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)


def tanimoto(mol: Chem.Mol | None, ref_fp, radius: int = 2, nbits: int = 2048) -> float:
    try:
        fp = morgan_fp(mol, radius=radius, nbits=nbits)
        if fp is None or ref_fp is None:
            return 0.0
        return float(DataStructs.TanimotoSimilarity(fp, ref_fp))
    except Exception:
        return 0.0


def murcko_scaffold_smiles(mol: Chem.Mol | None) -> str:
    m = standardize_for_scoring(mol, remove_hs=True)
    if m is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(scaf, canonical=True) if scaf is not None else ""
    except Exception:
        return ""


def clip01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def safe_geom_mean(vals: Iterable[float], eps: float = 1e-12) -> float:
    vals = [max(float(v), eps) for v in vals]
    if not vals:
        return 0.0
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


# -----------------------------
# MPO reward definitions
# -----------------------------

def r_qed(qed: float) -> float:
    # 0 at QED <= 0.4; 1 at QED >= 0.8
    return clip01((qed - 0.40) / (0.80 - 0.40))


def r_sa(sa: float) -> float:
    # Full reward at SA <= 3.0; exponential decay above 3.0.
    if not math.isfinite(sa):
        return 0.0
    return 1.0 if sa <= 3.0 else float(math.exp(-(sa - 3.0)))


def r_logp(logp: float) -> float:
    # Full reward in [1, 4]; exponential decay outside.
    if not math.isfinite(logp):
        return 0.0
    if 1.0 <= logp <= 4.0:
        return 1.0
    if logp < 1.0:
        return float(math.exp(-(1.0 - logp)))
    return float(math.exp(-(logp - 4.0)))


def r_tpsa_ref(tpsa: float, ref_tpsa: float) -> float:
    # Full reward within +-20; exponential decay beyond.
    if not math.isfinite(tpsa) or not math.isfinite(ref_tpsa):
        return 0.0
    return float(math.exp(-max(abs(tpsa - ref_tpsa) - 20.0, 0.0) / 20.0))


def r_size_ref(n_heavy: float, ref_heavy: float) -> float:
    if not math.isfinite(n_heavy) or not math.isfinite(ref_heavy):
        return 0.0
    return float(math.exp(-abs(n_heavy - ref_heavy) / 5.0))


def r_sim_A(sim: float) -> float:
    # Medium/high-support task: 0 at 0.20; full at 0.45.
    return clip01((sim - 0.20) / (0.45 - 0.20))


def r_sim_B(sim: float) -> float:
    # Weak-support task: 0 at 0.10; full at 0.35.
    return clip01((sim - 0.10) / (0.35 - 0.10))


def score_ref_mpo_A(row: dict[str, Any]) -> float:
    return safe_geom_mean([
        r_sim_A(row["sim_morgan"]),
        r_qed(row["qed"]),
        r_sa(row["sa"]),
        r_logp(row["logp"]),
    ])


def score_ref_mpo_B(row: dict[str, Any], ref: dict[str, Any]) -> float:
    return safe_geom_mean([
        r_sim_B(row["sim_morgan"]),
        r_qed(row["qed"]),
        r_sa(row["sa"]),
        r_tpsa_ref(row["tpsa"], ref["tpsa"]),
        r_size_ref(row["heavy_atoms"], ref["heavy_atoms"]),
    ])


def success_A(row: dict[str, Any], strict: bool = False) -> bool:
    sim_thr = 0.45 if strict else 0.40
    return (
        row["valid"]
        and row["sim_morgan"] > sim_thr
        and row["qed"] > 0.60
        and row["sa"] < 3.50
        and 1.0 < row["logp"] < 4.0
    )


def success_B(row: dict[str, Any], ref: dict[str, Any], strict: bool = False) -> bool:
    sim_thr = 0.35 if strict else 0.30
    return (
        row["valid"]
        and row["sim_morgan"] > sim_thr
        and row["qed"] > 0.60
        and row["sa"] < 3.50
        and abs(row["tpsa"] - ref["tpsa"]) < 20.0
        and abs(row["heavy_atoms"] - ref["heavy_atoms"]) <= 5
    )


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class CandidateRef:
    ref_id: int
    test_index: int
    smiles: str
    heavy_atoms: int
    atom_count_for_matching: int
    qed: float
    sa: float
    logp: float
    tpsa: float
    mw: float
    scaffold: str


# -----------------------------
# Model/data helpers
# -----------------------------

def mg_to_mol(mg) -> Chem.Mol | None:
    try:
        if hasattr(mg, "to_rdkit"):
            return standardize_for_scoring(mg.to_rdkit(), remove_hs=True)
    except Exception:
        return None
    return None


def mg_count_for_matching(mg, with_h: bool) -> int:
    mol = mg_to_mol(mg)
    if mol is not None:
        return num_atoms(mol, with_h=with_h)
    if hasattr(mg, "natoms"):
        return int(mg.natoms)
    return 0


def build_model_and_data(args):
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    gp = Update_PARAMS(GP, str(config_path))
    os.environ["CUDA_VISIBLE_DEVICES"] = gp.CUDA_VISIBLE_DEVICES

    L.seed_everything(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = MolGen_LFPOModel(
        d_model=gp.D_MODEL,
        atom_tokens=gp.TOKENS,
        n_bond_types=gp.N_BOND_TYPES,
        coord_std=gp.COORDS_STD_DEV,
        scale_ot=gp.SCALE_OT,
        self_cond=True,
        coord_noise_std=0.2,
        formulation="endpoint",
        eval_3D_props=False,
        ot_bond_weight=1,
        objective_name="QED_SA",  # only a label; scoring here is done manually.
    )
    lm = model.create_lightning_module(load_ckpt=args.load_ckpt)
    lm.eval().to(device)

    datasets_dir = Path(args.datasets_dir) if args.datasets_dir else Path(__file__).resolve().parent.parent / "datasets"
    dm_cls = __import__("sot_mol.data.datamodule", fromlist=["MGDataModule"]).MGDataModule
    model.data_module = dm_cls(
        model.vocab,
        model.n_bond_types,
        train_datafile=datasets_dir / "train.smol",
        val_datafile=datasets_dir / "val.smol",
        test_datafile=datasets_dir / "test.smol",
        max_atoms=model.max_atoms,
        coord_std=model.coord_std,
        scale_ot=model.scale_ot,
        scale_ot_factor=0.2,
        batchsize=args.batch_size,
        mini_batchsize=args.mini_batch_size,
        with_Hs=model.with_Hs,
        ot_geo_weight=model.ot_geo_weight,
        ot_type_weight=model.ot_type_weight,
        ot_bond_weight=model.ot_bond_weight,
    )
    model.data_module.setup(stage="test")
    full_test_mgs = list(model.data_module.testset.MGs)
    return model, lm, full_test_mgs, device


def collect_reference_candidates(args, full_test_mgs) -> list[CandidateRef]:
    candidates: list[CandidateRef] = []
    seen_smiles: set[str] = set()

    indices = list(range(len(full_test_mgs)))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)

    for idx in indices:
        mol = mg_to_mol(full_test_mgs[idx])
        if mol is None:
            continue
        smi = canonical_smiles(mol)
        if not smi or smi in seen_smiles:
            continue
        seen_smiles.add(smi)

        try:
            props = mol_props(mol)
        except Exception:
            continue

        h = int(props["heavy_atoms"])
        if not (args.ref_min_heavy <= h <= args.ref_max_heavy):
            continue
        if not (args.ref_min_qed <= props["qed"] <= args.ref_max_qed):
            continue
        if props["sa"] > args.ref_max_sa:
            continue
        if not (args.ref_min_logp <= props["logp"] <= args.ref_max_logp):
            continue
        if not (args.ref_min_tpsa <= props["tpsa"] <= args.ref_max_tpsa):
            continue

        candidates.append(
            CandidateRef(
                ref_id=len(candidates),
                test_index=idx,
                smiles=smi,
                heavy_atoms=h,
                atom_count_for_matching=mg_count_for_matching(full_test_mgs[idx], args.atom_count_with_h),
                qed=props["qed"],
                sa=props["sa"],
                logp=props["logp"],
                tpsa=props["tpsa"],
                mw=props["mw"],
                scaffold=murcko_scaffold_smiles(mol),
            )
        )
        if len(candidates) >= args.max_candidate_refs:
            break

    return candidates

def standardize_for_scoring(mol: Chem.Mol | None, remove_hs: bool = True) -> Chem.Mol | None:
    if mol is None:
        return None
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m)
        if remove_hs:
            m = Chem.RemoveHs(m, sanitize=True)
            Chem.SanitizeMol(m)
        return m
    except Exception:
        return None

def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def parse_bool_like(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def candidate_from_summary_row(row: dict[str, Any]) -> CandidateRef:
    """Reconstruct a CandidateRef from reference_profile_summary.csv.

    This enables a two-step workflow: run a cheap coarse screen once, inspect or
    select references, then fine-profile only selected references without
    regenerating the coarse screen.
    """
    return CandidateRef(
        ref_id=int(row["ref_id"]),
        test_index=int(row["test_index"]),
        smiles=str(row["ref_smiles"]),
        heavy_atoms=int(float(row["ref_heavy_atoms"])),
        atom_count_for_matching=int(float(row["ref_atom_count_for_matching"])),
        qed=float(row["ref_qed"]),
        sa=float(row["ref_sa"]),
        logp=float(row["ref_logp"]),
        tpsa=float(row["ref_tpsa"]),
        mw=float(row["ref_mw"]),
        scaffold=str(row.get("ref_scaffold", "")),
    )


def select_summary_rows_for_fine(
    rows: list[dict[str, Any]],
    selection: str,
    top_k_per_group: int,
    explicit_ids: str = "",
) -> list[dict[str, Any]]:
    """Select rows from a coarse summary CSV for fine profiling."""
    if selection == "ids":
        ids = {int(x.strip()) for x in explicit_ids.split(",") if x.strip()}
        return [r for r in rows if int(r["ref_id"]) in ids]

    k = max(1, int(top_k_per_group))
    selected: list[dict[str, Any]] = []

    if selection in {"rankA", "rankAB"}:
        selected.extend(sorted(rows, key=lambda r: float(r["rank_A"]), reverse=True)[:k])
    if selection in {"rankB", "rankAB"}:
        selected.extend(sorted(rows, key=lambda r: float(r["rank_B"]), reverse=True)[:k])
    if selection in {"labelA", "labelAB"}:
        labelA = [r for r in rows if parse_bool_like(r.get("is_exp2A_medium_high_support", False))]
        selected.extend(sorted(labelA, key=lambda r: float(r["rank_A"]), reverse=True)[:k])
    if selection in {"labelB", "labelAB"}:
        labelB = [r for r in rows if parse_bool_like(r.get("is_exp2B_weak_support", False))]
        selected.extend(sorted(labelB, key=lambda r: float(r["rank_B"]), reverse=True)[:k])

    # De-duplicate while preserving order.
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for r in selected:
        rid = int(r["ref_id"])
        if rid in seen:
            continue
        seen.add(rid)
        out.append(r)
    return out


def start_pool_for_ref(
    ref: CandidateRef,
    full_test_mgs,
    test_atom_counts: list[int],
    tolerance: int,
    exclude_ref: bool,
) -> list[int]:
    pool = []
    for i, c in enumerate(test_atom_counts):
        if exclude_ref and i == ref.test_index:
            continue
        if abs(c - ref.atom_count_for_matching) <= tolerance:
            pool.append(i)
    return pool


def generate_for_reference(
    args,
    model,
    lm,
    full_test_mgs,
    start_pool_indices: list[int],
    num_samples: int,
) -> list[Chem.Mol | None]:
    out: list[Chem.Mol | None] = []
    generated = 0
    chunk = 0
    while generated < num_samples:
        chunk += 1
        cur_n = min(args.gen_chunk_size, num_samples - generated)
        try:
            chosen = np.random.choice(
                start_pool_indices,
                size=cur_n,
                replace=(cur_n > len(start_pool_indices)),
            )
            model.data_module.testset.MGs = [full_test_mgs[int(i)] for i in chosen]
            with torch.inference_mode():
                cur_mols, _ = model.generate_molecules(lm, model.data_module, model.max_steps, stabilities=False)
            out.extend(cur_mols[:cur_n])
        except torch.OutOfMemoryError:
            print("[OOM] CUDA out of memory during generation. Reduce --batch_size or --gen_chunk_size.")
            raise SystemExit(1)
        finally:
            generated += cur_n
            if args.cleanup_every > 0 and (chunk % args.cleanup_every == 0):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
    return out[:num_samples]


def score_generated_for_ref(ref: CandidateRef, mols: list[Chem.Mol | None]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # ref_mol = Chem.MolFromSmiles(ref.smiles)
    ref_mol = standardize_for_scoring(Chem.MolFromSmiles(ref.smiles), remove_hs=True)
    if ref_mol is None:
        raise ValueError(f"Invalid reference SMILES: {ref.smiles}")
    ref_fp = morgan_fp(ref_mol)
    ref_dict = asdict(ref)

    rows: list[dict[str, Any]] = []
    for i, mol0 in enumerate(mols):
        # mol = sanitize_mol(mol0)
        mol = standardize_for_scoring(mol0, remove_hs=True)
        valid = mol is not None
        row: dict[str, Any] = {
            "local_idx": i,
            "valid": valid,
            "connected": is_connected(mol),
            "smiles": canonical_smiles(mol),
            "sim_morgan": float("nan"),
            "qed": float("nan"),
            "sa": float("nan"),
            "logp": float("nan"),
            "tpsa": float("nan"),
            "mw": float("nan"),
            "heavy_atoms": float("nan"),
            "scaffold": "",
        }
        if valid and mol is not None:
            try:
                props = mol_props(mol)
                row.update(props)
                row["sim_morgan"] = tanimoto(mol, ref_fp)
                row["scaffold"] = murcko_scaffold_smiles(mol)
            except Exception:
                row["valid"] = False
                row["connected"] = False
        row["score_A"] = score_ref_mpo_A(row) if row["valid"] else 0.0
        row["score_B"] = score_ref_mpo_B(row, ref_dict) if row["valid"] else 0.0
        row["success_A"] = success_A(row, strict=False)
        row["success_A_strict"] = success_A(row, strict=True)
        row["success_B"] = success_B(row, ref_dict, strict=False)
        row["success_B_strict"] = success_B(row, ref_dict, strict=True)
        rows.append(row)

    summary = summarize_rows(ref, rows)
    return rows, summary


def frac(vals: list[bool]) -> float:
    return float(sum(vals) / max(1, len(vals)))


def top_mean(vals: np.ndarray, frac_or_n: float | int) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    vals = np.sort(vals)[::-1]
    if isinstance(frac_or_n, float):
        n = max(1, int(math.ceil(vals.size * frac_or_n)))
    else:
        n = int(frac_or_n)
    return float(vals[: min(n, vals.size)].mean())


def summarize_rows(ref: CandidateRef, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [r for r in rows if r["valid"]]
    sim_vals = np.array([r["sim_morgan"] for r in valid_rows], dtype=float) if valid_rows else np.array([])
    score_A_vals = np.array([r["score_A"] for r in valid_rows], dtype=float) if valid_rows else np.array([])
    score_B_vals = np.array([r["score_B"] for r in valid_rows], dtype=float) if valid_rows else np.array([])

    def prop_mean(name: str) -> float:
        if not valid_rows:
            return 0.0
        vals = np.array([r[name] for r in valid_rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else 0.0

    out: dict[str, Any] = {
        "ref_id": ref.ref_id,
        "test_index": ref.test_index,
        "ref_smiles": ref.smiles,
        "ref_heavy_atoms": ref.heavy_atoms,
        "ref_atom_count_for_matching": ref.atom_count_for_matching,
        "ref_qed": ref.qed,
        "ref_sa": ref.sa,
        "ref_logp": ref.logp,
        "ref_tpsa": ref.tpsa,
        "ref_mw": ref.mw,
        "ref_scaffold": ref.scaffold,
        "num_samples": len(rows),
        "num_valid": len(valid_rows),
        "validity": float(len(valid_rows) / max(1, len(rows))),
        "connected_frac_valid": float(sum(r["connected"] for r in valid_rows) / max(1, len(valid_rows))),
        "sim_mean": float(sim_vals.mean()) if sim_vals.size else 0.0,
        "sim_max": float(sim_vals.max()) if sim_vals.size else 0.0,
        "sim_top1pct_mean": top_mean(sim_vals, 0.01),
        "sim_top5pct_mean": top_mean(sim_vals, 0.05),
        "sim_top10_mean": top_mean(sim_vals, 10),
        "sim_top100_mean": top_mean(sim_vals, 100),
        "frac_sim_gt_0p20": float(np.mean(sim_vals > 0.20)) if sim_vals.size else 0.0,
        "frac_sim_gt_0p30": float(np.mean(sim_vals > 0.30)) if sim_vals.size else 0.0,
        "frac_sim_gt_0p35": float(np.mean(sim_vals > 0.35)) if sim_vals.size else 0.0,
        "frac_sim_gt_0p40": float(np.mean(sim_vals > 0.40)) if sim_vals.size else 0.0,
        "frac_sim_gt_0p45": float(np.mean(sim_vals > 0.45)) if sim_vals.size else 0.0,
        "qed_mean": prop_mean("qed"),
        "sa_mean": prop_mean("sa"),
        "logp_mean": prop_mean("logp"),
        "tpsa_mean": prop_mean("tpsa"),
        "score_A_mean": float(score_A_vals.mean()) if score_A_vals.size else 0.0,
        "score_A_top100_mean": top_mean(score_A_vals, 100),
        "score_B_mean": float(score_B_vals.mean()) if score_B_vals.size else 0.0,
        "score_B_top100_mean": top_mean(score_B_vals, 100),
        "success_A_frac_valid": float(sum(r["success_A"] for r in valid_rows) / max(1, len(valid_rows))),
        "success_A_strict_frac_valid": float(sum(r["success_A_strict"] for r in valid_rows) / max(1, len(valid_rows))),
        "success_B_frac_valid": float(sum(r["success_B"] for r in valid_rows) / max(1, len(valid_rows))),
        "success_B_strict_frac_valid": float(sum(r["success_B_strict"] for r in valid_rows) / max(1, len(valid_rows))),
        "qed_gt_0p6_frac_valid": float(sum((r["qed"] > 0.60) for r in valid_rows) / max(1, len(valid_rows))),
        "sa_lt_3p5_frac_valid": float(sum((r["sa"] < 3.50) for r in valid_rows) / max(1, len(valid_rows))),
        "logp_1_4_frac_valid": float(sum((1.0 < r["logp"] < 4.0) for r in valid_rows) / max(1, len(valid_rows))),
        "tpsa_match_20_frac_valid": float(sum((abs(r["tpsa"] - ref.tpsa) < 20.0) for r in valid_rows) / max(1, len(valid_rows))),
        "size_match_5_frac_valid": float(sum((abs(r["heavy_atoms"] - ref.heavy_atoms) <= 5) for r in valid_rows) / max(1, len(valid_rows))),
    }

    out.update(classify_reference(out))
    return out


def classify_reference(s: dict[str, Any]) -> dict[str, Any]:
    """Assign strict labels and continuous ranking scores for Exp 2A/2B."""
    max_sim = s["sim_max"]
    top1 = s["sim_top1pct_mean"]
    p30 = s["frac_sim_gt_0p30"]
    p35 = s["frac_sim_gt_0p35"]
    p40 = s["frac_sim_gt_0p40"]
    p45 = s["frac_sim_gt_0p45"]
    succA = s["success_A_frac_valid"]
    succB = s["success_B_frac_valid"]

    is_A = (
        max_sim >= 0.50
        and top1 >= 0.38
        and p35 >= 0.02
        and p45 >= 0.002
        and 0.03 <= succA <= 0.20
    )
    is_B = (
        0.35 <= max_sim < 0.45
        and 0.25 <= top1 < 0.35
        and 0.003 <= p30 <= 0.02
        and p40 < 0.002
        and 0.005 <= succB <= 0.08
    )

    # Ranking scores are softer than labels; useful when no candidate perfectly matches thresholds.
    # A favors high but not trivial support and non-trivial MPO difficulty.
    rank_A = (
        2.0 * min(max_sim, 0.60)
        + 2.0 * min(top1, 0.50)
        + 10.0 * min(p35, 0.05)
        + 20.0 * min(p45, 0.01)
        - 2.0 * abs(succA - 0.10)
    )
    # B favors weak-but-nonzero support.
    rank_B = (
        2.0 * (1.0 - abs(max_sim - 0.40) / 0.40)
        + 2.0 * (1.0 - abs(top1 - 0.30) / 0.30)
        + 30.0 * min(p30, 0.02)
        - 20.0 * max(p40 - 0.002, 0.0)
        - 2.0 * abs(succB - 0.03)
    )
    return {
        "is_exp2A_medium_high_support": bool(is_A),
        "is_exp2B_weak_support": bool(is_B),
        "rank_A": float(rank_A),
        "rank_B": float(rank_B),
    }


def select_rows(summary_rows: list[dict[str, Any]], key: str, n: int) -> list[dict[str, Any]]:
    return sorted(summary_rows, key=lambda r: r[key], reverse=True)[:n]


def run_screen(args, model, lm, full_test_mgs, candidates: list[CandidateRef], samples_per_ref: int, tag: str):
    outdir = Path(args.output_dir) / tag
    outdir.mkdir(parents=True, exist_ok=True)

    test_atom_counts = [mg_count_for_matching(mg, args.atom_count_with_h) for mg in full_test_mgs]
    summaries: list[dict[str, Any]] = []

    for j, ref in enumerate(candidates, start=1):
        pool = start_pool_for_ref(
            ref,
            full_test_mgs,
            test_atom_counts,
            tolerance=args.atom_count_tolerance,
            exclude_ref=args.exclude_reference_from_start_pool,
        )
        if len(pool) < args.min_start_pool_size:
            print(
                f"[WARN] ref_id={ref.ref_id} start pool too small: {len(pool)} < {args.min_start_pool_size}; skipping."
            )
            continue

        print(
            f"[{tag}] {j}/{len(candidates)} ref_id={ref.ref_id} test_idx={ref.test_index} "
            f"atoms={ref.atom_count_for_matching} pool={len(pool)} samples={samples_per_ref}"
        )
        mols = generate_for_reference(args, model, lm, full_test_mgs, pool, samples_per_ref)
        rows, summary = score_generated_for_ref(ref, mols)
        summary["start_pool_size"] = len(pool)
        summary["start_pool_atom_min"] = int(min(test_atom_counts[i] for i in pool))
        summary["start_pool_atom_mean"] = float(np.mean([test_atom_counts[i] for i in pool]))
        summary["start_pool_atom_max"] = int(max(test_atom_counts[i] for i in pool))
        summaries.append(summary)

        if args.save_per_ref_csv:
            ref_path = outdir / "per_reference" / f"ref_{ref.ref_id:04d}_generated.csv"
            rows_to_write = []
            for r in rows:
                rr = {
                    "ref_id": ref.ref_id,
                    "test_index": ref.test_index,
                    "ref_smiles": ref.smiles,
                    **r,
                }
                rows_to_write.append(rr)
            write_csv(ref_path, rows_to_write)

    summary_path = outdir / "reference_profile_summary.csv"
    write_csv(summary_path, summaries)
    (outdir / "reference_profile_summary.json").write_text(json.dumps(summaries, indent=2))

    selected_A_label = [r for r in summaries if r["is_exp2A_medium_high_support"]]
    selected_B_label = [r for r in summaries if r["is_exp2B_weak_support"]]
    selected_A_rank = select_rows(summaries, "rank_A", args.select_top_k)
    selected_B_rank = select_rows(summaries, "rank_B", args.select_top_k)

    write_csv(outdir / "selected_exp2A_label_matches.csv", selected_A_label)
    write_csv(outdir / "selected_exp2B_label_matches.csv", selected_B_label)
    write_csv(outdir / "selected_exp2A_ranked_top.csv", selected_A_rank)
    write_csv(outdir / "selected_exp2B_ranked_top.csv", selected_B_rank)

    print(f"[{tag}] saved summary: {summary_path}")
    print(f"[{tag}] strict Exp2A label matches: {len(selected_A_label)}")
    print(f"[{tag}] strict Exp2B label matches: {len(selected_B_label)}")
    print(f"[{tag}] top-ranked Exp2A refs:", [r["ref_id"] for r in selected_A_rank])
    print(f"[{tag}] top-ranked Exp2B refs:", [r["ref_id"] for r in selected_B_rank])
    return summaries, selected_A_rank, selected_B_rank


def main():
    ap = argparse.ArgumentParser()

    # Model / generation args
    ap.add_argument("--config", type=str, default="rl.json")
    ap.add_argument("--load_ckpt", type=str, required=True)
    ap.add_argument("--datasets_dir", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--mini_batch_size", type=int, default=1,
                    help="MGDataModule mini_batchsize. Increase if generation is GPU-underutilized and memory allows.")
    ap.add_argument("--gen_chunk_size", type=int, default=100)
    ap.add_argument("--cleanup_every", type=int, default=20,
                    help="Run torch.cuda.empty_cache() and gc.collect() every N generation chunks. 0 disables explicit cleanup.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)

    # Output args
    ap.add_argument("--output_dir", type=str, default="prior_profile/Ref_MPO_support")
    ap.add_argument("--save_per_ref_csv", action="store_true")

    # Candidate reference filtering
    ap.add_argument("--max_candidate_refs", type=int, default=200)
    ap.add_argument("--ref_min_heavy", type=int, default=20)
    ap.add_argument("--ref_max_heavy", type=int, default=45)
    ap.add_argument("--ref_min_qed", type=float, default=0.45)
    ap.add_argument("--ref_max_qed", type=float, default=0.85)
    ap.add_argument("--ref_max_sa", type=float, default=4.5)
    ap.add_argument("--ref_min_logp", type=float, default=1.0)
    ap.add_argument("--ref_max_logp", type=float, default=5.0)
    ap.add_argument("--ref_min_tpsa", type=float, default=30.0)
    ap.add_argument("--ref_max_tpsa", type=float, default=130.0)

    # Size-matched profiling
    ap.add_argument("--atom_count_tolerance", type=int, default=5)
    ap.add_argument("--atom_count_with_h", action="store_true")
    ap.add_argument("--min_start_pool_size", type=int, default=100)
    ap.add_argument("--exclude_reference_from_start_pool", action="store_true")

    # Coarse/fine profiling
    ap.add_argument("--samples_per_ref", type=int, default=500)
    ap.add_argument("--fine_samples_per_ref", type=int, default=5000)
    ap.add_argument("--fine_top_k_per_group", type=int, default=0)
    ap.add_argument("--select_top_k", type=int, default=10)

    # Two-step workflow: fine-profile references selected from an existing coarse CSV.
    # If --fine_from_csv is provided, the script skips the coarse screening stage.
    ap.add_argument("--fine_from_csv", type=str, default=None,
                    help="Path to an existing coarse/reference_profile_summary.csv. If set, skip coarse screening and fine-profile selected refs from this CSV.")
    ap.add_argument("--fine_selection", type=str, default="rankAB",
                    choices=["rankA", "rankB", "rankAB", "labelA", "labelB", "labelAB", "ids"],
                    help="How to select references from --fine_from_csv.")
    ap.add_argument("--fine_ref_ids", type=str, default="",
                    help="Comma-separated ref_id list when --fine_selection ids is used.")

    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps(vars(args), indent=2))

    model, lm, full_test_mgs, device = build_model_and_data(args)
    print(f"Loaded test MGs: {len(full_test_mgs)}")

    if args.fine_from_csv:
        coarse_csv = Path(args.fine_from_csv)
        rows = read_csv_dicts(coarse_csv)
        selected_rows = select_summary_rows_for_fine(
            rows,
            selection=args.fine_selection,
            top_k_per_group=args.fine_top_k_per_group or args.select_top_k,
            explicit_ids=args.fine_ref_ids,
        )
        fine_candidates = [candidate_from_summary_row(r) for r in selected_rows]
        print(f"Loaded coarse summary: {coarse_csv}")
        print(f"Fine selection mode: {args.fine_selection}")
        print(f"Fine profiling selected refs: {[c.ref_id for c in fine_candidates]}")
        if not fine_candidates:
            print("No references selected for fine profiling. Check --fine_selection/--fine_ref_ids.")
            return
        run_screen(
            args,
            model,
            lm,
            full_test_mgs,
            fine_candidates,
            args.fine_samples_per_ref,
            tag="fine_from_csv",
        )
        print("Done.")
        print(f"Outputs saved under: {outdir}")
        return

    candidates = collect_reference_candidates(args, full_test_mgs)
    print(f"Candidate references after property filters: {len(candidates)}")
    candidate_rows = [asdict(c) for c in candidates]
    write_csv(outdir / "candidate_references.csv", candidate_rows)

    if not candidates:
        print("No reference candidates. Relax --ref_* filters.")
        return

    summaries, topA, topB = run_screen(
        args, model, lm, full_test_mgs, candidates, args.samples_per_ref, tag="coarse"
    )

    # Optional fine profiling on top-ranked candidates from each group.
    # This uses the coarse results generated in this same process. If you want to
    # run fine profiling later from an already-saved coarse CSV, use --fine_from_csv.
    if args.fine_top_k_per_group and args.fine_top_k_per_group > 0:
        selected_ids = []
        for r in topA[: args.fine_top_k_per_group] + topB[: args.fine_top_k_per_group]:
            selected_ids.append(int(r["ref_id"]))
        selected_ids = sorted(set(selected_ids))
        id_to_ref = {c.ref_id: c for c in candidates}
        fine_candidates = [id_to_ref[i] for i in selected_ids if i in id_to_ref]
        print(f"Fine profiling selected refs: {selected_ids}")
        run_screen(
            args,
            model,
            lm,
            full_test_mgs,
            fine_candidates,
            args.fine_samples_per_ref,
            tag="fine",
        )

    print("Done.")
    print(f"Outputs saved under: {outdir}")


if __name__ == "__main__":
    main()
