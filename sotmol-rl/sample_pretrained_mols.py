from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path

import lightning as L
import torch
from rdkit import Chem

from sot_mol.comparm import GP, Update_PARAMS
from sot_mol.models.rl_lfpo_interface import MolGen_LFPOModel
from sot_mol.data.datamodule import MGDataModule


def build_model(gp, objective_name: str) -> MolGen_LFPOModel:
    return MolGen_LFPOModel(
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
        objective_name=objective_name,
    )


def build_datamodule(model: MolGen_LFPOModel, datasets_dir: Path, batch_size: int) -> MGDataModule:
    dm = MGDataModule(
        model.vocab,
        model.n_bond_types,
        train_datafile=datasets_dir / "train.smol",
        val_datafile=datasets_dir / "val.smol",
        test_datafile=datasets_dir / "test.smol",
        max_atoms=model.max_atoms,
        coord_std=model.coord_std,
        scale_ot=model.scale_ot,
        scale_ot_factor=0.2,
        batchsize=batch_size,
        mini_batchsize=1,
        with_Hs=model.with_Hs,
        ot_geo_weight=model.ot_geo_weight,
        ot_type_weight=model.ot_type_weight,
        ot_bond_weight=model.ot_bond_weight,
    )
    dm.setup(stage="test")
    return dm


def cuda_cleanup(device: torch.device) -> None:
    gc.collect()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample molecules from a pretrained SOT-Mol/LFPO checkpoint and stream outputs to disk."
    )
    ap.add_argument("--config", type=str, default="rl.json")
    ap.add_argument("--load_ckpt", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--gen_chunk_size", type=int, default=128)
    ap.add_argument("--output_dir", type=str, default="samples/pretrained")
    ap.add_argument("--sdf_name", type=str, default="generated.sdf")
    ap.add_argument("--csv_name", type=str, default="generated_smiles.csv")
    ap.add_argument("--summary_name", type=str, default="summary.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--datasets_dir", type=str, default=None)
    ap.add_argument(
        "--objective_name",
        type=str,
        default="QED",
        help="Kept only for model-wrapper compatibility. No scoring/filtering is performed.",
    )
    args = ap.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    if args.gen_chunk_size <= 0:
        raise ValueError("--gen_chunk_size must be positive.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    sdf_path = outdir / args.sdf_name
    csv_path = outdir / args.csv_name
    summary_path = outdir / args.summary_name

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    gp = Update_PARAMS(GP, str(config_path))
    if hasattr(gp, "CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gp.CUDA_VISIBLE_DEVICES)

    L.seed_everything(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = build_model(gp, objective_name=args.objective_name)
    lm = model.create_lightning_module(load_ckpt=args.load_ckpt)
    lm.eval().to(device)
    lm.zero_grad(set_to_none=True)

    datasets_dir = Path(args.datasets_dir) if args.datasets_dir is not None else script_dir.parent / "datasets"
    model.data_module = build_datamodule(model, datasets_dir=datasets_dir, batch_size=args.batch_size)

    # Important: keep a CPU-side copy of all test MolGraphs once. Do not call dm.setup() in every chunk;
    # repeated setup reparses the whole .smol file and can cause unnecessary memory growth.
    full_test_mgs = list(model.data_module.testset.MGs)
    if len(full_test_mgs) == 0:
        raise RuntimeError(f"No test MolGraphs found in {datasets_dir / 'test.smol'}")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)

    total_records = 0
    total_non_none = 0
    total_valid_smiles = 0
    chunk_idx = 0

    sdf_writer = Chem.SDWriter(str(sdf_path))
    with csv_path.open("w", newline="") as f_csv:
        csv_writer = csv.DictWriter(f_csv, fieldnames=["idx", "valid", "smiles"])
        csv_writer.writeheader()

        while total_records < args.num_samples:
            chunk_idx += 1
            cur_n = min(args.gen_chunk_size, args.num_samples - total_records)

            # Sample molecular sizes/start graphs from the already-loaded test pool.
            indices = torch.randint(len(full_test_mgs), (cur_n,), generator=rng).tolist()
            model.data_module.testset.MGs = [full_test_mgs[int(i)] for i in indices]

            cuda_cleanup(device)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            try:
                with torch.no_grad():
                    gen_result = model.generate_molecules(
                        lm,
                        model.data_module,
                        model.max_steps,
                        stabilities=False,
                    )
                cur_mols = list(gen_result[0][:cur_n])
                # Drop the second return value immediately. Some implementations return tensors/diagnostics there.
                del gen_result
            except torch.OutOfMemoryError as exc:
                print("[OOM] CUDA out of memory during generation.")
                print("Reduce --gen_chunk_size first; if still OOM, reduce --batch_size.")
                raise SystemExit(1) from exc

            chunk_non_none = 0
            chunk_valid_smiles = 0

            for mol in cur_mols:
                global_idx = total_records
                total_records += 1

                valid = False
                smiles = ""
                if mol is not None:
                    chunk_non_none += 1
                    total_non_none += 1
                    try:
                        smiles = Chem.MolToSmiles(mol)
                        valid = True
                    except Exception:
                        smiles = ""
                        valid = False

                    # Write every non-None molecule to SDF, even if SMILES conversion failed.
                    m = Chem.Mol(mol)
                    m.SetProp("idx", str(global_idx))
                    sdf_writer.write(m)

                if valid:
                    chunk_valid_smiles += 1
                    total_valid_smiles += 1

                csv_writer.writerow({"idx": global_idx, "valid": bool(valid), "smiles": smiles})

            f_csv.flush()
            del cur_mols
            cuda_cleanup(device)

            if torch.cuda.is_available() and device.type == "cuda":
                alloc = torch.cuda.memory_allocated() / (1024**2)
                reserved = torch.cuda.memory_reserved() / (1024**2)
                peak = torch.cuda.max_memory_allocated() / (1024**2)
                print(
                    f"[chunk {chunk_idx}] generated {total_records}/{args.num_samples}, "
                    f"non_none {chunk_non_none}/{cur_n}, valid_smiles {chunk_valid_smiles}/{cur_n}, "
                    f"cuda allocated/reserved/peak(MB)={alloc:.1f}/{reserved:.1f}/{peak:.1f}"
                )
            else:
                print(
                    f"[chunk {chunk_idx}] generated {total_records}/{args.num_samples}, "
                    f"non_none {chunk_non_none}/{cur_n}, valid_smiles {chunk_valid_smiles}/{cur_n}"
                )

    sdf_writer.close()

    summary = {
        "load_ckpt": str(args.load_ckpt),
        "config": str(config_path),
        "datasets_dir": str(datasets_dir),
        "num_samples_requested": int(args.num_samples),
        "num_records": int(total_records),
        "num_non_none_mols": int(total_non_none),
        "num_valid_smiles": int(total_valid_smiles),
        "sdf_path": str(sdf_path),
        "csv_path": str(csv_path),
        "seed": int(args.seed),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "gen_chunk_size": int(args.gen_chunk_size),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Saved:")
    print(f"- SDF: {sdf_path}")
    print(f"- SMILES CSV: {csv_path}")
    print(f"- summary: {summary_path}")


if __name__ == "__main__":
    main()
