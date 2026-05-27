from dataclasses import dataclass
from typing import Any
import warnings
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Crippen, rdMolDescriptors, AllChem, rdFreeSASA
from rdkit.Chem.AtomPairs import Pairs
from .desirability import gaussian, max_gaussian, min_gaussian, thresholded, weighted_geometric_mean, weighted_linear_sum, tchebycheff_score
from .diversity import canonical_smiles, murcko_scaffold_smiles, morgan_fp

@dataclass
class ScoringResult:
    score: torch.Tensor
    component_scores: dict[str, torch.Tensor]
    raw_properties: dict[str, torch.Tensor]
    feasible: torch.BoolTensor
    severe_violation: torch.BoolTensor
    valid: torch.BoolTensor
    connected: torch.BoolTensor
    smiles: list[str | None]
    canonical_smiles: list[str | None]
    scaffolds: list[str | None]
    fps: list[Any]
    mols: list[Any]
    metadata: dict

class BaseObjective:
    name = "base"
    component_names = []
    pareto_component_names = []
    def __init__(self, cfg=None): self.cfg = cfg or {}
    def score_mols(self, mols, device=None, dtype=torch.float32): raise NotImplementedError



# -----------------------------------------------------------------------------
# xTB force-reward helpers
# -----------------------------------------------------------------------------
def _xtb_force_worker(payload):
    """Compute a single-molecule xTB force RMS in a subprocess-safe way.

    The payload is intentionally a tuple/list of plain Python objects so this
    function can be used with ProcessPoolExecutor. The reward follows the
    convention in xtb_reward_workflow.md for the raw physical cost. The worker
    returns force_rms and legacy neg-force reward; the objective can then map it
    to a bounded 0-1 score.
    """
    symbols, positions, method, force_norm, fail_penalty, reward_scale = payload
    try:
        import numpy as _np
        from ase import Atoms as _Atoms
        from xtb.ase.calculator import XTB as _XTB

        atoms = _Atoms(symbols=list(symbols), positions=_np.asarray(positions, dtype=float))
        atoms.calc = _XTB(method=str(method))
        forces = _np.asarray(atoms.get_forces(), dtype=float)
        energy = float(atoms.get_potential_energy())
        if forces.ndim != 2 or forces.shape[1] != 3 or (not _np.all(_np.isfinite(forces))):
            raise RuntimeError("xTB returned invalid force array")

        if str(force_norm).lower() in {"component_rms", "component", "xyz_rms"}:
            force_rms = float(_np.sqrt(_np.mean(forces ** 2)))
        else:
            # sqrt(1/N * sum_i ||F_i||^2), matching the workflow document.
            force_rms = float(_np.sqrt(_np.mean(_np.sum(forces ** 2, axis=1))))

        reward = -float(reward_scale) * force_rms
        return {
            "reward": reward,
            "force_rms": force_rms,
            "energy": energy,
            "success": True,
            "error": "",
        }
    except Exception as exc:
        fail_force = abs(float(fail_penalty)) / max(abs(float(reward_scale)), 1e-12)
        return {
            "reward": float(fail_penalty),
            "force_rms": float(fail_force),
            "energy": 0.0,
            "success": False,
            "error": repr(exc)[:300],
        }


def _xtb_process_entry(payload, queue):
    """Child-process entry used to enforce a real per-molecule timeout.

    ASE/xTB calls can occasionally hang inside native code on malformed generated
    geometries. concurrent.futures timeout does not kill a stuck worker, so we run
    each molecule in a short-lived child process when timeout is enabled.
    """
    try:
        queue.put(_xtb_force_worker(payload))
    except Exception as exc:
        queue.put({
            "reward": float(payload[4]),
            "force_rms": abs(float(payload[4])) / max(abs(float(payload[5])), 1e-12),
            "energy": 0.0,
            "success": False,
            "error": repr(exc)[:300],
        })


def _xtb_failure_result(payload, error="xTB failed"):
    fail_penalty = float(payload[4])
    reward_scale = float(payload[5])
    fail_force = abs(fail_penalty) / max(abs(reward_scale), 1e-12)
    return {
        "reward": fail_penalty,
        "force_rms": fail_force,
        "energy": 0.0,
        "success": False,
        "error": str(error)[:300],
    }


def _run_xtb_force_job_hard_timeout(payload, timeout):
    """Run one xTB job with a killable wall-time timeout."""
    if timeout is None or float(timeout) <= 0:
        return _xtb_force_worker(payload)

    import multiprocessing as _mp

    try:
        ctx = _mp.get_context("fork")
    except ValueError:
        ctx = _mp.get_context()

    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_xtb_process_entry, args=(payload, queue))
    proc.daemon = True
    proc.start()
    proc.join(float(timeout))

    if proc.is_alive():
        proc.terminate()
        proc.join(2.0)
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
            proc.join(2.0)
        return _xtb_failure_result(payload, error=f"xTB hard timeout after {timeout}s")

    if proc.exitcode != 0:
        return _xtb_failure_result(payload, error=f"xTB child exitcode={proc.exitcode}")

    try:
        return queue.get_nowait()
    except Exception:
        return _xtb_failure_result(payload, error="xTB child produced no result")


def _run_xtb_force_jobs(payloads, max_workers=1, timeout=None):
    """Run xTB jobs with optional hard per-molecule timeout.

    If timeout is provided, every molecule is isolated in a killable child process.
    This avoids a single pathological generated geometry blocking the whole RL step
    for hours. Parallelism is implemented at the Python thread level, with each
    thread supervising one child process.
    """
    max_workers = max(1, int(max_workers or 1))
    if len(payloads) == 0:
        return []

    if max_workers <= 1 or len(payloads) <= 1:
        return [_run_xtb_force_job_hard_timeout(p, timeout) for p in payloads]

    from concurrent.futures import ThreadPoolExecutor

    results = [None] * len(payloads)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_run_xtb_force_job_hard_timeout, p, timeout) for p in payloads]
        for i, fut in enumerate(futures):
            try:
                results[i] = fut.result(timeout=(None if timeout is None else float(timeout) + 10.0))
            except Exception as exc:
                results[i] = _xtb_failure_result(payloads[i], error=repr(exc))
    return results

class QEDObjective(BaseObjective):
    name = "qed"; component_names=["qed"]
    def score_mols(self,mols,device=None,dtype=torch.float32):
        with torch.no_grad():
            vals=[]; valid=[]; conn=[]; cs=[]; sc=[]; fps=[]
            for m in mols:
                v=m is not None; c=v and len(Chem.GetMolFrags(m))==1
                vals.append(QED.qed(m) if v else 0.0); valid.append(v); conn.append(c)
                cs.append(canonical_smiles(m) if v else None); sc.append(murcko_scaffold_smiles(m) if v else None); fps.append(morgan_fp(m) if v else None)
            s=torch.tensor(vals,dtype=dtype,device=device); vb=torch.tensor(valid,dtype=torch.bool,device=device); cb=torch.tensor(conn,dtype=torch.bool,device=device)
            return ScoringResult(s,{"qed":s},{"QED":s},vb&cb,(~vb)|(~cb),vb,cb,cs,cs,sc,fps,mols,{"official_guacamol":False})


class TargetSimilarityObjective(BaseObjective):
    """Single-target Tanimoto-similarity objective with a small extension hook.

    Default use case here is Celecoxib similarity-guided RL:
        objective_name="celecoxib_similarity"
        objective_config={"target_smiles": CELECOXIB_SMILES, "fp_type": "morgan"}

    By default, the optimized score is raw Tanimoto similarity in [0, 1]. If
    similarity_threshold is set, the score becomes min(sim, threshold)/threshold,
    matching the capped similarity reward used in some scaffold-hopping papers.

    Extra RDKit properties are computed and logged as raw_properties so that QED,
    TPSA, or logP can later be promoted into component_scores without touching the
    training loop.
    """
    name = "target_similarity"
    component_names = ["similarity"]
    pareto_component_names = ["similarity"]

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.target_smiles = self.cfg.get("target_smiles") or self.cfg.get("reference_smiles")
        if not self.target_smiles:
            raise ValueError("TargetSimilarityObjective requires objective_config['target_smiles'].")
        self.target = Chem.MolFromSmiles(self.target_smiles)
        if self.target is None:
            raise ValueError(f"Invalid target_smiles for TargetSimilarityObjective: {self.target_smiles}")
        self.target = Chem.RemoveHs(self.target)
        Chem.SanitizeMol(self.target)
        self.ref_canonical = canonical_smiles(self.target)

        self.target_name = str(self.cfg.get("target_name", "target")).lower()
        self.fp_type = str(self.cfg.get("fp_type", "morgan")).lower()
        self.radius = int(self.cfg.get("radius", 2))
        self.n_bits = int(self.cfg.get("n_bits", 2048))
        threshold = self.cfg.get("similarity_threshold", self.cfg.get("threshold", None))
        self.similarity_threshold = None if threshold is None else float(threshold)
        self.sim_key = str(self.cfg.get("similarity_key", f"sim_{self.target_name}"))
        self.component_names = [self.sim_key]
        self.pareto_component_names = self.cfg.get("pareto_component_names", [self.sim_key])
        self.target_fp = self._fingerprint(self.target)

    def _fingerprint(self, mol):
        if mol is None:
            return None
        if self.fp_type in {"morgan", "ecfp", "ecfp4"}:
            return AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        if self.fp_type in {"ap", "atom_pair", "atom-pair"}:
            return Pairs.GetAtomPairFingerprint(mol)
        if self.fp_type in {"rdkit", "rdk"}:
            return Chem.RDKFingerprint(mol, fpSize=self.n_bits)
        raise ValueError(f"Unsupported fp_type={self.fp_type}; use morgan, ap, or rdkit.")

    def _similarity_score(self, raw_sim: torch.Tensor) -> torch.Tensor:
        if self.similarity_threshold is None:
            return raw_sim.clamp(0, 1)
        thr = max(float(self.similarity_threshold), 1e-8)
        return (torch.minimum(raw_sim, torch.tensor(thr, dtype=raw_sim.dtype, device=raw_sim.device)) / thr).clamp(0, 1)

    def score_mols(self, mols, device=None, dtype=torch.float32):
        with torch.no_grad():
            sim_vals=[]; qed_vals=[]; tpsa_vals=[]; logp_vals=[]; hac_vals=[]
            valid=[]; conn=[]; cs=[]; sc=[]; fps=[]
            for m in mols:
                v = m is not None
                c = v and len(Chem.GetMolFrags(m)) == 1
                if v:
                    try:
                        m = Chem.RemoveHs(Chem.Mol(m))
                        Chem.SanitizeMol(m)
                    except Exception:
                        v = False
                        c = False

                valid.append(v); conn.append(c)
                cs.append(canonical_smiles(m) if v else None)
                sc.append(murcko_scaffold_smiles(m) if v else None)
                fps.append(morgan_fp(m) if v else None)

                if not (v and c):
                    sim_vals.append(0.0); qed_vals.append(0.0); tpsa_vals.append(0.0); logp_vals.append(0.0); hac_vals.append(0.0)
                    continue

                fp = self._fingerprint(m)
                sim = DataStructs.TanimotoSimilarity(fp, self.target_fp) if fp is not None else 0.0
                sim_vals.append(float(sim))
                qed_vals.append(float(QED.qed(m)))
                tpsa_vals.append(float(rdMolDescriptors.CalcTPSA(m)))
                logp_vals.append(float(Crippen.MolLogP(m)))
                hac_vals.append(float(m.GetNumHeavyAtoms()))

            raw_sim = torch.tensor(sim_vals, dtype=dtype, device=device).clamp(0, 1)
            comp_sim = self._similarity_score(raw_sim)
            raw_qed = torch.tensor(qed_vals, dtype=dtype, device=device).clamp(0, 1)
            raw_tpsa = torch.tensor(tpsa_vals, dtype=dtype, device=device)
            raw_logp = torch.tensor(logp_vals, dtype=dtype, device=device)
            raw_hac = torch.tensor(hac_vals, dtype=dtype, device=device)
            vb = torch.tensor(valid, dtype=torch.bool, device=device)
            cb = torch.tensor(conn, dtype=torch.bool, device=device)

            score = torch.where(vb & cb, comp_sim, torch.zeros_like(comp_sim))
            component_scores = {self.sim_key: score}
            raw_properties = {
                self.sim_key: raw_sim,
                "QED": raw_qed,
                "TPSA": raw_tpsa,
                "logP": raw_logp,
                "heavy_atom_count": raw_hac,
            }
            metadata = {
                "official_guacamol": False,
                "objective": "target_similarity",
                "target_name": self.target_name,
                "reference_canonical_smiles": self.ref_canonical,
                "fp_type": self.fp_type,
                "radius": self.radius,
                "n_bits": self.n_bits,
                "similarity_threshold": self.similarity_threshold,
                "geometric_score": score,
                "linear_score": score,
                "tchebycheff_score": score,
                "min_component_score": score,
            }
            return ScoringResult(score, component_scores, raw_properties, vb & cb, (~vb) | (~cb), vb, cb, cs, cs, sc, fps, mols, metadata)


class PerindoprilSimilarityAromaticObjective(BaseObjective):
    """Perindopril similarity + aromatic-ring-count objective.

    Final score is a weighted linear combination:
        score = w_sim * sim(perindopril, mol) + w_arom * aromatic_component

    aromatic_component:
        1.0  if number of aromatic rings == target_aromatic_rings, default 2
        0.5  if number of aromatic rings is in near_aromatic_rings, default {1, 3}
        0.0  otherwise

    This keeps similarity as the dominant term while giving an explicit scaffold-hopping
    incentive toward aromatic-ring-containing scaffolds.
    """
    name = "perindopril_similarity_aromatic"
    component_names = ["sim_perindopril", "aromatic_ring_reward"]
    pareto_component_names = ["sim_perindopril", "aromatic_ring_reward"]

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.target_smiles = self.cfg.get(
            "target_smiles",
            "CCC[C@@H](C(=O)OCC)N[C@@H](C)C(=O)N1[C@H]2CCCC[C@H]2C[C@H]1C(=O)O",
        )
        self.target = Chem.MolFromSmiles(self.target_smiles)
        if self.target is None:
            raise ValueError(f"Invalid Perindopril target_smiles: {self.target_smiles}")
        self.target = Chem.RemoveHs(self.target)
        Chem.SanitizeMol(self.target)
        self.ref_canonical = canonical_smiles(self.target)

        self.fp_type = str(self.cfg.get("fp_type", "morgan")).lower()
        self.radius = int(self.cfg.get("radius", 2))
        self.n_bits = int(self.cfg.get("n_bits", 2048))
        threshold = self.cfg.get("similarity_threshold", self.cfg.get("threshold", None))
        self.similarity_threshold = None if threshold is None else float(threshold)

        self.sim_key = str(self.cfg.get("similarity_key", "sim_perindopril"))
        self.arom_key = str(self.cfg.get("aromatic_key", "aromatic_ring_reward"))
        self.raw_arom_key = str(self.cfg.get("raw_aromatic_key", "num_aromatic_rings"))

        self.target_aromatic_rings = int(self.cfg.get("target_aromatic_rings", 2))
        near = self.cfg.get("near_aromatic_rings", [1, 3])
        if isinstance(near, str):
            near = [int(x.strip()) for x in near.split(",") if x.strip()]
        self.near_aromatic_rings = {int(x) for x in near}
        self.near_aromatic_score = float(self.cfg.get("near_aromatic_score", 0.5))

        self.similarity_weight = float(self.cfg.get("similarity_weight", 0.8))
        self.aromatic_weight = float(self.cfg.get("aromatic_weight", 0.2))
        weight_sum = max(self.similarity_weight + self.aromatic_weight, 1e-8)
        self.similarity_weight = self.similarity_weight / weight_sum
        self.aromatic_weight = self.aromatic_weight / weight_sum

        self.component_names = [self.sim_key, self.arom_key]
        self.pareto_component_names = self.cfg.get("pareto_component_names", [self.sim_key, self.arom_key])
        self.target_fp = self._fingerprint(self.target)

    def _fingerprint(self, mol):
        if mol is None:
            return None
        if self.fp_type in {"morgan", "ecfp", "ecfp4"}:
            return AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        if self.fp_type in {"ap", "atom_pair", "atom-pair"}:
            return Pairs.GetAtomPairFingerprint(mol)
        if self.fp_type in {"rdkit", "rdk"}:
            return Chem.RDKFingerprint(mol, fpSize=self.n_bits)
        raise ValueError(f"Unsupported fp_type={self.fp_type}; use morgan, ap, or rdkit.")

    def _similarity_score(self, raw_sim: torch.Tensor) -> torch.Tensor:
        if self.similarity_threshold is None:
            return raw_sim.clamp(0, 1)
        thr = max(float(self.similarity_threshold), 1e-8)
        return (torch.minimum(raw_sim, torch.tensor(thr, dtype=raw_sim.dtype, device=raw_sim.device)) / thr).clamp(0, 1)

    def _aromatic_component_value(self, n_arom: int) -> float:
        if int(n_arom) == self.target_aromatic_rings:
            return 1.0
        if int(n_arom) in self.near_aromatic_rings:
            return self.near_aromatic_score
        return 0.0

    def score_mols(self, mols, device=None, dtype=torch.float32):
        with torch.no_grad():
            sim_vals = []
            arom_counts = []
            arom_scores = []
            qed_vals = []
            tpsa_vals = []
            logp_vals = []
            hac_vals = []
            valid = []
            conn = []
            cs = []
            sc = []
            fps = []

            for m in mols:
                v = m is not None
                c = v and len(Chem.GetMolFrags(m)) == 1
                if v:
                    try:
                        m = Chem.RemoveHs(Chem.Mol(m))
                        Chem.SanitizeMol(m)
                    except Exception:
                        v = False
                        c = False

                valid.append(v)
                conn.append(c)
                cs.append(canonical_smiles(m) if v else None)
                sc.append(murcko_scaffold_smiles(m) if v else None)
                fps.append(morgan_fp(m) if v else None)

                if not (v and c):
                    sim_vals.append(0.0)
                    arom_counts.append(0.0)
                    arom_scores.append(0.0)
                    qed_vals.append(0.0)
                    tpsa_vals.append(0.0)
                    logp_vals.append(0.0)
                    hac_vals.append(0.0)
                    continue

                fp = self._fingerprint(m)
                sim = DataStructs.TanimotoSimilarity(fp, self.target_fp) if fp is not None else 0.0
                n_arom = int(rdMolDescriptors.CalcNumAromaticRings(m))
                sim_vals.append(float(sim))
                arom_counts.append(float(n_arom))
                arom_scores.append(float(self._aromatic_component_value(n_arom)))
                qed_vals.append(float(QED.qed(m)))
                tpsa_vals.append(float(rdMolDescriptors.CalcTPSA(m)))
                logp_vals.append(float(Crippen.MolLogP(m)))
                hac_vals.append(float(m.GetNumHeavyAtoms()))

            raw_sim = torch.tensor(sim_vals, dtype=dtype, device=device).clamp(0, 1)
            sim_component = self._similarity_score(raw_sim)
            arom_raw = torch.tensor(arom_counts, dtype=dtype, device=device)
            arom_component = torch.tensor(arom_scores, dtype=dtype, device=device).clamp(0, 1)
            raw_qed = torch.tensor(qed_vals, dtype=dtype, device=device).clamp(0, 1)
            raw_tpsa = torch.tensor(tpsa_vals, dtype=dtype, device=device)
            raw_logp = torch.tensor(logp_vals, dtype=dtype, device=device)
            raw_hac = torch.tensor(hac_vals, dtype=dtype, device=device)
            vb = torch.tensor(valid, dtype=torch.bool, device=device)
            cb = torch.tensor(conn, dtype=torch.bool, device=device)

            score = self.similarity_weight * sim_component + self.aromatic_weight * arom_component
            score = torch.where(vb & cb, score.clamp(0, 1), torch.zeros_like(score))

            component_scores = {
                self.sim_key: torch.where(vb & cb, sim_component, torch.zeros_like(sim_component)),
                self.arom_key: torch.where(vb & cb, arom_component, torch.zeros_like(arom_component)),
            }
            raw_properties = {
                self.sim_key: raw_sim,
                self.raw_arom_key: arom_raw,
                "QED": raw_qed,
                "TPSA": raw_tpsa,
                "logP": raw_logp,
                "heavy_atom_count": raw_hac,
            }
            metadata = {
                "official_guacamol": False,
                "objective": "perindopril_similarity_aromatic",
                "reference_canonical_smiles": self.ref_canonical,
                "fp_type": self.fp_type,
                "radius": self.radius,
                "n_bits": self.n_bits,
                "similarity_threshold": self.similarity_threshold,
                "target_aromatic_rings": self.target_aromatic_rings,
                "near_aromatic_rings": sorted(self.near_aromatic_rings),
                "near_aromatic_score": self.near_aromatic_score,
                "similarity_weight": self.similarity_weight,
                "aromatic_weight": self.aromatic_weight,
                "geometric_score": score,
                "linear_score": score,
                "tchebycheff_score": score,
                "min_component_score": torch.stack([component_scores[self.sim_key], component_scores[self.arom_key]], dim=1).min(dim=1).values,
            }
            return ScoringResult(score, component_scores, raw_properties, vb & cb, (~vb) | (~cb), vb, cb, cs, cs, sc, fps, mols, metadata)




class XTBForceObjective(BaseObjective):
    """GFN-xTB force-RMS objective for physically stable 3D generation.

    The raw physical cost is force_rms = sqrt(mean_i ||F_i||^2) by default.
    For this codebase the default optimized score is bounded to [0, 1]:

        score = 1 / (1 + force_rms / force_scale)

    so smaller residual xTB forces produce larger rewards, and successful
    near-equilibrium conformations approach 1. Failed xTB calculations receive
    fail_score=0 by default and are marked infeasible for top selection.

    For strict reproduction of negative-force rewards, set
    reward_transform="neg_force"; then score = -reward_scale * force_rms and
    failed samples receive fail_penalty.
    """
    name = "xtb_force"
    component_names = ["xtb_reward"]
    pareto_component_names = ["xtb_reward"]

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.method = str(self.cfg.get("xtb_method", self.cfg.get("method", "GFN2-xTB")))
        self.force_norm = str(self.cfg.get("force_norm", "atom_rms")).lower()
        self.fail_penalty = float(self.cfg.get("fail_penalty", -5.0))
        self.fail_score = float(self.cfg.get("fail_score", 0.0))
        self.max_workers = int(self.cfg.get("max_workers", 1))
        timeout = self.cfg.get("timeout", 120.0)
        self.timeout = None if timeout is None or float(timeout) <= 0 else float(timeout)
        self.reward_scale = float(self.cfg.get("reward_scale", 1.0))
        self.reward_transform = str(self.cfg.get("reward_transform", "bounded_inverse")).lower()
        self.force_scale = max(float(self.cfg.get("force_scale", 1.0)), 1e-12)
        self.linear_cutoff = max(float(self.cfg.get("linear_cutoff", self.force_scale)), 1e-12)
        self.require_connected = bool(self.cfg.get("require_connected", True))
        self.require_xtb_success_for_feasible = bool(self.cfg.get("require_xtb_success_for_feasible", True))
        self.score_key = str(self.cfg.get("score_key", "xtb_reward"))
        self.component_names = [self.score_key]
        self.pareto_component_names = self.cfg.get("pareto_component_names", [self.score_key])

    def _mol_to_payload(self, mol):
        if mol is None:
            return None
        if mol.GetNumConformers() == 0:
            return None
        conf = mol.GetConformer(0)
        symbols = []
        positions = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            p = conf.GetAtomPosition(idx)
            symbols.append(atom.GetSymbol())
            positions.append([float(p.x), float(p.y), float(p.z)])
        if len(symbols) == 0:
            return None
        return (symbols, positions, self.method, self.force_norm, self.fail_penalty, self.reward_scale)

    def _force_to_score(self, force_t: torch.Tensor, success_t: torch.Tensor) -> torch.Tensor:
        mode = self.reward_transform
        if mode in {"neg_force", "negative", "raw_negative", "minus_force"}:
            score = -float(self.reward_scale) * force_t
            fail_value = float(self.fail_penalty)
        elif mode in {"bounded_inverse", "inverse", "01_inverse", "one_over_one_plus"}:
            score = 1.0 / (1.0 + force_t / float(self.force_scale))
            fail_value = float(self.fail_score)
        elif mode in {"bounded_exp", "exp", "01_exp", "exponential"}:
            score = torch.exp(-force_t / float(self.force_scale))
            fail_value = float(self.fail_score)
        elif mode in {"linear_cutoff", "cutoff", "01_linear"}:
            score = (1.0 - force_t / float(self.linear_cutoff)).clamp(0.0, 1.0)
            fail_value = float(self.fail_score)
        else:
            raise ValueError(
                f"Unsupported xTB reward_transform={self.reward_transform}. "
                "Use bounded_inverse, bounded_exp, linear_cutoff, or neg_force."
            )

        fail_tensor = torch.full_like(score, fail_value)
        return torch.where(success_t, score, fail_tensor)

    def score_mols(self, mols, device=None, dtype=torch.float32):
        with torch.no_grad():
            valid=[]; conn=[]; cs=[]; sc=[]; fps=[]; payloads=[]; payload_map=[]
            for i, m in enumerate(mols):
                v = m is not None
                c = v and len(Chem.GetMolFrags(m)) == 1
                if v:
                    try:
                        # Keep explicit Hs and generated conformer coordinates for xTB.
                        m = Chem.Mol(m)
                        Chem.SanitizeMol(m)
                    except Exception:
                        v = False; c = False
                valid.append(v); conn.append(c)
                cs.append(canonical_smiles(m) if v else None)
                sc.append(murcko_scaffold_smiles(m) if v else None)
                fps.append(morgan_fp(m) if v else None)

                if not v or (self.require_connected and not c):
                    continue
                payload = self._mol_to_payload(m)
                if payload is not None:
                    payload_map.append(i)
                    payloads.append(payload)

            # Large finite placeholder only for raw logging of failed/no-conformer samples.
            fail_force = float(self.cfg.get("fail_force_rms", 999.0))
            force_rms=[float(fail_force)] * len(mols)
            energies=[0.0] * len(mols)
            success=[False] * len(mols)
            errors=[""] * len(mols)

            if payloads:
                xtb_results = _run_xtb_force_jobs(payloads, max_workers=self.max_workers, timeout=self.timeout)
                for idx, res in zip(payload_map, xtb_results):
                    force_rms[idx] = float(res.get("force_rms", fail_force))
                    energies[idx] = float(res.get("energy", 0.0))
                    success[idx] = bool(res.get("success", False))
                    errors[idx] = str(res.get("error", ""))

            force_t=torch.tensor(force_rms,dtype=dtype,device=device)
            energy_t=torch.tensor(energies,dtype=dtype,device=device)
            success_t=torch.tensor(success,dtype=torch.bool,device=device)
            success_f=success_t.to(dtype=dtype)
            vb=torch.tensor(valid,dtype=torch.bool,device=device)
            cb=torch.tensor(conn,dtype=torch.bool,device=device)
            score=self._force_to_score(force_t, success_t)

            feasible = vb.clone()
            if self.require_connected:
                feasible = feasible & cb
            if self.require_xtb_success_for_feasible:
                feasible = feasible & success_t
            severe = (~vb)
            if self.require_connected:
                severe = severe | (~cb)
            if self.require_xtb_success_for_feasible:
                severe = severe | (~success_t)

            component_scores={self.score_key: score}
            raw_properties={
                "xtb_force_rms": force_t,
                "xtb_energy": energy_t,
                "xtb_success": success_f,
                "xtb_fail": 1.0 - success_f,
                "xtb_score_01": score if self.reward_transform not in {"neg_force", "negative", "raw_negative", "minus_force"} else 1.0 / (1.0 + force_t / float(self.force_scale)),
            }
            metadata={
                "official_guacamol":False,
                "objective":"xtb_force",
                "xtb_method":self.method,
                "force_norm":self.force_norm,
                "reward_definition":(
                    "score = 1 / (1 + force_rms / force_scale)"
                    if self.reward_transform in {"bounded_inverse", "inverse", "01_inverse", "one_over_one_plus"}
                    else "score = exp(-force_rms / force_scale)"
                    if self.reward_transform in {"bounded_exp", "exp", "01_exp", "exponential"}
                    else "score = clamp(1 - force_rms / linear_cutoff, 0, 1)"
                    if self.reward_transform in {"linear_cutoff", "cutoff", "01_linear"}
                    else "score = -reward_scale * force_rms"
                ),
                "reward_transform":self.reward_transform,
                "force_scale":self.force_scale,
                "linear_cutoff":self.linear_cutoff,
                "reward_scale":self.reward_scale,
                "fail_score":self.fail_score,
                "fail_penalty":self.fail_penalty,
                "max_workers":self.max_workers,
                "timeout":self.timeout,
                "require_connected":self.require_connected,
                "require_xtb_success_for_feasible":self.require_xtb_success_for_feasible,
                "geometric_score":score,
                "linear_score":score,
                "tchebycheff_score":score,
                "min_component_score":score,
                "last_xtb_error":"; ".join([e for e in errors if e][:3]),
            }
            return ScoringResult(score,component_scores,raw_properties,feasible,severe,vb,cb,cs,cs,sc,fps,mols,metadata)


class PSA3DMinObjective(BaseObjective):
    """3D PSA minimization objective.

    This follows the PSA definition described in RL-semla: PSA is the solvent-accessible
    surface area (SASA) contribution of polar atoms, computed with RDKit rdFreeSASA and
    van der Waals radii. By default, attached hydrogens on polar atoms are also included.

    Reward transform:
        score = sigmoid(-k * (PSA - center))

    Thus lower PSA gives higher reward. Defaults are intentionally configurable because
    the paper states that sigmoid/reverse-sigmoid parameters are selected empirically from
    the prior distribution and desired training speed.
    """
    name = "psa_min"
    component_names = ["PSA"]
    pareto_component_names = ["PSA"]
    _polar_symbols_default = ("N", "O", "S", "P")

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.center = float(self.cfg.get("psa_center", self.cfg.get("psa_b", 75.0)))
        self.k = float(self.cfg.get("psa_k", 0.05))
        self.add_hs = bool(self.cfg.get("psa_add_hs", True))
        self.include_polar_hs = bool(self.cfg.get("psa_include_polar_hs", True))
        self.embed_if_missing = bool(self.cfg.get("psa_embed_if_missing", False))
        self.mmff_opt_if_embedded = bool(self.cfg.get("psa_mmff_opt_if_embedded", False))
        polar_symbols = self.cfg.get("psa_polar_symbols", self._polar_symbols_default)
        if isinstance(polar_symbols, str):
            polar_symbols = [x.strip() for x in polar_symbols.split(",") if x.strip()]
        self.polar_symbols = set(polar_symbols)

    def _prepare_mol_for_sasa(self, mol):
        if mol is None:
            return None
        m = Chem.Mol(mol)
        if self.add_hs:
            try:
                m = Chem.AddHs(m, addCoords=True)
            except Exception:
                m = Chem.AddHs(m)

        if m.GetNumConformers() == 0:
            if not self.embed_if_missing:
                return None
            try:
                params = AllChem.ETKDGv3()
                params.randomSeed = int(self.cfg.get("psa_embed_seed", 0xC0FFEE))
                if AllChem.EmbedMolecule(m, params) != 0:
                    return None
                if self.mmff_opt_if_embedded and AllChem.MMFFHasAllMoleculeParams(m):
                    AllChem.MMFFOptimizeMolecule(m, maxIters=int(self.cfg.get("psa_mmff_max_iters", 100)))
            except Exception:
                return None
        return m

    def _atom_is_polar_surface_atom(self, atom):
        symbol = atom.GetSymbol()
        if symbol in self.polar_symbols:
            return True
        if self.include_polar_hs and atom.GetAtomicNum() == 1:
            nbrs = atom.GetNeighbors()
            if len(nbrs) == 1 and nbrs[0].GetSymbol() in self.polar_symbols:
                return True
        return False

    def _calc_psa(self, mol):
        m = self._prepare_mol_for_sasa(mol)
        if m is None:
            return 0.0, False
        try:
            pt = Chem.GetPeriodicTable()
            radii = [float(pt.GetRvdw(atom.GetAtomicNum())) for atom in m.GetAtoms()]
            rdFreeSASA.CalcSASA(m, radii, confIdx=int(self.cfg.get("psa_conf_idx", 0)))
            psa = 0.0
            for atom in m.GetAtoms():
                if self._atom_is_polar_surface_atom(atom) and atom.HasProp("SASA"):
                    psa += float(atom.GetDoubleProp("SASA"))
            return psa, True
        except Exception:
            return 0.0, False

    def _reverse_sigmoid(self, raw_psa: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(-self.k * (raw_psa - self.center))

    def score_mols(self, mols, device=None, dtype=torch.float32):
        with torch.no_grad():
            vals=[]; valid=[]; conn=[]; psa_ok=[]; cs=[]; sc=[]; fps=[]; natoms=[]
            for m in mols:
                v=m is not None; c=v and len(Chem.GetMolFrags(m))==1
                valid.append(v); conn.append(c)
                cs.append(canonical_smiles(m) if v else None)
                sc.append(murcko_scaffold_smiles(m) if v else None)
                fps.append(morgan_fp(m) if v else None)
                natoms.append(float(m.GetNumAtoms()) if v else 0.0)
                if v:
                    psa, ok = self._calc_psa(m)
                    vals.append(float(psa)); psa_ok.append(bool(ok))
                else:
                    vals.append(0.0); psa_ok.append(False)

            raw_psa=torch.tensor(vals,dtype=dtype,device=device)
            vb=torch.tensor(valid,dtype=torch.bool,device=device)
            cb=torch.tensor(conn,dtype=torch.bool,device=device)
            okb=torch.tensor(psa_ok,dtype=torch.bool,device=device)
            nat=torch.tensor(natoms,dtype=dtype,device=device)
            score=self._reverse_sigmoid(raw_psa).clamp(0,1)
            score=torch.where(vb & cb & okb, score, torch.zeros_like(score))
            metadata={
                "official_guacamol":False,
                "property":"PSA",
                "psa_definition":"3D polar SASA from RDKit rdFreeSASA using periodic-table vdW radii; summed over polar atoms and, by default, attached polar hydrogens.",
                "score_transform":"reverse_sigmoid",
                "lower_is_better":True,
                "psa_center":self.center,
                "psa_k":self.k,
                "psa_add_hs":self.add_hs,
                "psa_include_polar_hs":self.include_polar_hs,
                "psa_polar_symbols":sorted(self.polar_symbols),
                "psa_embed_if_missing":self.embed_if_missing,
                "psa_ok":okb,
            }
            return ScoringResult(
                score=score,
                component_scores={"PSA":score},
                raw_properties={"PSA":raw_psa,"num_atoms":nat},
                feasible=vb&cb&okb,
                severe_violation=(~vb)|(~cb)|(~okb),
                valid=vb,
                connected=cb,
                smiles=cs,
                canonical_smiles=cs,
                scaffolds=sc,
                fps=fps,
                mols=mols,
                metadata=metadata,
            )


class QEDSAObjective(BaseObjective):
    """Dual-objective optimization: maximize QED and minimize synthetic accessibility score.

    Raw properties:
      - QED: RDKit QED.qed(mol), higher is better, already in [0, 1].
      - SA: Ertl-Schuffenhauer synthetic accessibility score from RDKit Contrib SA_Score
            (typically 1 = easy synthesis, 10 = hard synthesis), lower is better.

    Component scores:
      - qed component = raw QED.
      - SA component = transformed SA desirability, by default linear mapping
            sa_component = clamp((10 - SA) / 9, 0, 1)
        so SA=1 gives 1 and SA=10 gives 0.
      - Optional reverse sigmoid can be used by objective_config["sa_transform"] = "reverse_sigmoid".

    Final score:
      - weighted geometric mean of QED component and SA component by default.
    """
    name = "qed_sa"
    component_names = ["qed", "SA"]
    pareto_component_names = ["qed", "SA"]
    _sascorer = None
    _sascorer_error = None

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.sa_transform = str(self.cfg.get("sa_transform", "linear")).lower()
        self.sa_center = float(self.cfg.get("sa_center", 3.0))
        self.sa_k = float(self.cfg.get("sa_k", 1.0))
        self.component_weights = self.cfg.get("component_weights", {"qed": 1.0, "SA": 1.0})
        self.aggregate = str(self.cfg.get("aggregate", "geometric")).lower()
        self.allow_sa_proxy = bool(self.cfg.get("sa_allow_proxy", False))

    @classmethod
    def _load_sascorer(cls):
        if cls._sascorer is not None:
            return cls._sascorer
        if cls._sascorer_error is not None:
            raise cls._sascorer_error
        try:
            from rdkit.Contrib.SA_Score import sascorer
            cls._sascorer = sascorer
            return cls._sascorer
        except Exception as e1:
            try:
                from rdkit import RDConfig
                import sys
                from pathlib import Path as _Path
                sa_path = _Path(RDConfig.RDContribDir) / "SA_Score"
                if str(sa_path) not in sys.path:
                    sys.path.append(str(sa_path))
                import sascorer
                cls._sascorer = sascorer
                return cls._sascorer
            except Exception as e2:
                cls._sascorer_error = ImportError(
                    "Cannot import RDKit Contrib SA_Score/sascorer. "
                    "Install/enable RDKit Contrib, or set objective_config['sa_allow_proxy']=True "
                    "to use a crude fallback proxy. Original errors: "
                    f"{repr(e1)} ; {repr(e2)}"
                )
                raise cls._sascorer_error

    def _proxy_sa_score(self, mol):
        """Crude fallback only; not Ertl SA. Disabled by default."""
        heavy = max(1, mol.GetNumHeavyAtoms())
        rings = rdMolDescriptors.CalcNumRings(mol)
        hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
        chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        raw = 1.0 + 0.04 * heavy + 0.25 * rings + 0.15 * hetero + 0.25 * chiral + 0.3 * spiro + 0.3 * bridge
        return float(max(1.0, min(10.0, raw)))

    def _calc_sa(self, mol):
        try:
            scorer = self._load_sascorer()
            return float(scorer.calculateScore(mol)), False
        except Exception:
            if self.allow_sa_proxy:
                return self._proxy_sa_score(mol), True
            raise

    def _sa_desirability(self, raw_sa: torch.Tensor) -> torch.Tensor:
        if self.sa_transform in {"reverse_sigmoid", "sigmoid", "rsigmoid"}:
            return torch.sigmoid(-self.sa_k * (raw_sa - self.sa_center)).clamp(0, 1)
        if self.sa_transform in {"min_gaussian", "mingaussian"}:
            return min_gaussian(raw_sa, self.sa_center, self.sa_k).clamp(0, 1)
        # Default: SA is usually in [1, 10], where 1 is easy and 10 is hard.
        return ((10.0 - raw_sa) / 9.0).clamp(0, 1)

    def score_mols(self, mols, device=None, dtype=torch.float32):
        with torch.no_grad():
            qed_vals=[]; sa_vals=[]; proxy_flags=[]; valid=[]; conn=[]; cs=[]; sc=[]; fps=[]
            for m in mols:
                v=m is not None; c=v and len(Chem.GetMolFrags(m))==1
                valid.append(v); conn.append(c)
                
                if v:
                    try:
                        m = Chem.RemoveHs(m)
                        Chem.SanitizeMol(m)
                    except Exception:
                        v = False
    
                cs.append(canonical_smiles(m) if v else None)
                sc.append(murcko_scaffold_smiles(m) if v else None)
                fps.append(morgan_fp(m) if v else None)
                if not v:
                    qed_vals.append(0.0); sa_vals.append(10.0); proxy_flags.append(False)
                    continue
                qed_vals.append(float(QED.qed(m)))
                sa, used_proxy = self._calc_sa(m)
                sa_vals.append(float(sa)); proxy_flags.append(bool(used_proxy))

            raw_qed=torch.tensor(qed_vals,dtype=dtype,device=device).clamp(0,1)
            raw_sa=torch.tensor(sa_vals,dtype=dtype,device=device)
            sa_score=self._sa_desirability(raw_sa)
            comp_t={"qed":raw_qed,"SA":sa_score}
            if self.aggregate == "linear":
                score=weighted_linear_sum(comp_t,self.component_weights)
            elif self.aggregate == "tchebycheff":
                score=tchebycheff_score(comp_t,self.component_weights)
            else:
                score=weighted_geometric_mean(comp_t,self.component_weights)
            vb=torch.tensor(valid,dtype=torch.bool,device=device)
            cb=torch.tensor(conn,dtype=torch.bool,device=device)
            score=torch.where(vb & cb, score.clamp(0,1), torch.zeros_like(score))
            proxy_t=torch.tensor(proxy_flags,dtype=torch.bool,device=device)
            metadata={
                "official_guacamol":False,
                "objective":"QED_SA",
                "lower_is_better_for_SA":True,
                "SA_definition":"Ertl-Schuffenhauer synthetic accessibility score via RDKit Contrib SA_Score/sascorer; lower is easier to synthesize.",
                "sa_transform":self.sa_transform,
                "sa_center":self.sa_center,
                "sa_k":self.sa_k,
                "component_weights":self.component_weights,
                "aggregate":self.aggregate,
                "used_sa_proxy":proxy_t,
                "geometric_score":weighted_geometric_mean(comp_t,self.component_weights),
                "linear_score":weighted_linear_sum(comp_t,self.component_weights),
                "tchebycheff_score":tchebycheff_score(comp_t,self.component_weights),
                "min_component_score":torch.stack([comp_t["qed"], comp_t["SA"]],dim=1).min(dim=1).values,
            }
            return ScoringResult(
                score=score,
                component_scores=comp_t,
                raw_properties={"QED":raw_qed,"SA":raw_sa},
                feasible=vb&cb,
                severe_violation=(~vb)|(~cb),
                valid=vb,
                connected=cb,
                smiles=cs,
                canonical_smiles=cs,
                scaffolds=sc,
                fps=fps,
                mols=mols,
                metadata=metadata,
            )

class MPOObjective(BaseObjective):
    _warned=False
    def __init__(self,name,target_smiles,components,cfg=None):
        super().__init__(cfg); self.name=name; self.component_defs=components; self.component_names=[c[0] for c in components]
        self.pareto_component_names=self.cfg.get("pareto_component_names", self.component_names)
        self.target = Chem.MolFromSmiles(target_smiles)
        self.target_ap=Pairs.GetAtomPairFingerprint(self.target)
        self.ref_canonical=canonical_smiles(self.target)
        self.ap_self=DataStructs.TanimotoSimilarity(self.target_ap, self.target_ap)
    def _official_scores(self, smiles):
        if not self.cfg.get("use_official_guacamol", True): return None
        try:
            import guacamol.standard_benchmarks as sb
            fn = getattr(sb, self.name.lower(), None)
            if fn is None: return None
            obj = fn().objective
            if hasattr(obj, "score_list"):
                out = obj.score_list(smiles)
                return out if isinstance(out, list) else list(out)
            return [obj.score(s) for s in smiles]
        except Exception:
            if not MPOObjective._warned:
                warnings.warn("Using local fallback scorer (official GuacaMol unavailable/API mismatch).")
                MPOObjective._warned=True
            return None
    def score_mols(self,mols,device=None,dtype=torch.float32):
        with torch.no_grad():
            valid=[]; conn=[]; cs=[]; scaf=[]; fps=[]
            raws={"sim_ranolazine_AP":[],"logP":[],"TPSA":[],"num_F":[]}
            comps={k:[] for k in self.component_names}
            for m in mols:
                v=m is not None; c=v and len(Chem.GetMolFrags(m))==1; valid.append(v); conn.append(c)
                cs.append(canonical_smiles(m) if v else None); scaf.append(murcko_scaffold_smiles(m) if v else None); fps.append(morgan_fp(m) if v else None)
                if not v:
                    for k in raws: raws[k].append(0.0)
                    for k in comps: comps[k].append(0.0)
                    continue
                ap = Pairs.GetAtomPairFingerprint(m)
                sim_ap = DataStructs.TanimotoSimilarity(ap, self.target_ap)
                logp = Crippen.MolLogP(m); tpsa = rdMolDescriptors.CalcTPSA(m); nF = float(sum(1 for a in m.GetAtoms() if a.GetSymbol()=="F"))
                raws["sim_ranolazine_AP"].append(sim_ap); raws["logP"].append(logp); raws["TPSA"].append(tpsa); raws["num_F"].append(nF)
                local={"sim_AP":sim_ap,"logP":logp,"TPSA":tpsa,"num_F":nF,"formula":rdMolDescriptors.CalcMolFormula(m)}
                for n, fn in self.component_defs: comps[n].append(float(fn(local)))
            comp_t={k:torch.tensor(v,dtype=dtype,device=device).clamp(0,1) for k,v in comps.items()}
            raw_t={k:torch.tensor(v,dtype=dtype,device=device) for k,v in raws.items()}
            geo=weighted_geometric_mean(comp_t,self.cfg.get("component_weights"))
            lin=weighted_linear_sum(comp_t,self.cfg.get("component_weights"))
            tch=tchebycheff_score(comp_t,self.cfg.get("component_weights"))
            agg=self.cfg.get("aggregate","geometric")
            score={"geometric":geo,"linear":lin,"tchebycheff":tch}.get(agg,geo)
            official=self._official_scores([s or "" for s in cs]); off=False
            if agg=="official" and official is not None:
                score=torch.tensor(official,dtype=dtype,device=device).clamp(0,1); off=True
            vb=torch.tensor(valid,dtype=torch.bool,device=device); cb=torch.tensor(conn,dtype=torch.bool,device=device)
            metadata={"official_guacamol":off,"using_local_fallback":not off,"reference_canonical_smiles":self.ref_canonical,"ranolazine_AP_self_similarity":float(self.ap_self),"official_score":torch.tensor(official,dtype=dtype,device=device).clamp(0,1) if official is not None else None,"geometric_score":geo,"linear_score":lin,"tchebycheff_score":tch,"min_component_score":torch.stack([comp_t[k] for k in self.pareto_component_names if k in comp_t],dim=1).min(dim=1).values}
            return ScoringResult(score,comp_t,raw_t,vb&cb,(~vb)|(~cb),vb,cb,cs,cs,scaf,fps,mols,metadata)

def build_objective(name, objective_config=None):
    cfg=objective_config or {}; n=(name or "qed").lower()
    if n=="qed": return QEDObjective(cfg)
    if n in {"xtb", "xtb_force", "xtb_stability", "gfn2_xtb", "gfn2_xtb_force", "xtb_force_rms"}: return XTBForceObjective(cfg)
    if n in {"celecoxib_similarity", "celecoxib_sim", "celecoxib_scaffold_hopping"}:
        celecoxib_smiles = "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"
        local_cfg = {"target_name": "celecoxib", "target_smiles": celecoxib_smiles, "fp_type": "morgan", "radius": 2, "n_bits": 2048, **cfg}
        return TargetSimilarityObjective(local_cfg)
    if n in {"perindopril_similarity_aromatic", "perindopril_aromatic", "perindopril_scaffold_hopping"}:
        perindopril_smiles = "CCC[C@@H](C(=O)OCC)N[C@@H](C)C(=O)N1[C@H]2CCCC[C@H]2C[C@H]1C(=O)O"
        local_cfg = {
            "target_smiles": perindopril_smiles,
            "fp_type": "morgan",
            "radius": 2,
            "n_bits": 2048,
            "similarity_key": "sim_perindopril",
            "aromatic_key": "aromatic_ring_reward",
            "raw_aromatic_key": "num_aromatic_rings",
            "target_aromatic_rings": 2,
            "near_aromatic_rings": [1, 3],
            "near_aromatic_score": 0.5,
            "similarity_weight": 0.8,
            "aromatic_weight": 0.2,
            **cfg,
        }
        return PerindoprilSimilarityAromaticObjective(local_cfg)
    if n in {"target_similarity", "similarity", "tanimoto_similarity"}: return TargetSimilarityObjective(cfg)
    if n in {"psa_min", "psa3d_min", "polar_surface_area_min", "psamin"}: return PSA3DMinObjective(cfg)
    if n in {"qed_sa", "qed_sa_min", "qedsa", "qed_sa_mpo"}: return QEDSAObjective(cfg)
    tasks={"ranolazine_mpo":("Ranolazine_MPO","COc1ccc2nc(S(N)(=O)=O)sc2c1CCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",[("sim_ranolazine_AP",lambda p: thresholded(torch.tensor(p["sim_AP"]),0.7)),("logP",lambda p:max_gaussian(torch.tensor(p["logP"]),7,1)),("TPSA",lambda p:max_gaussian(torch.tensor(p["TPSA"]),95,20)),("num_F",lambda p:gaussian(torch.tensor(p["num_F"]),1,1))]),
"osimertinib_mpo":("Osimertinib_MPO","COc1cc(Nc2ncnc3cc(OCCCN4CCOCC4)c(OC)c23)ccc1N(C)C",[("sim_osimertinib_FCFC4",lambda p: thresholded(torch.tensor(p.get("sim_ECFC4",0.0)),0.8)),("sim_osimertinib_ECFC6",lambda p:min_gaussian(torch.tensor(p.get("sim_ECFC6",0.0)),0.85,2)),("TPSA",lambda p:max_gaussian(torch.tensor(p["TPSA"]),100,2)),("logP",lambda p:min_gaussian(torch.tensor(p["logP"]),1,2))])}
    if n in tasks: t=tasks[n]; return MPOObjective(t[0],t[1],t[2],cfg)
    raise ValueError(f"Unknown objective: {name}")
