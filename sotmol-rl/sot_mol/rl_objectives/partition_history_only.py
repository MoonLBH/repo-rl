from dataclasses import dataclass
import math
import warnings

import torch
from rdkit import DataStructs

from .desirability import tchebycheff_score


@dataclass
class PartitionResult:
    top_mask: torch.BoolTensor
    bottom_mask: torch.BoolTensor
    selected_mask: torch.Tensor
    top_weights: torch.Tensor
    bottom_weights: torch.Tensor
    pareto_rank: torch.LongTensor | None
    dominated_count: torch.LongTensor | None
    bottom_reasons: list[str]
    top_reasons: list[str]
    diagnostics: dict[str, torch.Tensor]


class PartitionSelector:
    """History-aware feasible top selector.

    Simplified selection policy:
      1. Top candidates are restricted to scoring.feasible == True.
      2. Candidates are ordered by top_score.
      3. A candidate is skipped if it is too similar to any molecule selected
         as top in previous training steps.
         - scaffold mode: skip if scaffold has appeared in previous top set.
         - fingerprint mode: skip if max Tanimoto against previous top fps is
           >= history_fingerprint_threshold.
      4. No Pareto rank is computed.
      5. No within-batch diversity selection is performed.

    Bottom samples are still selected to keep the LFPO-F top/bottom loss usable.
    They are chosen from non-top samples by invalid/severe priority and then low score.
    """

    def __init__(self, cfg):
        self.cfg = cfg or {}
        self.history_scaffolds: set[str] = set()
        self.history_fps = []
        self.history_smiles: set[str] = set()
        self.history_scores: list[float] = []
        self.history_max_size = int(self.cfg.get("history_max_size", 4096))

    def _top_score(self, scoring):
        base = scoring.score.clone()
        mode = self.cfg.get("top_selection_score_mode", "score")
        if mode == "score":
            return base

        comp_names = list(scoring.component_scores.keys())
        if not comp_names:
            return base

        comp_stack = torch.stack([scoring.component_scores[k] for k in comp_names], dim=1)
        min_comp = comp_stack.min(dim=1).values

        if mode in ("component_balanced", "score_plus_min_component"):
            for k, w in self.cfg.get("top_selection_component_weights", {}).items():
                if k in scoring.component_scores:
                    base = base + float(w) * scoring.component_scores[k]

        if self.cfg.get("use_min_component_bonus", False) or mode in ("component_balanced", "score_plus_min_component"):
            base = base + float(self.cfg.get("min_component_weight", 1.0)) * min_comp

        if mode == "tchebycheff":
            base = tchebycheff_score(scoring.component_scores, self.cfg.get("top_selection_component_weights"))

        return base

    def _history_mode(self) -> str:
        # Keep this separate from old batch-level diversity_mode semantics.
        mode = str(self.cfg.get("history_diversity_mode", self.cfg.get("diversity_mode", "scaffold"))).lower()
        if mode not in {"none", "scaffold", "fingerprint"}:
            warnings.warn(f"Unknown history_diversity_mode={mode}; fallback to scaffold")
            mode = "scaffold"
        return mode

    def _is_history_duplicate(self, idx: int, scoring, mode: str) -> tuple[bool, str]:
        if mode == "none":
            return False, "none"

        if mode == "scaffold":
            scaf = scoring.scaffolds[idx] if idx < len(scoring.scaffolds) else None
            if scaf and scaf in self.history_scaffolds:
                return True, "history_scaffold"
            return False, "none"

        if mode == "fingerprint":
            fp = scoring.fps[idx] if idx < len(scoring.fps) else None
            if fp is None or len(self.history_fps) == 0:
                return False, "none"
            threshold = float(self.cfg.get("history_fingerprint_threshold", 0.70))
            try:
                sims = DataStructs.BulkTanimotoSimilarity(fp, self.history_fps)
                max_sim = max(sims) if sims else 0.0
            except Exception:
                max_sim = 0.0
            if max_sim >= threshold:
                return True, "history_fingerprint"
            return False, "none"

        return False, "none"

    def _trim_history(self):
        if self.history_max_size <= 0:
            return
        while len(self.history_scores) > self.history_max_size:
            # Keep implementation conservative: when capacity is exceeded,
            # rebuild from the most recent entries for fps/smiles. Scaffold set
            # cannot be exactly decremented without a multiset, so we keep it as
            # a broad historical memory. This is intentional for strict scaffold
            # novelty across training.
            self.history_scores.pop(0)
            if self.history_fps:
                self.history_fps.pop(0)
            break

    def _update_history(self, picked: list[int], scoring, top_score: torch.Tensor):
        mode = self._history_mode()
        for idx in picked:
            smi = scoring.canonical_smiles[idx] if idx < len(scoring.canonical_smiles) else None
            if smi:
                self.history_smiles.add(smi)

            scaf = scoring.scaffolds[idx] if idx < len(scoring.scaffolds) else None
            if scaf:
                self.history_scaffolds.add(scaf)

            if mode == "fingerprint":
                fp = scoring.fps[idx] if idx < len(scoring.fps) else None
                if fp is not None:
                    self.history_fps.append(fp)
                    self.history_scores.append(float(top_score[idx].detach().cpu()))

        self._trim_history()

    def _select_top_history(self, scoring, top_score, top_n: int):
        device = scoring.score.device
        feasible_idx = torch.where(scoring.feasible)[0]
        if feasible_idx.numel() == 0 or top_n <= 0:
            return [], [], []

        order = feasible_idx[torch.argsort(top_score.index_select(0, feasible_idx), descending=True)].tolist()
        mode = self._history_mode()
        picked = []
        excluded = []
        excluded_reasons = []

        for idx in order:
            is_dup, reason = self._is_history_duplicate(idx, scoring, mode)
            if is_dup:
                excluded.append(idx)
                excluded_reasons.append(reason)
                continue
            picked.append(idx)
            if len(picked) >= top_n:
                break

        self._update_history(picked, scoring, top_score)
        return picked, excluded, excluded_reasons

    def _select_bottom(self, scoring, top_mask, bottom_n: int):
        score = scoring.score
        b = score.numel()
        bottom_mask = torch.zeros_like(top_mask)
        reasons = ["middle"] * b
        if bottom_n <= 0:
            return bottom_mask, reasons

        severe = scoring.severe_violation
        invalid = (~scoring.feasible) | (~scoring.valid) | (~scoring.connected)

        priority_masks = [
            ("invalid", invalid),
            ("severe", severe),
            ("low_score", torch.ones_like(top_mask)),
        ]

        for reason, mask in priority_masks:
            if int(bottom_mask.sum().item()) >= bottom_n:
                break
            cand = torch.where(mask & (~top_mask) & (~bottom_mask))[0]
            if cand.numel() == 0:
                continue
            order = cand[torch.argsort(score.index_select(0, cand), descending=False)]
            take = min(bottom_n - int(bottom_mask.sum().item()), order.numel())
            picked = order[:take]
            bottom_mask[picked] = True
            for i in picked.tolist():
                reasons[i] = reason

        return bottom_mask, reasons

    def select(self, scoring):
        score = scoring.score
        b = score.numel()
        top_n = min(b, int(math.ceil(b * float(self.cfg.get("top_ratio", 0.25)))))
        bottom_n = min(b, int(math.ceil(b * float(self.cfg.get("bottom_ratio", 0.25)))))

        mode = self.cfg.get("mode", "feasible_pareto")
        if mode != "feasible_pareto":
            warnings.warn(
                f"This simplified PartitionSelector only implements feasible_pareto; got mode={mode}. "
                "Proceeding with feasible_pareto semantics."
            )

        top_score = self._top_score(scoring)
        top_mask = torch.zeros(b, dtype=torch.bool, device=score.device)
        top_reasons = ["middle"] * b

        picked, excluded, excluded_reasons = self._select_top_history(scoring, top_score, top_n)
        if picked:
            top_mask[torch.tensor(picked, dtype=torch.long, device=score.device)] = True
            for i in picked:
                top_reasons[i] = "top_history_unique"

        bottom_mask, bottom_reasons = self._select_bottom(scoring, top_mask, bottom_n)
        selected_mask = (top_mask | bottom_mask).float()

        feasible_count = int(scoring.feasible.sum().item())
        excluded_count = len(excluded)
        candidate_denom = max(1, feasible_count)

        # bottom reason diagnostics
        reason_keys = ["invalid", "severe", "low_score"]
        diagnostics = {
            "top_count": torch.tensor(float(top_mask.sum().item()), device=score.device),
            "bottom_count": torch.tensor(float(bottom_mask.sum().item()), device=score.device),
            "selected_count": torch.tensor(float(selected_mask.sum().item()), device=score.device),
            "top_frac": top_mask.float().mean(),
            "bottom_frac": bottom_mask.float().mean(),
            "selected_frac": selected_mask.mean(),
            "top_candidate_count": torch.tensor(float(feasible_count), device=score.device),
            "history_size_scaffold": torch.tensor(float(len(self.history_scaffolds)), device=score.device),
            "history_size_fingerprint": torch.tensor(float(len(self.history_fps)), device=score.device),
            "history_excluded_count": torch.tensor(float(excluded_count), device=score.device),
            "history_excluded_frac_of_feasible": torch.tensor(float(excluded_count) / candidate_denom, device=score.device),
            "history_mode_scaffold": torch.tensor(1.0 if self._history_mode() == "scaffold" else 0.0, device=score.device),
            "history_mode_fingerprint": torch.tensor(1.0 if self._history_mode() == "fingerprint" else 0.0, device=score.device),
        }
        for rk in reason_keys:
            cnt = sum(1 for r in bottom_reasons if r == rk)
            diagnostics[f"bottom_reason_{rk}_frac"] = torch.tensor(float(cnt) / max(1, b), device=score.device)

        # Mode-specific exclusion diagnostics.
        diagnostics["history_excluded_scaffold_count"] = torch.tensor(
            float(sum(1 for r in excluded_reasons if r == "history_scaffold")), device=score.device
        )
        diagnostics["history_excluded_fingerprint_count"] = torch.tensor(
            float(sum(1 for r in excluded_reasons if r == "history_fingerprint")), device=score.device
        )

        return PartitionResult(
            top_mask=top_mask,
            bottom_mask=bottom_mask,
            selected_mask=selected_mask,
            top_weights=torch.ones_like(score),
            bottom_weights=torch.ones_like(score),
            pareto_rank=None,
            dominated_count=None,
            bottom_reasons=bottom_reasons,
            top_reasons=top_reasons,
            diagnostics=diagnostics,
        )


def build_partition_selector(partition_config):
    return PartitionSelector(partition_config or {})
