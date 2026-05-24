"""
src/evaluation/compare_extractions.py
Compare two ExtractionResult objects (e.g. human vs Mistral).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.evaluation.compare_to_shap import extraction_top_k_set
from src.evaluation.extraction_parser import ExtractionResult


@dataclass
class ExtractionAgreement:
    narrative_id: str = ""
    human_feature_count: int = 0
    mistral_feature_count: int = 0
    shared_feature_count: int = 0
    feature_set_jaccard: float = 0.0
    sign_agreement: Optional[float] = None
    rank_exact_match: Optional[float] = None
    rank_spearman: Optional[float] = None
    top_k_match: Optional[int] = None
    unknown_match: int = 0
    human_features: str = ""
    mistral_features: str = ""
    human_unknown: str = ""
    mistral_unknown: str = ""
    notes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "narrative_id": self.narrative_id,
            "human_feature_count": self.human_feature_count,
            "mistral_feature_count": self.mistral_feature_count,
            "shared_feature_count": self.shared_feature_count,
            "feature_set_jaccard": self.feature_set_jaccard,
            "sign_agreement": self.sign_agreement,
            "rank_exact_match": self.rank_exact_match,
            "rank_spearman": self.rank_spearman,
            "top_k_match": self.top_k_match,
            "unknown_match": self.unknown_match,
            "human_features": self.human_features,
            "mistral_features": self.mistral_features,
            "human_unknown": self.human_unknown,
            "mistral_unknown": self.mistral_unknown,
            "notes": self.notes,
        }


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _spearman_rho(ranks_a: Dict[str, int], ranks_b: Dict[str, int]) -> Optional[float]:
    """Spearman correlation on shared feature names."""
    shared = set(ranks_a) & set(ranks_b)
    if len(shared) < 2:
        return None

    names = sorted(shared)
    ra = [ranks_a[n] for n in names]
    rb = [ranks_b[n] for n in names]
    n = len(names)

    def _rank_data(values: List[int]) -> List[float]:
        sorted_idx = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[sorted_idx[k]] = avg_rank
            i = j + 1
        return out

    rra = _rank_data(ra)
    rrb = _rank_data(rb)

    mean_a = sum(rra) / n
    mean_b = sum(rrb) / n
    num = sum((rra[i] - mean_a) * (rrb[i] - mean_b) for i in range(n))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in rra))
    den_b = math.sqrt(sum((x - mean_b) ** 2 for x in rrb))
    if den_a == 0 or den_b == 0:
        return None
    return num / (den_a * den_b)


def compare_extractions(
    human: ExtractionResult,
    mistral: ExtractionResult,
    *,
    narrative_id: str = "",
    top_k_features: int = 3,
) -> ExtractionAgreement:
    """Compute agreement metrics between human and Mistral extractions."""
    human_names = set(human.features.keys())
    mistral_names = set(mistral.features.keys())
    shared = human_names & mistral_names

    result = ExtractionAgreement(
        narrative_id=narrative_id,
        human_feature_count=len(human_names),
        mistral_feature_count=len(mistral_names),
        shared_feature_count=len(shared),
        feature_set_jaccard=_jaccard(human_names, mistral_names),
        unknown_match=int(
            set(human.unknown_features) == set(mistral.unknown_features)
        ),
        human_features=", ".join(sorted(human_names)),
        mistral_features=", ".join(sorted(mistral_names)),
        human_unknown=", ".join(human.unknown_features),
        mistral_unknown=", ".join(mistral.unknown_features),
    )

    if shared:
        sign_matches = sum(
            1 for name in shared
            if human.features[name].sign == mistral.features[name].sign
        )
        result.sign_agreement = sign_matches / len(shared)

        rank_matches = sum(
            1 for name in shared
            if human.features[name].rank == mistral.features[name].rank
        )
        result.rank_exact_match = rank_matches / len(shared)

        human_ranks = {n: human.features[n].rank for n in shared}
        mistral_ranks = {n: mistral.features[n].rank for n in shared}
        result.rank_spearman = _spearman_rho(human_ranks, mistral_ranks)

    human_top = extraction_top_k_set(human, top_k_features)
    mistral_top = extraction_top_k_set(mistral, top_k_features)
    if human_top is not None and mistral_top is not None:
        result.top_k_match = int(human_top == mistral_top)
    else:
        result.notes["top_k_match"] = (
            f"Skipped: human has {len(human_names)} mentioned, "
            f"mistral has {len(mistral_names)} (need {top_k_features})"
        )

    return result
