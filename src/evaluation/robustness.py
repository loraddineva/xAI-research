"""
src/evaluation/robustness.py
Multi-sample extraction agreement for extraction-model reliability.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.evaluation.extraction_parser import ExtractionResult, parse_extraction_response


def _proportion_agreement(values: Sequence[Any]) -> float:
    """Share of runs that match the plurality value (e.g. 3/5 -> 0.6)."""
    if not values:
        return 0.0
    _count = Counter(values)
    return _count.most_common(1)[0][1] / len(values)


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isfinite(value):
            return round(value, 6)
        return value
    return str(value).strip()


@dataclass
class FeatureAgreement:
    sign_agreement: float
    rank_agreement: float
    value_agreement: Optional[float]


@dataclass
class RobustnessResult:
    n_successful_runs: int
    n_requested_runs: int
    per_feature: Dict[str, FeatureAgreement] = field(default_factory=dict)
    narrative_reliability_score: Optional[float] = None
    flagged_low_reliability: bool = False
    extraction_unreliable: bool = False
    parse_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_successful_runs": self.n_successful_runs,
            "n_requested_runs": self.n_requested_runs,
            "per_feature": {
                name: {
                    "sign_agreement": fa.sign_agreement,
                    "rank_agreement": fa.rank_agreement,
                    "value_agreement": fa.value_agreement,
                }
                for name, fa in self.per_feature.items()
            },
            "narrative_reliability_score": self.narrative_reliability_score,
            "flagged_low_reliability": self.flagged_low_reliability,
            "extraction_unreliable": self.extraction_unreliable,
        }


def compute_robustness(
    extractions: List[ExtractionResult],
    *,
    n_requested_runs: int,
    min_successful_runs: int = 3,
    reliability_threshold: float = 0.8,
) -> RobustnessResult:
    """
    Compute per-field and narrative-level agreement across parsed extractions.

    Args:
        extractions: Successfully parsed extraction results (may be fewer than requested).
        n_requested_runs: Number of extraction calls attempted.
        min_successful_runs: Below this count, mark extraction_unreliable and skip scoring.
        reliability_threshold: Narratives with score below this are flagged_low_reliability.
    """
    n_ok = len(extractions)
    result = RobustnessResult(
        n_successful_runs=n_ok,
        n_requested_runs=n_requested_runs,
    )

    if n_ok < min_successful_runs:
        result.extraction_unreliable = True
        return result

    feature_runs: Dict[str, int] = {}
    for ext in extractions:
        for name in ext.features:
            feature_runs[name] = feature_runs.get(name, 0) + 1

    majority_cutoff = n_ok / 2
    scored_features: List[str] = []

    for feature_name, mention_count in feature_runs.items():
        if mention_count <= majority_cutoff:
            continue

        signs: List[int] = []
        ranks: List[int] = []
        values: List[Any] = []
        has_non_null_value = False

        for ext in extractions:
            if feature_name not in ext.features:
                continue
            feat = ext.features[feature_name]
            signs.append(feat.sign)
            ranks.append(feat.rank)
            norm_val = _normalize_value(feat.value)
            values.append(norm_val)
            if norm_val is not None:
                has_non_null_value = True

        value_agreement: Optional[float] = None
        if has_non_null_value:
            value_agreement = _proportion_agreement(values)

        result.per_feature[feature_name] = FeatureAgreement(
            sign_agreement=_proportion_agreement(signs),
            rank_agreement=_proportion_agreement(ranks),
            value_agreement=value_agreement,
        )
        scored_features.append(feature_name)

    if scored_features:
        sign_scores = [result.per_feature[f].sign_agreement for f in scored_features]
        rank_scores = [result.per_feature[f].rank_agreement for f in scored_features]
        result.narrative_reliability_score = (sum(sign_scores) + sum(rank_scores)) / (
            2 * len(scored_features)
        )
        result.flagged_low_reliability = (
            result.narrative_reliability_score < reliability_threshold
        )

    return result


def try_parse_extraction(
    raw: str,
    feature_names: List[str],
) -> tuple[Optional[ExtractionResult], Optional[str]]:
    """Parse extraction response; return (result, error_message)."""
    try:
        return parse_extraction_response(raw, feature_names), None
    except ValueError as exc:
        return None, str(exc)
