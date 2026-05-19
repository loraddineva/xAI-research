"""
src/evaluation/robustness.py
Multi-sample extraction agreement for extraction-model reliability.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.evaluation.compare_to_shap import extraction_top_k_set
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
    value_agreement: Optional[float]


@dataclass
class RobustnessResult:
    n_successful_runs: int
    n_requested_runs: int
    per_feature: Dict[str, FeatureAgreement] = field(default_factory=dict)
    top_k_set_agreement: Optional[float] = None
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
                    "value_agreement": fa.value_agreement,
                }
                for name, fa in self.per_feature.items()
            },
            "top_k_set_agreement": self.top_k_set_agreement,
            "narrative_reliability_score": self.narrative_reliability_score,
            "flagged_low_reliability": self.flagged_low_reliability,
            "extraction_unreliable": self.extraction_unreliable,
        }


def compute_robustness(
    extractions: List[ExtractionResult],
    *,
    n_requested_runs: int,
    top_k_features: int = 3,
    min_successful_runs: int = 3,
    reliability_threshold: float = 0.8,
) -> RobustnessResult:
    """
    Compute per-field and narrative-level agreement across parsed extractions.

    Rank reliability uses the same top-*k* **set** rule as faithfulness evaluation
    (lowest ``rank`` values, order within the set ignored), not per-feature rank integers.

    Args:
        extractions: Successfully parsed extraction results (may be fewer than requested).
        n_requested_runs: Number of extraction calls attempted.
        top_k_features: Size of the importance set (default 3, matches evaluation).
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
        values: List[Any] = []
        has_non_null_value = False

        for ext in extractions:
            if feature_name not in ext.features:
                continue
            feat = ext.features[feature_name]
            if not feat.exists:
                continue
            signs.append(feat.sign)
            norm_val = _normalize_value(feat.value)
            values.append(norm_val)
            if norm_val is not None:
                has_non_null_value = True

        value_agreement: Optional[float] = None
        if has_non_null_value:
            value_agreement = _proportion_agreement(values)

        result.per_feature[feature_name] = FeatureAgreement(
            sign_agreement=_proportion_agreement(signs),
            value_agreement=value_agreement,
        )
        scored_features.append(feature_name)

    top_k_sets: List[frozenset[str]] = []
    for ext in extractions:
        top_set = extraction_top_k_set(ext, top_k_features)
        if top_set is not None:
            top_k_sets.append(top_set)

    if top_k_sets:
        result.top_k_set_agreement = _proportion_agreement(top_k_sets)

    score_parts: List[float] = []
    if scored_features:
        score_parts.extend(
            result.per_feature[f].sign_agreement for f in scored_features
        )
    if result.top_k_set_agreement is not None:
        score_parts.append(result.top_k_set_agreement)

    if score_parts:
        result.narrative_reliability_score = sum(score_parts) / len(score_parts)
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
