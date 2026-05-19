"""
src/evaluation/compare_to_shap.py
Deterministic comparison of extraction output to ground-truth SHAP values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.evaluation.extraction_parser import ExtractionResult

HALLUCINATION_TYPES = [
    "sign_inversion",
    "rank_swap",
    "feature_fabrication",
    "omission",
]

_SHAP_EPS = 1e-9


@dataclass
class ComparisonResult:
    sign_inversion: int = 0
    rank_swap: int = 0
    feature_fabrication: int = 0
    omission: int = 0
    any_hallucination: int = 0
    notes: Dict[str, str] = field(default_factory=dict)

    def flags_dict(self) -> Dict[str, int]:
        return {
            "sign_inversion": self.sign_inversion,
            "rank_swap": self.rank_swap,
            "feature_fabrication": self.feature_fabrication,
            "omission": self.omission,
            "any_hallucination": self.any_hallucination,
        }


def _shap_lookup(shap_values_sorted: List[List[Any]]) -> Dict[str, float]:
    return {str(item[0]): float(item[1]) for item in shap_values_sorted}


def _top_k_by_abs_shap(
    shap_map: Dict[str, float],
    k: int,
) -> List[Tuple[str, float]]:
    items = list(shap_map.items())
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return items[:k]


def extraction_top_k_set(
    extraction: ExtractionResult,
    top_k_features: int = 3,
) -> Optional[frozenset[str]]:
    """
    Top-k feature names by narrative ``rank`` (lowest ranks first), as a set.

    Returns ``None`` when fewer than *top_k_features* are mentioned, matching
    the rank-swap rule in :func:`compare_to_shap`.
    """
    mentioned = {
        name: feat
        for name, feat in extraction.features.items()
        if feat.exists
    }
    if len(mentioned) < top_k_features:
        return None
    narrative_top_k = sorted(
        mentioned.items(),
        key=lambda x: x[1].rank,
    )[:top_k_features]
    return frozenset(name for name, _ in narrative_top_k)


def compare_to_shap(
    extraction: ExtractionResult,
    shap_values_sorted: List[List[Any]],
    top_k_features: int = 3,
) -> ComparisonResult:
    """
    Derive four binary hallucination flags from extraction vs. SHAP ground truth.
    """
    result = ComparisonResult()
    shap_map = _shap_lookup(shap_values_sorted)
    notes: Dict[str, str] = {}

    # Feature fabrication
    if extraction.unknown_features:
        result.feature_fabrication = 1
        notes["feature_fabrication"] = (
            f"Unknown features in narrative: {extraction.unknown_features}"
        )

    mentioned = {
        name: feat
        for name, feat in extraction.features.items()
        if feat.exists
    }

    # Sign inversion
    sign_failures: List[str] = []
    for name, feat in mentioned.items():
        shap_val = shap_map.get(name)
        if shap_val is None or abs(shap_val) < _SHAP_EPS:
            continue
        shap_sign = 1 if shap_val > 0 else -1
        if feat.sign != shap_sign:
            sign_failures.append(
                f"{name}: narrative sign={feat.sign}, SHAP sign={shap_sign}"
            )
    if sign_failures:
        result.sign_inversion = 1
        notes["sign_inversion"] = "; ".join(sign_failures)

    # Rank swap: top-k narrative features (by rank) vs top-k SHAP by |SHAP| as sets
    narrative_top_set: Optional[Set[str]] = extraction_top_k_set(
        extraction, top_k_features
    )
    if narrative_top_set is not None and shap_map:
        shap_top_k = _top_k_by_abs_shap(shap_map, top_k_features)
        shap_top_set = {name for name, _ in shap_top_k}
        if narrative_top_set != shap_top_set:
            result.rank_swap = 1
            notes["rank_swap"] = (
                f"Narrative top-{top_k_features}: {sorted(narrative_top_set)}; "
                f"SHAP top-{top_k_features} by |value|: {sorted(shap_top_set)}"
            )

    # Omission: top-k SHAP features not in extraction
    top_k = _top_k_by_abs_shap(shap_map, top_k_features)
    mentioned_names = set(mentioned.keys())
    omitted = [name for name, _ in top_k if name not in mentioned_names]
    if omitted:
        result.omission = 1
        notes["omission"] = f"Top-{top_k_features} SHAP features not mentioned: {omitted}"

    result.notes = notes
    result.any_hallucination = int(
        any(
            getattr(result, t) == 1
            for t in HALLUCINATION_TYPES
        )
    )
    return result
