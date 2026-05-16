"""
src/evaluation/extraction_parser.py
Parse and validate JSON returned by the extraction LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class FeatureExtraction:
    exists: bool
    rank: int
    sign: int
    value: Any
    assumption: str


@dataclass
class ExtractionResult:
    features: Dict[str, FeatureExtraction]
    unknown_features: List[str] = field(default_factory=list)


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def parse_extraction_response(
    raw: str,
    feature_names: List[str],
) -> ExtractionResult:
    """
    Parse extraction LLM output and validate against the dataset feature list.

    Raises:
        ValueError: if JSON is invalid or violates schema rules.
    """
    allowed: Set[str] = set(feature_names)
    text = _strip_markdown_fences(raw)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in extraction response: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Extraction root must be a JSON object")

    features_raw = data.get("features")
    if features_raw is None:
        raise ValueError("Missing 'features' key in extraction JSON")
    if not isinstance(features_raw, dict):
        raise ValueError("'features' must be an object")

    unknown_raw = data.get("unknown_features", [])
    if unknown_raw is None:
        unknown_raw = []
    if not isinstance(unknown_raw, list):
        raise ValueError("'unknown_features' must be a list")
    unknown_features = [str(x) for x in unknown_raw]

    overlap = set(unknown_features) & allowed
    if overlap:
        raise ValueError(
            f"unknown_features overlaps valid feature names: {sorted(overlap)}"
        )

    features: Dict[str, FeatureExtraction] = {}
    for name, entry in features_raw.items():
        if name not in allowed:
            raise ValueError(
                f"Feature '{name}' in extraction is not in the dataset feature list"
            )
        if not isinstance(entry, dict):
            raise ValueError(f"Feature entry for '{name}' must be an object")

        exists = bool(entry.get("exists", True))
        rank = entry.get("rank")
        if rank is None or not isinstance(rank, (int, float)):
            raise ValueError(f"Feature '{name}' missing valid integer 'rank'")
        rank = int(rank)
        if rank < 0:
            raise ValueError(f"Feature '{name}' rank must be >= 0")

        sign = entry.get("sign")
        if sign not in (1, -1, "1", "-1"):
            raise ValueError(f"Feature '{name}' sign must be 1 or -1")
        sign = int(sign)

        assumption = entry.get("assumption", "")
        if not isinstance(assumption, str) or not assumption.strip():
            raise ValueError(f"Feature '{name}' requires non-empty 'assumption'")

        value = entry.get("value")
        if "value" not in entry:
            value = None

        features[name] = FeatureExtraction(
            exists=exists,
            rank=rank,
            sign=sign,
            value=value,
            assumption=assumption.strip(),
        )

    return ExtractionResult(features=features, unknown_features=unknown_features)
