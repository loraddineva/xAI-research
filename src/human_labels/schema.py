"""
src/human_labels/schema.py
Validate human extractions (sign, rank, unknown_features only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from src.evaluation.extraction_parser import ExtractionResult, FeatureExtraction


@dataclass
class HumanFeatureLabel:
    rank: int
    sign: int


@dataclass
class HumanLabelRecord:
    narrative_id: str
    run_id: str
    dataset: str
    instance_id: int
    prompt_strategy: str
    annotator: str
    features: Dict[str, HumanFeatureLabel]
    unknown_features: List[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "narrative_id": self.narrative_id,
            "run_id": self.run_id,
            "dataset": self.dataset,
            "instance_id": self.instance_id,
            "prompt_strategy": self.prompt_strategy,
            "annotator": self.annotator,
            "created_at": self.created_at,
            "features": {
                name: {"rank": feat.rank, "sign": feat.sign}
                for name, feat in self.features.items()
            },
            "unknown_features": list(self.unknown_features),
        }


def record_from_dict(data: dict) -> HumanLabelRecord:
    """Build a HumanLabelRecord from a JSONL row dict."""
    features_raw = data.get("features") or {}
    features: Dict[str, HumanFeatureLabel] = {}
    for name, entry in features_raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Feature entry for '{name}' must be an object")
        features[name] = HumanFeatureLabel(
            rank=int(entry["rank"]),
            sign=int(entry["sign"]),
        )

    unknown = data.get("unknown_features") or []
    if not isinstance(unknown, list):
        raise ValueError("'unknown_features' must be a list")

    return HumanLabelRecord(
        narrative_id=str(data["narrative_id"]),
        run_id=str(data.get("run_id", "")),
        dataset=str(data.get("dataset", "")),
        instance_id=int(data.get("instance_id", 0)),
        prompt_strategy=str(data.get("prompt_strategy", "")),
        annotator=str(data.get("annotator", "")),
        features=features,
        unknown_features=[str(x) for x in unknown],
        created_at=str(data.get("created_at", "")),
    )


def validate_human_label(
    record: HumanLabelRecord,
    feature_names: List[str],
) -> None:
    """
    Validate a human label against the dataset feature list.

    Raises:
        ValueError: on schema violations.
    """
    allowed: Set[str] = set(feature_names)

    for name in record.features:
        if name not in allowed:
            raise ValueError(
                f"Feature '{name}' is not in the dataset feature list"
            )

    ranks: List[int] = []
    for name, feat in record.features.items():
        if feat.sign not in (1, -1):
            raise ValueError(
                f"Feature '{name}' sign must be 1 or -1, got {feat.sign}"
            )
        if feat.rank < 0:
            raise ValueError(f"Feature '{name}' rank must be >= 0")
        ranks.append(feat.rank)

    if len(ranks) != len(set(ranks)):
        raise ValueError("Ranks must be unique among mentioned features")

    overlap = set(record.unknown_features) & allowed
    if overlap:
        raise ValueError(
            f"unknown_features overlaps valid feature names: {sorted(overlap)}"
        )


def human_label_to_extraction_result(
    record: HumanLabelRecord,
) -> ExtractionResult:
    """Convert human labels to ExtractionResult for shared comparison helpers."""
    features: Dict[str, FeatureExtraction] = {}
    for name, feat in record.features.items():
        features[name] = FeatureExtraction(
            exists=True,
            rank=feat.rank,
            sign=feat.sign,
            value=None,
            assumption="human",
        )
    return ExtractionResult(
        features=features,
        unknown_features=list(record.unknown_features),
    )


def new_label_record(
    narrative_id: str,
    run_id: str,
    dataset: str,
    instance_id: int,
    prompt_strategy: str,
    annotator: str,
    features: Dict[str, HumanFeatureLabel],
    unknown_features: List[str],
) -> HumanLabelRecord:
    """Create a record with a fresh UTC timestamp."""
    return HumanLabelRecord(
        narrative_id=narrative_id,
        run_id=run_id,
        dataset=dataset,
        instance_id=instance_id,
        prompt_strategy=prompt_strategy,
        annotator=annotator,
        features=features,
        unknown_features=unknown_features,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
