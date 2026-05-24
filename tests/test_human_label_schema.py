"""
tests/test_human_label_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.human_labels.schema import (
    HumanFeatureLabel,
    HumanLabelRecord,
    human_label_to_extraction_result,
    new_label_record,
    record_from_dict,
    validate_human_label,
)
from src.storage.human_labels_store import append_label, get_labeled_ids, load_labels

FEATURE_NAMES = ["age", "hours_per_week", "education"]


class TestHumanLabelSchema:
    def test_valid_record(self):
        record = new_label_record(
            narrative_id="n1",
            run_id="run1",
            dataset="adult",
            instance_id=1,
            prompt_strategy="martens",
            annotator="tester",
            features={
                "age": HumanFeatureLabel(rank=0, sign=1),
                "hours_per_week": HumanFeatureLabel(rank=1, sign=-1),
            },
            unknown_features=[],
        )
        validate_human_label(record, FEATURE_NAMES)
        ext = human_label_to_extraction_result(record)
        assert ext.features["age"].sign == 1
        assert ext.features["age"].assumption == "human"

    def test_rejects_invalid_sign(self):
        record = HumanLabelRecord(
            narrative_id="n1",
            run_id="run1",
            dataset="adult",
            instance_id=1,
            prompt_strategy="martens",
            annotator="tester",
            features={"age": HumanFeatureLabel(rank=0, sign=0)},
        )
        with pytest.raises(ValueError, match="sign must be"):
            validate_human_label(record, FEATURE_NAMES)

    def test_rejects_duplicate_ranks(self):
        record = HumanLabelRecord(
            narrative_id="n1",
            run_id="run1",
            dataset="adult",
            instance_id=1,
            prompt_strategy="martens",
            annotator="tester",
            features={
                "age": HumanFeatureLabel(rank=0, sign=1),
                "hours_per_week": HumanFeatureLabel(rank=0, sign=-1),
            },
        )
        with pytest.raises(ValueError, match="unique"):
            validate_human_label(record, FEATURE_NAMES)

    def test_rejects_unknown_overlap(self):
        record = HumanLabelRecord(
            narrative_id="n1",
            run_id="run1",
            dataset="adult",
            instance_id=1,
            prompt_strategy="martens",
            annotator="tester",
            features={},
            unknown_features=["age"],
        )
        with pytest.raises(ValueError, match="overlaps"):
            validate_human_label(record, FEATURE_NAMES)

    def test_record_from_dict(self):
        data = {
            "narrative_id": "n1",
            "run_id": "run1",
            "dataset": "adult",
            "instance_id": 5,
            "prompt_strategy": "martens",
            "annotator": "a",
            "features": {"age": {"rank": 0, "sign": 1}},
            "unknown_features": ["foo"],
        }
        record = record_from_dict(data)
        assert record.features["age"].rank == 0


class TestHumanLabelsStore:
    def test_append_upserts(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        r1 = new_label_record(
            "n1", "run", "adult", 1, "martens", "ann",
            {"age": HumanFeatureLabel(0, 1)}, [],
        )
        r2 = new_label_record(
            "n1", "run", "adult", 1, "martens", "ann",
            {"age": HumanFeatureLabel(1, -1)}, [],
        )
        append_label(path, r1)
        append_label(path, r2)
        loaded = load_labels(path)
        assert len(loaded) == 1
        assert loaded[0].features["age"].sign == -1
        assert get_labeled_ids(path, "ann") == {"n1"}
