"""
tests/test_compare_extractions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.compare_extractions import compare_extractions
from src.evaluation.extraction_parser import ExtractionResult, FeatureExtraction


def _ext(**features) -> ExtractionResult:
    parsed = {}
    for name, (rank, sign) in features.items():
        parsed[name] = FeatureExtraction(
            exists=True,
            rank=rank,
            sign=sign,
            value=None,
            assumption="test",
        )
    return ExtractionResult(features=parsed, unknown_features=[])


class TestCompareExtractions:
    def test_identical_extractions(self):
        human = _ext(age=(0, 1), hours_per_week=(1, -1), education=(2, 1))
        mistral = _ext(age=(0, 1), hours_per_week=(1, -1), education=(2, 1))
        ag = compare_extractions(human, mistral, top_k_features=3)
        assert ag.feature_set_jaccard == 1.0
        assert ag.sign_agreement == 1.0
        assert ag.rank_exact_match == 1.0
        assert ag.top_k_match == 1

    def test_sign_disagreement(self):
        human = _ext(age=(0, 1))
        mistral = _ext(age=(0, -1))
        ag = compare_extractions(human, mistral)
        assert ag.sign_agreement == 0.0

    def test_jaccard_partial_overlap(self):
        human = _ext(age=(0, 1), hours_per_week=(1, -1))
        mistral = _ext(age=(0, 1), education=(0, 1))
        ag = compare_extractions(human, mistral)
        assert ag.shared_feature_count == 1
        assert ag.feature_set_jaccard == pytest.approx(1 / 3)

    def test_top_k_skipped_when_insufficient(self):
        human = _ext(age=(0, 1))
        mistral = _ext(age=(0, 1))
        ag = compare_extractions(human, mistral, top_k_features=3)
        assert ag.top_k_match is None
        assert "top_k_match" in ag.notes

    def test_unknown_features_match(self):
        human = ExtractionResult(features={}, unknown_features=["foo"])
        mistral = ExtractionResult(features={}, unknown_features=["foo"])
        ag = compare_extractions(human, mistral)
        assert ag.unknown_match == 1
