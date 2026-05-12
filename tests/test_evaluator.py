"""
tests/test_evaluator.py
Handcrafted test cases for each of the five hallucination types.

Run with:
    pytest tests/test_evaluator.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EvaluationConfig
from src.evaluator import (
    EvaluationResult,
    evaluate_narrative,
    _check_sign_inversion,
    _check_rank_swap,
    _check_feature_fabrication,
    _check_magnitude_distortion,
    _check_omission,
)

# Default config used across tests
DEFAULT_CFG = EvaluationConfig(top_k_features=3, magnitude_threshold=0.5)

# Ground-truth SHAP values for a representative adult-income instance
SHAP_ADULT = {
    "age":           0.42,   # positive, large (top-1)
    "education_num": 0.31,   # positive, medium (top-2)
    "hours_per_week": 0.18,  # positive, medium (top-3)
    "capital_gain":  -0.05,  # negative, small
}

# ============================================================
# Helper
# ============================================================

def _eval(narrative: str, shap: dict = SHAP_ADULT) -> EvaluationResult:
    return evaluate_narrative(narrative, shap, DEFAULT_CFG)


# ============================================================
# 1. Sign inversion
# ============================================================

class TestSignInversion:
    def test_no_inversion_positive_feature(self):
        narrative = (
            "The applicant's age (52) increased the probability of high income, "
            "contributing positively and raising the model's output."
        )
        result = _eval(narrative)
        assert not result.sign_inversion

    def test_no_inversion_negative_feature(self):
        narrative = (
            "Capital gain had a slightly negative effect, reducing the predicted income class."
        )
        result = _eval(narrative)
        assert not result.sign_inversion

    def test_inversion_positive_shap_stated_as_negative(self):
        # age has positive SHAP (0.42) but narrative says it reduces / lowers
        narrative = (
            "The applicant's age significantly decreased the likelihood of high income, "
            "pushing the prediction lower."
        )
        result = _eval(narrative)
        assert result.sign_inversion, "Should flag: positive SHAP for age described as negative"

    def test_inversion_negative_shap_stated_as_positive(self):
        # capital_gain has negative SHAP (-0.05) but narrative says it boosted
        narrative = (
            "Capital gain boosted the prediction substantially, increasing the probability "
            "of high income."
        )
        shap = {**SHAP_ADULT, "capital_gain": -0.45}  # make it clearly large negative
        result = evaluate_narrative(narrative, shap, DEFAULT_CFG)
        assert result.sign_inversion, "Should flag: negative SHAP for capital_gain described as positive"

    def test_ambiguous_context_no_false_positive(self):
        # Narrative is neutral — no direction words near age
        narrative = "The applicant is 52 years old and has 13 years of education."
        result = _eval(narrative)
        assert not result.sign_inversion


# ============================================================
# 2. Rank swap
# ============================================================

class TestRankSwap:
    def test_no_rank_swap_correct_top(self):
        # age is correctly identified as most important
        narrative = (
            "Age was the most important factor, followed by education level "
            "and hours worked per week."
        )
        result = _eval(narrative)
        assert not result.rank_swap

    def test_rank_swap_wrong_feature_as_top(self):
        # capital_gain (bottom feature) described as most important
        narrative = (
            "Capital gain was the most important factor driving the prediction, "
            "with age and education playing secondary roles."
        )
        result = _eval(narrative)
        assert result.rank_swap, "Should flag: capital_gain described as most important but it is bottom-ranked"

    def test_rank_swap_second_feature_as_primary(self):
        # education_num described as primary driver, but age has higher |SHAP|
        # Use the exact feature name (or its underscore variant) so the proximity
        # search can find it near the superlative phrase.
        narrative = (
            "Education_num was the primary driver of the model's decision, "
            "with age contributing a smaller effect."
        )
        result = _eval(narrative)
        assert result.rank_swap, "Should flag: education_num described as primary, but age is top"

    def test_no_rank_swap_no_superlatives(self):
        narrative = (
            "Age, education, and hours worked all contributed positively "
            "to the model's prediction."
        )
        result = _eval(narrative)
        assert not result.rank_swap


# ============================================================
# 3. Feature fabrication
# ============================================================

class TestFeatureFabrication:
    def test_no_fabrication_valid_features(self):
        narrative = (
            "The applicant's age and education_num were the dominant contributors. "
            "Hours_per_week and capital_gain also played a role."
        )
        result = _eval(narrative)
        assert not result.feature_fabrication

    def test_fabrication_invented_feature(self):
        # 'marital_status' is not in SHAP_ADULT
        narrative = (
            "Age was important, and marital_status also contributed positively "
            "to the prediction."
        )
        result = _eval(narrative)
        assert result.feature_fabrication, "Should flag: marital_status not in SHAP dict"

    def test_fabrication_underscore_token_not_in_features(self):
        # 'job_type' not in SHAP_ADULT
        narrative = (
            "The model was heavily influenced by job_type, which is a key determinant."
        )
        result = _eval(narrative)
        assert result.feature_fabrication, "Should flag: job_type not in SHAP dict"


# ============================================================
# 4. Magnitude distortion
# ============================================================

class TestMagnitudeDistortion:
    def test_no_distortion_correct_labels(self):
        # age (large) described as significant, capital_gain (small) as minor
        narrative = (
            "Age had a significant positive effect. "
            "Capital gain had only a minor influence on the outcome."
        )
        result = _eval(narrative)
        assert not result.magnitude_distortion

    def test_distortion_large_feature_called_minor(self):
        # age has 0.42 SHAP (largest) but narrative calls it minor
        narrative = (
            "Age had only a slight and negligible effect on the prediction, "
            "contributing minimally to the output."
        )
        result = _eval(narrative)
        assert result.magnitude_distortion, "Should flag: large-SHAP age called minor"

    def test_distortion_small_feature_called_dominant(self):
        # capital_gain has 0.05 SHAP (smallest) but narrative calls it dominant
        narrative = (
            "Capital gain was the dominant and most powerful factor, "
            "substantially driving the model's decision."
        )
        result = _eval(narrative)
        assert result.magnitude_distortion, "Should flag: small-SHAP capital_gain called dominant"

    def test_no_distortion_medium_feature_neutral_label(self):
        narrative = (
            "Education level contributed to the prediction alongside other factors."
        )
        result = _eval(narrative)
        assert not result.magnitude_distortion


# ============================================================
# 5. Omission
# ============================================================

class TestOmission:
    def test_no_omission_all_top3_mentioned(self):
        narrative = (
            "Age, education_num, and hours_per_week all contributed to the model's "
            "prediction of high income."
        )
        result = _eval(narrative)
        assert not result.omission

    def test_omission_top1_not_mentioned(self):
        # age (top-1) completely absent
        narrative = (
            "Education and hours per week drove the prediction, with capital gain "
            "having a small negative effect."
        )
        result = _eval(narrative)
        assert result.omission, "Should flag: age (top-1 SHAP) not mentioned"

    def test_omission_top3_partial(self):
        # hours_per_week (top-3) missing
        narrative = (
            "Age was the strongest driver. Education level also contributed positively."
        )
        result = _eval(narrative)
        assert result.omission, "Should flag: hours_per_week (top-3) not mentioned"

    def test_no_omission_with_variant_names(self):
        # Test that 'education num' (space variant) is found for 'education_num'
        narrative = (
            "The applicant's age, education num, and hours per week all had positive "
            "effects on the predicted income."
        )
        result = _eval(narrative)
        assert not result.omission


# ============================================================
# EvaluationResult helpers
# ============================================================

class TestEvaluationResult:
    def test_any_hallucination_false_when_clean(self):
        r = EvaluationResult()
        assert not r.any_hallucination

    def test_any_hallucination_true_when_flagged(self):
        r = EvaluationResult(sign_inversion=True)
        assert r.any_hallucination

    def test_to_dict_has_all_keys(self):
        r = EvaluationResult(omission=True, notes=["omission: age missing"])
        d = r.to_dict()
        for key in ["sign_inversion", "rank_swap", "feature_fabrication",
                    "magnitude_distortion", "omission", "any_hallucination", "notes"]:
            assert key in d

    def test_notes_str_empty(self):
        r = EvaluationResult()
        assert r.notes_str() == ""

    def test_notes_str_joined(self):
        r = EvaluationResult(notes=["a", "b"])
        assert r.notes_str() == "a; b"
