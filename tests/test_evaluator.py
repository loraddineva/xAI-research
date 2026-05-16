"""
tests/test_evaluator.py
Tests for extraction parsing, SHAP comparison, and mocked evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.compare_to_shap import compare_to_shap
from src.evaluation.extraction_parser import (
    ExtractionResult,
    FeatureExtraction,
    parse_extraction_response,
)
from src.evaluation.evaluator import run_evaluation

FEATURE_NAMES = ["age", "education", "hours_per_week"]


def _extraction_json(**features: dict) -> str:
    body = {
        "features": features,
        "unknown_features": [],
    }
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TestExtractionParser:
    def test_valid_json(self):
        raw = _extraction_json(
            age={
                "exists": True,
                "rank": 0,
                "sign": 1,
                "value": 39,
                "assumption": "Age increases income.",
            }
        )
        result = parse_extraction_response(raw, FEATURE_NAMES)
        assert "age" in result.features
        assert result.features["age"].rank == 0
        assert result.features["age"].sign == 1

    def test_strips_markdown_fences(self):
        inner = _extraction_json(
            education={
                "exists": True,
                "rank": 0,
                "sign": -1,
                "value": None,
                "assumption": "Education lowers risk.",
            }
        )
        raw = f"```json\n{inner}\n```"
        result = parse_extraction_response(raw, FEATURE_NAMES)
        assert "education" in result.features

    def test_rejects_unknown_feature_key(self):
        raw = json.dumps({
            "features": {
                "not_a_feature": {
                    "exists": True,
                    "rank": 0,
                    "sign": 1,
                    "value": None,
                    "assumption": "Invalid.",
                }
            },
            "unknown_features": [],
        })
        with pytest.raises(ValueError, match="not in the dataset"):
            parse_extraction_response(raw, FEATURE_NAMES)

    def test_rejects_unknown_overlap(self):
        raw = json.dumps({
            "features": {},
            "unknown_features": ["age"],
        })
        with pytest.raises(ValueError, match="overlaps"):
            parse_extraction_response(raw, FEATURE_NAMES)

    def test_rejects_invalid_sign(self):
        raw = _extraction_json(
            age={
                "exists": True,
                "rank": 0,
                "sign": 2,
                "value": None,
                "assumption": "Bad sign.",
            }
        )
        with pytest.raises(ValueError, match="sign must be"):
            parse_extraction_response(raw, FEATURE_NAMES)


# ---------------------------------------------------------------------------
# compare_to_shap
# ---------------------------------------------------------------------------

def _shap(age: float = 0.5, education: float = -0.3, hours: float = 0.1):
    return [
        ["age", age],
        ["education", education],
        ["hours_per_week", hours],
    ]


class TestCompareToShap:
    def test_sign_inversion(self):
        ext = ExtractionResult(
            features={
                "age": FeatureExtraction(
                    exists=True, rank=0, sign=-1, value=None,
                    assumption="Age decreases income.",
                )
            }
        )
        cmp = compare_to_shap(ext, _shap(age=0.5), top_k_features=3)
        assert cmp.sign_inversion == 1

    def test_rank_swap(self):
        ext = ExtractionResult(
            features={
                "education": FeatureExtraction(
                    exists=True, rank=0, sign=-1, value=None,
                    assumption="Education is most important.",
                )
            }
        )
        cmp = compare_to_shap(ext, _shap(), top_k_features=3)
        assert cmp.rank_swap == 1

    def test_feature_fabrication(self):
        ext = ExtractionResult(
            features={},
            unknown_features=["salary_bracket"],
        )
        cmp = compare_to_shap(ext, _shap(), top_k_features=3)
        assert cmp.feature_fabrication == 1

    def test_omission(self):
        ext = ExtractionResult(
            features={
                "hours_per_week": FeatureExtraction(
                    exists=True, rank=0, sign=1, value=None,
                    assumption="Hours matter.",
                )
            }
        )
        cmp = compare_to_shap(ext, _shap(), top_k_features=2)
        assert cmp.omission == 1

    def test_no_hallucination_when_aligned(self):
        ext = ExtractionResult(
            features={
                "age": FeatureExtraction(
                    exists=True, rank=0, sign=1, value=None,
                    assumption="Age drives prediction.",
                ),
                "education": FeatureExtraction(
                    exists=True, rank=1, sign=-1, value=None,
                    assumption="Education reduces income.",
                ),
                "hours_per_week": FeatureExtraction(
                    exists=True, rank=2, sign=1, value=None,
                    assumption="Hours help.",
                ),
            }
        )
        cmp = compare_to_shap(ext, _shap(), top_k_features=3)
        assert cmp.any_hallucination == 0


# ---------------------------------------------------------------------------
# End-to-end (mocked LLM)
# ---------------------------------------------------------------------------

MOCK_EXTRACTION = json.dumps({
    "features": {
        "age": {
            "exists": True,
            "rank": 0,
            "sign": 1,
            "value": None,
            "assumption": "Older applicants have higher income.",
        }
    },
    "unknown_features": [],
})


class TestRunEvaluationMocked:
    @patch("src.evaluation.evaluator.LLMClient")
    @patch("src.evaluation.evaluator.load_narratives_csv")
    @patch("src.evaluation.evaluator.narratives_csv_path")
    @patch("src.evaluation.evaluator.generation_run_dir")
    def test_run_evaluation_writes_csv(
        self,
        mock_gen_run_dir,
        mock_narr_csv_path,
        mock_load_narr,
        mock_llm_cls,
    ):
        from src.config import load_config

        cfg = load_config()
        run_id = "test_run_eval"

        mock_gen_run_dir.return_value = Path("outputs/generation") / run_id
        mock_narr_csv_path.return_value = mock_gen_run_dir.return_value / "narratives.csv"

        mock_load_narr.return_value = __import__("pandas").DataFrame([{
            "narrative_id": "n1",
            "run_id": run_id,
            "dataset": "adult",
            "instance_id": 0,
            "model_id": "claude-opus",
            "prompt_strategy": "martens",
            "narrative_text": "Age drove the prediction.",
            "shap_values_sorted": json.dumps(_shap()),
            "error": "",
        }])

        mock_client = MagicMock()
        mock_client.generate.return_value = MOCK_EXTRACTION
        mock_llm_cls.return_value = mock_client

        with patch.object(Path, "exists", return_value=True):
            with patch("src.evaluation.evaluator._feature_names_for_dataset") as mock_fn:
                mock_fn.return_value = FEATURE_NAMES
                result = run_evaluation(cfg, run_id, dry_run=False, n_limit=1)

        assert result == run_id
        eval_csv = Path(cfg.evaluation.export_dir) / run_id / "evaluations.csv"
        assert eval_csv.exists()
        df = __import__("pandas").read_csv(eval_csv)
        assert len(df) >= 1
