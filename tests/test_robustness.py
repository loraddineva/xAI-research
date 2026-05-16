"""
tests/test_robustness.py
Tests for extraction robustness agreement scoring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.extraction_parser import (
    ExtractionResult,
    FeatureExtraction,
)
from src.evaluation.robustness import compute_robustness, try_parse_extraction

FEATURE_NAMES = ["age", "education", "hours_per_week"]


def _feat(
    rank: int = 0,
    sign: int = 1,
    value=None,
) -> FeatureExtraction:
    return FeatureExtraction(
        exists=True,
        rank=rank,
        sign=sign,
        value=value,
        assumption="Test assumption.",
    )


def _extraction(**features: FeatureExtraction) -> ExtractionResult:
    return ExtractionResult(features=dict(features))


class TestComputeRobustness:
    def test_full_agreement(self):
        runs = [
            _extraction(age=_feat(rank=0, sign=1)),
            _extraction(age=_feat(rank=0, sign=1)),
            _extraction(age=_feat(rank=0, sign=1)),
        ]
        result = compute_robustness(runs, n_requested_runs=5, min_successful_runs=3)
        assert result.n_successful_runs == 3
        assert not result.extraction_unreliable
        assert result.per_feature["age"].sign_agreement == 1.0
        assert result.per_feature["age"].rank_agreement == 1.0
        assert result.narrative_reliability_score == 1.0
        assert not result.flagged_low_reliability

    def test_split_agreement(self):
        runs = [
            _extraction(age=_feat(rank=0, sign=1)),
            _extraction(age=_feat(rank=0, sign=1)),
            _extraction(age=_feat(rank=0, sign=1)),
            _extraction(age=_feat(rank=1, sign=-1)),
            _extraction(age=_feat(rank=1, sign=-1)),
        ]
        result = compute_robustness(runs, n_requested_runs=5, min_successful_runs=3)
        assert result.per_feature["age"].sign_agreement == pytest.approx(0.6)
        assert result.per_feature["age"].rank_agreement == pytest.approx(0.6)
        assert result.narrative_reliability_score == pytest.approx(0.6)
        assert result.flagged_low_reliability

    def test_too_few_successful_runs(self):
        runs = [
            _extraction(age=_feat()),
            _extraction(age=_feat()),
        ]
        result = compute_robustness(runs, n_requested_runs=5, min_successful_runs=3)
        assert result.extraction_unreliable
        assert result.narrative_reliability_score is None
        assert result.per_feature == {}

    def test_value_agreement_null_when_no_values(self):
        runs = [_extraction(age=_feat(value=None)) for _ in range(3)]
        result = compute_robustness(runs, n_requested_runs=3, min_successful_runs=3)
        assert result.per_feature["age"].value_agreement is None

    def test_value_agreement_when_present(self):
        runs = [
            _extraction(age=_feat(value=39)),
            _extraction(age=_feat(value=39)),
            _extraction(age=_feat(value=40)),
        ]
        result = compute_robustness(runs, n_requested_runs=3, min_successful_runs=3)
        assert result.per_feature["age"].value_agreement == pytest.approx(2 / 3)

    def test_to_dict_schema(self):
        runs = [_extraction(age=_feat()) for _ in range(3)]
        result = compute_robustness(runs, n_requested_runs=5, min_successful_runs=3)
        d = result.to_dict()
        assert d["n_successful_runs"] == 3
        assert "per_feature" in d
        assert "age" in d["per_feature"]
        assert "sign_agreement" in d["per_feature"]["age"]


class TestTryParseExtraction:
    def test_valid_json(self):
        raw = json.dumps({
            "features": {
                "age": {
                    "exists": True,
                    "rank": 0,
                    "sign": 1,
                    "value": None,
                    "assumption": "Age increases income.",
                }
            },
            "unknown_features": [],
        })
        ext, err = try_parse_extraction(raw, FEATURE_NAMES)
        assert err is None
        assert ext is not None
        assert "age" in ext.features

    def test_invalid_json_returns_error(self):
        ext, err = try_parse_extraction("not json", FEATURE_NAMES)
        assert ext is None
        assert err is not None


class TestReliabilitySummary:
    def test_splits_by_threshold(self):
        import pandas as pd

        from src.evaluation.robustness_runner import reliability_summary

        df = pd.DataFrame([
            {"any_hallucination": 1, "narrative_reliability_score": 0.9},
            {"any_hallucination": 0, "narrative_reliability_score": 0.9},
            {"any_hallucination": 1, "narrative_reliability_score": 0.5},
        ])
        summary = reliability_summary(df)
        assert len(summary) == 2
        high = summary[summary["reliability_group"] == "high_reliability"]
        assert high["n"].iloc[0] == 2


class TestRunRobustnessMocked:
    @patch("src.evaluation.robustness_runner.LLMClient")
    @patch("src.evaluation.robustness_runner.load_narratives_csv")
    @patch("src.evaluation.robustness_runner.narratives_csv_path")
    @patch("src.evaluation.robustness_runner.run_dir")
    def test_writes_robustness_jsonl(
        self,
        mock_run_dir,
        mock_narr_csv_path,
        mock_load_narr,
        mock_llm_cls,
    ):
        from src.config import load_config
        from src.evaluation.robustness_runner import run_robustness

        cfg = load_config()
        run_id = "test_robustness_run"

        mock_run_dir.return_value = Path("outputs/generation") / run_id
        narr_path = MagicMock()
        narr_path.exists.return_value = True
        mock_narr_csv_path.return_value = narr_path

        mock_load_narr.return_value = __import__("pandas").DataFrame([{
            "narrative_id": "n1",
            "run_id": run_id,
            "dataset": "adult",
            "instance_id": 0,
            "model_id": "claude-opus",
            "prompt_strategy": "martens",
            "narrative_text": "Age drove the prediction.",
            "error": "",
        }])

        extraction_json = json.dumps({
            "features": {
                "age": {
                    "exists": True,
                    "rank": 0,
                    "sign": 1,
                    "value": None,
                    "assumption": "Age increases income.",
                }
            },
            "unknown_features": [],
        })

        mock_client = MagicMock()
        mock_client.generate.return_value = extraction_json
        mock_llm_cls.return_value = mock_client

        with patch(
            "src.evaluation.robustness_runner.evaluations_csv_path"
        ) as mock_eval_csv:
            mock_eval_csv.return_value = Path("/nonexistent/evaluations.csv")
            with patch(
                "src.evaluation.robustness_runner._feature_names_for_dataset"
            ) as mock_fn:
                mock_fn.return_value = FEATURE_NAMES
                run_robustness(cfg, run_id, dry_run=False, n_limit=1)

        out = Path(cfg.evaluation.export_dir) / run_id / "robustness.jsonl"
        assert out.exists()
        line = out.read_text(encoding="utf-8").strip()
        rec = json.loads(line)
        assert "robustness" in rec
        assert rec["robustness"]["n_successful_runs"] == 5
