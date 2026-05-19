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
from src.evaluation.robustness_runner import select_narratives_for_robustness

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
            _extraction(
                age=_feat(rank=0, sign=1),
                education=_feat(rank=1, sign=1),
                hours_per_week=_feat(rank=2, sign=1),
            ),
            _extraction(
                age=_feat(rank=0, sign=1),
                education=_feat(rank=1, sign=1),
                hours_per_week=_feat(rank=2, sign=1),
            ),
            _extraction(
                age=_feat(rank=0, sign=1),
                education=_feat(rank=1, sign=1),
                hours_per_week=_feat(rank=2, sign=1),
            ),
        ]
        result = compute_robustness(runs, n_requested_runs=5, min_successful_runs=3)
        assert result.n_successful_runs == 3
        assert not result.extraction_unreliable
        assert result.per_feature["age"].sign_agreement == 1.0
        assert result.top_k_set_agreement == 1.0
        assert result.narrative_reliability_score == 1.0
        assert not result.flagged_low_reliability

    def test_top_k_set_ignores_order_within_set(self):
        runs = [
            _extraction(
                age=_feat(rank=0, sign=1),
                education=_feat(rank=1, sign=1),
                hours_per_week=_feat(rank=2, sign=1),
            ),
            _extraction(
                education=_feat(rank=0, sign=1),
                hours_per_week=_feat(rank=1, sign=1),
                age=_feat(rank=2, sign=1),
            ),
            _extraction(
                hours_per_week=_feat(rank=0, sign=1),
                age=_feat(rank=1, sign=1),
                education=_feat(rank=2, sign=1),
            ),
        ]
        result = compute_robustness(runs, n_requested_runs=3, min_successful_runs=3)
        assert result.top_k_set_agreement == 1.0

    def test_split_agreement(self):
        runs = [
            _extraction(
                age=_feat(rank=0, sign=1),
                education=_feat(rank=1, sign=1),
                hours_per_week=_feat(rank=2, sign=1),
            ),
            _extraction(
                age=_feat(rank=0, sign=1),
                education=_feat(rank=1, sign=1),
                hours_per_week=_feat(rank=2, sign=1),
            ),
            _extraction(
                age=_feat(rank=0, sign=1),
                education=_feat(rank=1, sign=1),
                hours_per_week=_feat(rank=2, sign=1),
            ),
            _extraction(
                age=_feat(rank=0, sign=-1),
                education=_feat(rank=1, sign=-1),
                hours_per_week=_feat(rank=2, sign=-1),
            ),
            _extraction(
                age=_feat(rank=0, sign=-1),
                education=_feat(rank=1, sign=-1),
                hours_per_week=_feat(rank=2, sign=-1),
            ),
        ]
        result = compute_robustness(runs, n_requested_runs=5, min_successful_runs=3)
        assert result.per_feature["age"].sign_agreement == pytest.approx(0.6)
        assert result.top_k_set_agreement == 1.0
        assert result.narrative_reliability_score == pytest.approx(0.7)
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
        runs = [
            _extraction(
                age=_feat(rank=0),
                education=_feat(rank=1),
                hours_per_week=_feat(rank=2),
            )
            for _ in range(3)
        ]
        result = compute_robustness(runs, n_requested_runs=5, min_successful_runs=3)
        d = result.to_dict()
        assert d["n_successful_runs"] == 3
        assert "per_feature" in d
        assert "age" in d["per_feature"]
        assert "sign_agreement" in d["per_feature"]["age"]
        assert "rank_agreement" not in d["per_feature"]["age"]
        assert d["top_k_set_agreement"] == 1.0


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


class TestSelectNarrativesForRobustness:
    def _narratives(self) -> "pd.DataFrame":
        import pandas as pd

        rows = []
        for i, strategy in enumerate(["martens"] * 10 + ["chain_of_thought"] * 12):
            rows.append({
                "narrative_id": f"n{i}",
                "prompt_strategy": strategy,
                "error": "",
            })
        return pd.DataFrame(rows)

    def _evals(self, narrative_ids) -> "pd.DataFrame":
        import pandas as pd

        return pd.DataFrame({
            "narrative_id": narrative_ids,
            "extraction_json": ['{"features": {}}'] * len(narrative_ids),
            "parse_error": [""] * len(narrative_ids),
        })

    def test_balanced_subsample_equal_per_strategy(self):
        narr = self._narratives()
        evals = self._evals(narr["narrative_id"].tolist())
        selected, info = select_narratives_for_robustness(
            narr, evals, fraction=0.1, seed=42, balanced=True,
        )
        assert info["n_selected"] == 2
        assert info["per_strategy"]["martens"] == 1
        assert info["per_strategy"]["chain_of_thought"] == 1
        assert len(selected) == 2

    def test_ten_percent_of_272_eligible(self):
        import pandas as pd

        narr_rows = []
        eval_rows = []
        for i in range(130):
            nid = f"m{i}"
            narr_rows.append({
                "narrative_id": nid,
                "prompt_strategy": "martens",
                "error": "",
            })
            eval_rows.append({
                "narrative_id": nid,
                "extraction_json": "{}",
                "parse_error": "",
            })
        for i in range(142):
            nid = f"c{i}"
            narr_rows.append({
                "narrative_id": nid,
                "prompt_strategy": "chain_of_thought",
                "error": "",
            })
            eval_rows.append({
                "narrative_id": nid,
                "extraction_json": "{}",
                "parse_error": "",
            })
        narr = pd.DataFrame(narr_rows)
        evals = pd.DataFrame(eval_rows)
        selected, info = select_narratives_for_robustness(
            narr, evals, fraction=0.1, seed=42, balanced=True,
        )
        assert info["n_eligible"] == 272
        assert info["n_target"] == 27
        assert info["per_strategy"]["martens"] == 13
        assert info["per_strategy"]["chain_of_thought"] == 13
        assert len(selected) == 26


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
        import os
        from src.config import load_config
        from src.evaluation.robustness_runner import run_robustness

        project_root = Path(__file__).resolve().parent.parent
        os.chdir(project_root)
        cfg = load_config(project_root / "config/default.yaml")
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
            "model_id": "llama3-70b",
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

        import pandas as pd

        mock_evals_df = pd.DataFrame([{
            "narrative_id": "n1",
            "extraction_json": extraction_json,
            "parse_error": "",
        }])

        eval_path = Path(cfg.evaluation.export_dir) / run_id / "evaluations.csv"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        mock_evals_df.to_csv(eval_path, index=False)

        with patch(
            "src.evaluation.robustness_runner.evaluations_csv_path",
            return_value=eval_path,
        ):
            with patch.dict(
                "os.environ",
                {"HF_MISTRAL_ENDPOINT_URL": "https://test-mistral.endpoint.hf.cloud"},
            ):
                with patch(
                    "src.evaluation.robustness_runner._feature_names_for_dataset"
                ) as mock_fn:
                    mock_fn.return_value = FEATURE_NAMES
                    run_robustness(
                        cfg,
                        run_id,
                        dry_run=False,
                        subsample_fraction=1.0,
                    )

        out = Path(cfg.evaluation.export_dir) / run_id / "robustness.jsonl"
        assert out.exists()
        line = out.read_text(encoding="utf-8").strip()
        rec = json.loads(line)
        assert "robustness" in rec
        assert rec["robustness"]["n_successful_runs"] == 5
