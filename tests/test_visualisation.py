"""
tests/test_visualisation.py
Tests for hallucination and robustness visualisation helpers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.visualisation.export import export_evaluation_figures_complete
from src.visualisation.hallucination_analysis import (
    fabrication_list,
    missing_from_rank_set_by_feature,
    omissions_by_feature,
    parse_notes,
    sign_inversion_by_feature,
    summarize_hallucination_breakdown,
)
from src.visualisation.hallucination_rates import (
    HALLUCINATION_TYPES,
    TYPE_LABELS,
    plot_type_by_strategy,
    strategy_label,
)
from src.visualisation.robustness_plots import (
    attach_robustness_columns,
    filter_reliable_extractions,
    load_robustness_df,
)


def _robustness_json(score: float, flagged: bool = False, unreliable: bool = False) -> str:
    return json.dumps({
        "narrative_reliability_score": score,
        "flagged_low_reliability": flagged,
        "extraction_unreliable": unreliable,
        "n_successful_runs": 5,
        "n_requested_runs": 5,
    })


def _sample_evals_df(n: int = 40) -> pd.DataFrame:
    rows = []
    strategies = ["martens", "chain_of_thought"]
    for i in range(n):
        strat = strategies[i % 2]
        rows.append({
            "narrative_id": f"n{i}",
            "model_id": "llama3-70b",
            "prompt_strategy": strat,
            "dataset": "adult",
            "sign_inversion": i % 7 == 0,
            "rank_swap": i % 3 == 0,
            "feature_fabrication": i % 11 == 0,
            "omission": i % 5 == 0,
            "any_hallucination": i % 3 == 0 or i % 5 == 0,
            "robustness_json": _robustness_json(
                0.85 if i % 4 != 0 else 0.65,
                flagged=i % 4 == 0,
            ),
        })
    return pd.DataFrame(rows)


class TestStrategyLabel:
    def test_known_strategies(self):
        assert strategy_label("martens") == "Direct"
        assert strategy_label("chain_of_thought") == "Chain-of-thought"
        assert strategy_label("direct") == "Direct"
        assert strategy_label("cot") == "Chain-of-thought"

    def test_unknown_passthrough(self):
        assert strategy_label("custom") == "custom"


class TestPlotTypeByStrategy:
    def test_builds_all_type_strategy_pairs(self):
        df = _sample_evals_df(10)
        records = []
        for htype in HALLUCINATION_TYPES:
            for strategy, grp in df.groupby("prompt_strategy"):
                records.append({
                    "Hallucination type": TYPE_LABELS[htype],
                    "Prompt strategy": strategy_label(str(strategy)),
                    "Rate (%)": grp[htype].mean() * 100,
                })
        plot_df = pd.DataFrame(records)
        assert len(plot_df) == len(HALLUCINATION_TYPES) * 2
        assert set(plot_df["Prompt strategy"]) == {"Direct", "Chain-of-thought"}

    def test_figure_returns_without_error(self):
        fig = plot_type_by_strategy(_sample_evals_df(10))
        assert fig.axes
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestRobustnessColumns:
    def test_attach_robustness_columns(self):
        df = _sample_evals_df(5)
        out = attach_robustness_columns(df)
        assert "narrative_reliability_score" in out.columns
        assert out["narrative_reliability_score"].notna().all()

    def test_filter_reliable_extractions(self):
        df = _sample_evals_df(20)
        filtered = filter_reliable_extractions(df, threshold=0.8)
        assert (filtered["narrative_reliability_score"] >= 0.8).all()
        assert not filtered["extraction_unreliable"].any()

    def test_load_robustness_df_from_evals(self):
        df = _sample_evals_df(5)
        rb = load_robustness_df(df)
        assert len(rb) == 5
        assert "narrative_reliability_score" in rb.columns


class TestHallucinationAnalysis:
    def _sample_notes_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "eval_id": "e1",
                "narrative_id": "n1",
                "instance_id": 1,
                "model_id": "llama3-70b",
                "prompt_strategy": "martens",
                "notes": json.dumps({
                    "sign_inversion": (
                        "age: narrative sign=-1, SHAP sign=1; "
                        "sex_Male: narrative sign=1, SHAP sign=-1"
                    ),
                    "rank_swap": (
                        "Narrative top-3: ['age', 'sex_Male', 'hours_per_week']; "
                        "SHAP top-3 by |value|: ['hours_per_week', 'sex_Male', "
                        "'marital_status_Non_Married']"
                    ),
                    "omission": (
                        "Top-3 SHAP features not mentioned: "
                        "['marital_status_Non_Married']"
                    ),
                    "feature_fabrication": (
                        "Unknown features in narrative: ['education']"
                    ),
                }),
                "unknown_features": "[]",
            },
            {
                "eval_id": "e2",
                "narrative_id": "n2",
                "instance_id": 2,
                "model_id": "llama3-70b",
                "prompt_strategy": "martens",
                "notes": json.dumps({
                    "sign_inversion": "age: narrative sign=-1, SHAP sign=1",
                    "omission": (
                        "Top-3 SHAP features not mentioned: "
                        "['hours_per_week', 'marital_status_Non_Married']"
                    ),
                }),
                "unknown_features": '["bonus"]',
            },
        ])

    def test_sign_inversion_by_feature(self):
        summary = sign_inversion_by_feature(self._sample_notes_df())
        age_rows = summary[summary["feature"] == "age"]
        assert len(age_rows) == 1
        assert age_rows.iloc[0]["count"] == 2
        assert age_rows.iloc[0]["narrative_sign"] == -1
        assert age_rows.iloc[0]["shap_sign"] == 1

    def test_missing_from_rank_set_by_feature(self):
        summary = missing_from_rank_set_by_feature(self._sample_notes_df())
        row = summary.loc[summary["feature"] == "marital_status_Non_Married", "count"]
        assert row.iloc[0] == 1

    def test_omissions_by_feature(self):
        summary = omissions_by_feature(self._sample_notes_df())
        assert (
            summary.loc[summary["feature"] == "marital_status_Non_Married", "count"].iloc[0]
            == 2
        )
        assert summary.loc[summary["feature"] == "hours_per_week", "count"].iloc[0] == 1

    def test_fabrication_list(self):
        fab = fabrication_list(self._sample_notes_df())
        assert set(fab["feature"]) == {"education", "bonus"}
        assert len(fab) == 2

    def test_summarize_hallucination_breakdown(self):
        tables = summarize_hallucination_breakdown(self._sample_notes_df())
        assert set(tables) == {
            "sign_inversion_by_feature",
            "missing_from_rank_set_by_feature",
            "omissions_by_feature",
            "fabrication_list",
        }

    def test_parse_notes_empty(self):
        assert parse_notes("{}") == {}
        assert parse_notes(None) == {}


class TestExportMinNGuard:
    def test_exits_when_below_min_n(self, tmp_path):
        cfg = load_config(Path(__file__).resolve().parent.parent / "config" / "default.yaml")
        cfg.visualisation.min_narratives_for_figures = 30
        df = _sample_evals_df(5)
        with pytest.raises(SystemExit) as exc:
            export_evaluation_figures_complete(df, cfg, "test_run")
        assert exc.value.code == 1

    def test_passes_min_n_check(self, tmp_path, monkeypatch):
        cfg = load_config(Path(__file__).resolve().parent.parent / "config" / "default.yaml")
        cfg.visualisation.figure_dir = str(tmp_path / "figures") + "/"
        cfg.visualisation.min_narratives_for_figures = 10
        df = _sample_evals_df(40)

        saved_paths = []

        def fake_export_all(*args, **kwargs):
            return []

        def fake_export_robustness(*args, **kwargs):
            return []

        monkeypatch.setattr(
            "src.visualisation.export.export_all_figures",
            fake_export_all,
        )
        monkeypatch.setattr(
            "src.visualisation.export.export_robustness_figures",
            fake_export_robustness,
        )

        result = export_evaluation_figures_complete(df, cfg, "test_run")
        assert len(result) == 4
        assert all(p.name.endswith(".csv") for p in result)
        assert all("analysis" in str(p) for p in result)
