"""
tests/test_data_loader.py
Tests for random instance sampling in load_dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DatasetConfig
from src.data_loader import (
    format_instance_snapshot,
    format_shap_table,
    load_dataset,
)


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.csv"
    df = pd.DataFrame({
        "feat": range(20),
        "shap_feat": [float(i) / 20 for i in range(20)],
        "pred_proba": [0.5] * 20,
        "pred_label": [0] * 20,
    })
    df.to_csv(path, index=False)
    return path


def _cfg(path: Path, n_instances: int = 5) -> DatasetConfig:
    return DatasetConfig(
        name="test",
        path=str(path),
        shap_col_prefix="shap_",
        n_instances=n_instances,
        task_description="test task",
    )


class TestLoadDatasetSampling:
    def test_sample_is_reproducible_with_seed(self, tiny_csv: Path) -> None:
        cfg = _cfg(tiny_csv, n_instances=5)
        a = load_dataset(cfg, seed=7)
        b = load_dataset(cfg, seed=7)
        assert list(a.index) == list(b.index)

    def test_different_seeds_differ(self, tiny_csv: Path) -> None:
        cfg = _cfg(tiny_csv, n_instances=5)
        a = load_dataset(cfg, seed=1)
        b = load_dataset(cfg, seed=2)
        assert list(a.index) != list(b.index)

    def test_n_override(self, tiny_csv: Path) -> None:
        cfg = _cfg(tiny_csv, n_instances=10)
        df = load_dataset(cfg, n=3, seed=42)
        assert len(df) == 3

    def test_sample_false_returns_full(self, tiny_csv: Path) -> None:
        cfg = _cfg(tiny_csv, n_instances=5)
        df = load_dataset(cfg, sample=False)
        assert len(df) == 20

    def test_index_is_original_row_number(self, tiny_csv: Path) -> None:
        cfg = _cfg(tiny_csv, n_instances=5)
        df = load_dataset(cfg, seed=42)
        assert all(0 <= i < 20 for i in df.index)


class TestPromptFormatting:
    @pytest.fixture
    def adult_row(self) -> pd.Series:
        return pd.Series({
            "age": 55,
            "hours_per_week": 51,
            "sex_Male": 1,
            "marital_status_Non_Married": 1,
            "shap_age": 0.0017,
            "shap_hours_per_week": -0.0401,
            "shap_sex_Male": 0.0232,
            "shap_marital_status_Non_Married": 0.0976,
        })

    def test_instance_snapshot_uses_plain_labels(self, adult_row: pd.Series) -> None:
        text = format_instance_snapshot(adult_row, "shap_", dataset_name="adult")
        assert "Age (years): 55" in text
        assert "Hours worked per week: 51" in text
        assert "Sex: Male" in text
        assert "shap_" not in text
        assert "+0.0976" not in text

    def test_shap_table_is_contributions_only(self, adult_row: pd.Series) -> None:
        text = format_shap_table(adult_row, "shap_")
        assert "marital_status_Non_Married: +0.0976" in text
        assert "feature value" not in text
        assert "51" not in text
