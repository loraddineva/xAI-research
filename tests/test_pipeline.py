"""
tests/test_pipeline.py
Tests for the end-to-end pipeline orchestrator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.pipeline import run_pipeline


@pytest.fixture
def cfg(tmp_path):
    config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    return load_config(config_path)


class TestRunPipeline:
    @patch("src.pipeline.run_robustness", return_value="test_run")
    @patch("src.pipeline.run_evaluation", return_value="test_run")
    @patch("src.pipeline.run_generation", return_value="test_run")
    def test_runs_all_stages_in_order(
        self, mock_gen, mock_eval, mock_rb, cfg
    ):
        run_id = run_pipeline(cfg, dry_run=False)

        assert run_id == "test_run"
        mock_gen.assert_called_once()
        mock_eval.assert_called_once_with(
            cfg=cfg, run_id="test_run", dry_run=False, n_limit=None
        )
        mock_rb.assert_called_once_with(
            cfg=cfg,
            run_id="test_run",
            dry_run=False,
            n_limit=None,
            subsample_fraction=None,
        )

    @patch("src.pipeline.run_robustness", return_value="existing_run")
    @patch("src.pipeline.run_evaluation", return_value="existing_run")
    @patch("src.pipeline.run_generation")
    def test_skip_generation_uses_run_id(
        self, mock_gen, mock_eval, mock_rb, cfg
    ):
        run_id = run_pipeline(
            cfg,
            skip_generation=True,
            run_id="existing_run",
        )

        assert run_id == "existing_run"
        mock_gen.assert_not_called()
        mock_eval.assert_called_once_with(
            cfg=cfg, run_id="existing_run", dry_run=False, n_limit=None
        )

    @patch("src.pipeline.run_robustness")
    @patch("src.pipeline.run_evaluation")
    @patch("src.pipeline.run_generation", return_value="test_run")
    def test_skip_robustness(self, mock_gen, mock_eval, mock_rb, cfg):
        run_pipeline(cfg, skip_robustness=True)

        mock_gen.assert_called_once()
        mock_eval.assert_called_once()
        mock_rb.assert_not_called()

    def test_skip_generation_requires_run_id(self, cfg):
        with pytest.raises(ValueError, match="run_id is required"):
            run_pipeline(cfg, skip_generation=True)
