"""
src/prompt_renderer.py
Jinja2-based prompt template renderer.

The renderer loads templates from config/prompts/ using a jinja2.Environment,
injects the dataset name and SHAP table, and returns the final prompt string
ready for the LLM.

Template variables available in every template
-----------------------------------------------
    {{ dataset }}     — dataset name (e.g. "adult", "german_credit")
    {{ shap_table }}  — pre-formatted SHAP table string produced by format_shap_table()

Public API
----------
    PromptRenderer(cfg)
    renderer.render(strategy, dataset_name, row, shap_prefix) -> str
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.config import AppConfig
from src.data_loader import format_shap_table


class PromptRenderer:
    """
    Loads prompt templates from disk using Jinja2 and renders them with
    per-instance data.

    Args:
        cfg: The loaded AppConfig. Used to resolve the template directory.
    """

    def __init__(self, cfg: AppConfig) -> None:
        template_dir = Path(cfg.prompts.template_dir)
        if not template_dir.exists():
            raise FileNotFoundError(
                f"Prompt template directory not found: {template_dir.resolve()}"
            )

        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            # StrictUndefined raises an error if a variable used in a template
            # is not provided — catches typos early rather than silently rendering
            # an empty string.
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )

    def render(
        self,
        strategy: str,
        dataset_name: str,
        row: pd.Series,
        shap_prefix: str,
    ) -> str:
        """
        Render the prompt template for *strategy* with the given instance data.

        Args:
            strategy:     One of "zero_shot", "few_shot", "chain_of_thought".
            dataset_name: Human-readable dataset name injected into the template.
            row:          A single DataFrame row containing feature and SHAP columns.
            shap_prefix:  Column prefix for SHAP values (e.g. "shap_").

        Returns:
            The fully rendered prompt string.
        """
        template_file = f"{strategy}.txt"
        try:
            template = self._env.get_template(template_file)
        except Exception as exc:
            raise FileNotFoundError(
                f"Prompt template '{template_file}' not found in "
                f"{self._env.loader.searchpath}. "  # type: ignore[union-attr]
                f"Available strategies: zero_shot, few_shot, chain_of_thought."
            ) from exc

        shap_table = format_shap_table(row, shap_prefix)
        return template.render(dataset=dataset_name, shap_table=shap_table)
