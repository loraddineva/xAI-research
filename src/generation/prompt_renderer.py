"""
src/generation/prompt_renderer.py
Jinja2 renderer for narrative generation prompt strategies.

Public API
----------
    PromptRenderer(cfg)
    renderer.render(dataset_cfg, row, strategy_id) -> str
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.config import AppConfig, DatasetConfig, PromptStrategyConfig
from src.data_loader import format_shap_table


class PromptRenderer:
    """
    Loads all configured prompt templates and renders one per instance.

    Args:
        cfg: A loaded :class:`AppConfig`. Templates are listed under
             ``cfg.prompt.strategies``.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._template_names: dict[str, str] = {}
        search_dirs: set[str] = set()

        for strategy in cfg.prompt.strategies:
            template_path = Path(strategy.template)
            if not template_path.exists():
                raise FileNotFoundError(
                    f"Prompt template not found: {template_path.resolve()}"
                )
            search_dirs.add(str(template_path.parent))
            self._template_names[strategy.id] = template_path.name

        self._env = Environment(
            loader=FileSystemLoader(sorted(search_dirs)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )

    def get_strategy(self, strategy_id: str) -> PromptStrategyConfig:
        return self._cfg.prompt.get_strategy(strategy_id)

    def render(
        self,
        dataset_cfg: DatasetConfig,
        row: pd.Series,
        strategy_id: str,
    ) -> str:
        """
        Render the narrative prompt for one instance and strategy.

        The row must contain ``pred_proba`` and ``pred_label`` columns
        (produced by ``scripts/prepare_data.py``).
        """
        if strategy_id not in self._template_names:
            raise KeyError(
                f"Unknown prompt strategy '{strategy_id}'. "
                f"Configured: {list(self._template_names)}"
            )

        template = self._env.get_template(self._template_names[strategy_id])

        if "pred_proba" not in row or "pred_label" not in row:
            raise KeyError(
                f"Dataset '{dataset_cfg.name}' is missing 'pred_proba' or "
                f"'pred_label' columns. Re-run scripts/prepare_data.py to "
                f"add the model prediction columns required by the prompt."
            )

        pred_proba = float(row["pred_proba"])
        pred_label = int(row["pred_label"])
        pred_class_text = (
            dataset_cfg.positive_class_label
            if pred_label == 1
            else dataset_cfg.negative_class_label
        )

        shap_table = format_shap_table(
            row,
            dataset_cfg.shap_col_prefix,
            dataset_name=dataset_cfg.name,
        )

        return template.render(
            dataset=dataset_cfg.name,
            task_description=dataset_cfg.task_description,
            positive_class_label=dataset_cfg.positive_class_label,
            negative_class_label=dataset_cfg.negative_class_label,
            pred_proba=pred_proba,
            pred_label=pred_label,
            pred_class_text=pred_class_text,
            shap_table=shap_table,
        )
