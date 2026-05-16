"""
src/generation/prompt_renderer.py
Jinja2 renderer for the single Martens-style narrative prompt.

The renderer loads the template file specified by ``cfg.prompt.template``
and injects:
    - dataset name + per-dataset task description / class labels,
    - the model's predicted probability and predicted class label text,
    - a SHAP table sorted from most positive to most negative.

Public API
----------
    PromptRenderer(cfg)
    renderer.render(dataset_cfg, row) -> str
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.config import AppConfig, DatasetConfig
from src.data_loader import format_shap_table


class PromptRenderer:
    """
    Loads the configured prompt template once and renders it per instance.

    Args:
        cfg: A loaded :class:`AppConfig`. ``cfg.prompt.template`` is treated
             as the path to the Jinja2 template file.
    """

    def __init__(self, cfg: AppConfig) -> None:
        template_path = Path(cfg.prompt.template)
        if not template_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {template_path.resolve()}"
            )

        # FileSystemLoader needs a directory; we then look up the template
        # by its basename. This keeps the standard Jinja loader semantics
        # (search path, includes, etc.) without forcing a custom loader.
        self._env = Environment(
            loader=FileSystemLoader(str(template_path.parent)),
            # StrictUndefined raises an error if a variable used in a template
            # is not provided — catches typos early rather than silently
            # rendering an empty string.
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        self._template_name = template_path.name

    # ------------------------------------------------------------------
    # Public render
    # ------------------------------------------------------------------

    def render(self, dataset_cfg: DatasetConfig, row: pd.Series) -> str:
        """
        Render the narrative prompt for one instance.

        The row must contain ``pred_proba`` and ``pred_label`` columns
        (produced by the updated ``scripts/prepare_data.py``). If they are
        missing, prep the data again with the new script.
        """
        try:
            template = self._env.get_template(self._template_name)
        except Exception as exc:
            raise FileNotFoundError(
                f"Prompt template '{self._template_name}' not found in "
                f"{self._env.loader.searchpath}"  # type: ignore[union-attr]
            ) from exc

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
