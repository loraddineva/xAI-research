"""
src/evaluation/extraction_prompt_renderer.py
Jinja2 renderer for the narrative extraction prompt.
"""

from __future__ import annotations

from pathlib import Path

from src.config import AppConfig, DatasetConfig
from src.prompts.jinja_env import make_jinja_env


class ExtractionPromptRenderer:
    """Loads extract.j2 and renders one prompt per narrative."""

    def __init__(self, cfg: AppConfig) -> None:
        template_path = Path(cfg.evaluation.template)
        if not template_path.exists():
            raise FileNotFoundError(
                f"Extraction template not found: {template_path.resolve()}"
            )
        self._env = make_jinja_env([str(template_path.parent)])
        self._template_name = template_path.name

    def render(
        self,
        dataset_cfg: DatasetConfig,
        narrative_text: str,
        feature_names: list[str],
    ) -> str:
        template = self._env.get_template(self._template_name)
        names_block = "\n".join(f"  - {name}" for name in feature_names)
        return template.render(
            dataset=dataset_cfg.name,
            task_description=dataset_cfg.task_description,
            positive_class_label=dataset_cfg.positive_class_label,
            negative_class_label=dataset_cfg.negative_class_label,
            narrative_text=narrative_text,
            feature_names_list=names_block,
        )
