"""
src/evaluation/extraction_prompt_renderer.py
Jinja2 renderer for the narrative extraction prompt.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.config import AppConfig, DatasetConfig


class ExtractionPromptRenderer:
    """Loads extract.j2 and renders one prompt per narrative."""

    def __init__(self, cfg: AppConfig) -> None:
        template_path = Path(cfg.evaluation.template)
        if not template_path.exists():
            raise FileNotFoundError(
                f"Extraction template not found: {template_path.resolve()}"
            )
        self._env = Environment(
            loader=FileSystemLoader(str(template_path.parent)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
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
            narrative_text=narrative_text,
            feature_names_list=names_block,
        )
