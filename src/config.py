"""
src/config.py
Pydantic models for config/default.yaml.
Load with: cfg = load_config()  (reads config/default.yaml by default)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    name: str = "pilot_run"
    seed: int = 42


class DatasetConfig(BaseModel):
    """
    A single dataset entry.

    The narrative-prompt fields (`task_description`, `positive_class_label`,
    `negative_class_label`) are injected directly into the Martens-style
    template so the same prompt works across datasets without per-dataset
    template forking.
    """

    name: str
    path: str
    shap_col_prefix: str = "shap_"
    n_instances: int = 100
    task_description: str = ""
    positive_class_label: str = "positive class"
    negative_class_label: str = "negative class"


class ModelConfig(BaseModel):
    id: str
    provider: str                    # anthropic | together | mistral | ollama
    model_name: str
    max_tokens: int = 512
    temperature: float = 0.0
    base_url: Optional[str] = None   # used by ollama


class PromptConfig(BaseModel):
    """Single narrative prompt (no zero/few/CoT split)."""

    template: str = "config/prompts/narrative.j2"


class EvaluationConfig(BaseModel):
    top_k_features: int = 3
    magnitude_threshold: float = 0.5
    use_llm_judge: bool = False
    llm_judge_model: str = "claude-opus"


class StorageConfig(BaseModel):
    db_path: str = "outputs/results.db"
    generation_dir: str = "outputs/generation/"
    export_dir: str = "outputs/evaluations/"
    narrative_dir: str = "outputs/narratives/"      # legacy alias


class VisualisationConfig(BaseModel):
    figure_dir: str = "outputs/figures/"
    format: str = "png"
    dpi: int = 150


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    datasets: List[DatasetConfig] = Field(default_factory=list)
    models: List[ModelConfig] = Field(default_factory=list)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    visualisation: VisualisationConfig = Field(default_factory=VisualisationConfig)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def prompt_template_path(self) -> Path:
        """Resolve the absolute path of the narrative prompt template."""
        return Path(self.prompt.template)

    def load_prompt_template(self) -> str:
        """Read and return the raw text of the narrative prompt template."""
        path = self.prompt_template_path()
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    def get_model(self, model_id: str) -> ModelConfig:
        """Return the ModelConfig matching the given id."""
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"Model id '{model_id}' not found in config.")

    def get_dataset(self, dataset_name: str) -> DatasetConfig:
        """Return the DatasetConfig matching the given name."""
        for d in self.datasets:
            if d.name == dataset_name:
                return d
        raise KeyError(f"Dataset '{dataset_name}' not found in config.")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def load_config(path: str | Path | None = None) -> AppConfig:
    """
    Load AppConfig from a YAML file.

    Args:
        path: Path to the YAML config file. Defaults to config/default.yaml
              (resolved relative to the current working directory).

    Returns:
        A fully-validated AppConfig instance.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}. "
            "Run from the project root or pass an explicit path."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return AppConfig.model_validate(raw or {})
