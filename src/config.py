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


class PromptStrategyConfig(BaseModel):
    """One generation prompt strategy (template + optional token override)."""

    id: str
    template: str
    max_tokens: Optional[int] = None


class PromptConfig(BaseModel):
    """Narrative generation prompt strategies crossed in each run."""

    strategies: List[PromptStrategyConfig] = Field(
        default_factory=lambda: [
            PromptStrategyConfig(
                id="martens",
                template="config/prompts/narrative.j2",
            ),
            PromptStrategyConfig(
                id="chain_of_thought",
                template="config/prompts/chain_of_thought.j2",
                max_tokens=1024,
            ),
        ]
    )

    def get_strategy(self, strategy_id: str) -> PromptStrategyConfig:
        for s in self.strategies:
            if s.id == strategy_id:
                return s
        raise KeyError(f"Prompt strategy '{strategy_id}' not found in config.")


class StorageConfig(BaseModel):
    generation_dir: str = "outputs/generation/"


class VisualisationConfig(BaseModel):
    figure_dir: str = "outputs/figures/"
    format: str = "png"
    dpi: int = 150


class RobustnessConfig(BaseModel):
    """Multi-sample extraction agreement (semantic uncertainty check)."""

    n_runs: int = 5
    temperature: float = 0.9
    min_successful_runs: int = 3
    reliability_threshold: float = 0.8
    subsample_fraction: float = 1.0
    max_workers: int = 5


class EvaluationConfig(BaseModel):
    """LLM extraction + SHAP comparison evaluation settings."""

    extraction_model_id: str = "claude-opus"
    template: str = "config/prompts/extract.j2"
    top_k_features: int = 3
    export_dir: str = "outputs/evaluations/"
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    datasets: List[DatasetConfig] = Field(default_factory=list)
    models: List[ModelConfig] = Field(default_factory=list)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    visualisation: VisualisationConfig = Field(default_factory=VisualisationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def prompt_template_path(self, strategy_id: str = "martens") -> Path:
        """Resolve the absolute path of a strategy's prompt template."""
        return Path(self.prompt.get_strategy(strategy_id).template)

    def load_prompt_template(self, strategy_id: str = "martens") -> str:
        """Read and return the raw text of a strategy's prompt template."""
        path = self.prompt_template_path(strategy_id)
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
