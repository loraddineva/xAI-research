"""Evaluation: LLM extraction + SHAP comparison."""

from src.evaluation.evaluator import run_evaluation
from src.evaluation.robustness_runner import run_robustness

__all__ = ["run_evaluation", "run_robustness"]
