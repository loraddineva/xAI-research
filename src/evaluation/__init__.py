"""
src/evaluation
Evaluation subpackage. Currently exposes the existing rule-based evaluator
unchanged; will be revised once the new generation output format
(outputs/generation/<run_id>/) becomes the canonical input.
"""

from src.evaluation.evaluator import (
    EvaluationResult,
    evaluate_narrative,
    llm_judge,
)

__all__ = ["EvaluationResult", "evaluate_narrative", "llm_judge"]
