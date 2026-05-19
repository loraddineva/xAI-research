"""
src/pipeline.py
Run generation, evaluation, and robustness checks in sequence for one run_id.
"""

from __future__ import annotations

from typing import Optional

from src.config import AppConfig
from src.evaluation import run_evaluation, run_robustness
from src.generation import run_generation


def run_pipeline(
    cfg: AppConfig,
    *,
    dry_run: bool = False,
    run_id: Optional[str] = None,
    skip_generation: bool = False,
    skip_evaluation: bool = False,
    skip_robustness: bool = False,
    filter_model: Optional[str] = None,
    filter_dataset: Optional[str] = None,
    filter_strategy: Optional[str] = None,
    n_override: Optional[int] = None,
    eval_n_limit: Optional[int] = None,
    robustness_n_limit: Optional[int] = None,
    robustness_subsample: Optional[float] = None,
) -> str:
    """
    Run narrative generation, faithfulness evaluation, and extraction robustness
    for a single run_id.

    Each stage reads artefacts produced by the previous stage under the same
    ``outputs/generation/<run_id>/`` and ``outputs/evaluations/<run_id>/`` paths.

    Args:
        cfg: Loaded AppConfig.
        dry_run: Print prompts only; no LLM calls or disk writes in any stage.
        run_id: Existing generation run_id. Required when ``skip_generation`` is True.
        skip_generation: Skip generation and use ``run_id`` (must be set).
        skip_evaluation: Skip faithfulness evaluation.
        skip_robustness: Skip extraction robustness checks.
        filter_model: Restrict generation to this model id.
        filter_dataset: Restrict generation to this dataset name.
        filter_strategy: Restrict generation to this prompt strategy id.
        n_override: Override ``n_instances`` for generation.
        eval_n_limit: Evaluate only the first N successful narratives.
        robustness_n_limit: Robustness-check only the first N narratives.
        robustness_subsample: Fraction of narratives for robustness (overrides config).

    Returns:
        The run_id used for all stages.
    """
    if skip_generation:
        if not run_id:
            raise ValueError(
                "run_id is required when skip_generation is True "
                "(point to an existing outputs/generation/<run_id>/)."
            )
        active_run_id = run_id
        print(f"\n=== Pipeline: using existing run {active_run_id} ===\n")
    else:
        print("\n=== Pipeline: generation ===\n")
        active_run_id = run_generation(
            cfg=cfg,
            dry_run=dry_run,
            filter_model=filter_model,
            filter_dataset=filter_dataset,
            filter_strategy=filter_strategy,
            n_override=n_override,
        )

    if not skip_evaluation:
        print("\n=== Pipeline: evaluation ===\n")
        run_evaluation(
            cfg=cfg,
            run_id=active_run_id,
            dry_run=dry_run,
            n_limit=eval_n_limit,
        )

    if not skip_robustness:
        print("\n=== Pipeline: robustness ===\n")
        run_robustness(
            cfg=cfg,
            run_id=active_run_id,
            dry_run=dry_run,
            n_limit=robustness_n_limit,
            subsample_fraction=robustness_subsample,
        )

    print(f"\n=== Pipeline complete: {active_run_id} ===")
    if not dry_run:
        print(f"  Generation : outputs/generation/{active_run_id}/")
        if not skip_evaluation or not skip_robustness:
            print(f"  Evaluation : outputs/evaluations/{active_run_id}/")

    return active_run_id
