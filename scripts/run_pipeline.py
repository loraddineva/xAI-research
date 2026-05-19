"""
scripts/run_pipeline.py
CLI entry point: generation → evaluation → robustness for one run_id.

Usage
-----
    # Full pipeline (reads config/default.yaml)
    python scripts/run_pipeline.py

    # Dry-run all three stages (prompts only)
    python scripts/run_pipeline.py --dry-run

    # Quick smoke test
    python scripts/run_pipeline.py --model llama3-70b --dataset adult --n 3

    # Resume evaluation + robustness on an existing generation run
    python scripts/run_pipeline.py --run-id pilot_run_20260518T074326_cadfd5 --skip-generation

    # Calibrate robustness on 10% subsample
    python scripts/run_pipeline.py --subsample 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.config import load_config
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run generation, evaluation, and robustness checks in sequence "
            "for a single run_id."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts only; skip LLM calls and disk writes in all stages.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Existing generation run_id (required with --skip-generation).",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation; run evaluation and robustness on --run-id.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip faithfulness evaluation.",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip extraction robustness checks.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Generation: only this model id.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Generation: only this dataset name.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Generation: only this prompt strategy id.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Override generation n_instances; also limits evaluation and robustness.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="Robustness: fraction of narratives to check (e.g. 0.1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    run_pipeline(
        cfg=cfg,
        dry_run=args.dry_run,
        run_id=args.run_id,
        skip_generation=args.skip_generation,
        skip_evaluation=args.skip_evaluation,
        skip_robustness=args.skip_robustness,
        filter_model=args.model,
        filter_dataset=args.dataset,
        filter_strategy=args.strategy,
        n_override=args.n,
        eval_n_limit=args.n,
        robustness_n_limit=args.n,
        robustness_subsample=args.subsample,
    )


if __name__ == "__main__":
    main()
