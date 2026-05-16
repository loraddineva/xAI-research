"""
scripts/run_robustness.py
CLI entry point for extraction-model robustness checks.

Usage
-----
    python scripts/run_robustness.py --run-id <run_id>
    python scripts/run_robustness.py --run-id <run_id> --subsample 0.1
    python scripts/run_robustness.py --run-id <run_id> --dry-run --n 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.config import load_config
from src.evaluation.robustness_runner import run_robustness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-sample extraction robustness checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Generation run_id (under outputs/generation/).",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts only; no LLM calls or disk writes.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Check only the first N narratives (after subsampling).",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="Fraction of narratives to check (e.g. 0.1 for 10%% calibration).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_robustness(
        cfg=cfg,
        run_id=args.run_id,
        dry_run=args.dry_run,
        n_limit=args.n,
        subsample_fraction=args.subsample,
    )


if __name__ == "__main__":
    main()
