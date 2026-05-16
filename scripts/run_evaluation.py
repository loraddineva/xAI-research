"""
scripts/run_evaluation.py
CLI entry point for narrative faithfulness evaluation.

Usage
-----
    python scripts/run_evaluation.py --run-id <run_id>
    python scripts/run_evaluation.py --run-id <run_id> --dry-run
    python scripts/run_evaluation.py --run-id <run_id> --n 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.config import load_config
from src.evaluation import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate narratives via LLM extraction + SHAP comparison.",
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
        help="Print extraction prompts only; no LLM calls or disk writes.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Evaluate only the first N successful narratives.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_evaluation(
        cfg=cfg,
        run_id=args.run_id,
        dry_run=args.dry_run,
        n_limit=args.n,
    )


if __name__ == "__main__":
    main()
