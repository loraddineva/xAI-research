"""
scripts/run_generation.py
CLI entry point for narrative generation.

Usage
-----
    # Full run (reads config/default.yaml)
    python scripts/run_generation.py

    # Custom config file
    python scripts/run_generation.py --config config/my_config.yaml

    # Dry-run: print prompts, no LLM calls or DB writes
    python scripts/run_generation.py --dry-run

    # Scoped run: one model, one dataset, 5 instances
    python scripts/run_generation.py --model claude-opus --dataset adult --n 5

    # Dry-run scoped
    python scripts/run_generation.py --dry-run --model claude-opus --dataset adult --n 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Allow running as `python scripts/run_generation.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from src.config import load_config
from src.narrative_generator import run_generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM narratives for SHAP values.",
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
        help="Print prompts to stdout; skip all LLM calls and DB writes.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Run only for this model id (as defined in config models[].id).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Run only for this dataset name (as defined in config datasets[].name).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Override n_instances for all datasets (useful for quick tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    run_id = run_generation(
        cfg=cfg,
        dry_run=args.dry_run,
        filter_model=args.model,
        filter_dataset=args.dataset,
        n_override=args.n,
    )

    if not args.dry_run:
        print(f"\nTo evaluate this run:\n  python scripts/run_evaluation.py --run-id {run_id}")


if __name__ == "__main__":
    main()
