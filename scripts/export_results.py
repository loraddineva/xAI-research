"""
scripts/export_results.py
Dump a completed run from SQLite to CSV files for sharing / archiving.
Optionally also produce and save all figures.

Usage
-----
    # Export narratives + evaluations to CSV
    python scripts/export_results.py --run-id <run_id>

    # Also produce and save all figures
    python scripts/export_results.py --run-id <run_id> --figures

    # Custom config
    python scripts/export_results.py --run-id <run_id> --config config/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.config import load_config
from src.db import (
    db_connection,
    get_evaluations_for_run,
    get_narratives_for_run,
    get_run,
)


def export_run(run_id: str, cfg_path: str = "config/default.yaml", figures: bool = False) -> None:
    """Export all data for a run to CSV and optionally produce figures."""
    cfg = load_config(cfg_path)
    db_path = Path(cfg.storage.db_path)
    export_dir = Path(cfg.storage.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    with db_connection(db_path) as conn:
        run_meta = get_run(conn, run_id)
        narratives = get_narratives_for_run(conn, run_id)
        evaluations = get_evaluations_for_run(conn, run_id)

    if run_meta is None:
        print(f"Run '{run_id}' not found in database.")
        sys.exit(1)

    print(f"Run: {run_meta['run_name']} ({run_id})")
    print(f"  Narratives : {len(narratives)}")
    print(f"  Evaluations: {len(evaluations)}")

    # --- Export narratives ---
    narr_path = export_dir / f"{run_id}_narratives.csv"
    if narratives:
        with narr_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(narratives[0].keys()))
            writer.writeheader()
            writer.writerows(narratives)
        print(f"\nNarratives saved to: {narr_path}")

    # --- Export evaluations ---
    eval_path = export_dir / f"{run_id}_evaluations.csv"
    if evaluations:
        with eval_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(evaluations[0].keys()))
            writer.writeheader()
            writer.writerows(evaluations)
        print(f"Evaluations saved to: {eval_path}")

    # --- Optionally export figures ---
    if figures:
        if not evaluations:
            print("\nNo evaluations found — skipping figures.")
            return

        import pandas as pd
        from src.visualisation.export import export_all_figures

        evals_df = pd.DataFrame(evaluations)
        saved = export_all_figures(evals_df, cfg, run_id)
        print(f"\n{len(saved)} figures saved to {cfg.visualisation.figure_dir}{run_id}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a run's narratives and evaluations from SQLite to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-id", required=True, help="The run_id to export.")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path.")
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Also produce and save all visualisation figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_run(run_id=args.run_id, cfg_path=args.config, figures=args.figures)


if __name__ == "__main__":
    main()
