"""
scripts/export_results.py
Inspect a completed generation run and optionally produce dataset figures.

Usage
-----
    python scripts/export_results.py --run-id <run_id>
    python scripts/export_results.py --run-id <run_id> --figures
    python scripts/export_results.py --run-id <run_id> --config config/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.config import load_config
from src.data_loader import load_dataset
from src.storage.narratives_store import (
    get_run,
    load_narratives_csv,
    narratives_csv_path,
    run_dir,
)


def export_run(
    run_id: str,
    cfg_path: str = "config/default.yaml",
    figures: bool = False,
    eval_figures: bool = False,
) -> None:
    """Print run summary; optionally export dataset or evaluation figures."""
    cfg = load_config(cfg_path)
    run_meta = get_run(cfg.storage.generation_dir, run_id)

    if run_meta is None:
        print(f"Run '{run_id}' not found (no narratives.csv under generation dir).")
        sys.exit(1)

    csv_path = narratives_csv_path(run_dir(cfg, run_id))
    narratives_df = load_narratives_csv(csv_path)

    print(f"Run: {run_meta.get('run_name', run_id)} ({run_id})")
    print(f"  Path       : {run_meta['path']}")
    print(f"  Narratives : {len(narratives_df)}")
    if "error" in narratives_df.columns:
        n_failed = (narratives_df["error"].fillna("").astype(str) != "").sum()
        print(f"  Failed     : {n_failed}")

    if "dataset" in narratives_df.columns:
        print(f"  Datasets   : {', '.join(sorted(narratives_df['dataset'].unique()))}")
    if "model_id" in narratives_df.columns:
        print(f"  Models     : {', '.join(sorted(narratives_df['model_id'].unique()))}")

    print(f"\nCanonical CSV: {csv_path}")

    if eval_figures:
        from src.storage.evaluations_store import (
            eval_run_dir,
            evaluations_csv_path,
            load_evaluations_csv,
        )
        from src.visualisation.export import export_all_figures

        eval_dir = eval_run_dir(cfg.evaluation.export_dir, run_id)
        eval_csv = evaluations_csv_path(eval_dir)
        if not eval_csv.exists():
            print(
                f"\nNo evaluations found at {eval_csv}. "
                f"Run: python scripts/run_evaluation.py --run-id {run_id}"
            )
            sys.exit(1)
        evals_df = load_evaluations_csv(eval_csv)
        evals_df = evals_df[evals_df["parse_error"].fillna("").astype(str) == ""]
        print(f"\nEvaluations CSV: {eval_csv} ({len(evals_df)} rows)")
        export_all_figures(evals_df, cfg, run_id)
        return

    if not figures:
        return

    from src.visualisation.export import export_dataset_figures

    saved_all = []

    for dataset_cfg in cfg.datasets:
        if dataset_cfg.name not in narratives_df["dataset"].values:
            continue
        df = load_dataset(dataset_cfg)
        shap_prefix = dataset_cfg.shap_col_prefix
        shap_cols = [c for c in df.columns if c.startswith(shap_prefix)]
        feature_cols = [
            c for c in df.columns
            if not c.startswith(shap_prefix) and c not in ("label", "pred_proba", "pred_label")
        ]
        saved = export_dataset_figures(
            df, dataset_cfg.name, shap_cols, feature_cols, cfg, shap_prefix=shap_prefix
        )
        saved_all.extend(saved)

    print(f"\n{len(saved_all)} dataset figures saved under {cfg.visualisation.figure_dir}datasets/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a generation run from narratives.csv; optionally export figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-id", required=True, help="The run_id to inspect.")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path.")
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Produce dataset-level visualisation figures (no evaluation required).",
    )
    parser.add_argument(
        "--eval-figures",
        action="store_true",
        help="Produce hallucination rate figures from a completed evaluation run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_run(
        run_id=args.run_id,
        cfg_path=args.config,
        figures=args.figures,
        eval_figures=args.eval_figures,
    )


if __name__ == "__main__":
    main()
