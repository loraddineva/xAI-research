"""
Generate paper figures for the latest (or specified) evaluation run.

Produces:
  1. paired_flip_panels.png — 2×2 flip grids per error type (paired instances)
  2. rate_forest_k_sensitivity.png — Wilson CIs for rank swap and omission at K ∈ {2, 3, 5}

Usage
-----
    python scripts/generate_paper_figures.py
    python scripts/generate_paper_figures.py --run-id pilot_run_20260518T135815_bdad28
    python scripts/generate_paper_figures.py --k 2 3 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.config import load_config
from src.evaluation.compare_to_shap import compare_to_shap
from src.evaluation.extraction_parser import parse_extraction_response
from src.evaluation.evaluator import (
    _feature_names_for_dataset,
    _parse_shap_values_sorted,
)
from src.storage.evaluations_store import (
    eval_run_dir,
    evaluations_csv_path,
    list_eval_runs,
    load_evaluations_csv,
)
from src.storage.narratives_store import load_narratives_csv, narratives_csv_path, run_dir
from src.visualisation.export import _save
from src.visualisation.paper_figures import (
    build_rate_forest_rows,
    plot_paired_flip_panels,
    plot_rate_forest,
)

DEFAULT_K_VALUES = (2, 3, 5)
BASELINE_K = 3


def resolve_run_id(export_dir: str, run_id: str | None) -> str:
    if run_id:
        return run_id
    runs = list_eval_runs(export_dir)
    if not runs:
        raise FileNotFoundError(f"No evaluation runs under {export_dir}")
    for r in runs:
        rid = r["run_id"]
        if not rid.startswith("test_"):
            return rid
    return runs[0]["run_id"]


def recompute_k_sensitive_flags(
    cfg,
    eval_df: pd.DataFrame,
    narratives_df: pd.DataFrame,
    k_values: list[int],
) -> pd.DataFrame:
    """Per-narrative rank_swap and omission flags at each K."""
    shap_by_narrative = narratives_df.set_index("narrative_id")["shap_values_sorted"]
    feature_names_cache: dict[str, list[str]] = {}
    rows: list[dict] = []

    for _, row in eval_df.iterrows():
        narrative_id = str(row["narrative_id"])
        dataset_name = str(row["dataset"])
        if dataset_name not in feature_names_cache:
            feature_names_cache[dataset_name] = _feature_names_for_dataset(cfg, dataset_name)
        extraction = parse_extraction_response(
            str(row["extraction_json"]),
            feature_names_cache[dataset_name],
        )
        shap_sorted = _parse_shap_values_sorted(shap_by_narrative[narrative_id])
        record: dict = {"narrative_id": narrative_id}
        for k in k_values:
            comparison = compare_to_shap(extraction, shap_sorted, top_k_features=k)
            record[f"k{k}_rank_swap"] = comparison.rank_swap
            record[f"k{k}_omission"] = comparison.omission
        rows.append(record)

    return pd.DataFrame(rows)


def generate_paper_figures(
    run_id: str | None = None,
    config_path: str = "config/default.yaml",
    k_values: list[int] | None = None,
) -> list[Path]:
    cfg = load_config(config_path)
    if k_values is None:
        k_values = list(DEFAULT_K_VALUES)
    k_values = sorted(set(k_values) | {BASELINE_K})

    resolved_id = resolve_run_id(cfg.evaluation.export_dir, run_id)
    eval_dir = eval_run_dir(cfg.evaluation.export_dir, resolved_id)
    eval_csv = evaluations_csv_path(eval_dir)
    if not eval_csv.exists():
        raise FileNotFoundError(
            f"Evaluation run not found: {eval_csv.resolve()}\n"
            f"Run: python scripts/run_evaluation.py --run-id {resolved_id}"
        )

    evals_df = load_evaluations_csv(eval_csv)
    gen_csv = narratives_csv_path(run_dir(cfg, resolved_id))
    if not gen_csv.exists():
        raise FileNotFoundError(f"Generation run not found: {gen_csv.resolve()}")

    valid = evals_df[evals_df["parse_error"].fillna("").astype(str) == ""].copy()
    narratives_df = load_narratives_csv(gen_csv)

    print(f"Run ID:        {resolved_id}")
    print(f"Valid evals:   {len(valid)}")
    print(f"K values:      {k_values} (rank swap + omission)")

    recomputed = recompute_k_sensitive_flags(cfg, valid, narratives_df, k_values)

    out_dir = Path(cfg.visualisation.figure_dir) / resolved_id / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = cfg.visualisation.format
    dpi = cfg.visualisation.dpi
    saved: list[Path] = []

    fig_flip = plot_paired_flip_panels(evals_df)
    path_flip = _save(fig_flip, out_dir / "paired_flip_panels", dpi=dpi, fmt=fmt)
    saved.append(path_flip)
    print(f"  Saved: {path_flip}")

    forest_rows = build_rate_forest_rows(
        evals_df, recomputed, k_values=k_values, baseline_k=BASELINE_K
    )
    fig_forest = plot_rate_forest(forest_rows)
    path_forest = _save(fig_forest, out_dir / "rate_forest_k_sensitivity", dpi=dpi, fmt=fmt)
    saved.append(path_forest)
    print(f"  Saved: {path_forest}")

    csv_path = out_dir / "rate_forest_data.csv"
    forest_rows.to_csv(csv_path, index=False)
    saved.append(csv_path)
    print(f"  Saved: {csv_path}")

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired-flip and rate-forest figures for an evaluation run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Evaluation run_id (default: latest non-test run under export_dir).",
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        metavar="K",
        help="Top-k values for rank-swap and omission forest rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        generate_paper_figures(
            run_id=args.run_id,
            config_path=args.config,
            k_values=args.k,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
