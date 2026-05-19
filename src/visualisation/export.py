"""
src/visualisation/export.py
Save all standard figures for a completed run to outputs/figures/.

Two figure sets are produced:

    Dataset figures   — feature distributions, class balance, correlation
                        heatmap, and SHAP distribution plots. These are
                        dataset-level and do not require evaluation results.

    Evaluation figures — hallucination rate bar charts and model × strategy
                         heatmaps. These require a completed evaluation run.

Public API
----------
    export_dataset_figures(df, dataset_name, shap_cols, feature_cols, cfg) -> List[Path]
    export_all_figures(evals_df, cfg, run_id, subdir)                      -> List[Path]
    export_robustness_figures(evals_df, cfg, run_id)                         -> List[Path]
    export_evaluation_figures_complete(evals_df, cfg, run_id)               -> List[Path]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.config import AppConfig
from src.visualisation.dataset_overview import (
    plot_class_balance,
    plot_correlation_heatmap,
    plot_feature_distributions,
)
from src.visualisation.hallucination_analysis import export_hallucination_analysis_tables
from src.visualisation.hallucination_rates import (
    plot_rates_by_dataset,
    plot_rates_by_model,
    plot_rates_by_strategy,
    plot_rates_by_type,
    plot_type_by_model,
    plot_type_by_strategy,
)
from src.visualisation.heatmaps import (
    plot_all_datasets_heatmap,
    plot_model_strategy_heatmap,
    plot_type_heatmap,
    plot_type_strategy_heatmap,
)
from src.visualisation.robustness_plots import (
    attach_robustness_columns,
    filter_reliable_extractions,
    load_robustness_df,
    plot_hallucination_by_reliability_group,
    plot_low_reliability_rate_by_strategy,
    plot_reliability_by_strategy,
    plot_reliability_distribution,
)
from src.visualisation.shap_distributions import (
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_shap_scatter,
)


def _save(fig: plt.Figure, path: Path, dpi: int, fmt: str) -> Path:
    out = path.with_suffix(f".{fmt}")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Dataset-level figures (no evaluation results needed)
# ---------------------------------------------------------------------------

def export_dataset_figures(
    df: pd.DataFrame,
    dataset_name: str,
    shap_cols: List[str],
    feature_cols: List[str],
    cfg: AppConfig,
    shap_prefix: str = "shap_",
) -> List[Path]:
    """
    Generate and save exploratory figures for a single dataset.

    Produces:
        - Feature distribution grid (histograms / bar charts)
        - Class balance bar chart
        - Feature correlation heatmap
        - Global SHAP importance bar chart (mean |SHAP|)
        - SHAP beeswarm / strip plot
        - Per-feature SHAP scatter plots for the top 5 features by mean |SHAP|

    Args:
        df:           Processed DataFrame with feature and SHAP columns attached.
        dataset_name: Used to name the output subdirectory and figure titles.
        shap_cols:    List of SHAP column names (e.g. ["shap_age", ...]).
        feature_cols: List of raw feature column names.
        cfg:          AppConfig (provides figure_dir, format, dpi).
        shap_prefix:  Prefix on SHAP column names.

    Returns:
        List of Paths to saved figure files.
    """
    fig_dir = Path(cfg.visualisation.figure_dir) / "datasets" / dataset_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    fmt = cfg.visualisation.format
    dpi = cfg.visualisation.dpi
    saved: List[Path] = []

    def save(fig: plt.Figure, name: str) -> None:
        path = _save(fig, fig_dir / name, dpi=dpi, fmt=fmt)
        saved.append(path)
        print(f"  Saved: {path}")

    print(f"Exporting dataset figures for '{dataset_name}' to {fig_dir}/")

    save(plot_feature_distributions(df, feature_cols), "feature_distributions")
    if "label" in df.columns:
        save(plot_class_balance(df), "class_balance")
    save(plot_correlation_heatmap(df, feature_cols), "correlation_heatmap")
    save(plot_shap_bar(df, shap_cols, shap_prefix=shap_prefix), "shap_importance_bar")
    save(plot_shap_beeswarm(df, shap_cols, feature_cols=feature_cols, shap_prefix=shap_prefix), "shap_beeswarm")

    top5_shap = df[shap_cols].abs().mean().sort_values(ascending=False).index[:5]
    for shap_col in top5_shap:
        feat_col = shap_col[len(shap_prefix):]
        if feat_col in df.columns:
            save(
                plot_shap_scatter(df, feat_col, shap_col),
                f"shap_scatter_{feat_col}",
            )

    print(f"  {len(saved)} dataset figures saved.\n")
    return saved


# ---------------------------------------------------------------------------
# Evaluation figures (requires evaluation results)
# ---------------------------------------------------------------------------

def export_all_figures(
    evals_df: pd.DataFrame,
    cfg: AppConfig,
    run_id: str,
    *,
    subdir: str = "",
    title_suffix: str = "",
) -> List[Path]:
    """
    Generate and save the hallucination figure set for a run.

    Args:
        evals_df:      Evaluations DataFrame.
        cfg:           AppConfig.
        run_id:        Output subdirectory name under figure_dir.
        subdir:        Optional nested folder (e.g. 'unfiltered').
        title_suffix:  Appended to plot titles for sensitivity exports.

    Returns:
        List of Paths to saved figure files.
    """
    base = Path(cfg.visualisation.figure_dir) / run_id
    fig_dir = base / subdir if subdir else base
    fig_dir.mkdir(parents=True, exist_ok=True)

    fmt = cfg.visualisation.format
    dpi = cfg.visualisation.dpi
    saved: List[Path] = []
    suffix = f" ({title_suffix})" if title_suffix else ""

    def save(fig: plt.Figure, name: str) -> None:
        path = _save(fig, fig_dir / name, dpi=dpi, fmt=fmt)
        saved.append(path)
        print(f"  Saved: {path}")

    label = subdir or "main"
    print(f"Exporting evaluation figures [{label}] to {fig_dir}/")

    save(plot_rates_by_type(evals_df, title=f"Hallucination Rate by Type{suffix}"), "rates_by_type")
    save(plot_rates_by_model(evals_df, title=f"Overall Hallucination Rate by Model{suffix}"), "rates_by_model")
    save(
        plot_rates_by_strategy(evals_df, title=f"Overall Hallucination Rate by Prompt Strategy{suffix}"),
        "rates_by_strategy",
    )
    save(plot_rates_by_dataset(evals_df, title=f"Overall Hallucination Rate by Dataset{suffix}"), "rates_by_dataset")
    save(plot_type_by_model(evals_df, title=f"Hallucination Type Rate by Model{suffix}"), "type_by_model")
    save(
        plot_type_by_strategy(evals_df, title=f"Hallucination Type Rate by Prompt Strategy{suffix}"),
        "type_by_strategy",
    )

    save(plot_all_datasets_heatmap(evals_df), "heatmap_all_datasets")
    save(plot_type_heatmap(evals_df), "heatmap_type_by_model")
    save(plot_type_strategy_heatmap(evals_df), "heatmap_type_by_strategy")

    for dataset in sorted(evals_df["dataset"].unique()):
        safe_name = dataset.replace(" ", "_").lower()
        save(
            plot_model_strategy_heatmap(evals_df, dataset=dataset),
            f"heatmap_{safe_name}",
        )

    print(f"  {len(saved)} evaluation figures saved [{label}].")
    return saved


def export_robustness_figures(
    evals_df: pd.DataFrame,
    cfg: AppConfig,
    run_id: str,
) -> List[Path]:
    """Save robustness / reliability figures under <run_id>/robustness/."""
    threshold = cfg.evaluation.robustness.reliability_threshold
    rb_df = load_robustness_df(evals_df)

    fig_dir = Path(cfg.visualisation.figure_dir) / run_id / "robustness"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fmt = cfg.visualisation.format
    dpi = cfg.visualisation.dpi
    saved: List[Path] = []

    def save(fig: plt.Figure, name: str) -> None:
        path = _save(fig, fig_dir / name, dpi=dpi, fmt=fmt)
        saved.append(path)
        print(f"  Saved: {path}")

    print(f"Exporting robustness figures to {fig_dir}/")

    save(plot_reliability_distribution(rb_df, threshold=threshold), "reliability_score_distribution")
    save(plot_reliability_by_strategy(rb_df), "reliability_by_strategy")
    save(
        plot_hallucination_by_reliability_group(evals_df, threshold=threshold),
        "hallucination_by_reliability_group",
    )
    save(plot_low_reliability_rate_by_strategy(rb_df), "low_reliability_rate_by_strategy")

    print(f"  {len(saved)} robustness figures saved.")
    return saved


def _print_reliability_counts(evals_df: pd.DataFrame, threshold: float) -> None:
    if "robustness_json" not in evals_df.columns and "robustness" not in evals_df.columns:
        print("  Robustness: not available (no robustness_json column)")
        return

    df = attach_robustness_columns(evals_df)
    n_total = len(df)
    n_unreliable = int(df["extraction_unreliable"].sum())
    scored = df[df["narrative_reliability_score"].notna()]
    n_high = int((scored["narrative_reliability_score"] >= threshold).sum())
    n_low = int(
        (
            scored["narrative_reliability_score"].notna()
            & (scored["narrative_reliability_score"] < threshold)
        ).sum()
    ) if not scored.empty else 0
    print(f"  Reliability (threshold={threshold}):")
    print(f"    total narratives     : {n_total}")
    print(f"    extraction_unreliable: {n_unreliable}")
    print(f"    high reliability     : {n_high}")
    print(f"    low reliability      : {n_low}")


def export_evaluation_figures_complete(
    evals_df: pd.DataFrame,
    cfg: AppConfig,
    run_id: str,
) -> List[Path]:
    """
    Export filtered + unfiltered hallucination figures and robustness figures.

    Exits with code 1 if narrative count is below min_narratives_for_figures.
    """
    min_n = cfg.visualisation.min_narratives_for_figures
    threshold = cfg.evaluation.robustness.reliability_threshold

    n = len(evals_df)
    print(f"\nEvaluation export for run '{run_id}' ({n} narratives)")
    _print_reliability_counts(evals_df, threshold)

    if n < min_n:
        print(
            f"\nERROR: {n} narratives is below min_narratives_for_figures ({min_n}). "
            "Figures are intended for the full evaluation run; "
            "increase sample size or lower visualisation.min_narratives_for_figures in config."
        )
        sys.exit(1)

    has_robustness = "robustness_json" in evals_df.columns or "robustness" in evals_df.columns
    saved: List[Path] = []

    if has_robustness:
        filtered = filter_reliable_extractions(evals_df, threshold=threshold)
        print(f"\n  Filtered (high reliability): {len(filtered)} narratives")
        if filtered.empty:
            print(
                "  WARNING: no high-reliability narratives; "
                "skipping main (filtered) figure export."
            )
        else:
            saved.extend(
                export_all_figures(
                    filtered,
                    cfg,
                    run_id,
                    title_suffix="high-reliability extractions",
                )
            )
    else:
        print("  WARNING: no robustness data; exporting unfiltered set only to main folder.")
        saved.extend(export_all_figures(evals_df, cfg, run_id))

    saved.extend(
        export_all_figures(
            evals_df,
            cfg,
            run_id,
            subdir="unfiltered",
            title_suffix="all narratives",
        )
    )

    if has_robustness:
        saved.extend(export_robustness_figures(evals_df, cfg, run_id))

    analysis_dir = Path(cfg.visualisation.figure_dir) / run_id / "analysis"
    analysis_paths = export_hallucination_analysis_tables(evals_df, analysis_dir)
    for path in analysis_paths:
        print(f"  Saved: {path}")
    saved.extend(analysis_paths)

    print(f"\nTotal: {len(saved)} figures saved under {cfg.visualisation.figure_dir}{run_id}/")
    return saved
