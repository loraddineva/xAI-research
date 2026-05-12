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
    export_all_figures(evals_df, cfg, run_id)                               -> List[Path]
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from src.config import AppConfig
from src.visualisation.dataset_overview import (
    plot_class_balance,
    plot_correlation_heatmap,
    plot_feature_distributions,
)
from src.visualisation.hallucination_rates import (
    plot_rates_by_dataset,
    plot_rates_by_model,
    plot_rates_by_strategy,
    plot_rates_by_type,
    plot_type_by_model,
)
from src.visualisation.heatmaps import (
    plot_all_datasets_heatmap,
    plot_model_strategy_heatmap,
    plot_type_heatmap,
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

    # Per-feature scatter plots for top 5 features
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
) -> List[Path]:
    """
    Generate and save the full evaluation figure set for a run.

    Args:
        evals_df: Evaluations DataFrame (loaded from DB or CSV).
                  Must have columns: model_id, prompt_strategy, dataset,
                  any_hallucination, sign_inversion, rank_swap,
                  feature_fabrication, magnitude_distortion, omission.
        cfg:      AppConfig (provides figure_dir, format, dpi).
        run_id:   Used to name the output subdirectory.

    Returns:
        List of Paths to saved figure files.
    """
    fig_dir = Path(cfg.visualisation.figure_dir) / run_id
    fig_dir.mkdir(parents=True, exist_ok=True)

    fmt = cfg.visualisation.format
    dpi = cfg.visualisation.dpi
    saved: List[Path] = []

    def save(fig: plt.Figure, name: str) -> None:
        path = _save(fig, fig_dir / name, dpi=dpi, fmt=fmt)
        saved.append(path)
        print(f"  Saved: {path}")

    print(f"Exporting evaluation figures to {fig_dir}/")

    # Bar charts
    save(plot_rates_by_type(evals_df),     "rates_by_type")
    save(plot_rates_by_model(evals_df),    "rates_by_model")
    save(plot_rates_by_strategy(evals_df), "rates_by_strategy")
    save(plot_rates_by_dataset(evals_df),  "rates_by_dataset")
    save(plot_type_by_model(evals_df),     "type_by_model")

    # Heatmaps
    save(plot_all_datasets_heatmap(evals_df), "heatmap_all_datasets")
    save(plot_type_heatmap(evals_df),         "heatmap_type_by_model")

    # Per-dataset heatmaps
    for dataset in sorted(evals_df["dataset"].unique()):
        safe_name = dataset.replace(" ", "_").lower()
        save(
            plot_model_strategy_heatmap(evals_df, dataset=dataset),
            f"heatmap_{safe_name}",
        )

    print(f"\n{len(saved)} evaluation figures saved.")
    return saved
