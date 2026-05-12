"""
src/visualisation/shap_distributions.py
Visualisation of SHAP value distributions across a dataset.

Three plots are provided:

    plot_shap_bar       — horizontal bar chart of mean |SHAP| per feature
                          (the standard "global importance" summary).
    plot_shap_beeswarm  — strip / swarm plot showing the full SHAP distribution
                          for every feature, coloured by feature value magnitude.
                          This is the closest matplotlib/seaborn equivalent to
                          the SHAP library's beeswarm plot.
    plot_shap_scatter   — scatter plot of SHAP value vs raw feature value for a
                          single feature, useful for detecting non-linear effects.

All functions take a DataFrame that already has SHAP columns attached (as
produced by scripts/prepare_data.py) — no SHAP re-computation is needed.

Public API
----------
    plot_shap_bar(df, shap_cols, figsize)                          -> Figure
    plot_shap_beeswarm(df, shap_cols, feature_cols, figsize)       -> Figure
    plot_shap_scatter(df, feature_col, shap_col, figsize)          -> Figure
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


# ---------------------------------------------------------------------------
# Plot 1 — Global importance bar chart (mean |SHAP|)
# ---------------------------------------------------------------------------

def plot_shap_bar(
    df: pd.DataFrame,
    shap_cols: List[str],
    shap_prefix: str = "shap_",
    figsize: Tuple[float, float] = (8, 6),
    palette: str = "Blues_r",
) -> plt.Figure:
    """
    Horizontal bar chart of mean absolute SHAP value per feature.

    This gives the standard "global feature importance" view: features are
    ranked by how much they move the model's output on average across all
    instances in the dataset.

    Args:
        df:          DataFrame with SHAP columns.
        shap_cols:   List of SHAP column names (e.g. ["shap_age", "shap_education_num"]).
        shap_prefix: Prefix to strip from column names for display.
        figsize:     Figure size.
        palette:     Seaborn colour palette for the bars.

    Returns:
        A matplotlib Figure.
    """
    _set_style()
    mean_abs = df[shap_cols].abs().mean().sort_values(ascending=True)
    labels = [c[len(shap_prefix):] for c in mean_abs.index]
    colors = sns.color_palette(palette, len(labels))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, mean_abs.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Global Feature Importance (Mean |SHAP|)", fontweight="bold")
    ax.set_xlim(0, mean_abs.max() * 1.2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — Beeswarm / strip plot (SHAP distribution per feature)
# ---------------------------------------------------------------------------

def plot_shap_beeswarm(
    df: pd.DataFrame,
    shap_cols: List[str],
    feature_cols: Optional[List[str]] = None,
    shap_prefix: str = "shap_",
    figsize: Tuple[float, float] = (9, 7),
    max_display: int = 15,
    sample_n: int = 500,
) -> plt.Figure:
    """
    Strip plot showing the distribution of SHAP values for each feature.

    Points are coloured by the normalised raw feature value (blue = low,
    red = high), mimicking the standard SHAP beeswarm plot. Features are
    ordered by mean |SHAP| descending.

    Args:
        df:           DataFrame with SHAP and feature columns.
        shap_cols:    List of SHAP column names.
        feature_cols: Matching raw feature column names (same order as shap_cols).
                      Used for point colouring. If None, colouring is skipped.
        shap_prefix:  Prefix stripped from SHAP column names for display.
        figsize:      Figure size.
        max_display:  Maximum number of features shown (top by mean |SHAP|).
        sample_n:     Randomly sample this many rows to keep the plot readable.
                      Set to None to use all rows.

    Returns:
        A matplotlib Figure.
    """
    _set_style()

    # Rank features by mean |SHAP| and take top max_display
    mean_abs = df[shap_cols].abs().mean().sort_values(ascending=False)
    top_shap_cols = list(mean_abs.index[:max_display])
    top_labels = [c[len(shap_prefix):] for c in top_shap_cols]

    # Sample rows for readability
    plot_df = df.sample(min(sample_n, len(df)), random_state=42) if sample_n else df

    fig, ax = plt.subplots(figsize=figsize)

    for y_pos, (shap_col, label) in enumerate(zip(top_shap_cols, top_labels)):
        shap_vals = plot_df[shap_col].values

        # Colour by normalised feature value if available
        feat_col = shap_col[len(shap_prefix):]  # strip prefix to get feature name
        if feature_cols is not None and feat_col in plot_df.columns:
            raw = plot_df[feat_col].values.astype(float)
            rng = raw.max() - raw.min()
            norm = (raw - raw.min()) / rng if rng > 0 else np.zeros_like(raw)
            colors = plt.cm.RdBu_r(1 - norm)  # blue=low, red=high
        else:
            colors = "#4C72B0"

        # Add jitter on y-axis
        jitter = np.random.default_rng(42).uniform(-0.3, 0.3, size=len(shap_vals))
        ax.scatter(
            shap_vals,
            np.full_like(shap_vals, y_pos) + jitter,
            c=colors,
            alpha=0.5,
            s=8,
            linewidths=0,
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(top_labels)))
    ax.set_yticklabels(top_labels, fontsize=10)
    ax.set_xlabel("SHAP value", fontsize=11)
    ax.set_title("SHAP Value Distribution per Feature", fontweight="bold")

    # Colourbar legend
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Feature value\n(normalised)", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3 — SHAP vs feature value scatter
# ---------------------------------------------------------------------------

def plot_shap_scatter(
    df: pd.DataFrame,
    feature_col: str,
    shap_col: str,
    figsize: Tuple[float, float] = (6, 5),
    color: str = "#4C72B0",
) -> plt.Figure:
    """
    Scatter plot of SHAP value (y-axis) vs raw feature value (x-axis) for a
    single feature.

    Useful for detecting whether the feature has a linear, threshold, or
    non-linear effect on model output, and for checking that the direction of
    the relationship is consistent with domain knowledge.

    Args:
        df:          DataFrame with SHAP and feature columns.
        feature_col: Name of the raw feature column (x-axis).
        shap_col:    Name of the corresponding SHAP column (y-axis).
        figsize:     Figure size.
        color:       Point colour.

    Returns:
        A matplotlib Figure.
    """
    _set_style()
    valid = df[[feature_col, shap_col]].dropna()

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        valid[feature_col],
        valid[shap_col],
        alpha=0.4,
        s=12,
        color=color,
        linewidths=0,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(feature_col, fontsize=11)
    ax.set_ylabel(f"SHAP value for {feature_col}", fontsize=11)
    ax.set_title(f"SHAP vs Feature Value: {feature_col}", fontweight="bold")
    fig.tight_layout()
    return fig
