"""
src/visualisation/dataset_overview.py
Exploratory plots for raw dataset features.

Three plots are provided:

    plot_feature_distributions  — histogram (numeric) or bar chart (categorical)
                                  for every feature column, arranged in a grid.
    plot_class_balance          — bar chart showing the count of each target class.
    plot_correlation_heatmap    — Pearson correlation heatmap for numeric features.

All functions return a matplotlib Figure and do not call plt.show(), so they
can be used both in notebooks (display inline) and in scripts (save to disk).

Public API
----------
    plot_feature_distributions(df, feature_cols, figsize_per_plot) -> Figure
    plot_class_balance(df, label_col, figsize)                      -> Figure
    plot_correlation_heatmap(df, feature_cols, figsize)             -> Figure
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def _numeric_cols(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    return [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]


def _categorical_cols(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    return [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# Plot 1 — Feature distributions
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: List[str],
    figsize_per_plot: Tuple[float, float] = (3.0, 2.5),
    n_cols: int = 4,
    max_categories: int = 15,
) -> plt.Figure:
    """
    One subplot per feature column.

    - Numeric features: histogram with a KDE overlay.
    - Categorical features: bar chart of value counts (capped at max_categories).

    Args:
        df:               DataFrame containing the features.
        feature_cols:     List of column names to plot.
        figsize_per_plot: (width, height) of each individual subplot in inches.
        n_cols:           Number of subplots per row.
        max_categories:   Maximum number of categories shown for categorical features.

    Returns:
        A matplotlib Figure.
    """
    _set_style()
    n = len(feature_cols)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No feature columns provided.", ha="center", va="center")
        return fig

    n_rows = math.ceil(n / n_cols)
    fig_w = figsize_per_plot[0] * n_cols
    fig_h = figsize_per_plot[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for ax, col in zip(axes_flat, feature_cols):
        if pd.api.types.is_numeric_dtype(df[col]):
            ax.hist(df[col].dropna(), bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
            ax.set_xlabel(col, fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
        else:
            counts = df[col].value_counts().iloc[:max_categories]
            ax.bar(range(len(counts)), counts.values, color="#55A868", edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Count", fontsize=8)
            ax.set_xlabel(col, fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Feature Distributions", fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — Class balance
# ---------------------------------------------------------------------------

def plot_class_balance(
    df: pd.DataFrame,
    label_col: str = "label",
    figsize: Tuple[float, float] = (5, 4),
    palette: str = "Set2",
) -> plt.Figure:
    """
    Bar chart showing the count and percentage of each target class.

    Args:
        df:        DataFrame containing the label column.
        label_col: Name of the target/label column.
        figsize:   Figure size.
        palette:   Seaborn colour palette name.

    Returns:
        A matplotlib Figure.
    """
    _set_style()
    counts = df[label_col].value_counts().sort_index()
    total = counts.sum()
    colors = sns.color_palette(palette, len(counts))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        [str(c) for c in counts.index],
        counts.values,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, cnt in zip(bars, counts.values):
        pct = cnt / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{cnt:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Class Balance", fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3 — Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = "coolwarm",
) -> plt.Figure:
    """
    Pearson correlation heatmap for numeric feature columns.

    Args:
        df:           DataFrame containing the features.
        feature_cols: List of columns to include. If None, all numeric columns
                      in *df* are used.
        figsize:      Figure size.
        cmap:         Matplotlib colormap name.

    Returns:
        A matplotlib Figure.
    """
    _set_style()
    if feature_cols is None:
        feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    num_cols = _numeric_cols(df, feature_cols)
    if len(num_cols) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Not enough numeric features for correlation heatmap.",
                ha="center", va="center")
        return fig

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 8},
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    return fig
