"""
src/visualisation/heatmaps.py
Heatmaps showing hallucination rate across model × prompt strategy combinations,
optionally faceted by dataset.

Public API
----------
    plot_model_strategy_heatmap(evals_df, dataset, hallucination_col) -> Figure
    plot_all_datasets_heatmap(evals_df, hallucination_col)            -> Figure
    plot_type_heatmap(evals_df)                                       -> Figure
    plot_type_strategy_heatmap(evals_df)                              -> Figure
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _set_style() -> None:
    sns.set_theme(style="white", context="paper", font_scale=1.2)


# ---------------------------------------------------------------------------
# Heatmap 1 — Model × Prompt strategy for one dataset
# ---------------------------------------------------------------------------

def plot_model_strategy_heatmap(
    evals_df: pd.DataFrame,
    dataset: Optional[str] = None,
    hallucination_col: str = "any_hallucination",
    title: Optional[str] = None,
    figsize: tuple = (7, 5),
    cmap: str = "YlOrRd",
) -> plt.Figure:
    """
    Heatmap: rows = models, columns = prompt strategies.
    Cell value = proportion of narratives flagged (0–100%).

    Args:
        evals_df:         Evaluations DataFrame (from DB or CSV).
        dataset:          If provided, filter to this dataset only.
        hallucination_col: Column to aggregate (default: 'any_hallucination').
        title:            Figure title (auto-generated if None).
        figsize:          Figure size.
        cmap:             Matplotlib colormap name.

    Returns:
        A matplotlib Figure.
    """
    _set_style()
    df = evals_df.copy()
    if dataset:
        df = df[df["dataset"] == dataset]

    from src.visualisation.hallucination_rates import order_strategy_labels, strategy_label

    pivot = (
        df.groupby(["model_id", "prompt_strategy"])[hallucination_col]
        .mean()
        .mul(100)
        .unstack("prompt_strategy")
    )
    pivot.columns = [strategy_label(str(c)) for c in pivot.columns]
    pivot = pivot[order_strategy_labels(pivot.columns)]

    if title is None:
        suffix = f" — {dataset}" if dataset else ""
        title = f"Hallucination Rate (%) by Model × Prompt Strategy{suffix}"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Flagged narratives (%)"},
    )
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Prompt strategy")
    ax.set_ylabel("Model")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Heatmap 2 — All datasets side by side
# ---------------------------------------------------------------------------

def plot_all_datasets_heatmap(
    evals_df: pd.DataFrame,
    hallucination_col: str = "any_hallucination",
    figsize: tuple = (14, 5),
    cmap: str = "YlOrRd",
) -> plt.Figure:
    """
    Side-by-side heatmaps, one per dataset in *evals_df*.
    """
    _set_style()
    from src.visualisation.hallucination_rates import order_strategy_labels, strategy_label

    datasets = sorted(evals_df["dataset"].unique())
    ncols = len(datasets)

    fig, axes = plt.subplots(1, ncols, figsize=figsize, sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        subset = evals_df[evals_df["dataset"] == dataset]
        pivot = (
            subset.groupby(["model_id", "prompt_strategy"])[hallucination_col]
            .mean()
            .mul(100)
            .unstack("prompt_strategy")
        )
        pivot.columns = [strategy_label(str(c)) for c in pivot.columns]
        pivot = pivot[order_strategy_labels(pivot.columns)]
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            vmin=0,
            vmax=100,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            cbar=(ax is axes[-1]),
            cbar_kws={"label": "Flagged (%)"} if ax is axes[-1] else {},
        )
        ax.set_title(dataset, fontweight="bold")
        ax.set_xlabel("Prompt strategy")
        ax.set_ylabel("Model" if ax is axes[0] else "")

    fig.suptitle(
        "Hallucination Rate (%) — Model × Prompt Strategy by Dataset",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Heatmap 3 — Hallucination type × model
# ---------------------------------------------------------------------------

def plot_type_heatmap(
    evals_df: pd.DataFrame,
    figsize: tuple = (10, 5),
    cmap: str = "YlOrRd",
) -> plt.Figure:
    """
    Heatmap: rows = models, columns = hallucination types.
    Shows the rate for each (model, type) pair.
    """
    _set_style()
    from src.visualisation.hallucination_rates import HALLUCINATION_TYPES, TYPE_LABELS

    htypes = [t for t in HALLUCINATION_TYPES if t in evals_df.columns]
    records = []
    for htype in htypes:
        for model_id, grp in evals_df.groupby("model_id"):
            records.append({
                "model_id": model_id,
                "type": TYPE_LABELS[htype],
                "rate": grp[htype].mean() * 100,
            })

    pivot = (
        pd.DataFrame(records)
        .pivot(index="model_id", columns="type", values="rate")
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Flagged narratives (%)"},
    )
    ax.set_title("Hallucination Rate (%) by Model × Type", fontweight="bold", pad=12)
    ax.set_xlabel("Hallucination type")
    ax.set_ylabel("Model")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Heatmap 4 — Hallucination type × prompt strategy
# ---------------------------------------------------------------------------

def plot_type_strategy_heatmap(
    evals_df: pd.DataFrame,
    figsize: tuple = (8, 5),
    cmap: str = "YlOrRd",
) -> plt.Figure:
    """
    Heatmap: rows = hallucination types, columns = prompt strategies.
    Cell value = proportion of narratives flagged (0–100%).
    """
    _set_style()
    from src.visualisation.hallucination_rates import (
        HALLUCINATION_TYPES,
        TYPE_LABELS,
        order_strategy_labels,
        strategy_label,
    )

    if "prompt_strategy" not in evals_df.columns:
        raise ValueError("evals_df must contain a 'prompt_strategy' column.")

    htypes = [t for t in HALLUCINATION_TYPES if t in evals_df.columns]
    records = []
    for htype in htypes:
        for strategy, grp in evals_df.groupby("prompt_strategy"):
            records.append({
                "type": TYPE_LABELS[htype],
                "strategy": strategy_label(str(strategy)),
                "rate": grp[htype].mean() * 100,
            })

    pivot = (
        pd.DataFrame(records)
        .pivot(index="type", columns="strategy", values="rate")
    )
    pivot = pivot[order_strategy_labels(pivot.columns)]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Flagged narratives (%)"},
    )
    ax.set_title(
        "Hallucination Rate (%) by Type × Prompt Strategy",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Prompt strategy")
    ax.set_ylabel("Hallucination type")
    fig.tight_layout()
    return fig
