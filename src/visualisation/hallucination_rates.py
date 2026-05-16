"""
src/visualisation/hallucination_rates.py
Bar charts showing hallucination type frequencies broken down by model and
by prompt strategy.

Public API
----------
    plot_rates_by_type(evals_df)       -> matplotlib.figure.Figure
    plot_rates_by_model(evals_df)      -> matplotlib.figure.Figure
    plot_rates_by_strategy(evals_df)   -> matplotlib.figure.Figure
    plot_rates_by_dataset(evals_df)    -> matplotlib.figure.Figure
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Consistent ordering and labels
HALLUCINATION_TYPES = [
    "sign_inversion",
    "rank_swap",
    "feature_fabrication",
    "omission",
]

TYPE_LABELS = {
    "sign_inversion": "Sign\nInversion",
    "rank_swap": "Rank\nSwap",
    "feature_fabrication": "Feature\nFabrication",
    "omission": "Omission",
}

_PALETTE = "Set2"


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


# ---------------------------------------------------------------------------
# Plot 1 — Overall hallucination rate by type
# ---------------------------------------------------------------------------

def plot_rates_by_type(
    evals_df: pd.DataFrame,
    title: str = "Hallucination Rate by Type",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Grouped bar chart: one bar per hallucination type, height = proportion of
    narratives flagged for that type.

    Args:
        evals_df: DataFrame with boolean/int columns for each hallucination type.
        title:    Figure title.
        figsize:  Matplotlib figure size.

    Returns:
        A matplotlib Figure object.
    """
    _set_style()
    rates = {
        TYPE_LABELS[t]: evals_df[t].mean() * 100
        for t in HALLUCINATION_TYPES
        if t in evals_df.columns
    }

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        list(rates.keys()),
        list(rates.values()),
        color=sns.color_palette(_PALETTE, len(rates)),
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.set_ylabel("Narratives flagged (%)")
    ax.set_ylim(0, min(100, max(rates.values()) * 1.3 + 5))
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — Hallucination rate by model
# ---------------------------------------------------------------------------

def plot_rates_by_model(
    evals_df: pd.DataFrame,
    hallucination_col: str = "any_hallucination",
    title: str = "Overall Hallucination Rate by Model",
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """
    Bar chart: one bar per model, height = proportion of narratives with any
    hallucination (or a specified type).
    """
    _set_style()
    if "model_id" not in evals_df.columns:
        raise ValueError("evals_df must contain a 'model_id' column.")

    rates = (
        evals_df.groupby("model_id")[hallucination_col]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={hallucination_col: "rate"})
        .sort_values("rate", ascending=False)
    )

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        rates["model_id"],
        rates["rate"],
        color=sns.color_palette(_PALETTE, len(rates)),
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.set_ylabel("Narratives flagged (%)")
    ax.set_ylim(0, min(100, rates["rate"].max() * 1.3 + 5))
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Model")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3 — Hallucination rate by prompt strategy
# ---------------------------------------------------------------------------

def plot_rates_by_strategy(
    evals_df: pd.DataFrame,
    hallucination_col: str = "any_hallucination",
    title: str = "Overall Hallucination Rate by Prompt Strategy",
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Bar chart: one bar per prompt strategy."""
    _set_style()
    if "prompt_strategy" not in evals_df.columns:
        raise ValueError("evals_df must contain a 'prompt_strategy' column.")

    rates = (
        evals_df.groupby("prompt_strategy")[hallucination_col]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={hallucination_col: "rate"})
        .sort_values("rate", ascending=False)
    )

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        rates["prompt_strategy"],
        rates["rate"],
        color=sns.color_palette(_PALETTE, len(rates)),
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.set_ylabel("Narratives flagged (%)")
    ax.set_ylim(0, min(100, rates["rate"].max() * 1.3 + 5))
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Prompt strategy")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 4 — Per-type rates grouped by model
# ---------------------------------------------------------------------------

def plot_type_by_model(
    evals_df: pd.DataFrame,
    title: str = "Hallucination Type Rate by Model",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Grouped bar chart: x-axis = hallucination type, hue = model.
    Shows rates for each (type, model) combination.
    """
    _set_style()
    if "model_id" not in evals_df.columns:
        raise ValueError("evals_df must contain a 'model_id' column.")

    records = []
    for htype in HALLUCINATION_TYPES:
        if htype not in evals_df.columns:
            continue
        for model_id, grp in evals_df.groupby("model_id"):
            records.append({
                "Hallucination type": TYPE_LABELS[htype],
                "Model": model_id,
                "Rate (%)": grp[htype].mean() * 100,
            })
    plot_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=plot_df,
        x="Hallucination type",
        y="Rate (%)",
        hue="Model",
        palette=_PALETTE,
        ax=ax,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, min(100, plot_df["Rate (%)"].max() * 1.3 + 5))
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 5 — Hallucination rate by dataset
# ---------------------------------------------------------------------------

def plot_rates_by_dataset(
    evals_df: pd.DataFrame,
    hallucination_col: str = "any_hallucination",
    title: str = "Overall Hallucination Rate by Dataset",
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """Bar chart: one bar per dataset."""
    _set_style()
    if "dataset" not in evals_df.columns:
        raise ValueError("evals_df must contain a 'dataset' column.")

    rates = (
        evals_df.groupby("dataset")[hallucination_col]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={hallucination_col: "rate"})
        .sort_values("rate", ascending=False)
    )

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        rates["dataset"],
        rates["rate"],
        color=sns.color_palette(_PALETTE, len(rates)),
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.set_ylabel("Narratives flagged (%)")
    ax.set_ylim(0, min(100, rates["rate"].max() * 1.3 + 5))
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Dataset")
    fig.tight_layout()
    return fig
