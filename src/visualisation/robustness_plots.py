"""
src/visualisation/robustness_plots.py
Figures for extraction robustness / narrative reliability scores.

Public API
----------
    attach_robustness_columns(evals_df)              -> pd.DataFrame
    filter_reliable_extractions(evals_df, threshold) -> pd.DataFrame
    load_robustness_df(evals_df)                     -> pd.DataFrame
    plot_reliability_distribution(rb_df, threshold)  -> Figure
    plot_reliability_by_strategy(rb_df)              -> Figure
    plot_hallucination_by_reliability_group(evals_df) -> Figure
    plot_low_reliability_rate_by_strategy(rb_df)     -> Figure
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.evaluation.robustness_runner import reliability_summary
from src.visualisation.hallucination_rates import order_strategy_labels, strategy_label

_PALETTE = "Set2"


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def _parse_robustness_cell(cell: object) -> dict:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {}
    if isinstance(cell, dict):
        return cell.get("robustness", cell) if "robustness" in cell else cell
    if isinstance(cell, str) and cell.strip():
        try:
            data = json.loads(cell)
        except json.JSONDecodeError:
            return {}
        if isinstance(data, dict) and "robustness" in data:
            return data["robustness"]
        return data if isinstance(data, dict) else {}
    return {}


def attach_robustness_columns(evals_df: pd.DataFrame) -> pd.DataFrame:
    """Add narrative_reliability_score, flagged_low_reliability, extraction_unreliable."""
    df = evals_df.copy()
    if "robustness_json" in df.columns:
        parsed = df["robustness_json"].map(_parse_robustness_cell)
    elif "robustness" in df.columns:
        parsed = df["robustness"].map(
            lambda x: x if isinstance(x, dict) else _parse_robustness_cell(x)
        )
    else:
        raise ValueError(
            "evals_df needs 'robustness_json' or 'robustness' for robustness plots"
        )

    df["narrative_reliability_score"] = parsed.map(
        lambda r: r.get("narrative_reliability_score")
    )
    df["flagged_low_reliability"] = parsed.map(
        lambda r: bool(r.get("flagged_low_reliability", False))
    )
    df["extraction_unreliable"] = parsed.map(
        lambda r: bool(r.get("extraction_unreliable", False))
    )
    return df


def filter_reliable_extractions(
    evals_df: pd.DataFrame,
    threshold: float = 0.8,
) -> pd.DataFrame:
    """Keep rows with reliability score >= threshold and reliable extraction."""
    df = attach_robustness_columns(evals_df)
    return df[
        df["narrative_reliability_score"].notna()
        & (df["narrative_reliability_score"] >= threshold)
        & ~df["extraction_unreliable"]
    ].copy()


def load_robustness_df(
    source: Union[pd.DataFrame, Path, str],
) -> pd.DataFrame:
    """
    Flatten robustness to one row per narrative.

    Accepts evaluations DataFrame (with robustness_json) or robustness.jsonl path.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        records = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rb = rec.get("robustness", {})
                records.append({
                    "narrative_id": rec.get("narrative_id"),
                    "prompt_strategy": rec.get("prompt_strategy"),
                    "model_id": rec.get("model_id"),
                    "dataset": rec.get("dataset"),
                    "narrative_reliability_score": rb.get("narrative_reliability_score"),
                    "flagged_low_reliability": rb.get("flagged_low_reliability", False),
                    "extraction_unreliable": rb.get("extraction_unreliable", False),
                })
        return pd.DataFrame(records)

    df = attach_robustness_columns(source)
    cols = [
        c
        for c in [
            "narrative_id",
            "prompt_strategy",
            "model_id",
            "dataset",
            "narrative_reliability_score",
            "flagged_low_reliability",
            "extraction_unreliable",
        ]
        if c in df.columns
    ]
    return df[cols].copy()


def plot_reliability_distribution(
    rb_df: pd.DataFrame,
    threshold: float = 0.8,
    title: str = "Narrative Reliability Score Distribution",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Histogram of narrative_reliability_score with threshold line."""
    _set_style()
    scores = rb_df["narrative_reliability_score"].dropna()
    if scores.empty:
        raise ValueError("No reliability scores to plot.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        scores,
        bins=min(20, max(5, len(scores) // 5)),
        color=sns.color_palette(_PALETTE)[0],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.axvline(threshold, color="crimson", linestyle="--", linewidth=2, label=f"Threshold ({threshold})")
    ax.set_xlabel("Narrative reliability score")
    ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_reliability_by_strategy(
    rb_df: pd.DataFrame,
    title: str = "Narrative Reliability Score by Prompt Strategy",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Box plot of reliability score by prompt strategy."""
    _set_style()
    if "prompt_strategy" not in rb_df.columns:
        raise ValueError("rb_df must contain 'prompt_strategy'.")

    plot_df = rb_df.dropna(subset=["narrative_reliability_score"]).copy()
    plot_df["Prompt strategy"] = plot_df["prompt_strategy"].map(
        lambda s: strategy_label(str(s))
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=plot_df,
        x="Prompt strategy",
        y="narrative_reliability_score",
        order=order_strategy_labels(plot_df["Prompt strategy"].unique()),
        palette=_PALETTE,
        ax=ax,
    )
    ax.set_ylabel("Narrative reliability score")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_hallucination_by_reliability_group(
    evals_df: pd.DataFrame,
    threshold: float = 0.8,
    title: str = "Any Hallucination Rate by Extraction Reliability",
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Bar chart: high vs low reliability groups × any_hallucination rate."""
    _set_style()
    df = attach_robustness_columns(evals_df)
    summary = reliability_summary(df)
    if summary.empty:
        raise ValueError("No reliability groups with data to plot.")

    labels = {
        "high_reliability": f"High reliability (≥ {threshold})",
        "low_reliability": f"Low reliability (< {threshold})",
    }
    summary["label"] = summary["reliability_group"].map(labels)
    summary["rate_pct"] = summary["any_hallucination_rate"] * 100

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        summary["label"],
        summary["rate_pct"],
        color=sns.color_palette(_PALETTE, len(summary)),
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.set_ylabel("Narratives flagged (%)")
    ax.set_ylim(0, min(100, summary["rate_pct"].max() * 1.3 + 5))
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("")
    fig.tight_layout()
    return fig


def plot_low_reliability_rate_by_strategy(
    rb_df: pd.DataFrame,
    title: str = "Low-Reliability Extraction Rate by Prompt Strategy",
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Bar chart: % narratives flagged_low_reliability per strategy."""
    _set_style()
    if "prompt_strategy" not in rb_df.columns:
        raise ValueError("rb_df must contain 'prompt_strategy'.")

    rates = (
        rb_df.groupby("prompt_strategy")["flagged_low_reliability"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"flagged_low_reliability": "rate"})
    )
    rates["label"] = rates["prompt_strategy"].map(lambda s: strategy_label(str(s)))
    rates = rates.set_index("label").loc[order_strategy_labels(rates["label"])].reset_index()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        rates["label"],
        rates["rate"],
        color=sns.color_palette(_PALETTE, len(rates)),
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.set_ylabel("Narratives flagged (%)")
    ax.set_ylim(0, min(100, rates["rate"].max() * 1.3 + 5) if len(rates) else 100)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Prompt strategy")
    fig.tight_layout()
    return fig
