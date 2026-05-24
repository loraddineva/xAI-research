"""
Paper figures: paired strategy flip panels and rate forest plots with Wilson CIs.

Public API
----------
    flip_contingency(evals_df, error_col) -> dict[str, int]
    plot_paired_flip_panels(evals_df) -> matplotlib.figure.Figure
    build_rate_forest_rows(evals_df, recomputed_df | None, k_values) -> pd.DataFrame
    plot_rate_forest(rows_df) -> matplotlib.figure.Figure
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.evaluation.compare_to_shap import HALLUCINATION_TYPES
from src.visualisation.hallucination_rates import STRATEGY_COT, STRATEGY_DIRECT
from src.visualisation.stats import mcnemar_chi2, mcnemar_p_value, wilson_proportion_ci

# Paired flip panel styling
_CELL_BASE = "#F5F5F5"
_CELL_DISCORDANT = "#FFF4D6"
_CELL_EDGE = "#BBBBBB"
_FONT_SIZE_CELL = 13
_FONT_SIZE_LABEL = 9
_FONT_SIZE_STATS = 8
_MATRIX_XLO = -0.55
_MATRIX_XHI = 2.15
_MATRIX_YLO = -0.26
_MATRIX_YHI = 2.05
_STAT_Y = -0.19

# Paper Table 5.2 error types (fabrication omitted when all-zero).
FLIP_ERROR_TYPES = list(HALLUCINATION_TYPES) + ["any_hallucination"]

# K-sensitive types shown in the rate forest (rank swap + omission only).
K_SENSITIVITY_FOREST_TYPES = ("rank_swap", "omission")

_STRATEGY_ORDER = (STRATEGY_DIRECT, STRATEGY_COT)

_STRATEGY_COLORS = {
    "Zero-shot": "#4c72b0",
    "Chain-of-thought": "#55a868",
}

_PAPER_STRATEGY_LABELS = {
    STRATEGY_DIRECT: "Zero-shot",
    STRATEGY_COT: "Chain-of-thought",
    "direct": "Zero-shot",
    "cot": "Chain-of-thought",
}


def _paper_strategy_label(strategy_id: str) -> str:
    return _PAPER_STRATEGY_LABELS.get(strategy_id, strategy_id)


def _minimal_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _flag_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna(0).astype(int)


def paired_evaluations(evals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows with valid extractions under both prompt strategies (same instance_id).
    """
    valid = evals_df[evals_df["parse_error"].fillna("").astype(str) == ""].copy()
    if valid.empty:
        return valid

    counts = valid.groupby("instance_id")["prompt_strategy"].nunique()
    paired_ids = counts[counts >= 2].index
    valid = valid[valid["instance_id"].isin(paired_ids)]

    wide_rows: list[dict] = []
    for instance_id, grp in valid.groupby("instance_id"):
        by_strat = {str(r["prompt_strategy"]): r for _, r in grp.iterrows()}
        if STRATEGY_DIRECT not in by_strat or STRATEGY_COT not in by_strat:
            continue
        direct = by_strat[STRATEGY_DIRECT]
        cot = by_strat[STRATEGY_COT]
        row: dict = {"instance_id": instance_id}
        for col in FLIP_ERROR_TYPES:
            row[f"direct_{col}"] = int(direct[col])
            row[f"cot_{col}"] = int(cot[col])
        wide_rows.append(row)

    return pd.DataFrame(wide_rows)


def flip_contingency(paired_df: pd.DataFrame, error_col: str) -> dict[str, int]:
    """
    2×2 counts for (direct flagged × CoT flagged) on paired instances.

    Keys: both, direct_only, cot_only, neither.
    direct_only = McNemar b; cot_only = McNemar c.
    """
    d = _flag_series(paired_df, f"direct_{error_col}")
    c = _flag_series(paired_df, f"cot_{error_col}")
    return {
        "both": int(((d == 1) & (c == 1)).sum()),
        "direct_only": int(((d == 1) & (c == 0)).sum()),
        "cot_only": int(((d == 0) & (c == 1)).sum()),
        "neither": int(((d == 0) & (c == 0)).sum()),
    }


def _type_title(error_col: str) -> str:
    if error_col == "any_hallucination":
        return "Any hallucination"
    return error_col.replace("_", " ").title()


def _cell_label(row: int, col: int, value: int) -> str:
    text = str(value)
    if row == 0 and col == 1:
        return f"{text}\u1d9c"  # superscript c
    if row == 1 and col == 0:
        return f"{text}\u1d47"  # superscript b
    return text


def _mcnemar_figure_line(b: int, c: int) -> str:
    """Compact McNemar line (panel title carries the type name)."""
    chi2 = mcnemar_chi2(b, c)
    return rf"$\chi^2(1)$ = {chi2:.2f}, {mcnemar_p_value(chi2)}"


def _draw_contingency_matrix(
    ax: plt.Axes,
    counts: dict[str, int],
    *,
    show_ylabels: bool = False,
) -> None:
    """Draw a 2×2 contingency table with uniform fill and highlighted b/c cells."""
    values = [
        [counts["neither"], counts["cot_only"]],
        [counts["direct_only"], counts["both"]],
    ]
    discordant = {(0, 1), (1, 0)}

    ax.set_xlim(_MATRIX_XLO, _MATRIX_XHI)
    ax.set_ylim(_MATRIX_YLO, _MATRIX_YHI)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(2):
        for col in range(2):
            val = int(values[row][col])
            x, y = col, 1 - row
            face = _CELL_DISCORDANT if (row, col) in discordant else _CELL_BASE
            ax.add_patch(
                mpatches.Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=face,
                    edgecolor=_CELL_EDGE,
                    linewidth=1.0,
                    clip_on=False,
                )
            )
            ax.text(
                x + 0.5,
                y + 0.5,
                _cell_label(row, col, val),
                ha="center",
                va="center",
                fontsize=_FONT_SIZE_CELL,
                fontweight="normal",
                color="black",
                clip_on=False,
            )

    ax.text(0.5, -0.08, "CoT −", ha="center", va="top", fontsize=_FONT_SIZE_LABEL)
    ax.text(1.5, -0.08, "CoT +", ha="center", va="top", fontsize=_FONT_SIZE_LABEL)
    if show_ylabels:
        ax.text(
            -0.08,
            1.5,
            "Zero-shot −",
            ha="right",
            va="center",
            fontsize=_FONT_SIZE_LABEL,
        )
        ax.text(
            -0.08,
            0.5,
            "Zero-shot +",
            ha="right",
            va="center",
            fontsize=_FONT_SIZE_LABEL,
        )


def plot_paired_flip_panels(
    evals_df: pd.DataFrame,
    error_types: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Small-multiple McNemar contingency tables (paired instances only).

    Confusion-matrix layout: neither top-left, both bottom-right; off-diagonal
    cells b and c highlighted and superscripted; McNemar statistics below each panel.
    """
    paired = paired_evaluations(evals_df)
    if paired.empty:
        raise ValueError("No paired instances with valid extractions under both strategies.")

    types = list(error_types) if error_types is not None else FLIP_ERROR_TYPES
    if "feature_fabrication" in types:
        fab = flip_contingency(paired, "feature_fabrication")
        if sum(fab.values()) == fab["neither"]:
            types = [t for t in types if t != "feature_fabrication"]

    n = len(types)
    if figsize is None:
        figsize = (3.1 * n, 3.2)

    fig, axes_arr = plt.subplots(1, n, figsize=figsize, layout="constrained")
    axes = np.atleast_1d(axes_arr)

    for idx, (ax, error_col) in enumerate(zip(axes, types)):
        counts = flip_contingency(paired, error_col)
        _draw_contingency_matrix(ax, counts, show_ylabels=(idx == 0))
        ax.text(
            1.0,
            _STAT_Y,
            _mcnemar_figure_line(counts["direct_only"], counts["cot_only"]),
            ha="center",
            va="top",
            fontsize=_FONT_SIZE_STATS,
            clip_on=False,
        )

    return fig


def rate_with_ci(df: pd.DataFrame, col: str) -> tuple[int, int, float, float, float]:
    """Return k, n, p, lo, hi for column col in df."""
    n = len(df)
    k = int(_flag_series(df, col).sum()) if n else 0
    p, lo, hi = wilson_proportion_ci(k, n)
    return k, n, p, lo, hi


def build_rate_forest_rows(
    evals_df: pd.DataFrame,
    recomputed_df: pd.DataFrame | None = None,
    *,
    k_values: Sequence[int] = (2, 3, 5),
    baseline_k: int = 3,
    error_types: Sequence[str] = K_SENSITIVITY_FOREST_TYPES,
) -> pd.DataFrame:
    """
    One row per (error type, strategy, K) for forest plotting.

    Includes only K-sensitive types (rank swap and omission by default).
    Uses recomputed_df columns ``k{k}_{type}`` when present; otherwise stored
    flags at baseline K.
    """
    valid = evals_df[evals_df["parse_error"].fillna("").astype(str) == ""].copy()
    records: list[dict] = []
    ks = sorted(set(k_values) | {baseline_k})

    def add_row(
        error_col: str,
        strategy: str,
        k: int,
        subset: pd.DataFrame,
        value_col: str,
    ) -> None:
        k_count, n, p, lo, hi = rate_with_ci(subset, value_col)
        label = f"{_type_title(error_col)} (K = {k})"
        records.append({
            "error_col": error_col,
            "strategy": strategy,
            "strategy_label": _paper_strategy_label(strategy),
            "k": k,
            "label": f"{label} — {_paper_strategy_label(strategy)}",
            "k_count": k_count,
            "n": n,
            "rate": p * 100,
            "ci_lo": lo * 100,
            "ci_hi": hi * 100,
        })

    for error_col in error_types:
        for k in ks:
            recomp_col = f"k{k}_{error_col}"
            for strategy in _STRATEGY_ORDER:
                sub = valid[valid["prompt_strategy"] == strategy]
                if sub.empty:
                    continue
                if recomputed_df is not None and recomp_col in recomputed_df.columns:
                    merged = sub.merge(
                        recomputed_df[["narrative_id", recomp_col]],
                        on="narrative_id",
                        how="left",
                    )
                    col = recomp_col
                elif k == baseline_k:
                    merged = sub
                    col = error_col
                else:
                    continue
                add_row(error_col, strategy, k, merged, col)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    type_order = {t: i for i, t in enumerate(error_types)}
    df["_type_ord"] = df["error_col"].map(type_order)
    df["_k_ord"] = df["k"].fillna(-1)
    df["_strat_ord"] = df["strategy"].map({s: i for i, s in enumerate(_STRATEGY_ORDER)})
    df = df.sort_values(["_type_ord", "_k_ord", "_strat_ord"], ascending=[True, False, True])
    df = df.drop(columns=["_type_ord", "_k_ord", "_strat_ord"])
    df = df.reset_index(drop=True)
    return df


def plot_rate_forest(
    rows_df: pd.DataFrame,
    *,
    k_values: Sequence[int] = (2, 3, 5),
    figsize: tuple[float, float] = (7.5, 3.2),
) -> plt.Figure:
    """
    Two-panel forest plot: rank swap (left) and omission (right).

    Each panel has three rows (K = 2, 3, 5, top to bottom) with zero-shot and
    chain-of-thought as coloured points and Wilson 95% CI bars on a shared 0–100%
    x-axis.
    """
    if rows_df.empty:
        raise ValueError("No rows to plot.")

    from matplotlib.lines import Line2D

    k_order = sorted(k_values)
    panels = [
        ("rank_swap", "Rank swap"),
        ("omission", "Omission"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

    for ax, (error_col, panel_title) in zip(axes, panels):
        sub = rows_df[rows_df["error_col"] == error_col]
        for y_idx, k in enumerate(k_order):
            y = float(y_idx)
            for strategy in _STRATEGY_ORDER:
                match = sub[(sub["k"] == k) & (sub["strategy"] == strategy)]
                if match.empty:
                    continue
                row = match.iloc[0]
                color = _STRATEGY_COLORS.get(row["strategy_label"], "#333333")
                err_lo = row["rate"] - row["ci_lo"]
                err_hi = row["ci_hi"] - row["rate"]
                ax.errorbar(
                    row["rate"],
                    y,
                    xerr=[[err_lo], [err_hi]],
                    fmt="o",
                    color=color,
                    markersize=5,
                    capsize=2,
                    linewidth=1,
                )

        ax.set_yticks(range(len(k_order)))
        ax.set_yticklabels([str(k) for k in k_order], fontsize=9)
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlim(0, 100)
        _minimal_axes(ax)

    axes[0].set_ylabel("K", fontsize=10)
    fig.supxlabel("Narratives flagged (%)", fontsize=10)

    handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=_STRATEGY_COLORS["Zero-shot"],
            markersize=6, label="Zero-shot",
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=_STRATEGY_COLORS["Chain-of-thought"],
            markersize=6, label="Chain-of-thought",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    return fig
