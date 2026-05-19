"""
src/data_loader.py
Load a processed OpenXAI CSV (with SHAP columns attached) and return a
clean DataFrame ready for narrative generation.

Expected CSV layout
-------------------
Each row = one instance.
Feature columns: any name without the shap_col_prefix.
SHAP columns:    shap_<feature_name>  (e.g. shap_age, shap_income).

Public API
----------
    load_dataset(dataset_cfg, n, seed, sample)  -> pd.DataFrame
    get_shap_columns(df, prefix)                -> List[str]
    get_feature_columns(df, prefix)             -> List[str]
    top_k_shap_features(row, prefix, k)         -> List[Tuple[str, float]]
    format_instance_snapshot(row, prefix, dataset_name) -> str
    format_shap_table(row, prefix)                    -> str   (SHAP contributions only)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.config import DatasetConfig
from src.dataset_metadata import get_categorical_meaning, get_feature_label


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_cfg: DatasetConfig,
    n: Optional[int] = None,
    seed: int = 42,
    sample: bool = True,
) -> pd.DataFrame:
    """
    Load the processed CSV for a dataset config entry.

    Validates that:
    - The file exists.
    - At least one SHAP column (prefixed by dataset_cfg.shap_col_prefix) exists.

    When *sample* is True (default), returns a random subset of
    ``min(n or dataset_cfg.n_instances, len(df))`` rows, using ``random_state=seed``.
    The DataFrame index is the original row number in the CSV (used as ``instance_id``
    during generation). When *sample* is False, returns the full CSV unchanged.
    """
    path = Path(dataset_cfg.path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path.resolve()}\n"
            "Place the processed CSV at the path specified in config/default.yaml."
        )

    df = pd.read_csv(path)

    shap_cols = get_shap_columns(df, dataset_cfg.shap_col_prefix)
    if not shap_cols:
        raise ValueError(
            f"No SHAP columns found in '{path}' with prefix "
            f"'{dataset_cfg.shap_col_prefix}'. "
            "Check shap_col_prefix in config/default.yaml."
        )

    if not sample:
        return df

    requested = n if n is not None else dataset_cfg.n_instances
    n_take = min(requested, len(df))

    if len(df) < requested:
        import warnings
        warnings.warn(
            f"Dataset '{dataset_cfg.name}' has only {len(df)} rows "
            f"but requested {requested}. Using all available rows.",
            stacklevel=2,
        )

    if n_take < len(df):
        return df.sample(n=n_take, random_state=seed)

    return df


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def get_shap_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """Return all column names that start with *prefix*."""
    return [c for c in df.columns if c.startswith(prefix)]


def get_feature_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """Return all column names that do NOT start with *prefix*."""
    return [c for c in df.columns if not c.startswith(prefix)]


# ---------------------------------------------------------------------------
# Per-row helpers (used by narrative_generator)
# ---------------------------------------------------------------------------

def top_k_shap_features(
    row: pd.Series,
    prefix: str,
    k: int,
) -> List[Tuple[str, float]]:
    """
    Return the top-k features by absolute SHAP value for a single row.

    Returns
    -------
    List of (feature_name, shap_value) tuples, sorted by |shap_value| desc.
    feature_name has the prefix stripped (e.g. "shap_age" -> "age").
    """
    shap_items = [
        (col[len(prefix):], float(row[col]))
        for col in row.index
        if col.startswith(prefix)
    ]
    shap_items.sort(key=lambda x: abs(x[1]), reverse=True)
    return shap_items[:k]


def _format_feature_value(value: object) -> str:
    """
    Render a single feature value compactly:
      - Integer-valued floats are shown without a trailing ``.0``
        (so ``30.0`` becomes ``30``); this matches the way the raw UCI
        CSVs encode integers and avoids a stray decimal in the prompt.
      - All other floats are passed through ``str()``.
    """
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def _feature_names_for_row(row: pd.Series, prefix: str) -> List[str]:
    """Feature column names that have a matching SHAP column."""
    return [
        col[len(prefix):]
        for col in row.index
        if col.startswith(prefix) and col[len(prefix):] in row.index
    ]


def _format_instance_value(
    feature_name: str,
    raw_val: object,
    dataset_name: Optional[str],
) -> str:
    meaning = get_categorical_meaning(dataset_name, feature_name, raw_val)
    if meaning is not None:
        return meaning
    return _format_feature_value(raw_val)


def format_instance_snapshot(
    row: pd.Series,
    prefix: str,
    dataset_name: Optional[str] = None,
) -> str:
    """
    Render the instance's feature values as a human-readable profile.

    Describes *who this person is* (feature values only). Pair with
    :func:`format_shap_table` for SHAP contributions.
    """
    feature_names = sorted(_feature_names_for_row(row, prefix))
    lines: List[str] = []
    for feature_name in feature_names:
        raw_val = row.get(feature_name)
        if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)):
            continue
        label = get_feature_label(dataset_name, feature_name)
        display = _format_instance_value(feature_name, raw_val, dataset_name)
        lines.append(f"  {label}: {display}")
    return "\n".join(lines)


def format_shap_table(
    row: pd.Series,
    prefix: str,
    dataset_name: Optional[str] = None,
) -> str:
    """
    Render SHAP contributions for a row (attributions only, no feature values).

    Sort order is signed SHAP value descending (most positive → most negative)
    to match the Martens et al. (2024) narrative-XAI prompt convention.

    Output format (one feature per line):
        feature_name: +0.4200

    Use :func:`format_instance_snapshot` for the instance's feature values.
    """
    del dataset_name  # kept for API compatibility with earlier callers
    lines: List[str] = []
    shap_cols = [c for c in row.index if c.startswith(prefix)]
    shap_cols_sorted = sorted(shap_cols, key=lambda c: float(row[c]), reverse=True)

    for shap_col in shap_cols_sorted:
        feature_name = shap_col[len(prefix):]
        shap_val = float(row[shap_col])
        sign = "+" if shap_val >= 0 else ""
        lines.append(f"  {feature_name}: {sign}{shap_val:.4f}")

    return "\n".join(lines)
