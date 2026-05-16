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
    load_dataset(dataset_cfg)                   -> pd.DataFrame
    get_shap_columns(df, prefix)                -> List[str]
    get_feature_columns(df, prefix)             -> List[str]
    top_k_shap_features(row, prefix, k)         -> List[Tuple[str, float]]
    format_shap_table(row, prefix, dataset_name)-> str   (for prompt injection)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.config import DatasetConfig
from src.dataset_metadata import get_categorical_meaning


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_dataset(dataset_cfg: DatasetConfig) -> pd.DataFrame:
    """
    Load the processed CSV for a dataset config entry.

    Validates that:
    - The file exists.
    - At least one SHAP column (prefixed by dataset_cfg.shap_col_prefix) exists.
    - The number of rows is >= dataset_cfg.n_instances (warns if fewer).

    Returns a DataFrame with at most n_instances rows (first n rows are taken
    so results are reproducible given a fixed, pre-shuffled CSV).
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

    if len(df) < dataset_cfg.n_instances:
        import warnings
        warnings.warn(
            f"Dataset '{dataset_cfg.name}' has only {len(df)} rows "
            f"but n_instances={dataset_cfg.n_instances}. "
            "Using all available rows.",
            stacklevel=2,
        )
        return df.reset_index(drop=True)

    return df.iloc[: dataset_cfg.n_instances].reset_index(drop=True)


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


def format_shap_table(
    row: pd.Series,
    prefix: str,
    dataset_name: Optional[str] = None,
) -> str:
    """
    Render all SHAP values for a row as a human-readable table string
    suitable for injection into a prompt template.

    Sort order is signed SHAP value descending (most positive → most negative)
    to match the Martens et al. (2024) narrative-XAI prompt convention.

    Output format (one feature per line):
        feature_name: +0.4200 (feature value: 52)

    For categorical features (looked up against the registry in
    ``src.dataset_metadata`` keyed on *dataset_name*), the human-readable
    label is appended in square brackets after the integer code:

        checking_status: +0.0100 (feature value: 1 [< 0 DM])
    """
    lines: List[str] = []
    shap_cols = [c for c in row.index if c.startswith(prefix)]
    shap_cols_sorted = sorted(shap_cols, key=lambda c: float(row[c]), reverse=True)

    for shap_col in shap_cols_sorted:
        feature_name = shap_col[len(prefix):]
        shap_val = float(row[shap_col])
        sign = "+" if shap_val >= 0 else ""

        raw_val = row.get(feature_name, None)
        if raw_val is None:
            lines.append(f"  {feature_name}: {sign}{shap_val:.4f}")
            continue

        value_str = _format_feature_value(raw_val)
        meaning = get_categorical_meaning(dataset_name, feature_name, raw_val)
        if meaning is not None:
            display = f"{value_str} [{meaning}]"
        else:
            display = value_str

        lines.append(f"  {feature_name}: {sign}{shap_val:.4f} (feature value: {display})")

    return "\n".join(lines)
