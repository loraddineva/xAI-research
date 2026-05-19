"""
src/visualisation/hallucination_analysis.py
Per-feature breakdowns from evaluation notes (sign inversion, rank set, omission, fabrication).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_SIGN_INV_PATTERN = re.compile(
    r"([^:;]+): narrative sign=(-?\d+), SHAP sign=(-?\d+)"
)
_RANK_SWAP_PATTERN = re.compile(
    r"Narrative top-\d+: (\[[^\]]*\]); SHAP top-\d+ by \|value\|: (\[[^\]]*\])"
)
_OMISSION_PATTERN = re.compile(
    r"Top-\d+ SHAP features not mentioned: (\[[^\]]*\])"
)
_FABRICATION_PATTERN = re.compile(
    r"Unknown features in narrative: (\[[^\]]*\])"
)


def parse_notes(notes_cell: Any) -> Dict[str, str]:
    """Parse the ``notes`` column (JSON object string or dict)."""
    if notes_cell is None or (isinstance(notes_cell, float) and pd.isna(notes_cell)):
        return {}
    if isinstance(notes_cell, dict):
        return {str(k): str(v) for k, v in notes_cell.items()}
    text = str(notes_cell).strip()
    if not text or text == "{}":
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _parse_list_literal(text: str) -> List[str]:
    return list(ast.literal_eval(text))


def parse_sign_inversions(notes: Dict[str, str]) -> List[Dict[str, Any]]:
    raw = notes.get("sign_inversion", "")
    if not raw:
        return []
    cases: List[Dict[str, Any]] = []
    for match in _SIGN_INV_PATTERN.finditer(raw):
        narrative_sign = int(match.group(2))
        shap_sign = int(match.group(3))
        cases.append({
            "feature": match.group(1).strip(),
            "narrative_sign": narrative_sign,
            "shap_sign": shap_sign,
            "direction": f"narrative {narrative_sign:+d} vs SHAP {shap_sign:+d}",
        })
    return cases


def parse_rank_swap_sets(
    notes: Dict[str, str],
) -> Optional[Tuple[frozenset[str], frozenset[str]]]:
    raw = notes.get("rank_swap", "")
    if not raw:
        return None
    match = _RANK_SWAP_PATTERN.search(raw)
    if not match:
        return None
    narrative_set = frozenset(_parse_list_literal(match.group(1)))
    shap_set = frozenset(_parse_list_literal(match.group(2)))
    return narrative_set, shap_set


def parse_omissions(notes: Dict[str, str]) -> List[str]:
    raw = notes.get("omission", "")
    if not raw:
        return []
    match = _OMISSION_PATTERN.search(raw)
    if not match:
        return []
    return list(_parse_list_literal(match.group(1)))


def parse_fabrications(
    notes: Dict[str, str],
    unknown_features: Any = None,
) -> List[str]:
    fabricated: List[str] = []
    raw = notes.get("feature_fabrication", "")
    if raw:
        match = _FABRICATION_PATTERN.search(raw)
        if match:
            fabricated.extend(_parse_list_literal(match.group(1)))

    if unknown_features is None or (
        isinstance(unknown_features, float) and pd.isna(unknown_features)
    ):
        return fabricated

    if isinstance(unknown_features, list):
        extra = unknown_features
    else:
        text = str(unknown_features).strip()
        if not text or text == "[]":
            extra = []
        elif text.startswith("["):
            extra = json.loads(text)
        else:
            extra = ast.literal_eval(text)

    for name in extra:
        if name not in fabricated:
            fabricated.append(name)
    return fabricated


def sign_inversion_by_feature(evals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count sign-inversion cases per feature and direction pair.

    Columns: feature, narrative_sign, shap_sign, direction, count
    """
    records: List[Dict[str, Any]] = []
    for _, row in evals_df.iterrows():
        for case in parse_sign_inversions(parse_notes(row.get("notes"))):
            records.append(case)

    if not records:
        return pd.DataFrame(
            columns=["feature", "narrative_sign", "shap_sign", "direction", "count"]
        )

    detail = pd.DataFrame(records)
    return (
        detail.groupby(
            ["feature", "narrative_sign", "shap_sign", "direction"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )


def missing_from_rank_set_by_feature(evals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Features in SHAP top-k but absent from the narrative top-k set (rank-swap cases).

    Columns: feature, count
    """
    counts: Dict[str, int] = {}
    for _, row in evals_df.iterrows():
        parsed = parse_rank_swap_sets(parse_notes(row.get("notes")))
        if parsed is None:
            continue
        narrative_set, shap_set = parsed
        for feature in shap_set - narrative_set:
            counts[feature] = counts.get(feature, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["feature", "count"])

    return (
        pd.DataFrame(
            [{"feature": feature, "count": count} for feature, count in counts.items()]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def omissions_by_feature(evals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count top-k SHAP features omitted from the narrative, aggregated per feature.

    Columns: feature, count
    """
    counts: Dict[str, int] = {}
    for _, row in evals_df.iterrows():
        for feature in parse_omissions(parse_notes(row.get("notes"))):
            counts[feature] = counts.get(feature, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["feature", "count"])

    return (
        pd.DataFrame(
            [{"feature": feature, "count": count} for feature, count in counts.items()]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def fabrication_list(evals_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per fabricated (unknown) feature mention.

    Columns: feature, eval_id, narrative_id, instance_id, model_id, prompt_strategy
    """
    records: List[Dict[str, Any]] = []
    for _, row in evals_df.iterrows():
        notes = parse_notes(row.get("notes"))
        for feature in parse_fabrications(notes, row.get("unknown_features")):
            records.append({
                "feature": feature,
                "eval_id": row.get("eval_id"),
                "narrative_id": row.get("narrative_id"),
                "instance_id": row.get("instance_id"),
                "model_id": row.get("model_id"),
                "prompt_strategy": row.get("prompt_strategy"),
            })

    if not records:
        return pd.DataFrame(
            columns=[
                "feature",
                "eval_id",
                "narrative_id",
                "instance_id",
                "model_id",
                "prompt_strategy",
            ]
        )

    return pd.DataFrame(records).sort_values(
        ["feature", "instance_id"], ignore_index=True
    )


def summarize_hallucination_breakdown(
    evals_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Return all four breakdown tables in one dict."""
    return {
        "sign_inversion_by_feature": sign_inversion_by_feature(evals_df),
        "missing_from_rank_set_by_feature": missing_from_rank_set_by_feature(evals_df),
        "omissions_by_feature": omissions_by_feature(evals_df),
        "fabrication_list": fabrication_list(evals_df),
    }


def export_hallucination_analysis_tables(
    evals_df: pd.DataFrame,
    out_dir: Path,
) -> List[Path]:
    """Write breakdown CSVs to *out_dir*; return paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for name, table in summarize_hallucination_breakdown(evals_df).items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        saved.append(path)
    return saved
