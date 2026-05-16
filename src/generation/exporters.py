"""
src/generation/exporters.py
Writers that persist a generation run to disk.

Layout produced per run:

    outputs/generation/<run_id>/
        narratives.csv     — canonical tabular store (rewritten at end of run)
        narratives.jsonl   — one record per line; streamed during the run

Public API
----------
    NarrativeRecord          — dataclass capturing one row.
    append_jsonl(path, rec)
    append_csv_row(path, rec)
    write_csv(path, records)
    record_to_jsonl_dict(rec)
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence


# ---------------------------------------------------------------------------
# Record dataclass
# ---------------------------------------------------------------------------

@dataclass
class NarrativeRecord:
    """One generated narrative + everything needed to reproduce it."""

    narrative_id: str
    run_id: str
    dataset: str
    instance_id: int
    model_id: str
    prompt_strategy: str
    model_provider: str
    model_name: str
    temperature: float
    max_tokens: int
    pred_proba: float
    pred_label: int
    shap_values_sorted: List[List[Any]]   # [[feature, value], ...] most positive → most negative
    prompt: str
    narrative_text: str
    created_at: str
    error: str = ""                       # populated only if generation failed

    def to_dict(self) -> dict:
        return asdict(self)


# Stable column order for CSV output.
CSV_COLUMNS: List[str] = [
    "narrative_id",
    "run_id",
    "dataset",
    "instance_id",
    "model_id",
    "prompt_strategy",
    "model_provider",
    "model_name",
    "temperature",
    "max_tokens",
    "pred_proba",
    "pred_label",
    "shap_values_sorted",
    "prompt",
    "narrative_text",
    "error",
    "created_at",
]


def _serialize_cell(val: Any) -> Any:
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    return val


def _record_row(rec: NarrativeRecord) -> List[Any]:
    return [_serialize_cell(getattr(rec, col)) for col in CSV_COLUMNS]


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def record_to_jsonl_dict(rec: NarrativeRecord) -> dict:
    """Build the structured (nested) dict written to JSONL."""
    return {
        "narrative_id": rec.narrative_id,
        "run_id": rec.run_id,
        "dataset": rec.dataset,
        "instance_id": rec.instance_id,
        "prompt_strategy": rec.prompt_strategy,
        "model": {
            "id": rec.model_id,
            "provider": rec.model_provider,
            "model_name": rec.model_name,
            "max_tokens": rec.max_tokens,
            "temperature": rec.temperature,
        },
        "prediction": {
            "proba": rec.pred_proba,
            "label": rec.pred_label,
        },
        "shap_values_sorted": rec.shap_values_sorted,
        "prompt": rec.prompt,
        "narrative_text": rec.narrative_text,
        "error": rec.error,
        "created_at": rec.created_at,
    }


def append_jsonl(path: Path, rec: NarrativeRecord) -> None:
    """Append a single narrative record as one JSONL line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record_to_jsonl_dict(rec)) + "\n")


def write_jsonl(path: Path, records: Iterable[NarrativeRecord]) -> None:
    """Write all records to a JSONL file (one record per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(record_to_jsonl_dict(rec)) + "\n")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def init_csv(path: Path) -> None:
    """Create narratives.csv with a header row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)


def append_csv_row(path: Path, rec: NarrativeRecord) -> None:
    """Append one narrative row to the CSV (crash-safe incremental write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(CSV_COLUMNS)
        writer.writerow(_record_row(rec))


def write_csv(path: Path, records: Sequence[NarrativeRecord]) -> None:
    """Rewrite the full narratives CSV (includes failed rows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for rec in records:
            writer.writerow(_record_row(rec))
