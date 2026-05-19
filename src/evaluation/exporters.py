"""
src/evaluation/exporters.py
Persist evaluation run artefacts to disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

from src.storage.record_io import (
    append_csv_row as _append_csv_row,
    append_jsonl as _append_jsonl,
    init_csv as _init_csv,
    record_row,
    serialize_cell,
    write_csv as _write_csv,
)


EVAL_CSV_COLUMNS: List[str] = [
    "eval_id",
    "narrative_id",
    "run_id",
    "dataset",
    "instance_id",
    "model_id",
    "prompt_strategy",
    "sign_inversion",
    "rank_swap",
    "feature_fabrication",
    "omission",
    "any_hallucination",
    "notes",
    "extraction_json",
    "unknown_features",
    "extraction_model_id",
    "extraction_provider",
    "extraction_model_name",
    "extraction_prompt",
    "extraction_raw_response",
    "parse_error",
    "evaluated_at",
]


@dataclass
class EvaluationRecord:
    eval_id: str
    narrative_id: str
    run_id: str
    dataset: str
    instance_id: int
    model_id: str
    prompt_strategy: str
    sign_inversion: int
    rank_swap: int
    feature_fabrication: int
    omission: int
    any_hallucination: int
    notes: str
    extraction_json: str
    unknown_features: str
    extraction_model_id: str
    extraction_provider: str
    extraction_model_name: str
    extraction_prompt: str
    extraction_raw_response: str
    parse_error: str
    evaluated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def _record_row(rec: EvaluationRecord) -> List:
    return record_row(rec, EVAL_CSV_COLUMNS, serialize_cell)


def record_to_jsonl_dict(rec: EvaluationRecord) -> dict:
    d = rec.to_dict()
    try:
        d["notes"] = json.loads(rec.notes) if rec.notes else {}
    except json.JSONDecodeError:
        pass
    try:
        d["extraction_json"] = json.loads(rec.extraction_json) if rec.extraction_json else {}
    except json.JSONDecodeError:
        pass
    try:
        d["unknown_features"] = json.loads(rec.unknown_features) if rec.unknown_features else []
    except json.JSONDecodeError:
        pass
    return d


def append_jsonl(path: Path, rec: EvaluationRecord) -> None:
    _append_jsonl(path, record_to_jsonl_dict(rec))


def init_csv(path: Path) -> None:
    _init_csv(path, EVAL_CSV_COLUMNS)


def append_csv_row(path: Path, rec: EvaluationRecord) -> None:
    _append_csv_row(path, EVAL_CSV_COLUMNS, _record_row(rec))


def write_csv(path: Path, records: Sequence[EvaluationRecord]) -> None:
    _write_csv(path, EVAL_CSV_COLUMNS, [_record_row(rec) for rec in records])
