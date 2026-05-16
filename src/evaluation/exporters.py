"""
src/evaluation/exporters.py
Persist evaluation run artefacts to disk.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence


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


def _serialize_cell(val: Any) -> Any:
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    return val


def _record_row(rec: EvaluationRecord) -> List[Any]:
    return [_serialize_cell(getattr(rec, col)) for col in EVAL_CSV_COLUMNS]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record_to_jsonl_dict(rec), ensure_ascii=False) + "\n")


def init_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(EVAL_CSV_COLUMNS)


def append_csv_row(path: Path, rec: EvaluationRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(EVAL_CSV_COLUMNS)
        writer.writerow(_record_row(rec))


def write_csv(path: Path, records: Sequence[EvaluationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(EVAL_CSV_COLUMNS)
        for rec in records:
            writer.writerow(_record_row(rec))
