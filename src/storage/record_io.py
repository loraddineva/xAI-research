"""
src/storage/record_io.py
Generic CSV and JSONL helpers for run record persistence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, List, Sequence


def serialize_cell(val: Any) -> Any:
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    return val


def record_row(rec: Any, columns: Sequence[str], serialize: Callable[[Any], Any]) -> List[Any]:
    return [serialize(getattr(rec, col)) for col in columns]


def init_csv(path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(columns)


def append_csv_row(path: Path, columns: Sequence[str], row: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(columns)
        writer.writerow(row)


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
