"""
src/storage/human_labels_store.py
Read/write helpers for human extraction labels.

Layout per run:
    outputs/human_labels/<run_id>/labels.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.human_labels.schema import HumanLabelRecord, record_from_dict

LABELS_JSONL = "labels.jsonl"
DEFAULT_HUMAN_LABELS_DIR = "outputs/human_labels"


def labels_dir(base_dir: str | Path, run_id: str) -> Path:
    return Path(base_dir) / run_id


def labels_path(base_dir: str | Path, run_id: str) -> Path:
    return labels_dir(base_dir, run_id) / LABELS_JSONL


def load_labels(path: str | Path) -> List[HumanLabelRecord]:
    """Load all labels from a JSONL file (empty list if missing)."""
    path = Path(path)
    if not path.exists():
        return []

    records: List[HumanLabelRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(record_from_dict(json.loads(line)))
    return records


def _label_key(narrative_id: str, annotator: str) -> Tuple[str, str]:
    return (narrative_id, annotator)


def get_labeled_ids(
    path: str | Path,
    annotator: Optional[str] = None,
) -> Set[str]:
    """
    Return narrative_ids that have at least one label.

    If *annotator* is set, only count labels from that annotator.
    """
    records = load_labels(path)
    if annotator is None:
        return {r.narrative_id for r in records}
    return {
        r.narrative_id
        for r in records
        if r.annotator == annotator
    }


def _latest_by_key(records: List[HumanLabelRecord]) -> Dict[Tuple[str, str], HumanLabelRecord]:
    """Keep the last record per (narrative_id, annotator)."""
    out: Dict[Tuple[str, str], HumanLabelRecord] = {}
    for rec in records:
        out[_label_key(rec.narrative_id, rec.annotator)] = rec
    return out


def append_label(
    path: str | Path,
    record: HumanLabelRecord,
) -> None:
    """
    Upsert a label: replace any prior row with the same narrative_id + annotator,
    then rewrite the JSONL file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_labels(path)
    merged = _latest_by_key(existing)
    merged[_label_key(record.narrative_id, record.annotator)] = record

    with path.open("w", encoding="utf-8") as fh:
        for rec in merged.values():
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def load_label_for_narrative(
    path: str | Path,
    narrative_id: str,
    annotator: str,
) -> Optional[HumanLabelRecord]:
    """Return the latest label for a narrative/annotator pair, or None."""
    for rec in reversed(load_labels(path)):
        if rec.narrative_id == narrative_id and rec.annotator == annotator:
            return rec
    return None
