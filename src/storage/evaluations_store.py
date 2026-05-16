"""
src/storage/evaluations_store.py
Read/write helpers for evaluation run artefacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

EVALUATIONS_CSV = "evaluations.csv"
EVAL_METADATA_YAML = "eval_metadata.yaml"


def eval_run_dir(export_dir: str | Path, run_id: str) -> Path:
    return Path(export_dir) / run_id


def evaluations_csv_path(run_directory: Path) -> Path:
    return run_directory / EVALUATIONS_CSV


def eval_metadata_path(run_directory: Path) -> Path:
    return run_directory / EVAL_METADATA_YAML


def load_evaluations_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def evaluation_exists(csv_path: Path, narrative_id: str) -> bool:
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path, usecols=["narrative_id", "parse_error"])
    mask = (df["narrative_id"] == narrative_id) & (
        df["parse_error"].fillna("").astype(str) == ""
    )
    return bool(mask.any())


def write_eval_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(metadata, fh, default_flow_style=False, sort_keys=False)


def load_eval_metadata(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def list_eval_runs(export_dir: str | Path) -> List[dict]:
    root = Path(export_dir)
    if not root.exists():
        return []

    runs: List[dict] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        csv_path = child / EVALUATIONS_CSV
        if not csv_path.exists():
            continue
        try:
            n_rows = max(0, sum(1 for _ in open(csv_path, encoding="utf-8")) - 1)
        except OSError:
            n_rows = 0
        runs.append({
            "run_id": child.name,
            "path": str(child),
            "n_evaluations": n_rows,
        })

    runs.sort(key=lambda r: Path(r["path"]).stat().st_mtime, reverse=True)
    return runs
