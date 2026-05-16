"""
src/storage/narratives_store.py
Read/write helpers for generation run artefacts on disk.

Layout per run:
    outputs/generation/<run_id>/
        narratives.csv       — canonical tabular store
        narratives.jsonl     — crash-safe streaming log
        run_metadata.yaml    — config snapshot and run summary
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import yaml

from src.config import AppConfig

NARRATIVES_CSV = "narratives.csv"
RUN_METADATA_YAML = "run_metadata.yaml"


def run_dir(cfg: AppConfig, run_id: str) -> Path:
    """Resolve the output directory for a generation run."""
    return Path(cfg.storage.generation_dir) / run_id


def narratives_csv_path(run_directory: Path) -> Path:
    return run_directory / NARRATIVES_CSV


def run_metadata_path(run_directory: Path) -> Path:
    return run_directory / RUN_METADATA_YAML


def list_runs(generation_dir: str | Path) -> List[dict]:
    """
    Scan generation_dir for subfolders containing narratives.csv.

    Returns rows ordered by directory mtime (newest first):
        run_id, path, n_narratives
    """
    root = Path(generation_dir)
    if not root.exists():
        return []

    runs: List[dict] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        csv_path = child / NARRATIVES_CSV
        if not csv_path.exists():
            continue
        try:
            n_rows = max(0, sum(1 for _ in open(csv_path, encoding="utf-8")) - 1)
        except OSError:
            n_rows = 0
        runs.append({
            "run_id": child.name,
            "path": str(child),
            "n_narratives": n_rows,
        })

    runs.sort(key=lambda r: Path(r["path"]).stat().st_mtime, reverse=True)
    return runs


def load_narratives_csv(path: str | Path) -> pd.DataFrame:
    """Load the canonical narratives CSV for a run."""
    return pd.read_csv(path)


def narrative_exists(
    csv_path: Path,
    dataset: str,
    instance_id: int,
    model_id: str,
    prompt_strategy: str,
) -> bool:
    """
    Return True if a successful narrative row already exists for this key.

    Skips rows with a non-empty error field.
    """
    if not csv_path.exists():
        return False

    df = pd.read_csv(
        csv_path,
        usecols=["dataset", "instance_id", "model_id", "prompt_strategy", "error"],
    )
    if "prompt_strategy" not in df.columns:
        return False
    mask = (
        (df["dataset"] == dataset)
        & (df["instance_id"] == instance_id)
        & (df["model_id"] == model_id)
        & (df["prompt_strategy"] == prompt_strategy)
        & (df["error"].fillna("").astype(str) == "")
    )
    return bool(mask.any())


def write_run_metadata(path: Path, metadata: dict) -> None:
    """Write run metadata (config snapshot, counts) as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(metadata, fh, default_flow_style=False, sort_keys=False)


def load_run_metadata(path: Path) -> Optional[dict]:
    """Load run_metadata.yaml, or None if missing."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_run(
    generation_dir: str | Path,
    run_id: str,
) -> Optional[dict]:
    """
    Return run summary dict for run_id, or None if the run folder/CSV is missing.
    """
    root = Path(generation_dir)
    run_directory = root / run_id
    csv_path = narratives_csv_path(run_directory)
    if not csv_path.exists():
        return None

    meta = load_run_metadata(run_metadata_path(run_directory)) or {}
    return {
        "run_id": run_id,
        "run_name": meta.get("run_name", run_id),
        "created_at": meta.get("created_at", ""),
        "path": str(run_directory),
        "n_narratives": meta.get("n_records", 0),
        "n_failed": meta.get("n_failed", 0),
        "metadata": meta,
    }
