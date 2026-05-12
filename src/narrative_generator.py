"""
src/narrative_generator.py
Iterates dataset × model × prompt strategy, calls the LLM, and persists
results to both SQLite (via src.db) and JSON files in outputs/narratives/.

Public API
----------
    run_generation(cfg, dry_run, filter_model, filter_dataset, n_override)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.config import AppConfig, DatasetConfig, ModelConfig
from src.data_loader import load_dataset
from src.db import db_connection, init_db, insert_narrative, insert_run, narrative_exists
from src.llm_client import LLMClient
from src.prompt_renderer import PromptRenderer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_run_id(run_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{run_name}_{timestamp}_{short_uuid}"


# ---------------------------------------------------------------------------
# Narrative JSON saver
# ---------------------------------------------------------------------------

def _save_narrative_json(
    narrative_dir: Path,
    run_id: str,
    dataset: str,
    instance_id: int,
    model_id: str,
    strategy: str,
    prompt: str,
    narrative_text: str,
    narrative_id: str,
    created_at: str,
) -> None:
    """Append one narrative record to a per-run JSON-lines file."""
    narrative_dir.mkdir(parents=True, exist_ok=True)
    out_file = narrative_dir / f"{run_id}.jsonl"
    record = {
        "narrative_id": narrative_id,
        "run_id": run_id,
        "dataset": dataset,
        "instance_id": instance_id,
        "model_id": model_id,
        "prompt_strategy": strategy,
        "prompt": prompt,
        "narrative_text": narrative_text,
        "created_at": created_at,
    }
    with out_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_generation(
    cfg: AppConfig,
    dry_run: bool = False,
    filter_model: Optional[str] = None,
    filter_dataset: Optional[str] = None,
    n_override: Optional[int] = None,
) -> str:
    """
    Run the full narrative generation pipeline.

    Args:
        cfg:            Loaded AppConfig.
        dry_run:        If True, print prompts instead of calling LLMs; skip DB writes.
        filter_model:   If set, only run for this model id.
        filter_dataset: If set, only run for this dataset name.
        n_override:     Override n_instances for all datasets (useful for quick tests).

    Returns:
        The run_id string for the completed run.

    Resume behaviour
    ----------------
    If the script is interrupted and restarted with the same run_id (not yet
    supported via CLI — resume requires passing run_id explicitly), narratives
    that already exist in the DB are skipped. When starting a fresh run the
    run_id is generated automatically, so restarting always creates a new run.
    To resume a crashed run, use the --resume-run-id flag in run_generation.py.
    """
    run_id = _make_run_id(cfg.run.name)
    run_created_at = _now_iso()
    config_snapshot = cfg.model_dump()

    db_path = Path(cfg.storage.db_path)
    narrative_dir = Path(cfg.storage.narrative_dir)

    if not dry_run:
        init_db(db_path)
        with db_connection(db_path) as conn:
            insert_run(
                conn,
                run_id=run_id,
                run_name=cfg.run.name,
                config_json=config_snapshot,
                created_at=run_created_at,
            )

    print(f"Run ID: {run_id}")
    if dry_run:
        print("[DRY RUN] No LLM calls or DB writes will be made.\n")

    models = [m for m in cfg.models if filter_model is None or m.id == filter_model]
    datasets = [d for d in cfg.datasets if filter_dataset is None or d.name == filter_dataset]
    strategies = cfg.prompts.strategies

    if not models:
        raise ValueError(f"No models matched filter '{filter_model}'.")
    if not datasets:
        raise ValueError(f"No datasets matched filter '{filter_dataset}'.")

    client = LLMClient()
    renderer = PromptRenderer(cfg)

    total = sum(
        min(n_override or d.n_instances, d.n_instances) * len(models) * len(strategies)
        for d in datasets
    )

    # Open a single DB connection for the entire generation run.
    # This avoids the overhead of opening and closing a connection for every
    # narrative write (which would be ~3,600 open/close cycles in a full run).
    with db_connection(db_path) as conn:
        with tqdm(total=total, desc="Generating narratives") as pbar:
            for dataset_cfg in datasets:
                df = load_dataset(dataset_cfg)
                n = min(n_override, len(df)) if n_override is not None else len(df)
                df = df.iloc[:n]

                for model_cfg in models:
                    for strategy in strategies:
                        for instance_id, row in df.iterrows():

                            # Resume: skip if this narrative was already generated
                            if not dry_run and narrative_exists(
                                conn,
                                run_id=run_id,
                                dataset=dataset_cfg.name,
                                instance_id=int(instance_id),
                                model_id=model_cfg.id,
                                prompt_strategy=strategy,
                            ):
                                pbar.update(1)
                                continue

                            prompt = renderer.render(
                                strategy=strategy,
                                dataset_name=dataset_cfg.name,
                                row=row,
                                shap_prefix=dataset_cfg.shap_col_prefix,
                            )

                            if dry_run:
                                print(
                                    f"\n--- {dataset_cfg.name} | {model_cfg.id} | "
                                    f"{strategy} | instance {instance_id} ---"
                                )
                                print(prompt[:500] + ("..." if len(prompt) > 500 else ""))
                                pbar.update(1)
                                continue

                            try:
                                narrative_text = client.generate(prompt, model_cfg)
                            except Exception as exc:
                                tqdm.write(
                                    f"[ERROR] {dataset_cfg.name}/{model_cfg.id}/{strategy}"
                                    f"/instance {instance_id}: {exc}"
                                )
                                pbar.update(1)
                                continue

                            narrative_id = str(uuid.uuid4())
                            created_at = _now_iso()

                            insert_narrative(
                                conn,
                                narrative_id=narrative_id,
                                run_id=run_id,
                                dataset=dataset_cfg.name,
                                instance_id=int(instance_id),
                                model_id=model_cfg.id,
                                prompt_strategy=strategy,
                                narrative_text=narrative_text,
                                created_at=created_at,
                            )

                            _save_narrative_json(
                                narrative_dir=narrative_dir,
                                run_id=run_id,
                                dataset=dataset_cfg.name,
                                instance_id=int(instance_id),
                                model_id=model_cfg.id,
                                strategy=strategy,
                                prompt=prompt,
                                narrative_text=narrative_text,
                                narrative_id=narrative_id,
                                created_at=created_at,
                            )

                            pbar.update(1)

    print(f"\nDone. Run ID: {run_id}")
    return run_id
