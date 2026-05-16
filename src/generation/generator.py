"""
src/generation/generator.py
Iterates dataset × model × instance, calls the LLM with the single
Martens-style narrative prompt, and persists results.

Outputs per run
---------------
    outputs/generation/<run_id>/
        narratives.jsonl   — appended live during the run (crash-safe)
        run.json           — full config + all narratives, written at end
        narratives.xlsx    — flat spreadsheet, written at end

The SQLite database is also updated for backwards compatibility with the
existing evaluation script. The `prompt_strategy` column carries the
constant placeholder ``"narrative"`` since the multi-strategy split has
been retired.

Public API
----------
    run_generation(cfg, dry_run, filter_model, filter_dataset, n_override) -> str
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from src.config import AppConfig
from src.data_loader import load_dataset
from src.db import db_connection, init_db, insert_narrative, insert_run, narrative_exists
from src.generation.exporters import (
    NarrativeRecord,
    append_jsonl,
    write_run_json,
    write_xlsx,
)
from src.generation.llm_client import LLMClient
from src.generation.prompt_renderer import PromptRenderer

# Placeholder written into the legacy `prompt_strategy` column so existing
# DB queries / joins keep working without a schema migration.
_PROMPT_STRATEGY_PLACEHOLDER = "narrative"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_run_id(run_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{run_name}_{timestamp}_{short_uuid}"


def _shap_values_sorted(row: pd.Series, prefix: str) -> List[List]:
    """
    Return [[feature, value], ...] sorted from most positive to most
    negative SHAP — matches the order in the rendered prompt.
    """
    items = [
        (col[len(prefix):], float(row[col]))
        for col in row.index
        if col.startswith(prefix)
    ]
    items.sort(key=lambda x: x[1], reverse=True)
    return [[name, val] for name, val in items]


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
        dry_run:        Print prompts to stdout; skip LLM calls and disk writes.
        filter_model:   If set, only run for this model id.
        filter_dataset: If set, only run for this dataset name.
        n_override:     Override n_instances for all datasets.

    Returns:
        The run_id string for the completed run.
    """
    run_id = _make_run_id(cfg.run.name)
    run_created_at = _now_iso()
    config_snapshot = cfg.model_dump()

    db_path = Path(cfg.storage.db_path)
    run_dir = Path(cfg.storage.generation_dir) / run_id
    jsonl_path = run_dir / "narratives.jsonl"
    json_path = run_dir / "run.json"
    xlsx_path = run_dir / "narratives.xlsx"

    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
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
        print("[DRY RUN] No LLM calls or disk writes will be made.\n")
    else:
        print(f"Output dir: {run_dir}")

    models = [m for m in cfg.models if filter_model is None or m.id == filter_model]
    datasets = [d for d in cfg.datasets if filter_dataset is None or d.name == filter_dataset]

    if not models:
        raise ValueError(f"No models matched filter '{filter_model}'.")
    if not datasets:
        raise ValueError(f"No datasets matched filter '{filter_dataset}'.")

    client = LLMClient()
    renderer = PromptRenderer(cfg)

    total = sum(
        min(n_override, d.n_instances) if n_override is not None else d.n_instances
        for d in datasets
    ) * len(models)

    records: List[NarrativeRecord] = []

    # One DB connection for the whole run avoids ~N open/close cycles.
    with db_connection(db_path) as conn:
        with tqdm(total=total, desc="Generating narratives") as pbar:
            for dataset_cfg in datasets:
                df = load_dataset(dataset_cfg)
                n = min(n_override, len(df)) if n_override is not None else len(df)
                df = df.iloc[:n]

                for model_cfg in models:
                    for instance_id, row in df.iterrows():

                        if not dry_run and narrative_exists(
                            conn,
                            run_id=run_id,
                            dataset=dataset_cfg.name,
                            instance_id=int(instance_id),
                            model_id=model_cfg.id,
                        ):
                            pbar.update(1)
                            continue

                        prompt = renderer.render(
                            dataset_cfg=dataset_cfg,
                            row=row,
                        )

                        if dry_run:
                            print(
                                f"\n--- {dataset_cfg.name} | {model_cfg.id} | "
                                f"instance {instance_id} ---"
                            )
                            print(prompt[:500] + ("..." if len(prompt) > 500 else ""))
                            pbar.update(1)
                            continue

                        narrative_text = ""
                        error_msg = ""
                        try:
                            narrative_text = client.generate(prompt, model_cfg)
                        except Exception as exc:
                            error_msg = repr(exc)
                            tqdm.write(
                                f"[ERROR] {dataset_cfg.name}/{model_cfg.id}"
                                f"/instance {instance_id}: {exc}"
                            )

                        narrative_id = str(uuid.uuid4())
                        created_at = _now_iso()

                        record = NarrativeRecord(
                            narrative_id=narrative_id,
                            run_id=run_id,
                            dataset=dataset_cfg.name,
                            instance_id=int(instance_id),
                            model_id=model_cfg.id,
                            model_provider=model_cfg.provider,
                            model_name=model_cfg.model_name,
                            temperature=model_cfg.temperature,
                            max_tokens=model_cfg.max_tokens,
                            pred_proba=float(row["pred_proba"]),
                            pred_label=int(row["pred_label"]),
                            shap_values_sorted=_shap_values_sorted(
                                row, dataset_cfg.shap_col_prefix
                            ),
                            prompt=prompt,
                            narrative_text=narrative_text,
                            created_at=created_at,
                            error=error_msg,
                        )

                        # Persist to DB and stream-append to JSONL only when
                        # generation succeeded — failed rows still go to
                        # the in-memory records list so the final run.json /
                        # XLSX include them with their error message.
                        if not error_msg:
                            insert_narrative(
                                conn,
                                narrative_id=narrative_id,
                                run_id=run_id,
                                dataset=dataset_cfg.name,
                                instance_id=int(instance_id),
                                model_id=model_cfg.id,
                                prompt_strategy=_PROMPT_STRATEGY_PLACEHOLDER,
                                narrative_text=narrative_text,
                                created_at=created_at,
                            )
                            append_jsonl(jsonl_path, record)

                        records.append(record)
                        pbar.update(1)

    if not dry_run:
        # Final consolidated artefacts. Include the failed rows here so the
        # archive is a complete record of the run, not just the successes.
        run_metadata = {
            "run_id": run_id,
            "run_name": cfg.run.name,
            "created_at": run_created_at,
            "config": config_snapshot,
            "n_records": len(records),
            "n_failed": sum(1 for r in records if r.error),
        }
        write_run_json(json_path, run_metadata, records)
        write_xlsx(xlsx_path, records)
        print(f"\nDone. Run ID: {run_id}")
        print(f"  JSONL : {jsonl_path}")
        print(f"  JSON  : {json_path}")
        print(f"  XLSX  : {xlsx_path}")

    return run_id
