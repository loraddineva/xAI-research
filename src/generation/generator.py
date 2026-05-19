"""
src/generation/generator.py
Iterates dataset × model × prompt strategy × instance, calls the LLM,
and persists results.

Outputs per run
---------------
    outputs/generation/<run_id>/
        narratives.csv       — canonical tabular store (rewritten at end)
        narratives.jsonl     — appended live during the run (crash-safe)
        run_metadata.yaml    — config snapshot + run summary

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

from src.config import AppConfig, ModelConfig, PromptStrategyConfig
from src.data_loader import load_dataset
from src.generation.exporters import (
    NarrativeRecord,
    append_csv_row,
    append_jsonl,
    init_csv,
    write_csv,
)
from src.generation.llm_client import LLMClient
from src.generation.prompt_renderer import PromptRenderer
from src.storage.narratives_store import (
    narrative_exists,
    narratives_csv_path,
    run_metadata_path,
    write_run_metadata,
)


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


def _effective_max_tokens(
    model_cfg: ModelConfig,
    strategy_cfg: PromptStrategyConfig,
) -> int:
    if strategy_cfg.max_tokens is not None:
        return strategy_cfg.max_tokens
    return model_cfg.max_tokens


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_generation(
    cfg: AppConfig,
    dry_run: bool = False,
    filter_model: Optional[str] = None,
    filter_dataset: Optional[str] = None,
    filter_strategy: Optional[str] = None,
    n_override: Optional[int] = None,
) -> str:
    """
    Run the full narrative generation pipeline.

    Args:
        cfg:              Loaded AppConfig.
        dry_run:          Print prompts to stdout; skip LLM calls and disk writes.
        filter_model:     If set, only run for this model id.
        filter_dataset:   If set, only run for this dataset name.
        filter_strategy:  If set, only run for this prompt strategy id.
        n_override:       Override n_instances for all datasets.

    Returns:
        The run_id string for the completed run.
    """
    run_id = _make_run_id(cfg.run.name)
    run_created_at = _now_iso()
    config_snapshot = cfg.model_dump()

    run_dir = Path(cfg.storage.generation_dir) / run_id
    jsonl_path = run_dir / "narratives.jsonl"
    csv_path = narratives_csv_path(run_dir)
    metadata_path = run_metadata_path(run_dir)

    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        init_csv(csv_path)
        write_run_metadata(metadata_path, {
            "run_id": run_id,
            "run_name": cfg.run.name,
            "created_at": run_created_at,
            "config": config_snapshot,
        })

    print(f"Run ID: {run_id}")
    if dry_run:
        print("[DRY RUN] No LLM calls or disk writes will be made.\n")
    else:
        print(f"Output dir: {run_dir}")

    models = [
        m for m in cfg.generation_models()
        if filter_model is None or m.id == filter_model
    ]
    datasets = [d for d in cfg.datasets if filter_dataset is None or d.name == filter_dataset]
    strategies = [
        s for s in cfg.prompt.strategies
        if filter_strategy is None or s.id == filter_strategy
    ]

    if not models:
        raise ValueError(f"No models matched filter '{filter_model}'.")
    if not datasets:
        raise ValueError(f"No datasets matched filter '{filter_dataset}'.")
    if not strategies:
        raise ValueError(f"No prompt strategies matched filter '{filter_strategy}'.")

    client = LLMClient()
    renderer = PromptRenderer(cfg)

    n_per_dataset = [
        min(n_override, d.n_instances) if n_override is not None else d.n_instances
        for d in datasets
    ]
    total = sum(n_per_dataset) * len(models) * len(strategies)

    records: List[NarrativeRecord] = []

    with tqdm(total=total, desc="Generating narratives") as pbar:
        for dataset_cfg in datasets:
            df = load_dataset(
                dataset_cfg,
                n=n_override,
                seed=cfg.run.seed,
            )

            for strategy_cfg in strategies:
                for model_cfg in models:
                    for instance_id, row in df.iterrows():

                        if not dry_run and narrative_exists(
                            csv_path,
                            dataset=dataset_cfg.name,
                            instance_id=int(instance_id),
                            model_id=model_cfg.id,
                            prompt_strategy=strategy_cfg.id,
                        ):
                            pbar.update(1)
                            continue

                        prompt = renderer.render(
                            dataset_cfg=dataset_cfg,
                            row=row,
                            strategy_id=strategy_cfg.id,
                        )

                        if dry_run:
                            print(
                                f"\n--- {dataset_cfg.name} | {model_cfg.id} | "
                                f"{strategy_cfg.id} | instance {instance_id} ---"
                            )
                            print(
                                prompt[:500] + ("..." if len(prompt) > 500 else "")
                            )
                            pbar.update(1)
                            continue

                        narrative_text = ""
                        error_msg = ""
                        max_tokens = _effective_max_tokens(model_cfg, strategy_cfg)
                        try:
                            narrative_text = client.generate(
                                prompt,
                                model_cfg,
                                max_tokens=max_tokens,
                            )
                        except Exception as exc:
                            error_msg = repr(exc)
                            tqdm.write(
                                f"[ERROR] {dataset_cfg.name}/{model_cfg.id}/"
                                f"{strategy_cfg.id}/instance {instance_id}: {exc}"
                            )

                        narrative_id = str(uuid.uuid4())
                        created_at = _now_iso()

                        record = NarrativeRecord(
                            narrative_id=narrative_id,
                            run_id=run_id,
                            dataset=dataset_cfg.name,
                            instance_id=int(instance_id),
                            model_id=model_cfg.id,
                            prompt_strategy=strategy_cfg.id,
                            model_provider=model_cfg.provider,
                            model_name=model_cfg.model_name,
                            temperature=model_cfg.temperature,
                            max_tokens=max_tokens,
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

                        if not error_msg:
                            append_jsonl(jsonl_path, record)
                            append_csv_row(csv_path, record)

                        records.append(record)
                        pbar.update(1)

    if not dry_run:
        write_csv(csv_path, records)
        run_metadata = {
            "run_id": run_id,
            "run_name": cfg.run.name,
            "created_at": run_created_at,
            "config": config_snapshot,
            "n_records": len(records),
            "n_failed": sum(1 for r in records if r.error),
        }
        write_run_metadata(metadata_path, run_metadata)
        print(f"\nDone. Run ID: {run_id}")
        print(f"  CSV   : {csv_path}")
        print(f"  JSONL : {jsonl_path}")
        print(f"  Meta  : {metadata_path}")

    return run_id
