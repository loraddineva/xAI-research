"""
src/evaluation/evaluator.py
Run LLM extraction + SHAP comparison for a generation run.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.config import AppConfig
from src.data_loader import get_feature_columns, load_dataset
from src.evaluation.compare_to_shap import compare_to_shap
from src.evaluation.exporters import (
    EvaluationRecord,
    append_csv_row,
    append_jsonl,
    init_csv,
    write_csv,
)
from src.evaluation.extraction_parser import parse_extraction_response
from src.evaluation.extraction_prompt_renderer import ExtractionPromptRenderer
from src.generation.llm_client import LLMClient
from src.generation.narrative_text import narrative_text_for_evaluation
from src.storage.evaluations_store import (
    eval_metadata_path,
    eval_run_dir,
    evaluation_exists,
    evaluations_csv_path,
    write_eval_metadata,
)
from src.storage.narratives_store import (
    load_narratives_csv,
    narratives_csv_path,
    run_dir as generation_run_dir,
)

_META_COLS = frozenset({"label", "pred_proba", "pred_label"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _feature_names_for_dataset(cfg: AppConfig, dataset_name: str) -> List[str]:
    dataset_cfg = cfg.get_dataset(dataset_name)
    df = load_dataset(dataset_cfg)
    prefix = dataset_cfg.shap_col_prefix
    return [
        c
        for c in get_feature_columns(df, prefix)
        if c not in _META_COLS
    ]


def _parse_shap_values_sorted(cell: object) -> List[List]:
    if isinstance(cell, str):
        return json.loads(cell)
    return list(cell)


def run_evaluation(
    cfg: AppConfig,
    run_id: str,
    dry_run: bool = False,
    n_limit: Optional[int] = None,
) -> str:
    """
    Evaluate all successful narratives in a generation run.

    Returns:
        run_id (same as input; eval artefacts under evaluation.export_dir/run_id)
    """
    gen_dir = generation_run_dir(cfg, run_id)
    narratives_path = narratives_csv_path(gen_dir)
    if not narratives_path.exists():
        raise FileNotFoundError(
            f"Generation run not found: {narratives_path.resolve()}"
        )

    narratives_df = load_narratives_csv(narratives_path)
    narratives_df = narratives_df[
        narratives_df["error"].fillna("").astype(str) == ""
    ]
    if n_limit is not None:
        narratives_df = narratives_df.head(n_limit)

    if narratives_df.empty:
        raise ValueError(f"No successful narratives to evaluate in run '{run_id}'.")

    eval_dir = eval_run_dir(cfg.evaluation.export_dir, run_id)
    jsonl_path = eval_dir / "evaluations.jsonl"
    csv_path = evaluations_csv_path(eval_dir)
    metadata_path = eval_metadata_path(eval_dir)

    extraction_model = cfg.get_model(cfg.evaluation.extraction_model_id)
    feature_names_cache: Dict[str, List[str]] = {}

    if not dry_run:
        eval_dir.mkdir(parents=True, exist_ok=True)
        init_csv(csv_path)
        write_eval_metadata(metadata_path, {
            "run_id": run_id,
            "generation_run_id": run_id,
            "created_at": _now_iso(),
            "config": cfg.model_dump(),
            "extraction_model_id": extraction_model.id,
        })

    print(f"Evaluating run: {run_id}")
    print(f"  Narratives: {len(narratives_df)}")
    if dry_run:
        print("[DRY RUN] No LLM calls or disk writes.\n")
    else:
        print(f"  Output : {eval_dir}")

    client = LLMClient()
    renderer = ExtractionPromptRenderer(cfg)
    records: List[EvaluationRecord] = []

    with tqdm(total=len(narratives_df), desc="Evaluating narratives") as pbar:
        for _, row in narratives_df.iterrows():
            narrative_id = str(row["narrative_id"])
            dataset_name = str(row["dataset"])

            if not dry_run and evaluation_exists(csv_path, narrative_id):
                pbar.update(1)
                continue

            if dataset_name not in feature_names_cache:
                feature_names_cache[dataset_name] = _feature_names_for_dataset(
                    cfg, dataset_name
                )
            feature_names = feature_names_cache[dataset_name]

            prompt_strategy = str(
                row.get("prompt_strategy", "martens")
            )
            narrative_text = narrative_text_for_evaluation(
                str(row["narrative_text"]),
                prompt_strategy,
            )
            prompt = renderer.render(
                cfg.get_dataset(dataset_name),
                narrative_text=narrative_text,
                feature_names=feature_names,
            )

            if dry_run:
                print(f"\n--- {dataset_name} | {row['model_id']} | {narrative_id[:8]}... ---")
                print(prompt[:400] + ("..." if len(prompt) > 400 else ""))
                pbar.update(1)
                continue

            raw_response = ""
            parse_error = ""
            extraction_json_str = ""
            unknown_features_str = "[]"
            flags = {
                "sign_inversion": 0,
                "rank_swap": 0,
                "feature_fabrication": 0,
                "omission": 0,
                "any_hallucination": 0,
            }
            notes_str = "{}"

            try:
                raw_response = client.generate(prompt, extraction_model)
                extraction = parse_extraction_response(raw_response, feature_names)
                shap_sorted = _parse_shap_values_sorted(row["shap_values_sorted"])
                comparison = compare_to_shap(
                    extraction,
                    shap_sorted,
                    top_k_features=cfg.evaluation.top_k_features,
                )
                flags = comparison.flags_dict()
                notes_str = json.dumps(comparison.notes, ensure_ascii=False)
                extraction_json_str = json.dumps(
                    {
                        "features": {
                            k: {
                                "exists": v.exists,
                                "rank": v.rank,
                                "sign": v.sign,
                                "value": v.value,
                                "assumption": v.assumption,
                            }
                            for k, v in extraction.features.items()
                        },
                        "unknown_features": extraction.unknown_features,
                    },
                    ensure_ascii=False,
                )
                unknown_features_str = json.dumps(
                    extraction.unknown_features, ensure_ascii=False
                )
            except Exception as exc:
                parse_error = repr(exc)
                tqdm.write(f"[ERROR] {narrative_id}: {exc}")

            eval_record = EvaluationRecord(
                eval_id=str(uuid.uuid4()),
                narrative_id=narrative_id,
                run_id=run_id,
                dataset=dataset_name,
                instance_id=int(row["instance_id"]),
                model_id=str(row["model_id"]),
                prompt_strategy=prompt_strategy,
                sign_inversion=flags["sign_inversion"],
                rank_swap=flags["rank_swap"],
                feature_fabrication=flags["feature_fabrication"],
                omission=flags["omission"],
                any_hallucination=flags["any_hallucination"],
                notes=notes_str,
                extraction_json=extraction_json_str,
                unknown_features=unknown_features_str,
                extraction_model_id=extraction_model.id,
                extraction_provider=extraction_model.provider,
                extraction_model_name=extraction_model.model_name,
                extraction_prompt=prompt,
                extraction_raw_response=raw_response,
                parse_error=parse_error,
                evaluated_at=_now_iso(),
            )

            if not parse_error:
                append_jsonl(jsonl_path, eval_record)
                append_csv_row(csv_path, eval_record)

            records.append(eval_record)
            pbar.update(1)

    if not dry_run:
        write_csv(csv_path, records)
        write_eval_metadata(metadata_path, {
            "run_id": run_id,
            "generation_run_id": run_id,
            "created_at": _now_iso(),
            "config": cfg.model_dump(),
            "extraction_model_id": extraction_model.id,
            "n_records": len(records),
            "n_failed": sum(1 for r in records if r.parse_error),
            "n_any_hallucination": sum(1 for r in records if r.any_hallucination),
        })
        print(f"\nDone. Evaluations: {csv_path}")

    return run_id
