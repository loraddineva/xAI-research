"""
src/evaluation/robustness_runner.py
Run multi-sample extraction robustness checks for evaluated narratives.
"""

from __future__ import annotations

import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.config import AppConfig, ModelConfig
from src.evaluation.extraction_prompt_renderer import ExtractionPromptRenderer
from src.evaluation.robustness import compute_robustness, try_parse_extraction
from src.evaluation.evaluator import _feature_names_for_dataset
from src.generation.llm_client import LLMClient
from src.generation.narrative_text import narrative_text_for_evaluation
from src.storage.evaluations_store import (
    eval_run_dir,
    evaluations_csv_path,
    load_evaluations_csv,
)
from src.storage.narratives_store import load_narratives_csv, narratives_csv_path, run_dir


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_with_temperature(model_cfg: ModelConfig, temperature: float) -> ModelConfig:
    return model_cfg.model_copy(update={"temperature": temperature})


def _run_single_extraction(
    client: LLMClient,
    prompt: str,
    model_cfg: ModelConfig,
) -> tuple[str, Optional[str]]:
    try:
        return client.generate(prompt, model_cfg), None
    except Exception as exc:
        return "", repr(exc)


def _sample_extractions_parallel(
    client: LLMClient,
    prompt: str,
    model_cfg: ModelConfig,
    feature_names: List[str],
    n_runs: int,
    max_workers: int,
) -> tuple[List, List[str]]:
    """Run n_runs extractions in parallel; return (parsed_extractions, parse_errors)."""
    raw_responses: List[str] = [""] * n_runs
    api_errors: List[str] = []

    workers = min(max_workers, n_runs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_single_extraction, client, prompt, model_cfg): i
            for i in range(n_runs)
        }
        for future in as_completed(futures):
            idx = futures[future]
            raw, err = future.result()
            raw_responses[idx] = raw
            if err:
                api_errors.append(err)

    parsed = []
    parse_errors = list(api_errors)
    for raw in raw_responses:
        if not raw.strip():
            if not api_errors:
                parse_errors.append("empty response")
            continue
        extraction, err = try_parse_extraction(raw, feature_names)
        if extraction is not None:
            parsed.append(extraction)
        elif err:
            parse_errors.append(err)

    return parsed, parse_errors


def run_robustness(
    cfg: AppConfig,
    run_id: str,
    *,
    dry_run: bool = False,
    n_limit: Optional[int] = None,
    subsample_fraction: Optional[float] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Run extraction robustness checks for narratives in a generation run.

    Merges robustness results into evaluations.jsonl / evaluations.csv when present,
    or writes standalone robustness.jsonl if evaluation has not been run yet.

    Returns:
        run_id
    """
    rb_cfg = cfg.evaluation.robustness
    fraction = subsample_fraction if subsample_fraction is not None else rb_cfg.subsample_fraction
    rng_seed = seed if seed is not None else cfg.run.seed

    gen_dir = run_dir(cfg, run_id)
    narratives_path = narratives_csv_path(gen_dir)
    if not narratives_path.exists():
        raise FileNotFoundError(f"Generation run not found: {narratives_path.resolve()}")

    narratives_df = load_narratives_csv(narratives_path)
    narratives_df = narratives_df[
        narratives_df["error"].fillna("").astype(str) == ""
    ]

    if fraction < 1.0:
        n_sample = max(1, int(len(narratives_df) * fraction))
        narratives_df = narratives_df.sample(
            n=n_sample, random_state=rng_seed
        ).sort_index()

    if n_limit is not None:
        narratives_df = narratives_df.head(n_limit)

    if narratives_df.empty:
        raise ValueError(f"No narratives to check for run '{run_id}'.")

    eval_dir = eval_run_dir(cfg.evaluation.export_dir, run_id)
    eval_csv = evaluations_csv_path(eval_dir)
    evals_by_narrative: Dict[str, dict] = {}
    if eval_csv.exists():
        evals_df = load_evaluations_csv(eval_csv)
        evals_df = evals_df[evals_df["parse_error"].fillna("").astype(str) == ""]
        for _, row in evals_df.iterrows():
            evals_by_narrative[str(row["narrative_id"])] = row.to_dict()

    extraction_model = cfg.get_model(cfg.evaluation.extraction_model_id)
    robustness_model = _model_with_temperature(extraction_model, rb_cfg.temperature)

    print(f"Robustness check for run: {run_id}")
    print(f"  Narratives : {len(narratives_df)}")
    print(f"  Runs each  : {rb_cfg.n_runs} @ temperature={rb_cfg.temperature}")
    if fraction < 1.0:
        print(f"  Subsample  : {fraction:.0%}")
    if dry_run:
        print("[DRY RUN] No LLM calls or disk writes.\n")
    else:
        eval_dir.mkdir(parents=True, exist_ok=True)

    client = LLMClient()
    renderer = ExtractionPromptRenderer(cfg)
    feature_names_cache: Dict[str, List[str]] = {}
    robustness_records: List[dict] = []

    with tqdm(total=len(narratives_df), desc="Robustness checks") as pbar:
        for _, row in narratives_df.iterrows():
            narrative_id = str(row["narrative_id"])
            dataset_name = str(row["dataset"])

            if dataset_name not in feature_names_cache:
                feature_names_cache[dataset_name] = _feature_names_for_dataset(
                    cfg, dataset_name
                )
            feature_names = feature_names_cache[dataset_name]

            prompt_strategy = str(row.get("prompt_strategy", "martens"))
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
                print(
                    f"\n--- {dataset_name} | {row['model_id']} | {narrative_id[:8]}... ---"
                )
                pbar.update(1)
                continue

            parsed, parse_errors = _sample_extractions_parallel(
                client,
                prompt,
                robustness_model,
                feature_names,
                n_runs=rb_cfg.n_runs,
                max_workers=rb_cfg.max_workers,
            )

            robustness = compute_robustness(
                parsed,
                n_requested_runs=rb_cfg.n_runs,
                min_successful_runs=rb_cfg.min_successful_runs,
                reliability_threshold=rb_cfg.reliability_threshold,
            )
            robustness.parse_errors = parse_errors[:10]

            record = {
                "robustness_id": str(uuid.uuid4()),
                "narrative_id": narrative_id,
                "run_id": run_id,
                "dataset": dataset_name,
                "instance_id": int(row["instance_id"]),
                "model_id": str(row["model_id"]),
                "prompt_strategy": prompt_strategy,
                "robustness": robustness.to_dict(),
                "checked_at": _now_iso(),
            }
            robustness_records.append(record)

            if narrative_id in evals_by_narrative:
                evals_by_narrative[narrative_id]["robustness"] = robustness.to_dict()

            pbar.update(1)

    if dry_run:
        return run_id

    jsonl_path = eval_dir / "robustness.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in robustness_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if evals_by_narrative:
        _write_evaluations_with_robustness(eval_dir, evals_by_narrative, robustness_records)

    n_unreliable = sum(
        1 for r in robustness_records
        if r["robustness"].get("extraction_unreliable")
    )
    n_low = sum(
        1 for r in robustness_records
        if r["robustness"].get("flagged_low_reliability")
    )
    print(f"\nDone. Robustness: {jsonl_path}")
    print(f"  extraction_unreliable: {n_unreliable}/{len(robustness_records)}")
    print(f"  flagged_low_reliability: {n_low}/{len(robustness_records)}")

    return run_id


def _write_evaluations_with_robustness(
    eval_dir: Path,
    evals_by_narrative: Dict[str, dict],
    robustness_records: List[dict],
) -> None:
    """Rewrite evaluations artefacts with merged robustness blocks."""
    rb_by_id = {r["narrative_id"]: r["robustness"] for r in robustness_records}

    jsonl_in = eval_dir / "evaluations.jsonl"
    jsonl_out = eval_dir / "evaluations.jsonl"
    if jsonl_in.exists():
        merged_lines = []
        with jsonl_in.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                nid = obj.get("narrative_id")
                if nid in rb_by_id:
                    obj["robustness"] = rb_by_id[nid]
                merged_lines.append(obj)
        with jsonl_out.open("w", encoding="utf-8") as fh:
            for obj in merged_lines:
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    csv_path = eval_dir / "evaluations.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["robustness_json"] = df["narrative_id"].map(
            lambda nid: json.dumps(rb_by_id.get(str(nid), {}), ensure_ascii=False)
            if str(nid) in rb_by_id
            else ""
        )
        df.to_csv(csv_path, index=False)


def reliability_summary(evals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Split hallucination rates by extraction reliability (score >= threshold).

    Expects columns: any_hallucination, narrative_reliability_score (or robustness_json).
    """
    df = evals_df.copy()
    if "narrative_reliability_score" not in df.columns:
        if "robustness_json" not in df.columns:
            raise ValueError(
                "evals_df needs 'narrative_reliability_score' or 'robustness_json'"
            )

        def _score(cell: object) -> Optional[float]:
            if not cell or (isinstance(cell, float) and pd.isna(cell)):
                return None
            try:
                data = json.loads(cell) if isinstance(cell, str) else cell
            except json.JSONDecodeError:
                return None
            if isinstance(data, dict) and "narrative_reliability_score" in data:
                return data.get("narrative_reliability_score")
            if isinstance(data, dict) and "robustness" in data:
                return data["robustness"].get("narrative_reliability_score")
            return data.get("narrative_reliability_score")

        df["narrative_reliability_score"] = df["robustness_json"].map(_score)

    threshold = 0.8
    reliable = df[
        df["narrative_reliability_score"].notna()
        & (df["narrative_reliability_score"] >= threshold)
    ]
    unreliable = df[
        df["narrative_reliability_score"].notna()
        & (df["narrative_reliability_score"] < threshold)
    ]

    rows = []
    for label, subset in [("high_reliability", reliable), ("low_reliability", unreliable)]:
        if subset.empty:
            continue
        rows.append({
            "reliability_group": label,
            "n": len(subset),
            "any_hallucination_rate": subset["any_hallucination"].mean(),
        })
    return pd.DataFrame(rows)
