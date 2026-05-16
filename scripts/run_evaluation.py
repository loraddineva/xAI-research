"""
scripts/run_evaluation.py
CLI entry point for evaluating a completed generation run.

Usage
-----
    python scripts/run_evaluation.py --run-id <run_id>
    python scripts/run_evaluation.py --run-id <run_id> --config config/default.yaml
    python scripts/run_evaluation.py --run-id <run_id> --llm-judge
"""

from __future__ import annotations

import argparse
import csv
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.config import load_config
from src.db import (
    db_connection,
    get_narratives_for_run,
    insert_evaluation,
)
from src.evaluation import EvaluationResult, evaluate_narrative, llm_judge


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Dataset / SHAP helpers — loaded once per evaluation run
# ---------------------------------------------------------------------------

def _build_dataset_cache(
    narratives: List[dict],
    shap_prefix: str = "shap_",
) -> Dict[str, dict]:
    """
    Pre-load every dataset CSV referenced by the narratives list.

    Returns a dict keyed by dataset name:
        {
            "adult": {
                "df":      pd.DataFrame,          # full CSV
                "features": List[str],            # non-SHAP, non-label column names
            },
            ...
        }

    Loading once here avoids re-reading the same (potentially large) CSV once
    per narrative inside the evaluation loop.
    """
    import pandas as pd

    cfg = load_config()
    dataset_names = {n["dataset"] for n in narratives}
    cache: Dict[str, dict] = {}

    for name in dataset_names:
        try:
            dataset_cfg = cfg.get_dataset(name)
            df = pd.read_csv(dataset_cfg.path)
            features = [
                c for c in df.columns
                if not c.startswith(shap_prefix) and c != "label"
            ]
            cache[name] = {"df": df, "features": features}
        except Exception as exc:
            tqdm.write(f"[WARN] Could not load dataset '{name}': {exc}")

    return cache


def _get_shap_values(
    cache: Dict[str, dict],
    dataset: str,
    instance_id: int,
    shap_prefix: str = "shap_",
) -> Dict[str, float]:
    """Extract SHAP values for one instance from the pre-loaded cache."""
    df = cache[dataset]["df"]
    row = df.iloc[instance_id]
    shap_cols = [c for c in row.index if c.startswith(shap_prefix)]
    return {col[len(shap_prefix):]: float(row[col]) for col in shap_cols}


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_evaluation(
    run_id: str,
    cfg_path: str = "config/default.yaml",
    use_llm_judge_override: bool = False,
) -> Path:
    """
    Load all narratives for *run_id*, evaluate each, write results to DB and CSV.

    Returns the path to the exported CSV.
    """
    cfg = load_config(cfg_path)
    db_path = Path(cfg.storage.db_path)
    export_dir = Path(cfg.storage.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    use_judge = use_llm_judge_override or cfg.evaluation.use_llm_judge

    # Resolve judge model by id (llm_judge_model in config stores the model id)
    judge_model_cfg = None
    if use_judge:
        try:
            judge_model_cfg = cfg.get_model(cfg.evaluation.llm_judge_model)
        except KeyError:
            tqdm.write(
                f"[WARN] LLM judge model id '{cfg.evaluation.llm_judge_model}' "
                "not found in config models list — judge disabled."
            )
            use_judge = False

    with db_connection(db_path) as conn:
        narratives = get_narratives_for_run(conn, run_id)

    if not narratives:
        print(f"No narratives found for run_id '{run_id}'.")
        return export_dir

    print(f"Evaluating {len(narratives)} narratives for run '{run_id}'...")
    if use_judge and judge_model_cfg:
        print(f"LLM judge enabled: {judge_model_cfg.model_name}")

    # Pre-load all dataset CSVs once — avoids re-reading per narrative
    dataset_cache = _build_dataset_cache(narratives)

    csv_path = export_dir / f"{run_id}_evaluations.csv"
    csv_rows: List[dict] = []

    with db_connection(db_path) as conn:
        with tqdm(total=len(narratives), desc="Evaluating") as pbar:
            for narr in narratives:
                dataset = narr["dataset"]

                if dataset not in dataset_cache:
                    tqdm.write(
                        f"[WARN] Dataset '{dataset}' not in cache — "
                        f"skipping narrative {narr['narrative_id']}"
                    )
                    pbar.update(1)
                    continue

                try:
                    shap_values = _get_shap_values(
                        dataset_cache, dataset, narr["instance_id"]
                    )
                    all_features = dataset_cache[dataset]["features"]
                except Exception as exc:
                    tqdm.write(
                        f"[WARN] Could not extract SHAP for narrative "
                        f"{narr['narrative_id']}: {exc}"
                    )
                    pbar.update(1)
                    continue

                # Single Martens-style narrative — no CoT trimming needed.
                eval_text = narr["narrative_text"]

                result: EvaluationResult = evaluate_narrative(
                    narrative=eval_text,
                    shap_values=shap_values,
                    cfg=cfg.evaluation,
                    all_dataset_features=all_features,
                )

                # Optional LLM judge
                if use_judge and judge_model_cfg:
                    try:
                        judge_result = llm_judge(eval_text, shap_values, judge_model_cfg)
                        result.sign_inversion = result.sign_inversion or judge_result.sign_inversion
                        result.rank_swap = result.rank_swap or judge_result.rank_swap
                        result.feature_fabrication = result.feature_fabrication or judge_result.feature_fabrication
                        result.magnitude_distortion = result.magnitude_distortion or judge_result.magnitude_distortion
                        result.omission = result.omission or judge_result.omission
                        result.notes.extend(judge_result.notes)
                    except Exception as exc:
                        tqdm.write(f"[WARN] LLM judge failed for {narr['narrative_id']}: {exc}")

                eval_id = str(uuid.uuid4())
                evaluated_at = _now_iso()

                insert_evaluation(
                    conn,
                    eval_id=eval_id,
                    narrative_id=narr["narrative_id"],
                    sign_inversion=result.sign_inversion,
                    rank_swap=result.rank_swap,
                    feature_fabrication=result.feature_fabrication,
                    magnitude_distortion=result.magnitude_distortion,
                    omission=result.omission,
                    notes=result.notes_str(),
                    evaluated_at=evaluated_at,
                )

                csv_rows.append({
                    "eval_id": eval_id,
                    "narrative_id": narr["narrative_id"],
                    "run_id": run_id,
                    "dataset": dataset,
                    "instance_id": narr["instance_id"],
                    "model_id": narr["model_id"],
                    "prompt_strategy": narr.get("prompt_strategy", "narrative"),
                    **result.to_dict(),
                    "evaluated_at": evaluated_at,
                })

                pbar.update(1)

    # Write CSV
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nEvaluations saved to: {csv_path}")

    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a completed narrative generation run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-id", required=True, help="The run_id to evaluate.")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path.")
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Force enable LLM judge regardless of config setting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        run_id=args.run_id,
        cfg_path=args.config,
        use_llm_judge_override=args.llm_judge,
    )
    print(f"\nTo export all results to CSV:\n  python scripts/export_results.py --run-id {args.run_id}")


if __name__ == "__main__":
    main()
