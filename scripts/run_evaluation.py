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
from typing import Dict, List

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
from src.evaluator import EvaluationResult, evaluate_narrative, llm_judge


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_shap_values(narrative_row: dict, shap_prefix: str = "shap_") -> Dict[str, float]:
    """
    The DB narrative row does not store SHAP values directly — they live in
    the processed CSV. This helper re-loads the CSV row for the given instance.

    Returns a dict mapping feature_name → shap_value.
    """
    # Import here to avoid circular at module level
    import pandas as pd
    from src.config import load_config as _load_cfg

    cfg = _load_cfg()
    dataset_name = narrative_row["dataset"]
    instance_id = narrative_row["instance_id"]

    dataset_cfg = cfg.get_dataset(dataset_name)
    df = pd.read_csv(dataset_cfg.path)
    row = df.iloc[instance_id]

    shap_cols = [c for c in row.index if c.startswith(shap_prefix)]
    return {col[len(shap_prefix):]: float(row[col]) for col in shap_cols}


def _all_dataset_features(dataset_name: str, shap_prefix: str = "shap_") -> List[str]:
    """Return all non-SHAP feature names for a dataset."""
    import pandas as pd
    from src.config import load_config as _load_cfg

    cfg = _load_cfg()
    dataset_cfg = cfg.get_dataset(dataset_name)
    df = pd.read_csv(dataset_cfg.path, nrows=1)
    return [c for c in df.columns if not c.startswith(shap_prefix)]


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
    judge_model_cfg = cfg.get_model("claude-opus") if use_judge else None

    with db_connection(db_path) as conn:
        narratives = get_narratives_for_run(conn, run_id)

    if not narratives:
        print(f"No narratives found for run_id '{run_id}'.")
        return export_dir

    print(f"Evaluating {len(narratives)} narratives for run '{run_id}'...")
    if use_judge:
        print(f"LLM judge enabled: {cfg.evaluation.llm_judge_model}")

    csv_path = export_dir / f"{run_id}_evaluations.csv"
    csv_rows: List[dict] = []

    with tqdm(total=len(narratives), desc="Evaluating") as pbar:
        for narr in narratives:
            try:
                shap_values = _parse_shap_values(narr)
                all_features = _all_dataset_features(narr["dataset"])
            except Exception as exc:
                tqdm.write(f"[WARN] Could not load SHAP for narrative {narr['narrative_id']}: {exc}")
                pbar.update(1)
                continue

            # Rule-based evaluation
            result: EvaluationResult = evaluate_narrative(
                narrative=narr["narrative_text"],
                shap_values=shap_values,
                cfg=cfg.evaluation,
                all_dataset_features=all_features,
            )

            # Optional LLM judge (only on narratives flagged by rule-based OR all)
            if use_judge and judge_model_cfg:
                try:
                    judge_result = llm_judge(narr["narrative_text"], shap_values, judge_model_cfg)
                    # Merge: flag if either rule-based or judge detects
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

            with db_connection(db_path) as conn:
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
                "dataset": narr["dataset"],
                "instance_id": narr["instance_id"],
                "model_id": narr["model_id"],
                "prompt_strategy": narr["prompt_strategy"],
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
    csv_path = run_evaluation(
        run_id=args.run_id,
        cfg_path=args.config,
        use_llm_judge_override=args.llm_judge,
    )
    print(f"\nTo export all results to CSV:\n  python scripts/export_results.py --run-id {args.run_id}")


if __name__ == "__main__":
    main()
