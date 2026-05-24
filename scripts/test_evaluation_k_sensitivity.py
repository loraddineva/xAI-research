"""
Re-compute evaluation flags at alternate top-k values for a completed run.

Uses stored extraction JSON and SHAP values from the generation run (no LLM
calls). Compares K = 2 and K = 5 against the stored K = 3 results.

Usage
-----
    python scripts/test_evaluation_k_sensitivity.py
    python scripts/test_evaluation_k_sensitivity.py --run-id pilot_run_20260518T135815_bdad28
    python scripts/test_evaluation_k_sensitivity.py --k 2 5
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.config import load_config
from src.evaluation.compare_to_shap import HALLUCINATION_TYPES, compare_to_shap
from src.evaluation.extraction_parser import parse_extraction_response
from src.evaluation.evaluator import (
    _feature_names_for_dataset,
    _parse_shap_values_sorted,
)
from src.storage.evaluations_store import eval_run_dir, evaluations_csv_path
from src.storage.narratives_store import load_narratives_csv, narratives_csv_path, run_dir

DEFAULT_FINAL_RUN_ID = "pilot_run_20260518T135815_bdad28"
DEFAULT_K_VALUES = (2, 5)
BASELINE_K = 3
STRATEGIES = ("martens", "chain_of_thought")
K_SENSITIVE_TYPES = ("rank_swap", "omission", "any_hallucination")
FLAG_COLUMNS = tuple(HALLUCINATION_TYPES) + ("any_hallucination",)


def proportion_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = proportion_ci(k, n)
    return f"{100 * k / n:.1f}% (95% CI: {100 * lo:.1f}–{100 * hi:.1f}%)"


def recompute_flags(
    cfg,
    eval_df: pd.DataFrame,
    narratives_df: pd.DataFrame,
    k_values: list[int],
) -> tuple[pd.DataFrame, dict]:
    """
    Return per-narrative recomputed flags and mismatch counts vs stored K=3.
    """
    shap_by_narrative = narratives_df.set_index("narrative_id")["shap_values_sorted"]
    feature_names_cache: dict[str, list[str]] = {}
    rows: list[dict] = []
    mismatches: dict = {
        k: defaultdict(int) for k in k_values if k != BASELINE_K
    }

    for _, row in eval_df.iterrows():
        narrative_id = str(row["narrative_id"])
        dataset_name = str(row["dataset"])
        extraction_json = str(row["extraction_json"])

        if dataset_name not in feature_names_cache:
            feature_names_cache[dataset_name] = _feature_names_for_dataset(
                cfg, dataset_name
            )
        extraction = parse_extraction_response(
            extraction_json,
            feature_names_cache[dataset_name],
        )
        shap_sorted = _parse_shap_values_sorted(shap_by_narrative[narrative_id])

        stored = {col: int(row[col]) for col in FLAG_COLUMNS}

        recomputed: dict[int, dict[str, int]] = {}
        for k in k_values:
            comparison = compare_to_shap(extraction, shap_sorted, top_k_features=k)
            recomputed[k] = comparison.flags_dict()

        if BASELINE_K in recomputed:
            for col in FLAG_COLUMNS:
                if recomputed[BASELINE_K][col] != stored[col]:
                    mismatches.setdefault("baseline", defaultdict(int))
                    mismatches["baseline"][col] += 1

        for k in k_values:
            if k == BASELINE_K:
                continue
            for col in FLAG_COLUMNS:
                if recomputed[k][col] != stored[col]:
                    mismatches[k][col] += 1

        record = {
            "narrative_id": narrative_id,
            "prompt_strategy": str(row["prompt_strategy"]),
            "instance_id": int(row["instance_id"]),
            **{f"stored_{col}": stored[col] for col in HALLUCINATION_TYPES},
            "stored_any_hallucination": stored["any_hallucination"],
        }
        for k in k_values:
            for col in FLAG_COLUMNS:
                record[f"k{k}_{col}"] = recomputed[k][col]
        rows.append(record)

    return pd.DataFrame(rows), mismatches


def print_rate_table(recomputed_df: pd.DataFrame, k_values: list[int]) -> None:
    valid = recomputed_df
    by_strategy: dict[str, pd.DataFrame] = {
        s: valid[valid["prompt_strategy"] == s] for s in STRATEGIES
    }

    print("\n" + "=" * 78)
    print("HALLUCINATION RATES BY TOP-K (valid extractions only)")
    print("=" * 78)

    header = (
        f"{'Type':<22} "
        + " ".join(f"{'K=' + str(k):>22}" for k in k_values)
        + f" {'K=3 stored':>22}"
    )
    print("\nOverall (n = {})".format(len(valid)))
    print(header)
    print("-" * len(header))

    for col in FLAG_COLUMNS:
        label = col.replace("_", " ").capitalize()
        parts = []
        for k in k_values:
            key = f"k{k}_{col}"
            k_count = int(valid[key].sum())
            parts.append(f"{pct(k_count, len(valid)):>22}")
        stored_key = "stored_any_hallucination" if col == "any_hallucination" else f"stored_{col}"
        stored_count = int(valid[stored_key].sum())
        parts.append(f"{pct(stored_count, len(valid)):>22}")
        print(f"  {label:<20} " + " ".join(parts))

    for strat in STRATEGIES:
        sub = by_strategy[strat]
        if sub.empty:
            continue
        print(f"\n{strat} (n = {len(sub)})")
        print(header)
        print("-" * len(header))
        for col in FLAG_COLUMNS:
            label = col.replace("_", " ").capitalize()
            parts = []
            for k in k_values:
                key = f"k{k}_{col}"
                k_count = int(sub[key].sum())
                parts.append(f"{pct(k_count, len(sub)):>22}")
            stored_key = (
                "stored_any_hallucination" if col == "any_hallucination" else f"stored_{col}"
            )
            stored_count = int(sub[stored_key].sum())
            parts.append(f"{pct(stored_count, len(sub)):>22}")
            print(f"  {label:<20} " + " ".join(parts))


def print_k_transitions(recomputed_df: pd.DataFrame, k_values: list[int]) -> None:
    print("\n" + "=" * 78)
    print("ANY-HALLUCINATION TRANSITIONS (vs K = 3 stored)")
    print("=" * 78)

    stored = recomputed_df["stored_any_hallucination"].astype(int)
    for k in sorted(k for k in k_values if k != BASELINE_K):
        alt = recomputed_df[f"k{k}_any_hallucination"].astype(int)
        gained = int(((stored == 0) & (alt == 1)).sum())
        lost = int(((stored == 1) & (alt == 0)).sum())
        unchanged = int((stored == alt).sum())
        print(
            f"\nK = {k}: unchanged = {unchanged}, "
            f"newly flagged = {gained}, newly faithful = {lost}"
        )

        for col in ("rank_swap", "omission"):
            stored_col = f"stored_{col}"
            alt_col = f"k{k}_{col}"
            g = int(((recomputed_df[stored_col] == 0) & (recomputed_df[alt_col] == 1)).sum())
            l = int(((recomputed_df[stored_col] == 1) & (recomputed_df[alt_col] == 0)).sum())
            print(f"  {col}: +{g} flagged, -{l} unflagged vs K=3")


def print_verification(mismatches: dict) -> None:
    print("\n" + "=" * 78)
    print("RECOMPUTATION CHECKS")
    print("=" * 78)

    baseline = mismatches.get("baseline", {})
    if not baseline:
        print(f"\nK = {BASELINE_K} recomputation matches all stored flags.")
    else:
        print(f"\nWARNING: K = {BASELINE_K} recomputation differs from stored CSV:")
        for col, count in sorted(baseline.items()):
            print(f"  {col}: {count} mismatches")


def run_k_sensitivity(
    run_id: str,
    config_path: str = "config/default.yaml",
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    if k_values is None:
        k_values = list(DEFAULT_K_VALUES)

    k_values = sorted(set(k_values) | {BASELINE_K})

    eval_dir = eval_run_dir(cfg.evaluation.export_dir, run_id)
    eval_csv = evaluations_csv_path(eval_dir)
    if not eval_csv.exists():
        raise FileNotFoundError(
            f"Evaluation run not found: {eval_csv.resolve()}\n"
            f"Run: python scripts/run_evaluation.py --run-id {run_id}"
        )

    gen_csv = narratives_csv_path(run_dir(cfg, run_id))
    if not gen_csv.exists():
        raise FileNotFoundError(f"Generation run not found: {gen_csv.resolve()}")

    eval_df = pd.read_csv(eval_csv)
    eval_df = eval_df[eval_df["parse_error"].fillna("").astype(str) == ""]
    narratives_df = load_narratives_csv(gen_csv)

    print(f"Run ID:     {run_id}")
    print(f"Eval dir:   {eval_dir.resolve()}")
    print(f"Valid n:    {len(eval_df)} (of {len(pd.read_csv(eval_csv))} evaluated)")
    print(f"K values:   {k_values} (baseline K = {BASELINE_K} from stored CSV)")

    recomputed_df, mismatches = recompute_flags(
        cfg, eval_df, narratives_df, k_values
    )
    print_verification(mismatches)
    print_rate_table(recomputed_df, k_values)
    print_k_transitions(recomputed_df, k_values)

    return recomputed_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-compute rank_swap and omission at alternate K for a completed "
            "evaluation run (default: final pilot run)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_FINAL_RUN_ID,
        help="Generation / evaluation run_id.",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        metavar="K",
        help="Top-k values to evaluate (K=3 is always included for comparison).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_k_sensitivity(
        run_id=args.run_id,
        config_path=args.config,
        k_values=args.k,
    )


if __name__ == "__main__":
    main()
