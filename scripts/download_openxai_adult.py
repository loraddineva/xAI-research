"""
scripts/download_openxai_adult.py

Orchestrates three download/compute tasks for the Adult Income dataset:

  Task A (--task raw):
      Download the original UCI Adult Income dataset with string categorical
      values intact (not encoded/normalised). Saves to data/raw/adult_original.csv
      and a companion metadata JSON.

  Task B (--task ann):
      Compute SHAP values using OpenXAI's pre-trained ANN model and save
      to data/processed/adult_ann.csv (same schema as adult.csv / LR model).

  Task C (--task metrics):
      Use OpenXAI's Evaluator to compute PGI, PGU, RIS, RRS, ROS for SHAP
      explanations on both the LR model (adult.csv) and ANN model (adult_ann.csv).
      Saves results to outputs/xai_metrics/adult_openxai_metrics.csv.

  Default (--task all):
      Run A → B → C in sequence.

Usage
-----
    python scripts/download_openxai_adult.py
    python scripts/download_openxai_adult.py --task raw
    python scripts/download_openxai_adult.py --task ann --n 100
    python scripts/download_openxai_adult.py --task metrics
    python scripts/download_openxai_adult.py --task all --n 200

Notes
-----
- FA, RA, SA, SRA, PRA require ground-truth explanations available only for
  synthetic datasets. They are not computed here (Adult Income is real-world).
- OpenXAI must be installed from source:
      git clone https://github.com/AI4LIFE-GROUP/OpenXAI.git
      cd OpenXAI && pip install -e .
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse existing helpers from prepare_data.py
from scripts.prepare_data import prepare_dataset, validate_output  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# UCI Adult Income dataset URLs
UCI_TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
UCI_TEST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)

# Original 15 column names (including target)
UCI_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

# Feature type metadata
FEATURE_METADATA = {
    "age": {
        "type": "continuous",
        "description": "Age in years",
        "range": "17–90",
    },
    "workclass": {
        "type": "categorical",
        "description": "Employment class",
        "values": [
            "Private", "Self-emp-not-inc", "Self-emp-inc",
            "Federal-gov", "Local-gov", "State-gov",
            "Without-pay", "Never-worked",
        ],
    },
    "fnlwgt": {
        "type": "continuous",
        "description": "Final sampling weight (Census population estimate)",
        "range": "12285–1484705",
    },
    "education": {
        "type": "categorical",
        "description": "Highest educational attainment (string label)",
        "values": [
            "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th",
            "10th", "11th", "12th", "HS-grad", "Some-college",
            "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters",
            "Prof-school", "Doctorate",
        ],
    },
    "education_num": {
        "type": "ordinal",
        "description": "Numeric encoding of education (1=Preschool … 16=Doctorate)",
        "range": "1–16",
    },
    "marital_status": {
        "type": "categorical",
        "description": "Marital status",
        "values": [
            "Married-civ-spouse", "Divorced", "Never-married",
            "Separated", "Widowed", "Married-spouse-absent",
            "Married-AF-spouse",
        ],
    },
    "occupation": {
        "type": "categorical",
        "description": "Occupation type",
        "values": [
            "Tech-support", "Craft-repair", "Other-service", "Sales",
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
            "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
            "Transport-moving", "Priv-house-serv", "Protective-serv",
            "Armed-Forces",
        ],
    },
    "relationship": {
        "type": "categorical",
        "description": "Relationship role within the household",
        "values": [
            "Wife", "Own-child", "Husband",
            "Not-in-family", "Other-relative", "Unmarried",
        ],
    },
    "race": {
        "type": "categorical",
        "description": "Race / ethnicity",
        "values": [
            "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
            "Other", "Black",
        ],
    },
    "sex": {
        "type": "binary",
        "description": "Sex",
        "values": ["Male", "Female"],
    },
    "capital_gain": {
        "type": "continuous",
        "description": "Capital gains in USD (investment income)",
        "range": "0–99999",
    },
    "capital_loss": {
        "type": "continuous",
        "description": "Capital losses in USD",
        "range": "0–4356",
    },
    "hours_per_week": {
        "type": "continuous",
        "description": "Self-reported weekly working hours",
        "range": "1–99",
    },
    "native_country": {
        "type": "categorical",
        "description": "Country of origin (41 unique values; dropped by OpenXAI)",
        "notes": "Excluded from OpenXAI-processed version",
    },
    "income": {
        "type": "binary_target",
        "description": "Annual income relative to $50,000 threshold",
        "values": ["<=50K", ">50K"],
        "encoding": {"<=50K": 0, ">50K": 1},
    },
}


# ---------------------------------------------------------------------------
# Task A — Download raw original UCI dataset
# ---------------------------------------------------------------------------

def download_raw(output_dir: Path) -> Path:
    """
    Download the original UCI Adult Income dataset with original string
    categorical values (not encoded). Saves to output_dir/adult_original.csv
    and a companion adult_original_metadata.json.

    Returns path to saved CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "adult_original.csv"
    out_meta = output_dir / "adult_original_metadata.json"

    print(f"\n{'='*60}")
    print("Task A: Downloading raw UCI Adult Income dataset")
    print(f"{'='*60}")

    def _fetch(url: str, skip_rows: int = 0) -> pd.DataFrame:
        print(f"  Fetching: {url}")
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read().decode("utf-8")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        lines = lines[skip_rows:]
        df = pd.read_csv(
            StringIO("\n".join(lines)),
            header=None,
            names=UCI_COLUMNS,
            skipinitialspace=True,
        )
        return df

    df_train = _fetch(UCI_TRAIN_URL)
    # Test file has a header comment line ("| 1x3 Cross validator") — skip 1 row
    df_test = _fetch(UCI_TEST_URL, skip_rows=1)

    # The test file appends a trailing "." to income values — strip it
    df_test["income"] = df_test["income"].str.rstrip(".")

    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"  Combined: {len(df_train)} train + {len(df_test)} test = {len(df)} rows")

    # Strip leading/trailing whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # Summary
    print(f"  Label distribution: {df['income'].value_counts().to_dict()}")
    print(f"  Missing ('?'): {(df == '?').sum().sum()} cells")

    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # Write metadata JSON
    meta = {
        "description": (
            "Original UCI Adult Income dataset with string categorical values. "
            "No encoding, no normalisation. Includes native_country and income target."
        ),
        "source": "UCI ML Repository: https://archive.ics.uci.edu/dataset/2/adult",
        "citation": (
            "Becker, B., & Kohavi, R. (1996). Adult [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20"
        ),
        "train_url": UCI_TRAIN_URL,
        "test_url": UCI_TEST_URL,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "n_total": len(df),
        "n_features": 14,
        "n_with_target": 15,
        "features": FEATURE_METADATA,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved: {out_meta}")

    return out_csv


# ---------------------------------------------------------------------------
# Task B — ANN SHAP values
# ---------------------------------------------------------------------------

def download_ann(
    output_dir: Path,
    n_instances: int | None,
    shap_prefix: str,
) -> Path:
    """
    Compute SHAP values using OpenXAI's pre-trained ANN model for Adult Income.
    Saves to output_dir/adult_ann.csv using the same schema as adult.csv (LR).

    Returns path to saved CSV.
    """
    print(f"\n{'='*60}")
    print("Task B: Computing ANN SHAP values")
    print(f"{'='*60}")

    out_path = prepare_dataset(
        our_name="adult",
        ml_model="ann",
        split="test",
        n_instances=n_instances,
        output_dir=output_dir,
        shap_prefix=shap_prefix,
    )

    # Rename to adult_ann.csv to distinguish from LR file (adult.csv)
    ann_path = output_dir / "adult_ann.csv"
    if out_path != ann_path:
        import shutil
        shutil.copy(out_path, ann_path)
        print(f"  Copied to: {ann_path}")

    validate_output(ann_path, shap_prefix=shap_prefix)
    return ann_path


# ---------------------------------------------------------------------------
# Task C — OpenXAI faithfulness/stability metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    lr_csv: Path,
    ann_csv: Path,
    output_dir: Path,
) -> Path:
    """
    Compute OpenXAI evaluation metrics (PGI, PGU, RIS, RRS, ROS) for SHAP
    explanations on Adult Income using both LR and ANN models.

    Note: FA, RA, SA, SRA, PRA require ground-truth explanations only
    available for synthetic datasets. They are skipped here.

    Saves results to output_dir/adult_openxai_metrics.csv.

    Returns path to saved CSV.
    """
    print(f"\n{'='*60}")
    print("Task C: Computing OpenXAI evaluation metrics")
    print(f"{'='*60}")

    try:
        import torch
        from openxai import Evaluator, Explainer, LoadModel
        from openxai.dataloader import ReturnLoaders
    except ImportError as exc:
        raise ImportError(
            "OpenXAI is not installed. Install from source:\n"
            "  git clone https://github.com/AI4LIFE-GROUP/OpenXAI.git\n"
            "  cd OpenXAI && pip install -e ."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "adult_openxai_metrics.csv"

    # Load test data
    print("  Loading data via OpenXAI DataLoader...")
    loader_train, loader_test = ReturnLoaders(data_name="adult", download=True)

    X_list, y_list = [], []
    for batch_X, batch_y in loader_test:
        X_list.append(batch_X.numpy() if hasattr(batch_X, "numpy") else np.array(batch_X))
        y_list.append(batch_y.numpy() if hasattr(batch_y, "numpy") else np.array(batch_y))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    X_train_list = []
    for batch_X, _ in loader_train:
        X_train_list.append(
            batch_X.numpy() if hasattr(batch_X, "numpy") else np.array(batch_X)
        )
    X_train = np.concatenate(X_train_list, axis=0)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y.astype(int))
    X_train_tensor = torch.FloatTensor(X_train)

    print(f"  Test set: {X.shape[0]} instances, {X.shape[1]} features")

    results = []

    for ml_model, csv_path in [("lr", lr_csv), ("ann", ann_csv)]:
        if not csv_path.exists():
            print(f"  Skipping {ml_model.upper()}: {csv_path} not found")
            continue

        print(f"\n  --- Model: {ml_model.upper()} ---")

        # Load model
        model = LoadModel(data_name="adult", ml_model=ml_model, pretrained=True)

        # Load pre-computed SHAP values from our CSV
        df = pd.read_csv(csv_path)
        shap_cols = [c for c in df.columns if c.startswith("shap_")]
        shap_np = df[shap_cols].values

        # Use only the rows present in our CSV
        n = len(df)
        X_t = X_tensor[:n]
        y_t = y_tensor[:n]
        shap_t = torch.FloatTensor(shap_np)

        # Build explainer (used internally by Evaluator for perturbation-based metrics)
        explainer = Explainer(
            method="shap",
            model=model,
            param_dict={},
        )

        # ------------------------------------------------------------------
        # Compute metrics — try each individually; record errors gracefully
        # ------------------------------------------------------------------
        metrics_to_compute = {
            "PGI": "Prediction Gap on Important features — higher is better",
            "PGU": "Prediction Gap on Unimportant features — lower is better",
            "RIS": "Relative Input Stability — lower is better",
            "RRS": "Relative Representation Stability — lower is better",
            "ROS": "Relative Output Stability — lower is better",
        }

        for metric_name, metric_notes in metrics_to_compute.items():
            print(f"    Computing {metric_name}...", end=" ", flush=True)
            try:
                evaluator = Evaluator(
                    inputs=X_t,
                    labels=y_t,
                    model=model,
                    explainer=explainer,
                )
                value = evaluator.evaluate(metric=metric_name)
                # Evaluator may return a tensor or scalar
                if hasattr(value, "item"):
                    value = value.item()
                elif hasattr(value, "mean"):
                    value = float(value.mean())
                else:
                    value = float(value)
                print(f"{value:.6f}")
                results.append({
                    "model": ml_model,
                    "metric": metric_name,
                    "value": value,
                    "notes": metric_notes,
                    "n_instances": n,
                    "status": "ok",
                })
            except Exception as exc:
                msg = str(exc)
                print(f"ERROR — {msg}")
                results.append({
                    "model": ml_model,
                    "metric": metric_name,
                    "value": None,
                    "notes": metric_notes,
                    "n_instances": n,
                    "status": f"error: {msg}",
                })

        # Note: FA, RA, SA, SRA, PRA are skipped (require synthetic GT)
        for gt_metric in ["FA", "RA", "SA", "SRA", "PRA"]:
            results.append({
                "model": ml_model,
                "metric": gt_metric,
                "value": None,
                "notes": (
                    "Requires ground-truth explanations (synthetic datasets only). "
                    "Not applicable to Adult Income (real-world dataset)."
                ),
                "n_instances": n,
                "status": "skipped_no_ground_truth",
            })

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(out_path, index=False)
    print(f"\n  Metrics saved: {out_path}")
    print(metrics_df[["model", "metric", "value", "status"]].to_string(index=False))

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download/compute Adult Income data: raw UCI CSV, ANN SHAP values, "
            "and OpenXAI faithfulness metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=["raw", "ann", "metrics", "all"],
        default="all",
        help=(
            "Which task to run. "
            "'raw' = download original UCI CSV; "
            "'ann' = compute ANN SHAP values; "
            "'metrics' = compute OpenXAI evaluation metrics; "
            "'all' = run all three in sequence."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help=(
            "Cap number of instances for ANN SHAP computation (task 'ann'). "
            "None = use all test instances."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Output directory for raw UCI dataset (task 'raw').",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Output directory for processed CSV with SHAP values (task 'ann').",
    )
    parser.add_argument(
        "--metrics-dir",
        default="outputs/xai_metrics",
        help="Output directory for evaluation metrics CSV (task 'metrics').",
    )
    parser.add_argument(
        "--shap-prefix",
        default="shap_",
        help="Prefix for SHAP value columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir = PROJECT_ROOT / args.raw_dir
    processed_dir = PROJECT_ROOT / args.processed_dir
    metrics_dir = PROJECT_ROOT / args.metrics_dir

    tasks = (
        ["raw", "ann", "metrics"] if args.task == "all" else [args.task]
    )

    if "raw" in tasks:
        download_raw(raw_dir)

    if "ann" in tasks:
        download_ann(processed_dir, n_instances=args.n, shap_prefix=args.shap_prefix)

    if "metrics" in tasks:
        lr_csv  = processed_dir / "adult.csv"
        ann_csv = processed_dir / "adult_ann.csv"
        compute_metrics(lr_csv, ann_csv, metrics_dir)

    print(f"\n{'='*60}")
    print("All requested tasks complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
