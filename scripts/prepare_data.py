"""
scripts/prepare_data.py
Download OpenXAI datasets, compute per-instance SHAP values using the
OpenXAI pretrained model, and save processed CSVs to data/processed/.

Each output CSV has:
  - One row per instance (test split by default)
  - Feature columns with their original names (e.g. age, duration)
  - Corresponding SHAP columns prefixed with 'shap_' (e.g. shap_age)
  - A 'label' column with the ground-truth target

Prerequisites
-------------
OpenXAI is not on PyPI. Install it from source:
    git clone https://github.com/AI4LIFE-GROUP/OpenXAI.git
    cd OpenXAI && pip install -e .

Usage
-----
    # Prepare both datasets (default)
    python scripts/prepare_data.py

    # One dataset only
    python scripts/prepare_data.py --dataset adult
    python scripts/prepare_data.py --dataset german_credit

    # Choose ML model (lr = logistic regression, ann = neural net)
    python scripts/prepare_data.py --model lr

    # Cap instances (useful for quick smoke-tests)
    python scripts/prepare_data.py --n 20

    # Use training split instead of test split
    python scripts/prepare_data.py --split train

Output
------
    data/processed/adult.csv
    data/processed/german_credit.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# OpenXAI dataset name mapping
# Our config name  →  OpenXAI data_name
# ---------------------------------------------------------------------------

DATASET_MAP = {
    "adult":          "adult",
    "german_credit":  "german",
}

# Default output filenames (must match config/default.yaml paths)
OUTPUT_FILENAMES = {
    "adult":         "adult.csv",
    "german_credit": "german_credit.csv",
}


# ---------------------------------------------------------------------------
# Feature name extraction
# ---------------------------------------------------------------------------

def _get_feature_names(data_name: str, n_features: int) -> list[str]:
    """
    Return a list of feature names for the dataset.
    Tries to extract metadata from OpenXAI's dataloader; falls back to
    generic names (feature_0, feature_1, ...) if metadata is unavailable.
    """
    try:
        from openxai.dataloader import ReturnTrainTestX
        result = ReturnTrainTestX(
            data_name=data_name,
            download=True,
            return_feature_metadata=True,
        )
        # ReturnTrainTestX returns (X_train, X_test, y_train, y_test, metadata)
        # when return_feature_metadata=True
        if isinstance(result, tuple) and len(result) == 5:
            metadata = result[4]
            if hasattr(metadata, "columns"):
                cols = [c for c in metadata.columns if c != "label"]
                if len(cols) == n_features:
                    return cols
            if isinstance(metadata, (list, tuple)) and len(metadata) == n_features:
                return list(metadata)
    except Exception:
        pass

    # Fallback: use known feature names for the two target datasets
    known_features = {
        "adult": [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country",
        ],
        "german": [
            "checking_status", "duration", "credit_history", "purpose",
            "credit_amount", "savings_status", "employment",
            "installment_commitment", "personal_status", "other_parties",
            "residence_since", "property_magnitude", "age",
            "other_payment_plans", "housing", "existing_credits",
            "job", "num_dependents", "own_telephone", "foreign_worker",
        ],
    }
    if data_name in known_features:
        names = known_features[data_name]
        if len(names) == n_features:
            return names
        # Truncate or pad to match actual feature count
        if len(names) > n_features:
            return names[:n_features]
        names += [f"feature_{i}" for i in range(len(names), n_features)]
        return names

    # Generic fallback
    return [f"feature_{i}" for i in range(n_features)]


# ---------------------------------------------------------------------------
# Core preparation function
# ---------------------------------------------------------------------------

def prepare_dataset(
    our_name: str,
    ml_model: str = "lr",
    split: str = "test",
    n_instances: int | None = None,
    output_dir: Path = Path("data/processed"),
    shap_prefix: str = "shap_",
) -> Path:
    """
    Download, model, explain, and save one dataset.

    Args:
        our_name:     Dataset key used in our config ('adult' or 'german_credit').
        ml_model:     OpenXAI model type ('lr' or 'ann').
        split:        'test' or 'train' — which split to compute SHAP values for.
        n_instances:  If set, cap the number of rows. None = use all.
        output_dir:   Where to save the processed CSV.
        shap_prefix:  Column prefix for SHAP values.

    Returns:
        Path to the saved CSV.
    """
    try:
        import torch
        from openxai import Explainer, LoadModel
        from openxai.dataloader import ReturnLoaders, ReturnTrainTestX
    except ImportError as exc:
        raise ImportError(
            "OpenXAI is not installed.\n"
            "Install it from source:\n"
            "  git clone https://github.com/AI4LIFE-GROUP/OpenXAI.git\n"
            "  cd OpenXAI && pip install -e ."
        ) from exc

    openxai_name = DATASET_MAP[our_name]
    print(f"\n{'='*60}")
    print(f"Preparing dataset: {our_name!r}  (OpenXAI name: {openxai_name!r})")
    print(f"  Model : {ml_model}")
    print(f"  Split : {split}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Downloading / loading data...")
    trainloader, testloader = ReturnLoaders(
        data_name=openxai_name,
        download=True,
    )

    loader = testloader if split == "test" else trainloader

    # Collect all batches into numpy arrays
    X_list, y_list = [], []
    for batch_X, batch_y in loader:
        X_list.append(batch_X.numpy() if hasattr(batch_X, "numpy") else np.array(batch_X))
        y_list.append(batch_y.numpy() if hasattr(batch_y, "numpy") else np.array(batch_y))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if n_instances is not None:
        X = X[:n_instances]
        y = y[:n_instances]

    print(f"  Loaded {X.shape[0]} instances, {X.shape[1]} features")

    # ------------------------------------------------------------------
    # 2. Get training data (needed as SHAP background)
    # ------------------------------------------------------------------
    X_train_list = []
    for batch_X, _ in trainloader:
        X_train_list.append(
            batch_X.numpy() if hasattr(batch_X, "numpy") else np.array(batch_X)
        )
    X_train = np.concatenate(X_train_list, axis=0)

    # ------------------------------------------------------------------
    # 3. Load pretrained model
    # ------------------------------------------------------------------
    print(f"Loading pretrained {ml_model.upper()} model...")
    model = LoadModel(data_name=openxai_name, ml_model=ml_model, pretrained=True)

    # ------------------------------------------------------------------
    # 4. Compute SHAP values
    # ------------------------------------------------------------------
    print("Computing SHAP values (this may take a few minutes)...")

    X_tensor = torch.FloatTensor(X)
    X_train_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y.astype(int))

    explainer = Explainer(
        method="shap",
        model=model,
        param_dict={},
    )

    # get_explanations returns a tensor of shape (n_instances, n_features)
    try:
        shap_values = explainer.get_explanations(X_tensor, y_tensor)
    except TypeError:
        # Some versions use positional-only args
        shap_values = explainer.get_explanations(X_tensor)

    if hasattr(shap_values, "numpy"):
        shap_np = shap_values.detach().numpy()
    elif hasattr(shap_values, "detach"):
        shap_np = shap_values.detach().cpu().numpy()
    else:
        shap_np = np.array(shap_values)

    print(f"  SHAP values computed: shape {shap_np.shape}")

    # ------------------------------------------------------------------
    # 5. Build DataFrame
    # ------------------------------------------------------------------
    n_features = X.shape[1]
    feature_names = _get_feature_names(openxai_name, n_features)
    print(f"  Feature names: {feature_names}")

    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y.astype(int)

    for i, feat in enumerate(feature_names):
        df[f"{shap_prefix}{feat}"] = shap_np[:, i]

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / OUTPUT_FILENAMES[our_name]
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} rows to: {out_path}")
    print(f"  Columns: {list(df.columns)}")
    return out_path


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_output(csv_path: Path, shap_prefix: str = "shap_") -> None:
    """Print a quick validation summary for a processed CSV."""
    df = pd.read_csv(csv_path)
    shap_cols = [c for c in df.columns if c.startswith(shap_prefix)]
    feat_cols = [c for c in df.columns if not c.startswith(shap_prefix) and c != "label"]

    print(f"\nValidation: {csv_path.name}")
    print(f"  Rows          : {len(df)}")
    print(f"  Features      : {len(feat_cols)}")
    print(f"  SHAP columns  : {len(shap_cols)}")
    print(f"  Label dist    : {df['label'].value_counts().to_dict()}")
    print(f"  SHAP value range: [{df[shap_cols].values.min():.4f}, {df[shap_cols].values.max():.4f}]")
    print(f"  Null values   : {df.isnull().sum().sum()}")

    # Check feature / SHAP column parity
    feat_set = set(feat_cols)
    shap_feat_set = {c[len(shap_prefix):] for c in shap_cols}
    if feat_set != shap_feat_set:
        print(f"  WARNING: feature / SHAP column mismatch!")
        print(f"    Extra in features : {feat_set - shap_feat_set}")
        print(f"    Extra in SHAP     : {shap_feat_set - feat_set}")
    else:
        print(f"  Feature/SHAP columns match.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OpenXAI datasets with SHAP values for the hallucination pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_MAP.keys()) + ["all"],
        default="all",
        help="Which dataset to prepare. 'all' prepares both.",
    )
    parser.add_argument(
        "--model",
        choices=["lr", "ann"],
        default="lr",
        help="Pretrained model to use for SHAP computation.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "train"],
        default="test",
        help="Data split to compute SHAP values for.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Cap the number of instances (default: use all available).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to save processed CSVs.",
    )
    parser.add_argument(
        "--shap-prefix",
        default="shap_",
        help="Prefix for SHAP value columns (must match config/default.yaml shap_col_prefix).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Print a validation summary after saving each dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    datasets = list(DATASET_MAP.keys()) if args.dataset == "all" else [args.dataset]

    saved = []
    for name in datasets:
        path = prepare_dataset(
            our_name=name,
            ml_model=args.model,
            split=args.split,
            n_instances=args.n,
            output_dir=output_dir,
            shap_prefix=args.shap_prefix,
        )
        saved.append(path)
        if args.validate:
            validate_output(path, shap_prefix=args.shap_prefix)

    print(f"\nAll done. Processed files:")
    for p in saved:
        print(f"  {p}")
    print(
        "\nNext step:\n"
        "  python scripts/run_generation.py --dry-run --dataset adult --n 5"
    )


if __name__ == "__main__":
    main()
