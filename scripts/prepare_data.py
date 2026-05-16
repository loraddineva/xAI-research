"""
scripts/prepare_data.py
Download OpenXAI Adult Income dataset, compute per-instance SHAP values using the
OpenXAI pretrained model, and save processed CSV to data/processed/.

Each output CSV has:
  - One row per instance (test split by default)
  - Feature columns with their original (post-processing) names AND
    their original, unnormalised values (e.g. age in years).
  - Corresponding SHAP columns prefixed with 'shap_' (e.g. shap_age).
  - A 'label' column with the ground-truth target.
  - A 'pred_proba' column with the model's predicted probability of class 1.
  - A 'pred_label' column with the model's predicted class (argmax).

Dataset-specific post-processing:
  - Adult Income: OpenXAI returns 12 features (drops `education-num`
    and `native-country` from the raw 14-column UCI schema). We then
    drop `fnlwgt` (uninformative survey weight), leaving 11 features.

Prerequisites
-------------
OpenXAI is not on PyPI. Install it from source:
    git clone https://github.com/AI4LIFE-GROUP/OpenXAI.git
    cd OpenXAI && pip install -e .

Usage
-----
    python scripts/prepare_data.py
    python scripts/prepare_data.py --dataset adult
    python scripts/prepare_data.py --model lr
    python scripts/prepare_data.py --n 20
    python scripts/prepare_data.py --split train

Output
------
    data/processed/adult.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUR_NAME = "adult"
OPENXAI_NAME = "adult"
OUTPUT_FILENAME = "adult.csv"
RAW_CSV_DIR = Path("data/adult")


ADULT_OPENXAI_ORDER = [
    "age",
    "fnlwgt",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "sex_Male",
    "workclass_Private",
    "marital_status_Non_Married",
    "occupation_Other",
    "relationship_Non_Husband",
    "race_White",
    "native_country_US",
]

ADULT_RAW_CSV_NAMES = {
    "age": "age",
    "fnlwgt": "fnlwgt",
    "capital_gain": "capital-gain",
    "capital_loss": "capital-loss",
    "hours_per_week": "hours-per-week",
    "sex_Male": "sex_Male",
    "workclass_Private": "workclass_Private",
    "marital_status_Non_Married": "marital-status_Non-Married",
    "occupation_Other": "occupation_Other",
    "relationship_Non_Husband": "relationship_Non-Husband",
    "race_White": "race_White",
    "native_country_US": "native-country_US",
}

ADULT_DROP_COLUMNS = ["fnlwgt"]


def _resolve_feature_names(n_features: int) -> list[str]:
    if n_features != len(ADULT_OPENXAI_ORDER):
        raise ValueError(
            f"Adult expected {len(ADULT_OPENXAI_ORDER)} columns from OpenXAI, "
            f"got {n_features}. ADULT_OPENXAI_ORDER may be out of sync with OpenXAI."
        )
    return list(ADULT_OPENXAI_ORDER)


def _get_feature_names(data_name: str, n_features: int) -> list[str]:
    """Notebook helper — Adult only."""
    if data_name != OPENXAI_NAME:
        raise ValueError(f"Only '{OPENXAI_NAME}' supported; got {data_name!r}")
    return _resolve_feature_names(n_features)


def _load_raw_csv(split: str) -> pd.DataFrame:
    path = RAW_CSV_DIR / f"adult-{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw Adult CSV not found at {path.resolve()}. "
            "OpenXAI normally caches these on the first ReturnLoaders call."
        )
    return pd.read_csv(path)


def _denormalise_adult(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    shap_prefix: str = "shap_",
) -> pd.DataFrame:
    if len(df) != len(raw_df):
        raise ValueError(
            f"Adult row count mismatch: processed has {len(df)} rows, "
            f"raw CSV has {len(raw_df)}."
        )

    out = df.copy()
    for col in ADULT_OPENXAI_ORDER:
        raw_col = ADULT_RAW_CSV_NAMES[col]
        if raw_col not in raw_df.columns:
            raise KeyError(
                f"Expected column '{raw_col}' in raw Adult CSV (maps to '{col}')."
            )
        out[col] = raw_df[raw_col].values
    return out


def _drop_adult_uninformative(df: pd.DataFrame, shap_prefix: str = "shap_") -> pd.DataFrame:
    to_drop = []
    for col in ADULT_DROP_COLUMNS:
        if col in df.columns:
            to_drop.append(col)
        shap_col = f"{shap_prefix}{col}"
        if shap_col in df.columns:
            to_drop.append(shap_col)
    return df.drop(columns=to_drop)


def prepare_dataset(
    our_name: str = OUR_NAME,
    ml_model: str = "lr",
    split: str = "test",
    n_instances: int | None = None,
    output_dir: Path = Path("data/processed"),
    shap_prefix: str = "shap_",
) -> Path:
    """
    Download, model, predict, explain, and save the Adult Income dataset.

    Returns:
        Path to the saved CSV.
    """
    if our_name != OUR_NAME:
        raise ValueError(f"Only '{OUR_NAME}' is supported; got {our_name!r}.")

    try:
        import torch
        from openxai import Explainer, LoadModel
        from openxai.dataloader import ReturnLoaders
    except ImportError as exc:
        raise ImportError(
            "OpenXAI is not installed.\n"
            "  git clone https://github.com/AI4LIFE-GROUP/OpenXAI.git\n"
            "  cd OpenXAI && pip install -e ."
        ) from exc

    print(f"\n{'='*60}")
    print(f"Preparing dataset: {our_name!r}  (OpenXAI name: {OPENXAI_NAME!r})")
    print(f"  Model : {ml_model}")
    print(f"  Split : {split}")
    print(f"{'='*60}")

    print("Downloading / loading data...")
    trainloader, testloader = ReturnLoaders(data_name=OPENXAI_NAME, download=True)
    loader = testloader if split == "test" else trainloader

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

    raw_df = _load_raw_csv(split)
    if n_instances is not None:
        raw_df = raw_df.iloc[:n_instances].reset_index(drop=True)
    else:
        raw_df = raw_df.reset_index(drop=True)
    print(f"  Loaded raw adult-{split}.csv with {len(raw_df)} rows")

    print(f"Loading pretrained {ml_model.upper()} model...")
    model = LoadModel(data_name=OPENXAI_NAME, ml_model=ml_model, pretrained=True)

    print("Computing model predictions...")
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y.astype(int))

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred_proba = probs[:, 1]
    pred_label = probs.argmax(axis=1)
    print(f"  Predicted class balance: {dict(zip(*np.unique(pred_label, return_counts=True)))}")

    print("Computing SHAP values (this may take a few minutes)...")
    explainer = Explainer(method="shap", model=model, param_dict={})
    try:
        shap_values = explainer.get_explanations(X_tensor, y_tensor)
    except TypeError:
        shap_values = explainer.get_explanations(X_tensor)

    if hasattr(shap_values, "detach"):
        shap_np = shap_values.detach().cpu().numpy()
    elif hasattr(shap_values, "numpy"):
        shap_np = shap_values.numpy()
    else:
        shap_np = np.array(shap_values)

    print(f"  SHAP values computed: shape {shap_np.shape}")

    feature_names = _resolve_feature_names(X.shape[1])
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y.astype(int)
    df["pred_proba"] = pred_proba
    df["pred_label"] = pred_label.astype(int)
    for i, feat in enumerate(feature_names):
        df[f"{shap_prefix}{feat}"] = shap_np[:, i]

    df = _denormalise_adult(df, raw_df, shap_prefix=shap_prefix)
    df = _drop_adult_uninformative(df, shap_prefix=shap_prefix)
    print("  Replaced normalised Adult values with raw CSV values")
    print(f"  Dropped uninformative columns: {ADULT_DROP_COLUMNS}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / OUTPUT_FILENAME
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} rows to: {out_path}")
    print(f"  Columns: {list(df.columns)}")
    return out_path


def validate_output(csv_path: Path, shap_prefix: str = "shap_") -> None:
    """Print a quick validation summary for a processed CSV."""
    df = pd.read_csv(csv_path)
    shap_cols = [c for c in df.columns if c.startswith(shap_prefix)]
    meta_cols = {"label", "pred_proba", "pred_label"}
    feat_cols = [
        c for c in df.columns
        if not c.startswith(shap_prefix) and c not in meta_cols
    ]

    print(f"\nValidation: {csv_path.name}")
    print(f"  Rows          : {len(df)}")
    print(f"  Features      : {len(feat_cols)}")
    print(f"  SHAP columns  : {len(shap_cols)}")
    if "label" in df.columns:
        print(f"  Label dist    : {df['label'].value_counts().to_dict()}")
    if "pred_label" in df.columns:
        print(f"  Pred  dist    : {df['pred_label'].value_counts().to_dict()}")
    if "pred_proba" in df.columns:
        print(
            f"  Pred  proba   : "
            f"min={df['pred_proba'].min():.3f}  "
            f"mean={df['pred_proba'].mean():.3f}  "
            f"max={df['pred_proba'].max():.3f}"
        )
    print(f"  SHAP value range: [{df[shap_cols].values.min():.4f}, {df[shap_cols].values.max():.4f}]")
    print(f"  Null values   : {df.isnull().sum().sum()}")

    feat_set = set(feat_cols)
    shap_feat_set = {c[len(shap_prefix):] for c in shap_cols}
    if feat_set != shap_feat_set:
        print("  WARNING: feature / SHAP column mismatch!")
        print(f"    Extra in features : {feat_set - shap_feat_set}")
        print(f"    Extra in SHAP     : {shap_feat_set - feat_set}")
    else:
        print("  Feature/SHAP columns match.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Adult Income with SHAP values for the narrative pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["adult"],
        default="adult",
        help="Dataset to prepare (Adult Income only).",
    )
    parser.add_argument("--model", choices=["lr", "ann"], default="lr")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--shap-prefix", default="shap_")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = prepare_dataset(
        our_name=args.dataset,
        ml_model=args.model,
        split=args.split,
        n_instances=args.n,
        output_dir=Path(args.output_dir),
        shap_prefix=args.shap_prefix,
    )
    if args.validate:
        validate_output(path, shap_prefix=args.shap_prefix)

    print(
        f"\nAll done. Processed file:\n  {path}\n\n"
        "Next step:\n"
        "  python scripts/run_generation.py --dry-run --dataset adult --n 5"
    )


if __name__ == "__main__":
    main()
