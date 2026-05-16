"""
archive/german_credit/scripts/prepare_german_credit.py
Prepare German Credit with per-instance SHAP values (archived dataset).

Aggregates OpenXAI's 60 one-hot columns into 20 UCI features, substitutes
raw numeric values from the cached CSVs, and writes
archive/german_credit/data/processed/german_credit.csv.

Usage
-----
    python archive/german_credit/scripts/prepare_german_credit.py
    python archive/german_credit/scripts/prepare_german_credit.py --n 20 --validate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ARCHIVE_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUR_NAME = "german_credit"
OPENXAI_NAME = "german"
OUTPUT_FILENAME = "german_credit.csv"
RAW_CSV_DIR = ARCHIVE_ROOT / "data" / "german"
OUTPUT_DIR = ARCHIVE_ROOT / "data" / "processed"

GERMAN_RAW_COLUMNS = [
    "duration", "amount", "installment-rate", "present-residence", "age",
    "number-credits", "people-liable", "foreign-worker",
    "status_1", "status_2", "status_3", "status_4",
    "credit-history_0", "credit-history_1", "credit-history_2",
    "credit-history_3", "credit-history_4",
    "purpose_0", "purpose_1", "purpose_2", "purpose_3", "purpose_4",
    "purpose_5", "purpose_6", "purpose_7", "purpose_9", "purpose_10",
    "savings_1", "savings_2", "savings_3", "savings_4", "savings_5",
    "employment-duration_1", "employment-duration_2", "employment-duration_3",
    "employment-duration_4", "employment-duration_5",
    "personal-status-sex_1", "personal-status-sex_2",
    "personal-status-sex_3", "personal-status-sex_5",
    "other-debtors_1", "other-debtors_2", "other-debtors_3",
    "property_1", "property_2", "property_3", "property_4",
    "other-installment-plans_1", "other-installment-plans_2",
    "other-installment-plans_3",
    "housing_1", "housing_2", "housing_3",
    "job_1", "job_2", "job_3", "job_4",
    "telephone_1", "telephone_2",
]

GERMAN_NUMERIC_RENAME = {
    "duration": "duration",
    "amount": "credit_amount",
    "installment-rate": "installment_rate",
    "present-residence": "present_residence",
    "age": "age",
    "number-credits": "number_credits",
    "people-liable": "people_liable",
    "foreign-worker": "foreign_worker",
}

GERMAN_CATEGORICAL_GROUPS = [
    ("status_", "checking_status"),
    ("credit-history_", "credit_history"),
    ("purpose_", "purpose"),
    ("savings_", "savings_status"),
    ("employment-duration_", "employment"),
    ("personal-status-sex_", "personal_status"),
    ("other-debtors_", "other_parties"),
    ("property_", "property_magnitude"),
    ("other-installment-plans_", "other_payment_plans"),
    ("housing_", "housing"),
    ("job_", "job"),
    ("telephone_", "own_telephone"),
]


def _resolve_feature_names(n_features: int) -> list[str]:
    if n_features != len(GERMAN_RAW_COLUMNS):
        raise ValueError(
            f"German Credit expected {len(GERMAN_RAW_COLUMNS)} columns from OpenXAI, "
            f"got {n_features}."
        )
    return list(GERMAN_RAW_COLUMNS)


def _load_raw_csv(split: str) -> pd.DataFrame:
    path = RAW_CSV_DIR / f"german-{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw German CSV not found at {path.resolve()}. "
            "Run ReturnLoaders once so OpenXAI caches the file."
        )
    return pd.read_csv(path)


def _aggregate_german_one_hot(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    shap_prefix: str = "shap_",
) -> pd.DataFrame:
    if len(df) != len(raw_df):
        raise ValueError(
            f"German row count mismatch: processed has {len(df)} rows, "
            f"raw CSV has {len(raw_df)}."
        )

    out = pd.DataFrame(index=df.index)

    for raw_name, parent in GERMAN_NUMERIC_RENAME.items():
        if raw_name not in raw_df.columns:
            raise KeyError(f"Expected column '{raw_name}' in raw German CSV.")
        out[parent] = raw_df[raw_name].values
        out[f"{shap_prefix}{parent}"] = df[f"{shap_prefix}{raw_name}"].values

    for prefix, parent in GERMAN_CATEGORICAL_GROUPS:
        members = [c for c in GERMAN_RAW_COLUMNS if c.startswith(prefix)]
        levels = [int(c[len(prefix):]) for c in members]
        values_block = df[members].values
        active_idx = np.argmax(values_block, axis=1)
        out[parent] = [levels[i] for i in active_idx]
        shap_cols = [f"{shap_prefix}{m}" for m in members]
        out[f"{shap_prefix}{parent}"] = df[shap_cols].sum(axis=1).values

    for col in ("label", "pred_proba", "pred_label"):
        if col in df.columns:
            out[col] = df[col].values

    return out


def prepare_german_credit(
    ml_model: str = "lr",
    split: str = "test",
    n_instances: int | None = None,
    output_dir: Path = OUTPUT_DIR,
    shap_prefix: str = "shap_",
) -> Path:
    try:
        import torch
        from openxai import Explainer, LoadModel
        from openxai.dataloader import ReturnLoaders
    except ImportError as exc:
        raise ImportError(
            "OpenXAI is not installed. Install from:\n"
            "  https://github.com/AI4LIFE-GROUP/OpenXAI"
        ) from exc

    print(f"\nPreparing archived dataset: {OUR_NAME!r}  (OpenXAI: {OPENXAI_NAME!r})")

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

    raw_df = _load_raw_csv(split)
    if n_instances is not None:
        raw_df = raw_df.iloc[:n_instances].reset_index(drop=True)
    else:
        raw_df = raw_df.reset_index(drop=True)

    model = LoadModel(data_name=OPENXAI_NAME, ml_model=ml_model, pretrained=True)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y.astype(int))

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred_proba = probs[:, 1]
    pred_label = probs.argmax(axis=1)

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

    feature_names = _resolve_feature_names(X.shape[1])
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y.astype(int)
    df["pred_proba"] = pred_proba
    df["pred_label"] = pred_label.astype(int)
    for i, feat in enumerate(feature_names):
        df[f"{shap_prefix}{feat}"] = shap_np[:, i]

    df = _aggregate_german_one_hot(df, raw_df, shap_prefix=shap_prefix)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / OUTPUT_FILENAME
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    return out_path


def validate_output(csv_path: Path, shap_prefix: str = "shap_") -> None:
    from scripts.prepare_data import validate_output as _validate

    _validate(csv_path, shap_prefix=shap_prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare archived German Credit CSV.")
    parser.add_argument("--model", choices=["lr", "ann"], default="lr")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    path = prepare_german_credit(
        ml_model=args.model,
        split=args.split,
        n_instances=args.n,
        output_dir=Path(args.output_dir),
    )
    if args.validate:
        validate_output(path)


if __name__ == "__main__":
    main()
