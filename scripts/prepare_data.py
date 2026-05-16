"""
scripts/prepare_data.py
Download OpenXAI datasets, compute per-instance SHAP values using the
OpenXAI pretrained model, and save processed CSVs to data/processed/.

Each output CSV has:
  - One row per instance (test split by default)
  - Feature columns with their original (post-processing) names AND
    their original, unnormalised values (e.g. age in years, credit
    amount in DM, categorical level codes as plain integers). The
    OpenXAI dataloader normalises everything to [0, 1] for the model;
    we throw that representation away after SHAP computation and write
    the raw values back so the prompt and the human reader both see
    a meaningful number.
  - Corresponding SHAP columns prefixed with 'shap_' (e.g. shap_age).
    SHAP values are still computed against the normalised model input
    because that is what the trained model expects; per the linearity
    of SHAP, the *contribution to the prediction* is unchanged whether
    we display the feature itself in normalised or raw scale.
  - A 'label' column with the ground-truth target.
  - A 'pred_proba' column with the model's predicted probability of class 1.
  - A 'pred_label' column with the model's predicted class (argmax).

Dataset-specific post-processing:
  - Adult Income: OpenXAI returns 12 features (drops `education-num`
    and `native-country` from the raw 14-column UCI schema). The
    column order returned by ``ReturnLoaders`` is

        age, fnlwgt, capital-gain, capital-loss, hours-per-week,
        sex_Male, workclass_Private, marital-status_Non-Married,
        occupation_Other, relationship_Non-Husband, race_White,
        native-country_US

    We then drop `fnlwgt` (uninformative survey weight), leaving
    11 features. Names are normalised to snake_case for safe column
    access (``marital_status_Non_Married``).
  - German Credit: aggregates the 52 one-hot encoded columns
    (status_*, credit-history_*, purpose_*, ...) back into 12 parent
    categorical features by summing their SHAP contributions and
    recording the active dummy's index as the feature value. Combined
    with the 8 numeric columns this yields 20 features (matching the
    original UCI German Credit schema).

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# OpenXAI dataset name mapping
# Our config name  →  OpenXAI data_name
# ---------------------------------------------------------------------------

DATASET_MAP = {
    "adult":          "adult",
    "german_credit":  "german",
}

OUTPUT_FILENAMES = {
    "adult":         "adult.csv",
    "german_credit": "german_credit.csv",
}

# Where the unnormalised raw test/train CSVs live after OpenXAI's first
# download. We read these to recover the original feature scale that the
# normalised tensors from ``ReturnLoaders`` no longer carry.
RAW_CSV_DIRS = {
    "adult":          Path("data/adult"),
    "german_credit":  Path("data/german"),
}


# ---------------------------------------------------------------------------
# Adult Income — feature schema as returned by OpenXAI's dataloader.
#
# OpenXAI drops `education-num` and `native_country` from the original 14
# UCI columns and returns 12 features in the order below. We then drop
# `fnlwgt` (uninformative survey weight) before saving.
#
# `ADULT_OPENXAI_ORDER` mirrors the column order in
# ``data/adult/adult-test.csv`` minus `education-num`, which is what
# ReturnLoaders actually feeds into the model.
# ---------------------------------------------------------------------------

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

# Map the OpenXAI / our internal name -> column name in adult-{train,test}.csv.
# The raw CSVs use hyphenated names; pandas needs the original strings.
ADULT_RAW_CSV_NAMES = {
    "age":                          "age",
    "fnlwgt":                       "fnlwgt",
    "capital_gain":                 "capital-gain",
    "capital_loss":                 "capital-loss",
    "hours_per_week":               "hours-per-week",
    "sex_Male":                     "sex_Male",
    "workclass_Private":            "workclass_Private",
    "marital_status_Non_Married":   "marital-status_Non-Married",
    "occupation_Other":             "occupation_Other",
    "relationship_Non_Husband":     "relationship_Non-Husband",
    "race_White":                   "race_White",
    "native_country_US":            "native-country_US",
}

ADULT_RAW_LABEL_COL = "income"
ADULT_DROP_COLUMNS = ["fnlwgt"]


# ---------------------------------------------------------------------------
# German Credit — one-hot aggregation schema
#
# OpenXAI's processed German Credit has 60 columns: 8 numeric features +
# 52 one-hot dummies covering 12 categorical variables. The dummy column
# names follow the pattern `<group>_<level>`, e.g. status_1, status_2,
# credit-history_0, ..., telephone_2.
# ---------------------------------------------------------------------------

GERMAN_RAW_COLUMNS = [
    # Numeric (8)
    "duration", "amount", "installment-rate", "present-residence", "age",
    "number-credits", "people-liable", "foreign-worker",
    # status / checking_status (4 levels)
    "status_1", "status_2", "status_3", "status_4",
    # credit-history (5 levels)
    "credit-history_0", "credit-history_1", "credit-history_2",
    "credit-history_3", "credit-history_4",
    # purpose (10 levels — note non-contiguous: 0-7, 9, 10)
    "purpose_0", "purpose_1", "purpose_2", "purpose_3", "purpose_4",
    "purpose_5", "purpose_6", "purpose_7", "purpose_9", "purpose_10",
    # savings (5 levels)
    "savings_1", "savings_2", "savings_3", "savings_4", "savings_5",
    # employment-duration (5 levels)
    "employment-duration_1", "employment-duration_2", "employment-duration_3",
    "employment-duration_4", "employment-duration_5",
    # personal-status-sex (4 levels — non-contiguous: 1, 2, 3, 5)
    "personal-status-sex_1", "personal-status-sex_2",
    "personal-status-sex_3", "personal-status-sex_5",
    # other-debtors (3 levels)
    "other-debtors_1", "other-debtors_2", "other-debtors_3",
    # property (4 levels)
    "property_1", "property_2", "property_3", "property_4",
    # other-installment-plans (3 levels)
    "other-installment-plans_1", "other-installment-plans_2",
    "other-installment-plans_3",
    # housing (3 levels)
    "housing_1", "housing_2", "housing_3",
    # job (4 levels)
    "job_1", "job_2", "job_3", "job_4",
    # telephone (2 levels)
    "telephone_1", "telephone_2",
]

GERMAN_NUMERIC_RENAME = {
    "duration":          "duration",
    "amount":            "credit_amount",
    "installment-rate":  "installment_rate",
    "present-residence": "present_residence",
    "age":               "age",
    "number-credits":    "number_credits",
    "people-liable":     "people_liable",
    "foreign-worker":    "foreign_worker",
}

# (raw_prefix, parent_name) pairs — order is the order they appear in the
# output CSV's feature columns.
GERMAN_CATEGORICAL_GROUPS = [
    ("status_",                   "checking_status"),
    ("credit-history_",           "credit_history"),
    ("purpose_",                  "purpose"),
    ("savings_",                  "savings_status"),
    ("employment-duration_",      "employment"),
    ("personal-status-sex_",      "personal_status"),
    ("other-debtors_",            "other_parties"),
    ("property_",                 "property_magnitude"),
    ("other-installment-plans_",  "other_payment_plans"),
    ("housing_",                  "housing"),
    ("job_",                      "job"),
    ("telephone_",                "own_telephone"),
]

GERMAN_RAW_LABEL_COL = "credit-risk"


# ---------------------------------------------------------------------------
# Feature name resolution
# ---------------------------------------------------------------------------

def _resolve_feature_names(data_name: str, n_features: int) -> list[str]:
    """
    Return the list of column names produced by the OpenXAI dataloader.

    For German, *n_features* must be 60 (OpenXAI's processed shape) and we
    return GERMAN_RAW_COLUMNS in the canonical order.

    For Adult, *n_features* must be 12 (OpenXAI drops education-num) and
    we return ADULT_OPENXAI_ORDER.

    Raises ValueError if the shape does not match the expected schema —
    this catches cases where OpenXAI changes its preprocessing without
    silently producing meaningless `feature_X` placeholders.
    """
    if data_name == "german":
        if n_features != len(GERMAN_RAW_COLUMNS):
            raise ValueError(
                f"German Credit expected {len(GERMAN_RAW_COLUMNS)} columns "
                f"from OpenXAI, got {n_features}. The hardcoded schema in "
                f"GERMAN_RAW_COLUMNS is out of sync with OpenXAI."
            )
        return list(GERMAN_RAW_COLUMNS)

    if data_name == "adult":
        if n_features != len(ADULT_OPENXAI_ORDER):
            raise ValueError(
                f"Adult expected {len(ADULT_OPENXAI_ORDER)} columns from "
                f"OpenXAI, got {n_features}. The hardcoded schema in "
                f"ADULT_OPENXAI_ORDER is out of sync with OpenXAI."
            )
        return list(ADULT_OPENXAI_ORDER)

    return [f"feature_{i}" for i in range(n_features)]


# ---------------------------------------------------------------------------
# Raw CSV loading — used to substitute unnormalised values back onto rows
# whose SHAP values were computed from the OpenXAI-normalised tensor.
# ---------------------------------------------------------------------------

def _load_raw_csv(our_name: str, split: str) -> pd.DataFrame:
    """
    Load the raw, unnormalised CSV for a dataset / split.

    OpenXAI's ``ReturnLoaders`` reads these same CSVs from disk in row
    order without shuffling, so the i-th row of the returned tensor
    matches the i-th data row of the CSV — which is what makes the
    raw-value substitution downstream safe.
    """
    raw_dir = RAW_CSV_DIRS[our_name]
    if our_name == "adult":
        fname = f"adult-{split}.csv"
    else:
        fname = f"german-{split}.csv"

    path = raw_dir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Raw {our_name} CSV not found at {path.resolve()}. "
            "OpenXAI normally caches these here on the first ReturnLoaders call."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Adult — replace normalised values with the raw test-CSV values.
# ---------------------------------------------------------------------------

def _denormalise_adult(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    shap_prefix: str = "shap_",
) -> pd.DataFrame:
    """
    Overwrite each Adult feature column with its original (unnormalised)
    value from *raw_df*.

    The SHAP columns are left untouched. We only ever touch
    ADULT_OPENXAI_ORDER columns + the meta columns (label / pred_proba /
    pred_label).
    """
    if len(df) != len(raw_df):
        raise ValueError(
            f"Adult row count mismatch: processed has {len(df)} rows, "
            f"raw CSV has {len(raw_df)}. Row alignment is unsafe."
        )

    out = df.copy()
    for our_name in ADULT_OPENXAI_ORDER:
        raw_col = ADULT_RAW_CSV_NAMES[our_name]
        if raw_col not in raw_df.columns:
            raise KeyError(
                f"Expected column '{raw_col}' in raw Adult CSV "
                f"(maps to '{our_name}'). Got: {list(raw_df.columns)}"
            )
        out[our_name] = raw_df[raw_col].values
    return out


def _drop_adult_uninformative(df: pd.DataFrame, shap_prefix: str = "shap_") -> pd.DataFrame:
    """Drop columns named in ADULT_DROP_COLUMNS plus their SHAP twins."""
    to_drop = []
    for col in ADULT_DROP_COLUMNS:
        if col in df.columns:
            to_drop.append(col)
        shap_col = f"{shap_prefix}{col}"
        if shap_col in df.columns:
            to_drop.append(shap_col)
    return df.drop(columns=to_drop)


# ---------------------------------------------------------------------------
# German — aggregate one-hots and substitute raw numeric values.
# ---------------------------------------------------------------------------

def _aggregate_german_one_hot(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    shap_prefix: str = "shap_",
) -> pd.DataFrame:
    """
    Collapse the 60-column raw German Credit DataFrame into 20 aggregated
    feature columns + matching SHAP columns + label/pred columns.

    Numeric columns are renamed via GERMAN_NUMERIC_RENAME and their values
    are taken from the *raw* (unnormalised) CSV. SHAP values pass through
    unchanged.

    For each categorical group, the parent feature value is the level of
    the active dummy (the dummy with value > 0.5 in this row), and the
    parent SHAP value is the sum of all dummy SHAPs in that group.
    Summing is exact for one-hot inputs because only one dummy is ever
    active per row, so the inactive dummies contribute additively-zero
    expected SHAP under the standard SHAP linearity property.
    """
    if len(df) != len(raw_df):
        raise ValueError(
            f"German row count mismatch: processed has {len(df)} rows, "
            f"raw CSV has {len(raw_df)}. Row alignment is unsafe."
        )

    out = pd.DataFrame(index=df.index)

    # 1. Numeric features — rename and pull raw values from the CSV
    for raw_name, parent in GERMAN_NUMERIC_RENAME.items():
        if raw_name not in raw_df.columns:
            raise KeyError(
                f"Expected column '{raw_name}' in raw German CSV "
                f"(maps to '{parent}'). Got: {list(raw_df.columns)[:10]}..."
            )
        out[parent] = raw_df[raw_name].values
        out[f"{shap_prefix}{parent}"] = df[f"{shap_prefix}{raw_name}"].values

    # 2. Categorical features — find active level + sum SHAPs
    for prefix, parent in GERMAN_CATEGORICAL_GROUPS:
        members = [c for c in GERMAN_RAW_COLUMNS if c.startswith(prefix)]
        levels = [int(c[len(prefix):]) for c in members]

        # Active level per row: argmax across this group's dummies.
        # Using argmax (not == 1.0) is robust to any normalisation that
        # might leave a tiny residual on inactive dummies.
        values_block = df[members].values
        active_idx = np.argmax(values_block, axis=1)
        out[parent] = [levels[i] for i in active_idx]

        # Sum SHAPs across the group's dummies.
        shap_cols = [f"{shap_prefix}{m}" for m in members]
        out[f"{shap_prefix}{parent}"] = df[shap_cols].sum(axis=1).values

    # 3. Pass-through label / prediction columns
    for col in ("label", "pred_proba", "pred_label"):
        if col in df.columns:
            out[col] = df[col].values

    return out


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
    Download, model, predict, explain, and save one dataset.

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
        from openxai.dataloader import ReturnLoaders
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
    # 1. Load data (normalised tensors used for the model)
    # ------------------------------------------------------------------
    print("Downloading / loading data...")
    trainloader, testloader = ReturnLoaders(
        data_name=openxai_name,
        download=True,
    )

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

    # ------------------------------------------------------------------
    # 2. Load the matching raw CSV — needed to recover unnormalised values
    # ------------------------------------------------------------------
    raw_df = _load_raw_csv(our_name, split)
    if n_instances is not None:
        raw_df = raw_df.iloc[: n_instances].reset_index(drop=True)
    else:
        raw_df = raw_df.reset_index(drop=True)
    print(f"  Loaded raw {our_name}-{split}.csv with {len(raw_df)} rows")

    # ------------------------------------------------------------------
    # 3. Load pretrained model
    # ------------------------------------------------------------------
    print(f"Loading pretrained {ml_model.upper()} model...")
    model = LoadModel(data_name=openxai_name, ml_model=ml_model, pretrained=True)

    # ------------------------------------------------------------------
    # 4. Compute model predictions (probability of class 1 + argmax label)
    # ------------------------------------------------------------------
    print("Computing model predictions...")
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y.astype(int))

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        # Some OpenXAI models return raw logits, others return probabilities.
        # Apply softmax defensively — softmax of probabilities is numerically
        # close to the original (within ~1e-3) and softmax of logits is correct.
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred_proba = probs[:, 1]
    pred_label = probs.argmax(axis=1)
    print(f"  Predicted class balance: {dict(zip(*np.unique(pred_label, return_counts=True)))}")

    # ------------------------------------------------------------------
    # 5. Compute SHAP values
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 6. Build raw DataFrame using the canonical OpenXAI column names
    # ------------------------------------------------------------------
    feature_names = _resolve_feature_names(openxai_name, X.shape[1])
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y.astype(int)
    df["pred_proba"] = pred_proba
    df["pred_label"] = pred_label.astype(int)
    for i, feat in enumerate(feature_names):
        df[f"{shap_prefix}{feat}"] = shap_np[:, i]

    # ------------------------------------------------------------------
    # 7. Dataset-specific post-processing — substitute raw values, then
    #    apply any structural cleanup (drop columns / aggregate dummies).
    # ------------------------------------------------------------------
    if our_name == "adult":
        df = _denormalise_adult(df, raw_df, shap_prefix=shap_prefix)
        df = _drop_adult_uninformative(df, shap_prefix=shap_prefix)
        print(f"  Replaced normalised Adult values with raw CSV values")
        print(f"  Dropped uninformative columns: {ADULT_DROP_COLUMNS}")
    elif our_name == "german_credit":
        df = _aggregate_german_one_hot(df, raw_df, shap_prefix=shap_prefix)
        print(f"  Replaced normalised German numeric values with raw CSV values")
        print(
            f"  Aggregated 60 raw columns into "
            f"{len(GERMAN_NUMERIC_RENAME) + len(GERMAN_CATEGORICAL_GROUPS)} "
            f"parent features"
        )

    # ------------------------------------------------------------------
    # 8. Save
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
        description="Prepare OpenXAI datasets with SHAP values for the narrative pipeline.",
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
