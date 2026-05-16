"""
src/dataset_metadata.py
Static metadata for Adult Income: human-readable labels for categorical
features used by the data loader and prompt renderer.

Public API
----------
    get_categorical_meaning(dataset_name, feature_name, value) -> str | None
    is_categorical(dataset_name, feature_name) -> bool
"""

from __future__ import annotations

from typing import Dict, Optional, Set

ADULT_CATEGORICAL_MEANINGS: Dict[str, Dict[int, str]] = {
    "sex_Male": {
        0: "Female",
        1: "Male",
    },
    "workclass_Private": {
        0: "non-Private (Government / Self-employed / Without-pay / Never-worked)",
        1: "Private",
    },
    "marital_status_Non_Married": {
        0: "Married",
        1: "Non-Married (Never-married / Divorced / Separated / Widowed)",
    },
    "occupation_Other": {
        0: "main occupation category (Exec-managerial / Prof-specialty / Craft-repair / Sales / Adm-clerical)",
        1: "Other occupation (Machine-op-inspct / Transport-moving / Handlers-cleaners / Farming-fishing / etc.)",
    },
    "relationship_Non_Husband": {
        0: "Husband",
        1: "Non-Husband (Wife / Own-child / Not-in-family / Other-relative / Unmarried)",
    },
    "race_White": {
        0: "non-White (Black / Asian-Pac-Islander / Amer-Indian-Eskimo / Other)",
        1: "White",
    },
    "native_country_US": {
        0: "non-United-States",
        1: "United States",
    },
}

ADULT_CATEGORICAL_FEATURES: Set[str] = set(ADULT_CATEGORICAL_MEANINGS.keys())

_REGISTRY: Dict[str, Dict[str, Dict[int, str]]] = {
    "adult": ADULT_CATEGORICAL_MEANINGS,
}

_CATEGORICAL_FEATURES: Dict[str, Set[str]] = {
    "adult": ADULT_CATEGORICAL_FEATURES,
}


def is_categorical(dataset_name: Optional[str], feature_name: str) -> bool:
    """True if *feature_name* is in the categorical set for *dataset_name*."""
    if dataset_name is None:
        return False
    return feature_name in _CATEGORICAL_FEATURES.get(dataset_name, set())


def get_categorical_meaning(
    dataset_name: Optional[str],
    feature_name: str,
    value: object,
) -> Optional[str]:
    """
    Look up the human-readable label for a categorical feature value.

    Returns ``None`` when the dataset is unknown, the feature is not
    categorical, or the value is missing / cannot be coerced to int.
    """
    if dataset_name is None:
        return None

    mapping = _REGISTRY.get(dataset_name, {}).get(feature_name)
    if not mapping:
        return None

    try:
        key = int(float(value))
    except (TypeError, ValueError):
        return None

    return mapping.get(key)
