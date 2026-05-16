"""
src/dataset_metadata.py
Static, per-dataset metadata used by the data loader and the prompt
renderer to display feature values in their original (unnormalized)
form alongside human-readable category labels.

Two things are exposed for each supported dataset:

1. ``CATEGORICAL_MEANINGS`` — dict mapping ``feature_name -> {value: label}``
   so the prompt can show ``checking_status: 1 [< 0 DM]`` instead of a
   bare integer code.
2. ``CATEGORICAL_FEATURES`` — set of feature names that are categorical
   (used to decide whether to coerce the raw value to ``int`` before
   looking up its label).

For Adult, the categorical names use the OpenXAI binary-indicator
convention (``sex_Male``, ``workclass_Private``, ...). The "1" level
is the explicit category in the column name; the "0" level is the
collapsed residual category that OpenXAI's preprocessing groups
everything else into.

For German Credit, the category codes follow the standard UCI Statlog
German Credit attribute documentation
(https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).

Public API
----------
    get_categorical_meaning(dataset_name, feature_name, value) -> str | None
    is_categorical(dataset_name, feature_name) -> bool
"""

from __future__ import annotations

from typing import Dict, Optional, Set


# ---------------------------------------------------------------------------
# Adult Income — binary indicators produced by OpenXAI preprocessing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# German Credit — UCI Statlog category codes
# ---------------------------------------------------------------------------
# Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
# (the OpenXAI processed version preserves the original A-prefixed level
# numbers as plain integers, e.g. A11..A14 -> status_1..status_4 -> "1".."4")
# ---------------------------------------------------------------------------

GERMAN_CATEGORICAL_MEANINGS: Dict[str, Dict[int, str]] = {
    "checking_status": {
        1: "< 0 DM",
        2: "0 to 200 DM",
        3: ">= 200 DM or salary assignments for >= 1 year",
        4: "no checking account",
    },
    "credit_history": {
        0: "no credits taken / all credits paid back duly",
        1: "all credits at this bank paid back duly",
        2: "existing credits paid back duly until now",
        3: "delay in paying off in the past",
        4: "critical account / other credits existing (not at this bank)",
    },
    "purpose": {
        0: "car (new)",
        1: "car (used)",
        2: "furniture / equipment",
        3: "radio / television",
        4: "domestic appliances",
        5: "repairs",
        6: "education",
        7: "retraining",
        9: "business",
        10: "others",
    },
    "savings_status": {
        1: "< 100 DM",
        2: "100 to 500 DM",
        3: "500 to 1000 DM",
        4: ">= 1000 DM",
        5: "unknown / no savings account",
    },
    "employment": {
        1: "unemployed",
        2: "< 1 year",
        3: "1 to 4 years",
        4: "4 to 7 years",
        5: ">= 7 years",
    },
    "personal_status": {
        1: "male, divorced / separated",
        2: "female, divorced / separated / married",
        3: "male, single",
        5: "male, married / widowed",
    },
    "other_parties": {
        1: "none",
        2: "co-applicant",
        3: "guarantor",
    },
    "property_magnitude": {
        1: "real estate",
        2: "building society savings agreement / life insurance",
        3: "car or other",
        4: "unknown / no property",
    },
    "other_payment_plans": {
        1: "bank",
        2: "stores",
        3: "none",
    },
    "housing": {
        1: "rent",
        2: "own",
        3: "for free",
    },
    "job": {
        1: "unemployed / unskilled non-resident",
        2: "unskilled resident",
        3: "skilled employee / official",
        4: "management / self-employed / highly qualified",
    },
    "own_telephone": {
        1: "none",
        2: "yes, registered under the customer's name",
    },
    # Numeric-but-discrete UCI fields. Keeping these in the categorical map
    # lets the prompt expose the original yes/no semantics that get lost
    # when the raw codes (1, 2) are read as numbers.
    "foreign_worker": {
        1: "yes",
        2: "no",
    },
    "people_liable": {
        1: "1 person",
        2: "2 or more people",
    },
}

GERMAN_CATEGORICAL_FEATURES: Set[str] = set(GERMAN_CATEGORICAL_MEANINGS.keys())


# ---------------------------------------------------------------------------
# Registry + lookup helpers
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Dict[str, Dict[int, str]]] = {
    "adult": ADULT_CATEGORICAL_MEANINGS,
    "german_credit": GERMAN_CATEGORICAL_MEANINGS,
}

_CATEGORICAL_FEATURES: Dict[str, Set[str]] = {
    "adult": ADULT_CATEGORICAL_FEATURES,
    "german_credit": GERMAN_CATEGORICAL_FEATURES,
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

    # CSVs may give us float ("1.0"), str ("1"), or int — normalise to int.
    try:
        key = int(float(value))
    except (TypeError, ValueError):
        return None

    return mapping.get(key)
