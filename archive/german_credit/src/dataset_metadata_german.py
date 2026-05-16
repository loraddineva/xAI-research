"""
archive/german_credit/src/dataset_metadata_german.py
Categorical label mappings for German Credit (UCI Statlog).

Archived when the project scope was narrowed to Adult Income only.
To restore, merge GERMAN_CATEGORICAL_* into src/dataset_metadata.py and
re-add "german_credit" to the registries.
"""

from __future__ import annotations

from typing import Dict, Set

# Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

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
