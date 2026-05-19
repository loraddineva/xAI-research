"""Audit married / husband prose vs marital_status_Non_Married extraction."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import format_instance_snapshot
from src.dataset_metadata import get_categorical_meaning


def _married_prose(text: str) -> bool:
    return bool(re.search(r"\bmarried\b", text, re.I))


def _husband_prose(text: str) -> bool:
    return bool(re.search(r"\bhusband\b", text, re.I))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()

    eval_path = Path(f"outputs/evaluations/{args.run_id}/evaluations.csv")
    gen_path = Path(f"outputs/generation/{args.run_id}/narratives.csv")
    eval_df = pd.read_csv(eval_path)
    gen_df = pd.read_csv(gen_path)
    adult = pd.read_csv("data/processed/adult.csv")

    print(f"Run: {args.run_id}\n")
    for _, ev in eval_df.iterrows():
        gen = gen_df[gen_df["narrative_id"] == ev["narrative_id"]].iloc[0]
        narrative = str(gen.get("narrative_text", gen.get("narrative", "")) or "")
        inst = int(ev["instance_id"])
        row = adult.loc[inst]
        non_married_val = int(row["marital_status_Non_Married"])
        rel_val = int(row["relationship_Non_Husband"])
        married_gt = get_categorical_meaning("adult", "marital_status_Non_Married", non_married_val)
        rel_gt = get_categorical_meaning("adult", "relationship_Non_Husband", rel_val)

        ext = {}
        raw_ext = ev.get("extraction_json")
        if raw_ext is not None and pd.notna(raw_ext) and str(raw_ext).strip():
            ext = json.loads(str(raw_ext)).get("features", {})

        marital_ext = ext.get("marital_status_Non_Married")
        rel_ext = ext.get("relationship_Non_Husband")

        issues = []
        if _married_prose(narrative) and non_married_val == 0:
            if not marital_ext:
                issues.append("narrative says married but marital_status_Non_Married not extracted")
            elif marital_ext.get("sign") == -1:
                issues.append(
                    "narrative says married (value=0) but extraction sign=-1 "
                    "(reads as non-married hurts income)"
                )
        if _husband_prose(narrative) and rel_val == 0:
            if not rel_ext:
                issues.append("narrative says husband but relationship_Non_Husband not extracted")
        if marital_ext and _husband_prose(narrative) and not rel_ext:
            issues.append("marital extracted but husband prose mapped only to marital (confusion)")

        if not issues and not (_married_prose(narrative) or _husband_prose(narrative)):
            continue

        print("=" * 70)
        print(f"instance {inst}  parse_error={ev.get('parse_error', '') or 'none'}")
        print(f"  Ground truth: marital_status_Non_Married={non_married_val} ({married_gt})")
        print(f"                relationship_Non_Husband={rel_val} ({rel_gt})")
        print(f"  Prose: married={_married_prose(narrative)}  husband={_husband_prose(narrative)}")
        if marital_ext:
            print(f"  Extracted marital_status_Non_Married: sign={marital_ext.get('sign')}")
            print(f"    assumption: {marital_ext.get('assumption', '')[:100]}")
        if rel_ext:
            print(f"  Extracted relationship_Non_Husband: sign={rel_ext.get('sign')}")
        for msg in issues:
            print(f"  ISSUE: {msg}")
        print()


if __name__ == "__main__":
    main()
