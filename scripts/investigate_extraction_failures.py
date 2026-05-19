"""Inspect extraction parse failures for a run_id."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.extraction_parser import parse_extraction_response


def _probe_raw(raw: str, feature_names: list[str]) -> None:
    text = raw.strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        print("  (no JSON object found in raw response)")
        return
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        print(f"  JSON decode error: {exc}")
        return

    for fname, ent in data.get("features", {}).items():
        if not isinstance(ent, dict):
            print(f"  {fname}: entry is not an object: {ent!r}")
            continue
        sign = ent.get("sign")
        rank = ent.get("rank")
        exists = ent.get("exists", True)
        if sign not in (1, -1, "1", "-1"):
            print(f"  INVALID sign  {fname}: {sign!r} (type {type(sign).__name__})")
        if exists is False:
            print(f"  exists=false {fname} (skipped by parser)")
        if rank is None:
            print(f"  MISSING rank {fname}")
        if not str(ent.get("assumption", "")).strip():
            print(f"  EMPTY assumption {fname}")

    try:
        result = parse_extraction_response(raw, feature_names)
        print(f"  Re-parse: OK ({len(result.features)} features kept)")
        if len(result.features) < len(data.get("features", {})):
            skipped = set(data.get("features", {})) - set(result.features.keys())
            print(f"  Skipped (invalid sign / exists:false): {sorted(skipped)}")
    except ValueError as exc:
        print(f"  Re-parse: {exc}")


def _replay_flags(run_id: str, failed: pd.DataFrame, feature_names: list[str]) -> None:
    gen_path = Path(f"outputs/generation/{run_id}/narratives.csv")
    if not gen_path.exists():
        return
    gen_df = pd.read_csv(gen_path)
    print("REPLAY (parser fix applied to stored raw responses):")
    for _, row in failed.iterrows():
        raw = str(row.get("extraction_raw_response") or "")
        try:
            ext = parse_extraction_response(raw, feature_names)
            gen = gen_df[gen_df["narrative_id"] == row["narrative_id"]].iloc[0]
            shap = json.loads(gen["shap_values_sorted"])
            from src.evaluation.compare_to_shap import compare_to_shap

            cmp = compare_to_shap(ext, shap, top_k_features=3)
            print(
                f"  instance {row['instance_id']}: "
                f"{len(ext.features)} features -> {cmp.flags_dict()} "
                f"notes={cmp.notes}"
            )
        except ValueError as exc:
            print(f"  instance {row['instance_id']}: still fails: {exc}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--features", nargs="+", default=[
        "age", "capital_gain", "capital_loss", "hours_per_week", "sex_Male",
        "workclass_Private", "marital_status_Non_Married", "occupation_Other",
        "relationship_Non_Husband", "race_White", "native_country_US",
    ])
    args = parser.parse_args()

    csv_path = Path(f"outputs/evaluations/{args.run_id}/evaluations.csv")
    if not csv_path.exists():
        raise SystemExit(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path)
    failed = df[df["parse_error"].fillna("").astype(str).str.len() > 0]
    ok = df[df["parse_error"].fillna("").astype(str).str.len() == 0]

    print(f"Run: {args.run_id}")
    print(f"Total: {len(df)}  Failed: {len(failed)}  OK: {len(ok)}")
    print()

    for _, row in failed.iterrows():
        print("=" * 70)
        print(
            f"instance_id={row['instance_id']}  "
            f"strategy={row['prompt_strategy']}  "
            f"narrative_id={row['narrative_id'][:8]}..."
        )
        print(f"ERROR: {row['parse_error']}")
        raw = str(row.get("extraction_raw_response") or "")
        print(f"Raw length: {len(raw)} chars")
        print("-" * 40)
        print(raw[:3500])
        if len(raw) > 3500:
            print("... [truncated]")
        print("-" * 40)
        _probe_raw(raw, args.features)
        print()

    if len(failed):
        _replay_flags(args.run_id, failed, args.features)

    if len(ok):
        print("=" * 70)
        print("SUCCESSFUL EXTRACTIONS (summary)")
        for _, row in ok.iterrows():
            print(
                f"  instance {row['instance_id']}: "
                f"hall={row['any_hallucination']} "
                f"notes={row.get('notes', '')[:80]}"
            )


if __name__ == "__main__":
    main()
