"""
scripts/compare_human_to_mistral.py
Compare human extraction labels to cached Mistral extractions.

Usage:
    python scripts/compare_human_to_mistral.py \\
        --run-id pilot_run_20260518T135815_bdad28 \\
        --labels outputs/human_labels/pilot_run_20260518T135815_bdad28/labels.jsonl \\
        --eval-dir outputs/evaluations/pilot_run_20260518T135815_bdad28 \\
        --top-k 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.evaluation.compare_extractions import ExtractionAgreement, compare_extractions
from src.evaluation.evaluator import _feature_names_for_dataset
from src.evaluation.extraction_parser import parse_extraction_response
from src.human_labels.schema import (
    human_label_to_extraction_result,
    record_from_dict,
    validate_human_label,
)
from src.storage.evaluations_store import evaluations_csv_path
from src.storage.human_labels_store import DEFAULT_HUMAN_LABELS_DIR, labels_path


def _mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _load_mistral_extractions(
    eval_csv: Path,
    feature_names: List[str],
) -> Dict[str, Optional["ExtractionResult"]]:
    """Map narrative_id -> ExtractionResult or None if unavailable."""
    from src.evaluation.extraction_parser import ExtractionResult

    df = pd.read_csv(eval_csv)
    out: Dict[str, Optional[ExtractionResult]] = {}

    for _, row in df.iterrows():
        nid = str(row["narrative_id"])
        parse_err = str(row.get("parse_error", "") or "").strip()
        raw_json = str(row.get("extraction_json", "") or "").strip()

        if parse_err or not raw_json or raw_json in ("{}", "nan"):
            out[nid] = None
            continue

        try:
            out[nid] = parse_extraction_response(raw_json, feature_names)
        except ValueError:
            out[nid] = None

    return out


def run_comparison(
    labels_file: Path,
    eval_csv: Path,
    feature_names: List[str],
    top_k: int,
    report_path: Path,
) -> List[ExtractionAgreement]:
    with labels_file.open("r", encoding="utf-8") as fh:
        human_records = [record_from_dict(json.loads(line)) for line in fh if line.strip()]

    mistral_by_id = _load_mistral_extractions(eval_csv, feature_names)

    agreements: List[ExtractionAgreement] = []
    skipped_no_mistral = 0

    for record in human_records:
        validate_human_label(record, feature_names)
        human_ext = human_label_to_extraction_result(record)
        mistral_ext = mistral_by_id.get(record.narrative_id)

        if mistral_ext is None:
            skipped_no_mistral += 1
            continue

        agreement = compare_extractions(
            human_ext,
            mistral_ext,
            narrative_id=record.narrative_id,
            top_k_features=top_k,
        )
        agreements.append(agreement)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ExtractionAgreement().to_dict().keys())
    with report_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for ag in agreements:
            row = ag.to_dict()
            row["notes"] = json.dumps(row.get("notes", {}))
            writer.writerow(row)

    print(f"Human labels loaded: {len(human_records)}")
    print(f"Compared successfully: {len(agreements)}")
    print(f"Skipped (no valid Mistral extraction): {skipped_no_mistral}")

    if agreements:
        sign_ag = [a.sign_agreement for a in agreements if a.sign_agreement is not None]
        rank_ag = [a.rank_exact_match for a in agreements if a.rank_exact_match is not None]
        spearman = [a.rank_spearman for a in agreements if a.rank_spearman is not None]
        top_k = [a.top_k_match for a in agreements if a.top_k_match is not None]
        jaccard = [a.feature_set_jaccard for a in agreements]
        unknown = [a.unknown_match for a in agreements]

        print("\n--- Agreement summary ---")
        print(f"Mean feature-set Jaccard:     {_mean(jaccard):.3f}" if _mean(jaccard) else "Mean feature-set Jaccard:     n/a")
        if _mean(sign_ag) is not None:
            print(f"Mean sign agreement:          {_mean(sign_ag):.3f}")
        if _mean(rank_ag) is not None:
            print(f"Mean rank exact match:        {_mean(rank_ag):.3f}")
        if _mean(spearman) is not None:
            print(f"Mean rank Spearman:           {_mean(spearman):.3f}")
        if top_k:
            print(
                f"Top-{args.top_k} set match rate:  "
                f"{sum(top_k)}/{len(top_k)} ({100 * sum(top_k) / len(top_k):.1f}%)"
            )
        if unknown:
            print(f"Unknown-features exact match: {sum(unknown)}/{len(unknown)} ({100*sum(unknown)/len(unknown):.1f}%)")

    print(f"\nReport written to: {report_path}")
    return agreements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare human labels to cached Mistral extractions.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--labels",
        default=None,
        help="Path to labels.jsonl (default: outputs/human_labels/<run_id>/labels.jsonl).",
    )
    parser.add_argument(
        "--eval-dir",
        default=None,
        help="Evaluation run directory (default: outputs/evaluations/<run_id>).",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--report",
        default=None,
        help="Output CSV path (default: outputs/human_labels/<run_id>/agreement_report.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    labels_file = Path(args.labels) if args.labels else labels_path(
        DEFAULT_HUMAN_LABELS_DIR, args.run_id
    )
    if not labels_file.exists():
        raise SystemExit(f"Labels file not found: {labels_file}")

    eval_dir = Path(args.eval_dir) if args.eval_dir else Path(cfg.evaluation.export_dir) / args.run_id
    eval_csv = evaluations_csv_path(eval_dir)
    if not eval_csv.exists():
        raise SystemExit(f"Evaluations CSV not found: {eval_csv}")

    with labels_file.open("r", encoding="utf-8") as fh:
        first = json.loads(next(line for line in fh if line.strip()))
    dataset_name = first.get("dataset", "adult")
    feature_names = _feature_names_for_dataset(cfg, dataset_name)

    report_path = (
        Path(args.report)
        if args.report
        else labels_file.parent / "agreement_report.csv"
    )

    run_comparison(
        labels_file=labels_file,
        eval_csv=eval_csv,
        feature_names=feature_names,
        top_k=args.top_k,
        report_path=report_path,
    )


if __name__ == "__main__":
    main()
