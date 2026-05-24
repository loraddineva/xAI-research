"""
scripts/human_extraction_ui.py
Gradio UI for human extraction labeling (sign, rank, unknown_features).

Usage:
    python scripts/human_extraction_ui.py --run-id pilot_run_20260518T135815_bdad28
    python scripts/human_extraction_ui.py --run-id <id> --annotator lora --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.dataset_metadata import get_feature_label
from src.evaluation.evaluator import _feature_names_for_dataset
from src.generation.narrative_text import narrative_text_for_evaluation
from src.human_labels.schema import (
    HumanFeatureLabel,
    new_label_record,
    validate_human_label,
)
from src.storage.human_labels_store import (
    DEFAULT_HUMAN_LABELS_DIR,
    append_label,
    get_labeled_ids,
    labels_path,
    load_label_for_narrative,
)
from src.storage.narratives_store import list_runs, load_narratives_csv, run_dir

SIGN_CHOICES = [
    ("", ""),
    ("+1 → income above $50k", "1"),
    ("-1 → income at or below $50k", "-1"),
]
SIGN_LABELS = [c[0] for c in SIGN_CHOICES]
SIGN_VALUES = [c[1] for c in SIGN_CHOICES]


def _sign_value_to_label(value: str) -> str:
    for label, val in SIGN_CHOICES:
        if val == value:
            return label
    return ""


def _parse_unknown(text: str) -> List[str]:
    if not text or not str(text).strip():
        return []
    parts = [p.strip() for p in str(text).split(",")]
    return [p for p in parts if p]


class LabelingSession:
    """Holds narratives queue and navigation state for one run."""

    def __init__(
        self,
        narratives_df: pd.DataFrame,
        feature_names: List[str],
        labels_file: Path,
        annotator: str,
        seed: int,
        positive_label: str,
        negative_label: str,
    ) -> None:
        self.narratives_df = narratives_df.reset_index(drop=True)
        self.feature_names = feature_names
        self.labels_file = labels_file
        self.annotator = annotator
        self.seed = seed
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.queue: List[int] = list(range(len(self.narratives_df)))
        random.Random(seed).shuffle(self.queue)
        self.position = 0

    def eligible_count(self) -> int:
        return len(self.narratives_df)

    def labeled_count(self) -> int:
        return len(get_labeled_ids(self.labels_file, self.annotator))

    def progress_text(self) -> str:
        return f"Labeled: {self.labeled_count()} / {self.eligible_count()}"

    def _row_at_queue_pos(self, pos: int) -> pd.Series:
        idx = self.queue[pos % len(self.queue)]
        return self.narratives_df.iloc[idx]

    def current_row(self) -> pd.Series:
        return self._row_at_queue_pos(self.position)

    def resample_unlabeled(self) -> None:
        labeled = get_labeled_ids(self.labels_file, self.annotator)
        unlabeled_idx = [
            i
            for i, row in self.narratives_df.iterrows()
            if row["narrative_id"] not in labeled
        ]
        if unlabeled_idx:
            rng = random.Random(self.seed + self.position)
            rng.shuffle(unlabeled_idx)
            self.queue = unlabeled_idx
            self.position = 0
        else:
            self.queue = list(range(len(self.narratives_df)))
            random.Random(self.seed).shuffle(self.queue)
            self.position = 0

    def advance(self) -> None:
        self.position = (self.position + 1) % max(len(self.queue), 1)

    def go_prev(self) -> None:
        self.position = (self.position - 1) % max(len(self.queue), 1)

    def go_next(self) -> None:
        self.advance()

    def narrative_meta(self, row: pd.Series) -> str:
        pred = int(row.get("pred_label", 0))
        proba = float(row.get("pred_proba", 0))
        pred_text = (
            self.positive_label if pred == 1 else self.negative_label
        )
        return (
            f"**Instance** `{row['instance_id']}` · "
            f"**Strategy** `{row['prompt_strategy']}` · "
            f"**Prediction** {proba:.0%} → {pred_text}"
        )

    def narrative_text(self, row: pd.Series) -> str:
        raw = str(row.get("narrative_text", ""))
        strategy = str(row.get("prompt_strategy", ""))
        return narrative_text_for_evaluation(raw, strategy)

    def load_existing_label(
        self,
        narrative_id: str,
    ) -> Tuple[List[bool], List[Optional[float]], List[str], str]:
        """Return (mentioned, ranks, sign_labels, unknown_text) for UI."""
        mentioned = [False] * len(self.feature_names)
        ranks: List[Optional[float]] = [None] * len(self.feature_names)
        signs = [""] * len(self.feature_names)
        unknown_text = ""

        existing = load_label_for_narrative(
            self.labels_file, narrative_id, self.annotator
        )
        if existing is None:
            return mentioned, ranks, signs, unknown_text

        for i, fname in enumerate(self.feature_names):
            if fname in existing.features:
                feat = existing.features[fname]
                mentioned[i] = True
                ranks[i] = float(feat.rank)
                signs[i] = _sign_value_to_label(str(feat.sign))

        unknown_text = ", ".join(existing.unknown_features)
        return mentioned, ranks, signs, unknown_text

    def build_label_from_ui(
        self,
        row: pd.Series,
        mentioned: List[bool],
        ranks: List[Any],
        signs: List[str],
        unknown_text: str,
    ):
        features: Dict[str, HumanFeatureLabel] = {}
        for i, fname in enumerate(self.feature_names):
            if not mentioned[i]:
                continue
            sign_str = signs[i]
            if sign_str not in ("1", "-1"):
                raise ValueError(
                    f"Feature '{get_feature_label(row['dataset'], fname)}' "
                    f"is mentioned but has no sign selected."
                )
            rank_val = ranks[i]
            if rank_val is None or rank_val == "":
                raise ValueError(
                    f"Feature '{get_feature_label(row['dataset'], fname)}' "
                    f"is mentioned but has no rank."
                )
            features[fname] = HumanFeatureLabel(
                rank=int(rank_val),
                sign=int(sign_str),
            )

        unknown = _parse_unknown(unknown_text)
        record = new_label_record(
            narrative_id=str(row["narrative_id"]),
            run_id=str(row["run_id"]),
            dataset=str(row["dataset"]),
            instance_id=int(row["instance_id"]),
            prompt_strategy=str(row["prompt_strategy"]),
            annotator=self.annotator,
            features=features,
            unknown_features=unknown,
        )
        validate_human_label(record, self.feature_names)
        return record


def _filter_narratives(
    df: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:
    mask = df["error"].fillna("").astype(str) == ""
    df = df[mask].copy()
    if strategy and strategy != "all":
        df = df[df["prompt_strategy"] == strategy]
    return df


def create_app(session: LabelingSession) -> gr.Blocks:
    n_feat = len(session.feature_names)

    with gr.Blocks(title="Human extraction labeling") as app:
        gr.Markdown("# Human extraction labeling")
        gr.Markdown(session.progress_text())

        progress_md = gr.Markdown(session.progress_text())
        meta_md = gr.Markdown()
        narrative_md = gr.Markdown()

        mentioned_cbs: List[gr.Checkbox] = []
        rank_nums: List[gr.Number] = []
        sign_dds: List[gr.Dropdown] = []

        gr.Markdown("### Features")
        gr.Markdown(
            "Check **Mentioned** for each feature discussed in the narrative, "
            "then set **Rank** (0 = most important) and **Sign**."
        )

        dataset_name = str(session.narratives_df.iloc[0]["dataset"])
        for fname in session.feature_names:
            label = get_feature_label(dataset_name, fname)
            with gr.Row():
                gr.Markdown(f"**{label}**  \n`{fname}`")
                mentioned = gr.Checkbox(label="Mentioned", value=False)
                rank = gr.Number(label="Rank", value=None, precision=0)
                sign = gr.Dropdown(
                    label="Sign",
                    choices=SIGN_LABELS,
                    value="",
                )
                mentioned_cbs.append(mentioned)
                rank_nums.append(rank)
                sign_dds.append(sign)

        unknown_tb = gr.Textbox(
            label="Unknown features (comma-separated, not in valid list)",
            placeholder="e.g. college_degree, years_education",
        )

        status_tb = gr.Textbox(label="Status", interactive=False)
        narrative_id_state = gr.State("")

        with gr.Row():
            prev_btn = gr.Button("← Previous")
            next_btn = gr.Button("Next →")
            sample_btn = gr.Button("Sample random unlabeled")
            save_btn = gr.Button("Save & next", variant="primary")

        def _fill_ui(row: pd.Series):
            mentioned, ranks, signs, unknown = session.load_existing_label(
                str(row["narrative_id"])
            )
            outputs: List[Any] = [
                session.progress_text(),
                session.narrative_meta(row),
                session.narrative_text(row),
                str(row["narrative_id"]),
                unknown,
                "",
            ]
            for i in range(n_feat):
                outputs.extend([mentioned[i], ranks[i], signs[i]])
            return outputs

        def load_current():
            row = session.current_row()
            return _fill_ui(row)

        def on_prev():
            session.go_prev()
            return load_current()

        def on_next():
            session.go_next()
            return load_current()

        def on_sample():
            session.resample_unlabeled()
            return load_current()

        def on_save(
            narrative_id: str,
            unknown_text: str,
            *ui_values: Any,
        ):
            mentioned_vals = list(ui_values[:n_feat])
            rank_vals = list(ui_values[n_feat : 2 * n_feat])
            sign_vals = list(ui_values[2 * n_feat : 3 * n_feat])

            sign_strs = []
            for label in sign_vals:
                idx = SIGN_LABELS.index(label) if label in SIGN_LABELS else 0
                sign_strs.append(SIGN_VALUES[idx])

            row = session.narratives_df[
                session.narratives_df["narrative_id"] == narrative_id
            ].iloc[0]

            try:
                record = session.build_label_from_ui(
                    row,
                    mentioned_vals,
                    rank_vals,
                    sign_strs,
                    unknown_text,
                )
                append_label(session.labels_file, record)
                session.advance()
                filled = _fill_ui(session.current_row())
                filled[-1] = f"Saved label for {narrative_id}."
                return filled
            except ValueError as exc:
                filled = _fill_ui(row)
                filled[-1] = f"Error: {exc}"
                return filled

        all_outputs = (
            [progress_md, meta_md, narrative_md, narrative_id_state, unknown_tb, status_tb]
            + [c for triple in zip(mentioned_cbs, rank_nums, sign_dds) for c in triple]
        )

        app.load(fn=load_current, outputs=all_outputs)

        prev_btn.click(fn=on_prev, outputs=all_outputs)
        next_btn.click(fn=on_next, outputs=all_outputs)
        sample_btn.click(fn=on_sample, outputs=all_outputs)

        save_btn.click(
            fn=on_save,
            inputs=[narrative_id_state, unknown_tb]
            + mentioned_cbs
            + rank_nums
            + sign_dds,
            outputs=all_outputs,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gradio UI for human narrative extraction labeling.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Generation run_id under outputs/generation/.",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--annotator",
        default="default",
        help="Annotator name stored with each label.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["all", "martens", "chain_of_thought"],
        help="Filter narratives by prompt strategy.",
    )
    parser.add_argument(
        "--labels-dir",
        default=DEFAULT_HUMAN_LABELS_DIR,
        help="Base directory for human labels output.",
    )
    parser.add_argument("--share", action="store_true", help="Create public Gradio link.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    gen_dir = Path(cfg.storage.generation_dir)
    run_path = run_dir(cfg, args.run_id)
    csv_path = run_path / "narratives.csv"
    if not csv_path.exists():
        runs = list_runs(gen_dir)
        ids = [r["run_id"] for r in runs[:5]]
        hint = f" Available: {ids}" if ids else ""
        raise SystemExit(f"Narratives not found: {csv_path}.{hint}")

    df = _filter_narratives(load_narratives_csv(csv_path), args.strategy)
    if df.empty:
        raise SystemExit("No narratives match the selected filters.")

    dataset_name = str(df.iloc[0]["dataset"])
    dataset_cfg = cfg.get_dataset(dataset_name)
    feature_names = _feature_names_for_dataset(cfg, dataset_name)

    labels_file = labels_path(args.labels_dir, args.run_id)
    session = LabelingSession(
        narratives_df=df,
        feature_names=feature_names,
        labels_file=labels_file,
        annotator=args.annotator,
        seed=args.seed,
        positive_label=dataset_cfg.positive_class_label,
        negative_label=dataset_cfg.negative_class_label,
    )

    app = create_app(session)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
