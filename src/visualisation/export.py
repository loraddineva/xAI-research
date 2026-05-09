"""
src/visualisation/export.py
Save all standard figures for a completed run to outputs/figures/.

Public API
----------
    export_all_figures(evals_df, cfg, run_id) -> List[Path]
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from src.config import AppConfig
from src.visualisation.hallucination_rates import (
    plot_rates_by_dataset,
    plot_rates_by_model,
    plot_rates_by_strategy,
    plot_rates_by_type,
    plot_type_by_model,
)
from src.visualisation.heatmaps import (
    plot_all_datasets_heatmap,
    plot_model_strategy_heatmap,
    plot_type_heatmap,
)


def _save(fig: plt.Figure, path: Path, dpi: int, fmt: str) -> Path:
    out = path.with_suffix(f".{fmt}")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def export_all_figures(
    evals_df: pd.DataFrame,
    cfg: AppConfig,
    run_id: str,
) -> List[Path]:
    """
    Generate and save the full figure set for a run.

    Args:
        evals_df: Evaluations DataFrame (loaded from DB or CSV).
                  Must have columns: model_id, prompt_strategy, dataset,
                  any_hallucination, sign_inversion, rank_swap,
                  feature_fabrication, magnitude_distortion, omission.
        cfg:      AppConfig (provides figure_dir, format, dpi).
        run_id:   Used to prefix output filenames.

    Returns:
        List of Paths to saved figure files.
    """
    fig_dir = Path(cfg.visualisation.figure_dir) / run_id
    fig_dir.mkdir(parents=True, exist_ok=True)

    fmt = cfg.visualisation.format
    dpi = cfg.visualisation.dpi
    saved: List[Path] = []

    def save(fig: plt.Figure, name: str) -> None:
        path = _save(fig, fig_dir / name, dpi=dpi, fmt=fmt)
        saved.append(path)
        print(f"  Saved: {path}")

    print(f"Exporting figures to {fig_dir}/")

    # --- Bar charts ---
    save(plot_rates_by_type(evals_df), "rates_by_type")
    save(plot_rates_by_model(evals_df), "rates_by_model")
    save(plot_rates_by_strategy(evals_df), "rates_by_strategy")
    save(plot_rates_by_dataset(evals_df), "rates_by_dataset")
    save(plot_type_by_model(evals_df), "type_by_model")

    # --- Heatmaps (all datasets combined) ---
    save(plot_all_datasets_heatmap(evals_df), "heatmap_all_datasets")
    save(plot_type_heatmap(evals_df), "heatmap_type_by_model")

    # --- Per-dataset heatmaps ---
    for dataset in sorted(evals_df["dataset"].unique()):
        safe_name = dataset.replace(" ", "_").lower()
        save(
            plot_model_strategy_heatmap(evals_df, dataset=dataset),
            f"heatmap_{safe_name}",
        )

    print(f"\n{len(saved)} figures saved.")
    return saved
