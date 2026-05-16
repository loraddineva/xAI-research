# German Credit (archived)

This project now runs **Adult Income only**. German Credit assets and preparation
logic were moved here on 2026-05-16 so the active pipeline stays focused on one dataset.

## Contents

| Path | Description |
|------|-------------|
| `data/german/` | Raw OpenXAI train/test CSVs |
| `data/processed/german_credit.csv` | Processed instances with SHAP values |
| `models/pretrained/` | OpenXAI pretrained LR and ANN weights |
| `notebooks/` | Notebook-local copies of the above |
| `scripts/prepare_german_credit.py` | Standalone data-prep script |
| `src/dataset_metadata_german.py` | Categorical label mappings for prompts |

## Re-enable German Credit

1. Copy `data/` and `models/` back to the repo root (or symlink).
2. Add a `german_credit` block to `config/default.yaml` pointing at `data/processed/german_credit.csv`.
3. Copy `GERMAN_CATEGORICAL_*` from `src/dataset_metadata_german.py` into `src/dataset_metadata.py`.
4. Run preparation if needed:

```bash
python archive/german_credit/scripts/prepare_german_credit.py --n 20 --validate
```
