# xAI Hallucination Detection

A research pipeline for investigating whether large language models faithfully translate pre-computed SHAP values into natural-language explanations, and whether failures can be detected and classified automatically.

This is Paper 1 of a three-paper PhD thesis on the nature, causes, and human consequences of faithfulness failures in LLM-generated explainable AI.

---

## What the project does

The pipeline generates approximately 3,600 natural-language narratives by crossing three dimensions:

| Dimension | Values |
|---|---|
| LLMs | Claude Opus, Llama 3 70B, Mistral 7B |
| Prompt strategies | Zero-shot, few-shot, chain-of-thought |
| Datasets | Adult Income, German Credit (OpenXAI benchmark) |

Each narrative explains a single model prediction by describing which input features drove it and why. The narratives are then evaluated against their ground-truth SHAP values using a five-type hallucination taxonomy:

| Hallucination type | What it means | How it is detected |
|---|---|---|
| **Sign inversion** | The narrative states the wrong direction of a feature's effect (e.g., says a feature pushed the prediction up when SHAP shows it pushed it down) | Direction words in the narrative context window are compared to the sign of the SHAP value |
| **Rank swap** | A non-top feature is described using superlatives ("most important", "primary driver") that should only apply to the highest-ranked feature | Superlative phrases are located in the narrative and the nearest feature name is compared to the true top-ranked feature by |SHAP| |
| **Feature fabrication** | The narrative mentions a feature that does not exist in the SHAP input for that instance | Underscore-joined tokens in the narrative are checked against the dataset's actual feature list |
| **Magnitude distortion** | A feature with large |SHAP| is described as minor, or a small-effect feature is described as major | Relative SHAP magnitude (normalised by the instance maximum) is compared against magnitude-signalling words near the feature mention |
| **Omission** | One of the top-k features by |SHAP| is not mentioned anywhere in the narrative | Each of the top-k feature names is searched (with normalised variants) in the narrative text |

---

## Project structure

```
xai-hallucination/
├── config/
│   ├── default.yaml                  # Master config — change values here, not in code
│   └── prompts/
│       ├── zero_shot.txt             # Zero-shot prompt template
│       ├── few_shot.txt              # Two-example few-shot template
│       └── chain_of_thought.txt      # Step-by-step CoT template
├── data/
│   ├── raw/                          # OpenXAI datasets as downloaded
│   └── processed/                    # CSVs with shap_<feature> columns attached
├── outputs/
│   ├── narratives/                   # JSONL export, one file per run
│   ├── evaluations/                  # CSV exports of hallucination labels
│   └── figures/                      # Plots, one subfolder per run
├── src/
│   ├── config.py                     # Pydantic AppConfig — loads default.yaml
│   ├── data_loader.py                # Loads CSVs, formats SHAP tables for prompts
│   ├── db.py                         # SQLite schema + read/write helpers
│   ├── llm_client.py                 # Unified LLM client (Anthropic/Together/Mistral/Ollama)
│   ├── narrative_generator.py        # Orchestrates dataset × model × prompt loop
│   ├── evaluator.py                  # Rule-based hallucination detector + LLM judge
│   └── visualisation/
│       ├── hallucination_rates.py    # Bar charts by type, model, strategy, dataset
│       ├── heatmaps.py               # Model × prompt strategy heatmaps
│       └── export.py                 # Save all figures for a run to disk
├── notebooks/
│   ├── 00_data_preparation.ipynb     # Interactive version of prepare_data.py
│   ├── 01_data_exploration.ipynb     # Explore SHAP distributions, feature importance
│   ├── 02_narrative_inspection.ipynb # Browse generated narratives against SHAP values
│   └── 03_results_visualisation.ipynb# Load evaluations and produce all figures
├── scripts/
│   ├── prepare_data.py               # Download OpenXAI data + compute SHAP values
│   ├── run_generation.py             # CLI: generate narratives
│   ├── run_evaluation.py             # CLI: evaluate a completed run
│   └── export_results.py             # CLI: export DB → CSV (+ optional figures)
├── tests/
│   ├── test_evaluator.py             # Handcrafted cases for all 5 hallucination types
│   └── test_llm_client.py            # Mocked provider API tests
├── .env.example
└── requirements.txt
```

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure API keys**

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
ANTHROPIC_API_KEY=...
TOGETHER_API_KEY=...
MISTRAL_API_KEY=...
```

**3. Prepare datasets**

Place processed CSVs in `data/processed/`. Each CSV must have:
- One row per instance
- Regular feature columns (e.g. `age`, `education_num`)
- Corresponding SHAP columns prefixed with `shap_` (e.g. `shap_age`, `shap_education_num`)

The column prefix is configurable in `config/default.yaml` via `shap_col_prefix`.

---

## Configuration

Everything is controlled by `config/default.yaml`. The file is loaded into a typed `AppConfig` object at startup — no hardcoded values exist in the source code.

```yaml
run:
  name: "pilot_run"       # Label stored in the DB; used to group results
  seed: 42

datasets:
  - name: "adult"
    path: "data/processed/adult.csv"
    shap_col_prefix: "shap_"
    n_instances: 100       # How many rows to use from this dataset

models:
  - id: "claude-opus"
    provider: "anthropic"  # anthropic | together | mistral | ollama
    model_name: "claude-opus-4-6"
    max_tokens: 512
    temperature: 0.0

prompts:
  strategies:
    - zero_shot
    - few_shot
    - chain_of_thought
  template_dir: "config/prompts/"

evaluation:
  top_k_features: 3        # Top-k SHAP features checked for omission and rank swap
  magnitude_threshold: 0.5 # Features with |SHAP| > threshold × max|SHAP| are "large"
  use_llm_judge: false     # Enable second-pass LLM judge on all narratives
  llm_judge_model: "claude-opus-4-6"

storage:
  db_path: "outputs/results.db"
  export_dir: "outputs/evaluations/"
  narrative_dir: "outputs/narratives/"

visualisation:
  figure_dir: "outputs/figures/"
  format: "png"            # png or pdf
  dpi: 150
```

To add a local Ollama model, uncomment and configure the entry in `models:`:

```yaml
- id: "llama3-local"
  provider: "ollama"
  model_name: "llama3:70b"
  base_url: "http://localhost:11434"
```

---

## Data preparation

Before running the generation pipeline, the processed CSVs must exist in `data/processed/`.
The `prepare_data.py` script handles this end-to-end using the OpenXAI benchmark library.

### Install OpenXAI

OpenXAI is not on PyPI. Install it from source once:

```bash
git clone https://github.com/AI4LIFE-GROUP/OpenXAI.git
cd OpenXAI && pip install -e .
cd ..   # return to project root
```

### Run data preparation

```bash
# Prepare both datasets (Adult Income + German Credit) — recommended default
python scripts/prepare_data.py

# One dataset only
python scripts/prepare_data.py --dataset adult
python scripts/prepare_data.py --dataset german_credit

# Choose the underlying ML model (default: lr = logistic regression)
python scripts/prepare_data.py --model ann

# Cap instances for a quick smoke-test
python scripts/prepare_data.py --n 20 --validate

# Use the training split instead of the test split
python scripts/prepare_data.py --split train
```

The script:
1. Downloads the OpenXAI dataset (cached to `data/raw/` on first run).
2. Loads the corresponding pretrained model (`lr` or `ann`) from OpenXAI.
3. Runs the OpenXAI SHAP explainer (`SHAPExplainerC`) over each instance, using the training set as the SHAP background distribution.
4. Saves a CSV to `data/processed/` with feature columns and matching `shap_<feature>` columns.

With `--validate`, it prints a summary showing row count, label distribution, SHAP value range, and a feature/SHAP column parity check.

An interactive version of the same steps is available in `notebooks/00_data_preparation.ipynb`.

### Output format

Each processed CSV has this layout:

| Column type | Example columns |
|---|---|
| Feature values | `age`, `education_num`, `hours_per_week`, ... |
| Ground-truth label | `label` |
| SHAP values | `shap_age`, `shap_education_num`, `shap_hours_per_week`, ... |

The `shap_` prefix must match `shap_col_prefix` in `config/default.yaml` (default: `shap_`).

---

## Running the pipeline

### Step 1 — Generate narratives

```bash
# Full run (all models, datasets, and strategies from config)
python scripts/run_generation.py

# Custom config file
python scripts/run_generation.py --config config/my_config.yaml

# Dry-run: prints prompts to stdout, makes no API calls, writes nothing to DB
python scripts/run_generation.py --dry-run

# Scoped run: one model, one dataset, 5 instances
python scripts/run_generation.py --model claude-opus --dataset adult --n 5
```

The script prints the `run_id` on completion (e.g. `pilot_run_20260509T141023_a3f7c2`).
Results are written to SQLite and a JSONL file in `outputs/narratives/`.

### Step 2 — Evaluate

```bash
python scripts/run_evaluation.py --run-id <run_id>

# Force the LLM judge on all narratives (regardless of config setting)
python scripts/run_evaluation.py --run-id <run_id> --llm-judge
```

Evaluation results are written to the `evaluations` table in SQLite and exported to
`outputs/evaluations/<run_id>_evaluations.csv`.

### Step 3 — Export

```bash
# Export narratives + evaluations to CSV
python scripts/export_results.py --run-id <run_id>

# Also save all visualisation figures
python scripts/export_results.py --run-id <run_id> --figures
```

### Step 4 — Visualise (notebook)

Open `notebooks/03_results_visualisation.ipynb` and set `RUN_ID` to your run.
The notebook loads evaluations from the DB, displays all charts inline, and
calls `export_all_figures()` to save them to `outputs/figures/<run_id>/`.

---

## Source modules

### `src/config.py` — Configuration loader

Parses `config/default.yaml` into a fully typed `AppConfig` object using Pydantic v2.

```python
from src.config import load_config

cfg = load_config()                          # reads config/default.yaml
cfg = load_config("config/custom.yaml")      # custom path

model = cfg.get_model("claude-opus")         # ModelConfig
dataset = cfg.get_dataset("adult")           # DatasetConfig
template = cfg.load_prompt_template("zero_shot")       # raw string
path    = cfg.prompt_template_path("zero_shot")        # Path object (without reading)
```

---

### `src/data_loader.py` — Dataset loading and SHAP formatting

Loads a processed CSV, validates that SHAP columns exist, and returns the first
`n_instances` rows. Also provides helpers for formatting per-instance SHAP tables
ready for injection into prompt templates.

```python
from src.data_loader import (
    load_dataset, format_shap_table, top_k_shap_features,
    get_shap_columns, get_feature_columns,
)

df = load_dataset(cfg.get_dataset("adult"))
row = df.iloc[0]

# Produces a sorted, human-readable SHAP table string
print(format_shap_table(row, prefix="shap_"))
# Output:
#   age: +0.4200 (feature value: 52)
#   education_num: +0.3100 (feature value: 13)
#   ...

# Top-3 features by |SHAP|
top3 = top_k_shap_features(row, prefix="shap_", k=3)
# [("age", 0.42), ("education_num", 0.31), ("hours_per_week", 0.18)]

# Column helpers (operate on the full DataFrame, not a single row)
shap_cols = get_shap_columns(df, prefix="shap_")   # ["shap_age", "shap_education_num", ...]
feat_cols  = get_feature_columns(df, prefix="shap_") # ["age", "education_num", ...]
```

---

### `src/db.py` — SQLite persistence

Creates and manages the three-table SQLite schema. Uses WAL mode and foreign keys.
All write functions are atomic (autocommit per call).

```python
from src.db import (
    init_db, db_connection, open_connection,
    insert_run, insert_narrative, insert_evaluation,
    get_run, get_narrative, get_narratives_for_run,
    get_evaluations_for_run, list_runs,
)

init_db("outputs/results.db")               # creates schema (idempotent)

with db_connection("outputs/results.db") as conn:
    insert_run(conn, run_id="abc123", run_name="pilot", config_json={...}, created_at="...")
    insert_narrative(conn, narrative_id="n1", run_id="abc123", ...)
    insert_evaluation(conn, eval_id="e1", narrative_id="n1",
                      sign_inversion=False, rank_swap=True, ..., evaluated_at="...")

    rows      = get_narratives_for_run(conn, "abc123")   # List[dict]
    evals     = get_evaluations_for_run(conn, "abc123")  # List[dict], joined with narrative cols
    run_meta  = get_run(conn, "abc123")                  # dict | None
    one_narr  = get_narrative(conn, "n1")                # dict | None
    all_runs  = list_runs(conn)                          # List[dict], most recent first
```

`open_connection()` is the non-context-manager alternative that returns a bare
`sqlite3.Connection` when you need manual lifetime control.

`get_evaluations_for_run` returns rows joined with `narratives`, so each dict
includes `dataset`, `instance_id`, `model_id`, and `prompt_strategy` alongside
the evaluation flags — useful for loading directly into a DataFrame.

**Schema:**

```sql
CREATE TABLE runs (
    run_id      TEXT PRIMARY KEY,
    run_name    TEXT NOT NULL,
    config_json TEXT NOT NULL,   -- full config snapshot for reproducibility
    created_at  TEXT NOT NULL
);

CREATE TABLE narratives (
    narrative_id    TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    dataset         TEXT NOT NULL,
    instance_id     INTEGER NOT NULL,
    model_id        TEXT NOT NULL,
    prompt_strategy TEXT NOT NULL,
    narrative_text  TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE evaluations (
    eval_id              TEXT PRIMARY KEY,
    narrative_id         TEXT NOT NULL REFERENCES narratives(narrative_id),
    sign_inversion       INTEGER NOT NULL DEFAULT 0,
    rank_swap            INTEGER NOT NULL DEFAULT 0,
    feature_fabrication  INTEGER NOT NULL DEFAULT 0,
    magnitude_distortion INTEGER NOT NULL DEFAULT 0,
    omission             INTEGER NOT NULL DEFAULT 0,
    any_hallucination    INTEGER NOT NULL DEFAULT 0,  -- 1 if any of the above is 1
    notes                TEXT,
    evaluated_at         TEXT NOT NULL
);
```

Four indexes are created automatically: `narratives(run_id)`, `narratives(model_id)`,
`narratives(dataset)`, and `evaluations(narrative_id)`.

---

### `src/llm_client.py` — Unified LLM client

Single `generate(prompt, model_cfg)` interface that dispatches to the correct
provider SDK based on `model_cfg.provider`. All providers share the same retry
policy (up to 5 attempts, exponential back-off) via `tenacity`.

```python
from src.llm_client import LLMClient

client = LLMClient()
text = client.generate(prompt="Explain this prediction.", model_cfg=cfg.get_model("claude-opus"))
```

Supported providers:

| Provider value | SDK used | Env var required |
|---|---|---|
| `anthropic` | `anthropic` | `ANTHROPIC_API_KEY` |
| `together` | `together` | `TOGETHER_API_KEY` |
| `mistral` | `mistralai` | `MISTRAL_API_KEY` |
| `ollama` | `urllib` (no extra SDK) | `base_url` in config |

---

### `src/narrative_generator.py` — Generation orchestrator

Iterates over every `(dataset, model, prompt_strategy, instance)` combination,
calls the LLM client, and persists each result to both the DB and a JSONL file.

```python
from src.narrative_generator import run_generation

run_id = run_generation(
    cfg,
    dry_run=False,
    filter_model="claude-opus",   # optional: restrict to one model id
    filter_dataset="adult",       # optional: restrict to one dataset name
    n_override=5,                 # optional: override n_instances for all datasets
)
```

A progress bar (via `tqdm`) tracks generation. Errors on individual instances are
logged without aborting the run, so a single API failure does not lose the entire batch.

---

### `src/evaluator.py` — Hallucination detector

Applies five independent rule-based checks to a single narrative given its
ground-truth SHAP values. Returns an `EvaluationResult` with per-type boolean
flags and a human-readable `notes` list explaining each flag.

```python
from src.evaluator import evaluate_narrative, llm_judge, EvaluationResult
from src.config import EvaluationConfig

shap_values = {"age": 0.42, "education_num": 0.31, "hours_per_week": 0.18, "capital_gain": -0.05}
cfg_eval = EvaluationConfig(top_k_features=3, magnitude_threshold=0.5)

result = evaluate_narrative(
    narrative="Age was the most important factor, increasing the predicted income.",
    shap_values=shap_values,
    cfg=cfg_eval,
    all_dataset_features=["age", "education_num", "hours_per_week", "capital_gain", "sex"],
    # ^ optional: full feature list of the dataset. When provided, fabrication is only
    #   flagged for feature-like tokens that do not exist in the dataset at all.
    #   When omitted, any token not in the per-instance shap_values dict is flagged
    #   (higher false-positive rate for zero-contribution features).
)

print(result.any_hallucination)   # True/False
print(result.sign_inversion)      # True/False
print(result.notes_str())         # "rank_swap: ..."
print(result.to_dict())           # {"sign_inversion": 0, "rank_swap": 1, ..., "notes": "..."}
```

When `use_llm_judge: true` is set in config (or `--llm-judge` is passed to the
script), a second LLM pass is run on each narrative. The judge is given the SHAP
values and asked to assess all five hallucination types in a structured format.
Its verdicts are merged with the rule-based results — a narrative is flagged if
either pass raises an alarm.

```python
# Call the LLM judge directly (e.g. for spot-checking a single narrative)
judge_result = llm_judge(
    narrative="Age was the most important factor...",
    shap_values=shap_values,
    model_cfg=cfg.get_model("claude-opus"),
)
```

---

### `src/visualisation/` — Charts and heatmaps

The visualisation module reads only from the evaluations DataFrame and produces
matplotlib figures. It has no dependencies on generation or evaluation logic.

**`hallucination_rates.py`** — bar charts:
- `plot_rates_by_type(evals_df)` — one bar per hallucination type
- `plot_rates_by_model(evals_df)` — overall rate per model
- `plot_rates_by_strategy(evals_df)` — overall rate per prompt strategy
- `plot_rates_by_dataset(evals_df)` — overall rate per dataset
- `plot_type_by_model(evals_df)` — grouped bars: type × model

**`heatmaps.py`** — heatmaps:
- `plot_model_strategy_heatmap(evals_df, dataset)` — model × prompt strategy grid for one dataset
- `plot_all_datasets_heatmap(evals_df)` — side-by-side heatmaps for all datasets
- `plot_type_heatmap(evals_df)` — model × hallucination type grid

**`export.py`** — saves the full figure set:

```python
from src.visualisation.export import export_all_figures

saved_paths = export_all_figures(evals_df, cfg, run_id)
# Saves to outputs/figures/<run_id>/rates_by_type.png, heatmap_adult.png, ...
```

---

## Prompt templates

Three templates live in `config/prompts/`. Each uses `{dataset}` and `{shap_table}`
as format placeholders, which are filled by `narrative_generator.py` at runtime.

| Template | Strategy | Description |
|---|---|---|
| `zero_shot.txt` | Zero-shot | Instructs the model to write a faithful narrative; lists explicit faithfulness requirements |
| `few_shot.txt` | Few-shot | Provides two worked examples (Adult Income and German Credit) before asking for the target narrative |
| `chain_of_thought.txt` | Chain-of-thought | Asks the model to explicitly rank features, note their directions, then draft a narrative — showing reasoning before the final answer |

---

## Running the tests

```bash
pytest tests/ -v
```

`test_evaluator.py` covers 20 handcrafted cases — clean narratives that should
not be flagged, and narratives with deliberate errors for each hallucination type.

`test_llm_client.py` mocks all four provider SDKs to verify dispatch logic,
parameter passing, and response parsing without making any real API calls.

---

## Build status

| Phase | Description | Status |
|---|---|---|
| 1 | Foundation — config, data loader, DB | Complete |
| 2 | Generation — LLM client, narrative generator, CLI script | Complete |
| 3 | Evaluation — rule-based detector, LLM judge, CLI script | Complete |
| 4 | Visualisation — bar charts, heatmaps, figure export, notebooks | Complete |
| 5 | Export + tests | Complete |

---

## Design principles

- **Config over code** — every tunable parameter lives in `default.yaml`. Scripts and source modules read from config; they do not hardcode values.
- **One run = one config snapshot** — each run stores the full config as JSON in the `runs` table, so results are always reproducible from the record alone.
- **Scripts are thin** — `scripts/` contains entry points only. All logic lives in `src/`.
- **No hardcoded paths** — all file paths resolve through the config object.
- **Notebooks for exploration, scripts for execution** — the full pipeline is never run from a notebook.
- **Visualisation is standalone** — `src/visualisation/` reads only from a DataFrame and has no imports from generation or evaluation modules.
- **Fail gracefully, log loudly** — individual narrative failures are caught and logged; the run continues. No silent data loss.
