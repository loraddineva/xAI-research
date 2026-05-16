# xAI Hallucination Detection

A research pipeline that investigates whether large language models (LLMs) faithfully translate
pre-computed SHAP values into natural-language explanations, and whether failures of faithfulness
can be detected and classified automatically.

This is Paper 1 of a PhD thesis on the nature, causes, and human consequences of faithfulness
failures in LLM-generated explainable AI (XAI).

---

## Research context

When an ML model makes a prediction, SHAP values tell us exactly which input features drove that
prediction and by how much. This project asks: if we give those SHAP values to an LLM and ask it
to write a human-readable explanation, does the LLM faithfully report what the numbers say?

To answer this, the pipeline:

1. Takes two tabular datasets from the [OpenXAI benchmark](https://github.com/AI4LIFE-GROUP/OpenXAI)
  (Adult Income and German Credit), each paired with a pretrained ML model (logistic regression),
   per-instance SHAP values, and the model's own predicted probability and class label.
2. Generates ~1,200 natural-language narratives by crossing three LLMs × two prompt strategies
   (Martens direct, chain-of-thought) × two datasets × 100 instances per dataset.
3. Stores every narrative in a per-run CSV (`outputs/generation/<run_id>/narratives.csv`) for
   downstream faithfulness evaluation (evaluator to be implemented separately).

---

## Hallucination taxonomy


| Type                     | What it means                                                                                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sign inversion**       | The narrative states the wrong direction of effect — e.g., says a feature pushed the prediction *up* when SHAP shows it pushed it *down*            |
| **Rank swap**            | A non-top feature is described with superlatives ("most important", "primary driver") that should only apply to the feature with the highest \|SHAP\| |
| **Feature fabrication**  | The narrative mentions a feature that does not exist in the input at all                                                                            |
| **Magnitude distortion** | A feature with large \|SHAP\| is described as minor, or a small-effect feature is described as major                                                  |
| **Omission**             | One of the top-k features by \|SHAP\| is not mentioned anywhere in the narrative                                                                      |


---

## Experimental design


| Dimension             | Values                                                                                          |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| Datasets              | Adult Income, German Credit (OpenXAI benchmark)                                                 |
| Models                | Claude Opus, Llama 3 70B (via Together AI), Mistral Small (via Mistral AI)                      |
| Prompt strategies     | `martens` (Martens et al. 2024 direct narrative), `chain_of_thought` (rank/sign steps + `Narrative:` section) |
| Instances per dataset | 100                                                                                             |
| Total narratives      | ~1,200                                                                                          |


**Martens (`narrative.j2`):** follows Martens et al. (2024), *Tell Me a Story! Narrative-Driven XAI
with Large Language Models* — task framing, predicted probability and class, SHAP table sorted
from most positive to most negative, and a fluent story focused on the most influential features.

**Chain-of-thought (`chain_of_thought.j2`):** same context and SHAP table, then structured
reasoning steps (rank top features, note negligible effects, verify top feature) before a final
`Narrative:` section. Evaluation strips the reasoning steps and scores only the narrative prose.
CoT uses `max_tokens: 1024` (Martens uses 512).


---

## Project structure

```
xai-hallucination/
├── config/
│   ├── default.yaml                  # Master config — change values here, not in code
│   └── prompts/
│       ├── narrative.j2              # Martens-style direct narrative (strategy: martens)
│       └── chain_of_thought.j2       # CoT narrative (strategy: chain_of_thought)
├── data/
│   ├── raw/                          # OpenXAI datasets as downloaded (auto-created)
│   └── processed/                    # CSVs with feature + shap_<feature> + pred_proba/pred_label
├── outputs/
│   ├── generation/<run_id>/          # Per-run: narratives.csv, narratives.jsonl, run_metadata.yaml
│   └── figures/                      # Plots — datasets/ subfolder for exploratory figures
├── src/
│   ├── config.py                     # Pydantic AppConfig — parses default.yaml
│   ├── data_loader.py                # Loads CSVs; formats SHAP tables for prompts
│   ├── storage/
│   │   └── narratives_store.py       # List/load runs from narratives.csv on disk
│   ├── generation/                   # Generation subpackage
│   │   ├── llm_client.py             # Unified LLM client (Anthropic/Together/Mistral/Ollama)
│   │   ├── prompt_renderer.py        # Jinja2 renderer per prompt strategy
│   │   ├── narrative_text.py         # Strips CoT reasoning before evaluation
│   │   ├── generator.py              # Orchestrates dataset × strategy × model × instance loop
│   │   └── exporters.py              # CSV + JSONL writers
│   └── visualisation/
│       ├── dataset_overview.py       # Feature distributions, class balance, correlation heatmap
│       ├── shap_distributions.py     # SHAP importance bar, beeswarm, scatter plots
│       ├── hallucination_rates.py    # Bar charts by type / model / strategy / dataset
│       ├── heatmaps.py               # Model × prompt strategy heatmaps
│       └── export.py                 # Save all figures for a run or dataset to disk
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Feature distributions + SHAP visualisations
│   ├── 02_narrative_inspection.ipynb # Browse generated narratives alongside SHAP values
│   └── 03_results_visualisation.ipynb# Placeholder for future evaluation figures
├── scripts/
│   ├── prepare_data.py               # Download OpenXAI data, compute predictions + SHAP, save CSVs
│   ├── run_generation.py             # CLI: generate narratives for a run
│   └── export_results.py             # CLI: inspect run CSV (+ optional dataset figures)
├── tests/
│   └── test_llm_client.py            # Mocked provider API tests (no real API calls)
├── .env.example
└── requirements.txt
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys (only the providers you plan to use are required):

```
ANTHROPIC_API_KEY=sk-ant-...
TOGETHER_API_KEY=...
MISTRAL_API_KEY=...
```

### 3. Install OpenXAI

OpenXAI is not on PyPI. Install it from source once:

```bash
git clone https://github.com/AI4LIFE-GROUP/OpenXAI.git
cd OpenXAI && pip install -e .
cd ..
```

---

## Data preparation

Before running the generation pipeline, processed CSVs must exist in `data/processed/`.
The `prepare_data.py` script handles this end-to-end.

```bash
# Prepare both datasets (default — recommended for a full run)
python scripts/prepare_data.py

# One dataset only
python scripts/prepare_data.py --dataset adult
python scripts/prepare_data.py --dataset german_credit

# Use the neural-network model instead of logistic regression
python scripts/prepare_data.py --model ann

# Quick smoke-test: only prepare 20 instances and print a validation summary
python scripts/prepare_data.py --n 20 --validate
```

**What the script does:**

1. Downloads the OpenXAI dataset via `ReturnLoaders` (cached to `data/raw/` on first run).
2. Loads the corresponding pretrained model from OpenXAI (`lr` by default).
3. Computes the model's predicted probability of class 1 and predicted class label per instance.
4. Runs the OpenXAI SHAP explainer over each instance.
5. Applies dataset-specific post-processing:
   - **Adult Income** — drops `fnlwgt` (the uninformative survey weight column) and its SHAP twin.
   - **German Credit** — aggregates the 52 one-hot encoded dummy columns
     (`status_*`, `credit-history_*`, `purpose_*`, ...) back into 12 parent categorical
     features by summing dummy SHAPs and recording the active dummy's index as the
     feature value, yielding the original 20-feature schema.
6. Saves a CSV to `data/processed/`.

**Output CSV layout:**


| Column type           | Example names                                                |
| --------------------- | ------------------------------------------------------------ |
| Raw feature values    | `age`, `education_num`, `hours_per_week`, ...                |
| Ground-truth label    | `label`                                                      |
| Model prediction      | `pred_proba` (P(class=1)), `pred_label` (argmax class)        |
| SHAP values           | `shap_age`, `shap_education_num`, `shap_hours_per_week`, ... |


The `shap_` prefix must match `shap_col_prefix` in `config/default.yaml` (default: `shap_`).
The `pred_proba` and `pred_label` columns are required by the narrative prompt; if they are
missing the renderer will raise a `KeyError` asking you to re-run `prepare_data.py`.

SHAP computation on the full Adult Income test split (~9,000 instances) takes several minutes.
Run this step directly in a terminal, not as a background task.

---

## Running the full pipeline

### Step 1 — Generate narratives

```bash
# Full run (reads all models and datasets from config/default.yaml)
python scripts/run_generation.py

# Dry-run: prints prompts to stdout; makes no API calls and writes nothing to disk
python scripts/run_generation.py --dry-run

# Restrict to one model, one dataset, one strategy, 5 instances
python scripts/run_generation.py --model claude-opus --dataset adult --strategy martens --n 5

# Custom config file
python scripts/run_generation.py --config config/my_config.yaml
```

The script prints the `run_id` (e.g. `pilot_run_20260510T141023_a3f7c2`) on completion.
Each run writes to `outputs/generation/<run_id>/`:

| File | Role |
| ---- | ---- |
| `narratives.csv` | **Canonical store** — one row per narrative; all scripts and notebooks read this |
| `narratives.jsonl` | Crash-safe stream — appended live during the run |
| `run_metadata.yaml` | Config snapshot, timestamps, record counts |

Every row includes the **full rendered prompt**, model provider/name, temperature,
max_tokens, sorted SHAP values, prediction fields, and narrative text (or an `error`
message if generation failed).

**Resume behaviour:** within a single run, already-generated rows are detected in
`narratives.csv` and skipped. Restarting the script without a resume flag creates a new `run_id`.

### Step 2 — Inspect and export figures

```bash
# Print run summary from narratives.csv
python scripts/export_results.py --run-id <run_id>

# Also produce dataset-level visualisation figures (SHAP, feature distributions)
python scripts/export_results.py --run-id <run_id> --figures
```

Dataset figures are saved to `outputs/figures/datasets/<dataset_name>/`.

### Step 3 — Visualise (notebooks)

- `notebooks/02_narrative_inspection.ipynb` — browse narratives from `narratives.csv`
- `notebooks/01_data_exploration.ipynb` — feature distributions and SHAP plots
- `notebooks/03_results_visualisation.ipynb` — placeholder for future evaluation figures

---

## Configuration reference

Everything is controlled by `config/default.yaml`. No parameters are hardcoded in `src/`.

```yaml
run:
  name: "pilot_run"        # Human-readable label; stored in run_metadata.yaml
  seed: 42                 # Reserved for reproducibility hooks

datasets:
  - name: "adult"
    path: "data/processed/adult.csv"
    shap_col_prefix: "shap_"   # SHAP columns are named shap_<feature>
    n_instances: 100            # Number of rows to use from this dataset
    # Per-dataset narrative metadata — injected into both prompt templates:
    task_description: "predict whether a person's annual income exceeds $50,000, based on demographic and employment data"
    positive_class_label: "income above $50,000"
    negative_class_label: "income at or below $50,000"

models:
  - id: "claude-opus"           # Short id used in CLI flags and CSV records
    provider: "anthropic"       # anthropic | together | mistral | ollama
    model_name: "claude-opus-4-6"
    max_tokens: 512
    temperature: 0.0

prompt:
  strategies:
    - id: martens
      template: "config/prompts/narrative.j2"
    - id: chain_of_thought
      template: "config/prompts/chain_of_thought.j2"
      max_tokens: 1024

storage:
  generation_dir: "outputs/generation/"     # Per-run subfolders: csv + jsonl + metadata

visualisation:
  figure_dir: "outputs/figures/"
  format: "png"                 # png or pdf
  dpi: 150
```

To add a local Ollama model, uncomment and configure:

```yaml
- id: "llama3-local"
  provider: "ollama"
  model_name: "llama3:70b"
  base_url: "http://localhost:11434"
```

---

## Source module reference

### `src/config.py`

Parses `config/default.yaml` into a fully typed `AppConfig` object (Pydantic v2).

```python
from src.config import load_config

cfg = load_config()                           # reads config/default.yaml
cfg = load_config("config/custom.yaml")       # custom path

model   = cfg.get_model("claude-opus")        # → ModelConfig
dataset = cfg.get_dataset("adult")            # → DatasetConfig
```

---

### `src/data_loader.py`

Loads a processed CSV and provides helpers for formatting SHAP tables as prompt text.

```python
from src.data_loader import (
    load_dataset, format_shap_table, top_k_shap_features,
    get_shap_columns, get_feature_columns,
)

df   = load_dataset(cfg.get_dataset("adult"))
row  = df.iloc[0]

print(format_shap_table(row, prefix="shap_"))
# Output (sorted from most positive to most negative SHAP, one feature per line —
# matches the order shown to the LLM in the Martens-style prompt):
#   age: +0.4200 (feature value: 52)
#   education_num: +0.3100 (feature value: 13)
#   hours_per_week: +0.1800 (feature value: 45)
#   capital_gain: -0.0500 (feature value: 0)

top3 = top_k_shap_features(row, prefix="shap_", k=3)
# → [("age", 0.42), ("education_num", 0.31), ("hours_per_week", 0.18)]
# (top_k_shap_features still ranks by |SHAP|; format_shap_table uses signed order)

shap_cols = get_shap_columns(df, prefix="shap_")    # ["shap_age", "shap_education_num", ...]
feat_cols  = get_feature_columns(df, prefix="shap_") # ["age", "education_num", ...]
```

---

### `src/generation/prompt_renderer.py`

Renders a narrative prompt for a given strategy (`martens` or `chain_of_thought`).
Templates are listed under `cfg.prompt.strategies`. Per-dataset metadata plus the row's
`pred_proba` and `pred_label` columns drive the rendering.

```python
from src.generation import PromptRenderer

renderer = PromptRenderer(cfg)
prompt = renderer.render(
    dataset_cfg=cfg.get_dataset("adult"),
    row=df.iloc[0],
    strategy_id="chain_of_thought",
)
```

`StrictUndefined` is set on the Jinja2 environment, so a typo in a template variable
raises an error immediately rather than silently rendering an empty string.

---

### `src/storage/narratives_store.py`

Read/write helpers for generation run artefacts on disk.

```python
from src.storage import (
    list_runs, load_narratives_csv, get_run,
    narratives_csv_path, run_dir, narrative_exists,
)

runs = list_runs(cfg.storage.generation_dir)   # scan for folders with narratives.csv
meta = get_run(cfg.storage.generation_dir, "pilot_run_20260510T141023_a3f7c2")
df   = load_narratives_csv(narratives_csv_path(run_dir(cfg, meta["run_id"])))
```

---

### `src/generation/llm_client.py`

Single `generate(prompt, model_cfg)` interface dispatching to the correct provider.
All providers share the same retry policy: up to 5 attempts with exponential back-off,
using `tenacity`. The provider SDK client is instantiated once per `generate()` call;
retries reuse the same client object.

```python
from src.generation import LLMClient

client = LLMClient()
text = client.generate(
    prompt="Explain this prediction.",
    model_cfg=cfg.get_model("claude-opus"),
)
```


| `provider` value | SDK               | Required env var     |
| ---------------- | ----------------- | -------------------- |
| `anthropic`      | `anthropic`       | `ANTHROPIC_API_KEY`  |
| `together`       | `together`        | `TOGETHER_API_KEY`   |
| `mistral`        | `mistralai`       | `MISTRAL_API_KEY`    |
| `ollama`         | `urllib` (stdlib) | `base_url` in config |


---

### `src/generation/generator.py`

Iterates every `(dataset, prompt_strategy, model, instance)` combination, renders the
configured prompt template, calls the LLM, and persists each result to `narratives.csv` and a streaming
`narratives.jsonl` under `outputs/generation/<run_id>/`. The CSV is rewritten at the end
of the run so failed instances are included.

```python
from src.generation import run_generation

run_id = run_generation(
    cfg,
    dry_run=False,
    filter_model="claude-opus",   # optional: restrict to one model id
    filter_dataset="adult",       # optional: restrict to one dataset name
    n_override=5,                 # optional: override n_instances for all datasets
)
```

Individual API failures on a single instance are caught, logged, and recorded in
the CSV with their `error` field populated, without aborting the run.

---

### `src/generation/exporters.py`

CSV and JSONL writers for generation runs. Every row carries the full rendered prompt,
model configuration, sorted SHAP values, and prediction fields.

```python
from src.generation import append_csv_row, append_jsonl, write_csv
from src.generation.exporters import NarrativeRecord, CSV_COLUMNS

append_jsonl(path, record)   # streaming append during the run
append_csv_row(path, record) # incremental CSV append
write_csv(path, records)     # full rewrite at end of run
```

---

### `src/visualisation/`

All visualisation functions return a `matplotlib.Figure` and never call `plt.show()`, so
they work both in notebooks (displayed inline) and in scripts (saved to disk). The module
has no imports from generation or evaluation logic — it reads only from DataFrames.

#### `dataset_overview.py` — exploratory dataset plots

```python
from src.visualisation.dataset_overview import (
    plot_feature_distributions,
    plot_class_balance,
    plot_correlation_heatmap,
)
from src.data_loader import get_feature_columns

feat_cols = get_feature_columns(df, prefix="shap_")

fig1 = plot_feature_distributions(df, feat_cols)
# Grid of histograms (numeric features) and bar charts (categorical features)

fig2 = plot_class_balance(df, label_col="label")
# Bar chart of class counts with percentage labels

fig3 = plot_correlation_heatmap(df, feat_cols)
# Pearson correlation heatmap (numeric features only)
```

#### `shap_distributions.py` — SHAP summary plots

```python
from src.visualisation.shap_distributions import (
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_shap_scatter,
)
from src.data_loader import get_shap_columns, get_feature_columns

shap_cols = get_shap_columns(df, prefix="shap_")
feat_cols = get_feature_columns(df, prefix="shap_")

fig1 = plot_shap_bar(df, shap_cols)
# Horizontal bar chart: mean |SHAP| per feature (global importance ranking)

fig2 = plot_shap_beeswarm(df, shap_cols, feature_cols=feat_cols)
# Strip plot: SHAP value distribution per feature, coloured by feature magnitude
# (blue = low feature value, red = high) — equivalent to the SHAP library beeswarm

fig3 = plot_shap_scatter(df, feature_col="age", shap_col="shap_age")
# Scatter: SHAP value (y) vs raw feature value (x) for a single feature
# Reveals whether the feature has a linear, threshold, or non-linear effect
```

#### `hallucination_rates.py` — post-evaluation bar charts

```python
from src.visualisation.hallucination_rates import (
    plot_rates_by_type,      # one bar per hallucination type
    plot_rates_by_model,     # overall rate per model
    plot_rates_by_strategy,  # overall rate per prompt strategy
    plot_rates_by_dataset,   # overall rate per dataset
    plot_type_by_model,      # grouped: hallucination type × model
)
```

#### `heatmaps.py` — model × prompt strategy heatmaps

```python
from src.visualisation.heatmaps import (
    plot_model_strategy_heatmap,   # single dataset: rows=model, cols=strategy
    plot_all_datasets_heatmap,     # side-by-side heatmaps for all datasets
    plot_type_heatmap,             # rows=model, cols=hallucination type
)
```

#### `export.py` — save figures to disk

```python
from src.visualisation.export import export_dataset_figures, export_all_figures

# Dataset-level figures (no evaluation results needed)
saved = export_dataset_figures(
    df, dataset_name="adult", shap_cols=shap_cols, feature_cols=feat_cols, cfg=cfg
)
# Saves to outputs/figures/datasets/adult/

# Evaluation figures for a completed run
saved = export_all_figures(evals_df, cfg, run_id)
# Saves to outputs/figures/<run_id>/
```

---

## Prompt templates

Two Jinja2 templates are crossed in every full run (`cfg.prompt.strategies`):

| Strategy | File | Role |
| -------- | ---- | ---- |
| `martens` | `config/prompts/narrative.j2` | Martens et al. (2024) direct narrative |
| `chain_of_thought` | `config/prompts/chain_of_thought.j2` | Structured reasoning steps + `Narrative:` prose |

Both share per-dataset task description and class labels from `config/default.yaml`.
CoT responses are stored in full in `narratives.csv`; `src/generation/narrative_text.py`
strips everything before the final `Narrative:` heading before faithfulness evaluation.

**Template variables (all required — `StrictUndefined` raises on typos):**


| Variable                       | Content                                                                                  |
| ------------------------------ | ---------------------------------------------------------------------------------------- |
| `{{ dataset }}`                | Dataset name (e.g. `"adult"`)                                                            |
| `{{ task_description }}`       | Per-dataset task sentence from config (e.g. `"predict whether income exceeds $50,000"`)   |
| `{{ positive_class_label }}`   | Per-dataset positive class label (e.g. `"income above $50,000"`, `"good credit risk"`)   |
| `{{ negative_class_label }}`   | Per-dataset negative class label                                                         |
| `{{ pred_proba }}`             | Model's predicted probability of class 1, in `[0, 1]` (rendered as a percentage)         |
| `{{ pred_label }}`             | Model's predicted class label (`0` or `1`)                                               |
| `{{ pred_class_text }}`        | Resolved class label string for the predicted class                                      |
| `{{ shap_table }}`             | SHAP table sorted from most positive to most negative, one feature per line              |


To add or swap a strategy, add an entry under `prompt.strategies` with a unique `id`
and `template` path; optional `max_tokens` overrides the model default for that strategy.

---

## Running the tests

```bash
pytest tests/ -v
```

`test_llm_client.py` mocks all four provider SDKs to verify dispatch logic, parameter
passing, and response parsing without making any real API calls.

---

## Design principles

- **Config over code** — every tunable parameter lives in `config/default.yaml`. No value
that could plausibly change between runs is hardcoded in `src/`.
- **One run = one config snapshot** — the full config is written to `run_metadata.yaml`
at generation time, so results from any run can always be reproduced from the run folder.
- **Scripts are thin** — `scripts/` contains CLI entry points only. All logic lives in `src/`.
- **No hardcoded paths** — all file paths resolve through the config object.
- **Notebooks for exploration, scripts for execution** — the full pipeline is never run from
a notebook; notebooks call `src/` modules but do not contain pipeline logic themselves.
- **Visualisation is standalone** — `src/visualisation/` reads only from DataFrames and
has no imports from generation or evaluation logic.
- **Fail gracefully, log loudly** — individual narrative failures (e.g. API errors after
all retries) are caught and logged without aborting the run. No silent data loss.

---

## Phase completion


| Phase | Description                                                                                 | Status                  |
| ----- | ------------------------------------------------------------------------------------------- | ----------------------- |
| 1     | Foundation — config, data loader, CSV storage                                               | Complete                |
| 1b    | Data prep refresh — drop `fnlwgt`, add `pred_proba` / `pred_label`, aggregate German one-hot | Complete                |
| 2     | Generation — LLM client, Jinja2 renderer, martens + chain-of-thought prompts, CLI              | Complete                |
| 2b    | Output formats — canonical `narratives.csv`, JSONL stream, `run_metadata.yaml`              | Complete                |
| 3     | Evaluation — faithfulness metrics (deferred; rule-based detector removed)                  | Deferred                |
| 4     | Visualisation — dataset overview, SHAP distributions; hallucination charts when eval exists | Partial                 |
| 5     | Export + tests                                                                              | Complete                |


