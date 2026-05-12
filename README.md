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
  (Adult Income and German Credit), each paired with a pretrained ML model (logistic regression)
   and per-instance SHAP values.
2. Generates approximately 3,600 natural-language narratives by crossing three LLMs × three prompt
  strategies × two datasets × 100 instances per dataset.
3. Automatically evaluates each narrative against its ground-truth SHAP values using five
  hallucination types.

---

## Hallucination taxonomy


| Type                     | What it means                                                                                                                                       | How it is detected                                                                                                           |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Sign inversion**       | The narrative states the wrong direction of effect — e.g., says a feature pushed the prediction *up* when SHAP shows it pushed it *down*            | Direction words (increase/decrease etc.) in a context window around each feature mention are compared to the SHAP sign       |
| **Rank swap**            | A non-top feature is described with superlatives ("most important", "primary driver") that should only apply to the feature with the highest |SHAP| | Superlative phrases are located in the text; the nearest feature name is compared to the true top-ranked feature             |
| **Feature fabrication**  | The narrative mentions a feature that does not exist in the input at all                                                                            | Underscore-joined tokens in the narrative are checked against the dataset's actual feature list                              |
| **Magnitude distortion** | A feature with large |SHAP| is described as minor, or a small-effect feature is described as major                                                  | Each feature's |SHAP| is normalised by the instance maximum and compared against magnitude-signalling words near its mention |
| **Omission**             | One of the top-k features by |SHAP| is not mentioned anywhere in the narrative                                                                      | Each of the top-k feature names (and their normalised variants) is searched in the narrative text                            |


---

## Experimental design


| Dimension             | Values                                                                       |
| --------------------- | ---------------------------------------------------------------------------- |
| Datasets              | Adult Income, German Credit (OpenXAI benchmark)                              |
| Models                | Claude Sonnet, Llama 3 70B (via Together AI), Mistral Small (via Mistral AI) |
| Prompt strategies     | Zero-shot, few-shot, chain-of-thought                                        |
| Instances per dataset | 100                                                                          |
| Total narratives      | ~1, 800                                                                      |


---

## Project structure

```
xai-hallucination/
├── config/
│   ├── default.yaml                  # Master config — change values here, not in code
│   └── prompts/
│       ├── zero_shot.txt             # Zero-shot prompt template (Jinja2)
│       ├── few_shot.txt              # Two-example few-shot template (Jinja2)
│       └── chain_of_thought.txt      # Step-by-step CoT template (Jinja2)
├── data/
│   ├── raw/                          # OpenXAI datasets as downloaded (auto-created)
│   └── processed/                    # CSVs with feature + shap_<feature> columns
├── outputs/
│   ├── narratives/                   # JSONL file per run: prompt + narrative text
│   ├── evaluations/                  # CSV exports of hallucination labels per run
│   └── figures/                      # Plots — one subfolder per run or dataset
├── src/
│   ├── config.py                     # Pydantic AppConfig — parses default.yaml
│   ├── data_loader.py                # Loads CSVs; formats SHAP tables for prompts
│   ├── db.py                         # SQLite schema + read/write helpers
│   ├── llm_client.py                 # Unified LLM client (Anthropic/Together/Mistral/Ollama)
│   ├── prompt_renderer.py            # Jinja2-based prompt template renderer
│   ├── narrative_generator.py        # Orchestrates dataset × model × prompt loop
│   ├── evaluator.py                  # Rule-based hallucination detector + optional LLM judge
│   └── visualisation/
│       ├── dataset_overview.py       # Feature distributions, class balance, correlation heatmap
│       ├── shap_distributions.py     # SHAP importance bar, beeswarm, scatter plots
│       ├── hallucination_rates.py    # Bar charts by type / model / strategy / dataset
│       ├── heatmaps.py               # Model × prompt strategy heatmaps
│       └── export.py                 # Save all figures for a run or dataset to disk
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Feature distributions + SHAP visualisations
│   ├── 02_narrative_inspection.ipynb # Browse generated narratives alongside SHAP values
│   └── 03_results_visualisation.ipynb# Load evaluation results and produce all paper figures
├── scripts/
│   ├── prepare_data.py               # Download OpenXAI data, compute SHAP values, save CSVs
│   ├── run_generation.py             # CLI: generate narratives for a run
│   ├── run_evaluation.py             # CLI: evaluate a completed generation run
│   └── export_results.py             # CLI: export DB → CSV (+ optional figures)
├── tests/
│   ├── test_evaluator.py             # 20 handcrafted cases for all five hallucination types
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
3. Runs OpenXAI's SHAP explainer (`SHAPExplainerC`) over each instance.
4. Saves a CSV to `data/processed/` with one row per instance, containing both
  raw feature columns and matching `shap_<feature>` columns.

**Output CSV layout:**


| Column type        | Example names                                                |
| ------------------ | ------------------------------------------------------------ |
| Raw feature values | `age`, `education_num`, `hours_per_week`, ...                |
| Ground-truth label | `label`                                                      |
| SHAP values        | `shap_age`, `shap_education_num`, `shap_hours_per_week`, ... |


The `shap_` prefix must match `shap_col_prefix` in `config/default.yaml` (default: `shap_`).

SHAP computation on the full Adult Income test split (~9,000 instances) takes several minutes.
Run this step directly in a terminal, not as a background task.

---

## Running the full pipeline

### Step 1 — Generate narratives

```bash
# Full run (reads all models, datasets, and strategies from config/default.yaml)
python scripts/run_generation.py

# Dry-run: prints prompts to stdout; makes no API calls and writes nothing to DB
python scripts/run_generation.py --dry-run

# Restrict to one model and one dataset, 5 instances (useful for initial testing)
python scripts/run_generation.py --model claude-opus --dataset adult --n 5

# Custom config file
python scripts/run_generation.py --config config/my_config.yaml
```

The script prints the `run_id` (e.g. `pilot_run_20260510T141023_a3f7c2`) on completion.
Each narrative is written to:

- The `narratives` table in `outputs/results.db`
- A JSONL file at `outputs/narratives/<run_id>.jsonl` (contains the prompt and response)

**Resume behaviour:** if the script is interrupted, restarting it creates a new run.
Within a single run, already-generated narratives are detected by checking the DB and
skipped — so a run that crashes mid-way can be relaunched from the same `run_id` by
passing `--resume-run-id <run_id>` (planned CLI addition; currently, restart = new run_id).

### Step 2 — Evaluate

```bash
python scripts/run_evaluation.py --run-id <run_id>

# Force the LLM judge on all narratives (ignores the use_llm_judge config setting)
python scripts/run_evaluation.py --run-id <run_id> --llm-judge
```

The evaluator applies five rule-based checks to each narrative. For chain-of-thought
narratives, it automatically strips the reasoning steps and evaluates only the final
`Narrative:` section — this prevents false positives from reasoning text that is not
part of the model's stated claim.

Results are written to:

- The `evaluations` table in `outputs/results.db`
- `outputs/evaluations/<run_id>_evaluations.csv`

### Step 3 — Export

```bash
# Export narratives + evaluations to CSV files
python scripts/export_results.py --run-id <run_id>

# Also produce and save all visualisation figures
python scripts/export_results.py --run-id <run_id> --figures
```

Figures are saved to `outputs/figures/<run_id>/`.

### Step 4 — Visualise (notebooks)

Open `notebooks/03_results_visualisation.ipynb`, set `RUN_ID` to your run, and run all cells.
The notebook loads the evaluation CSV and renders all charts inline, then calls
`export_all_figures()` to save publication-ready PNG/PDF files.

For dataset-level exploration (feature distributions, SHAP plots), use
`notebooks/01_data_exploration.ipynb`.

---

## Configuration reference

Everything is controlled by `config/default.yaml`. No parameters are hardcoded in `src/`.

```yaml
run:
  name: "pilot_run"        # Human-readable label; stored in DB with every run
  seed: 42                 # Reserved for reproducibility hooks

datasets:
  - name: "adult"
    path: "data/processed/adult.csv"
    shap_col_prefix: "shap_"   # SHAP columns are named shap_<feature>
    n_instances: 100            # Number of rows to use from this dataset

models:
  - id: "claude-opus"           # Short id used in CLI flags and DB records
    provider: "anthropic"       # anthropic | together | mistral | ollama
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
  top_k_features: 3             # Features checked for omission and rank swap
  magnitude_threshold: 0.5      # Features with |SHAP| > threshold × max|SHAP| are "large"
  use_llm_judge: false          # Enable second-pass LLM judge on all narratives
  llm_judge_model: "claude-opus" # Model id (must match one of models[].id above)

storage:
  db_path: "outputs/results.db"
  export_dir: "outputs/evaluations/"
  narrative_dir: "outputs/narratives/"

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
# Output (sorted by |SHAP| descending, one feature per line):
#   age: +0.4200 (feature value: 52)
#   education_num: +0.3100 (feature value: 13)
#   hours_per_week: +0.1800 (feature value: 45)
#   capital_gain: -0.0500 (feature value: 0)

top3 = top_k_shap_features(row, prefix="shap_", k=3)
# → [("age", 0.42), ("education_num", 0.31), ("hours_per_week", 0.18)]

shap_cols = get_shap_columns(df, prefix="shap_")    # ["shap_age", "shap_education_num", ...]
feat_cols  = get_feature_columns(df, prefix="shap_") # ["age", "education_num", ...]
```

---

### `src/prompt_renderer.py`

Renders prompt templates using Jinja2. Templates live in `config/prompts/` and use
`{{ dataset }}` and `{{ shap_table }}` as placeholders.

```python
from src.prompt_renderer import PromptRenderer

renderer = PromptRenderer(cfg)
prompt = renderer.render(
    strategy="zero_shot",
    dataset_name="adult",
    row=df.iloc[0],
    shap_prefix="shap_",
)
```

`StrictUndefined` is set on the Jinja2 environment, so a typo in a template variable
raises an error immediately rather than silently rendering an empty string.

---

### `src/db.py`

Creates and manages the three-table SQLite schema. Uses WAL journal mode and enforces
foreign keys. The schema stores a full config snapshot with every run so results are
always reproducible.

```python
from src.db import (
    init_db, db_connection,
    insert_run, insert_narrative, insert_evaluation,
    get_run, get_narrative, get_narratives_for_run,
    get_evaluations_for_run, list_runs, narrative_exists,
)

init_db("outputs/results.db")    # creates schema (idempotent — safe to call repeatedly)

with db_connection("outputs/results.db") as conn:

    # Write
    insert_run(conn, run_id="abc123", run_name="pilot", config_json={...}, created_at="...")
    insert_narrative(conn, narrative_id="n1", run_id="abc123", dataset="adult",
                     instance_id=0, model_id="claude-opus", prompt_strategy="zero_shot",
                     narrative_text="...", created_at="...")
    insert_evaluation(conn, eval_id="e1", narrative_id="n1",
                      sign_inversion=False, rank_swap=True, feature_fabrication=False,
                      magnitude_distortion=False, omission=False,
                      notes="rank_swap: ...", evaluated_at="...")

    # Read
    narratives  = get_narratives_for_run(conn, "abc123")   # List[dict]
    evaluations = get_evaluations_for_run(conn, "abc123")  # List[dict] (joined with narrative cols)
    run_meta    = get_run(conn, "abc123")                  # dict | None
    all_runs    = list_runs(conn)                          # List[dict], most recent first

    # Resume check — True if this combination was already generated
    exists = narrative_exists(conn, "abc123", "adult", 0, "claude-opus", "zero_shot")
```

`get_evaluations_for_run` returns rows joined with `narratives`, so each dict includes
`dataset`, `instance_id`, `model_id`, and `prompt_strategy` alongside the five hallucination
flags — suitable for loading directly into a pandas DataFrame.

---

### `src/llm_client.py`

Single `generate(prompt, model_cfg)` interface dispatching to the correct provider.
All providers share the same retry policy: up to 5 attempts with exponential back-off,
using `tenacity`. The provider SDK client is instantiated once per `generate()` call;
retries reuse the same client object.

```python
from src.llm_client import LLMClient

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

### `src/narrative_generator.py`

Iterates every `(dataset, model, prompt_strategy, instance)` combination, calls the LLM,
and persists each result to the DB and a JSONL file. A single DB connection is held open
for the entire run to avoid per-narrative connection overhead.

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

Individual API failures on a single instance are caught and logged without aborting the run.

---

### `src/evaluator.py`

Applies five independent rule-based checks to a single narrative given its ground-truth
SHAP values. Returns an `EvaluationResult` with per-type boolean flags and a `notes` list
explaining each flag.

```python
from src.evaluator import evaluate_narrative, llm_judge, EvaluationResult
from src.config import EvaluationConfig

shap_values = {"age": 0.42, "education_num": 0.31, "hours_per_week": 0.18, "capital_gain": -0.05}
cfg_eval    = EvaluationConfig(top_k_features=3, magnitude_threshold=0.5)

result = evaluate_narrative(
    narrative="Age was the most important factor, increasing the predicted income.",
    shap_values=shap_values,
    cfg=cfg_eval,
    all_dataset_features=["age", "education_num", "hours_per_week", "capital_gain", "sex"],
)

result.any_hallucination    # True / False
result.sign_inversion       # True / False
result.rank_swap            # True / False
result.feature_fabrication  # True / False
result.magnitude_distortion # True / False
result.omission             # True / False
result.notes_str()          # "rank_swap: superlative 'most important' near 'age' ..."
result.to_dict()            # {"sign_inversion": 0, "rank_swap": 0, ..., "any_hallucination": 0}
```

`all_dataset_features` (optional) is the full feature list of the dataset. When provided,
the fabrication check only flags tokens that are not in the dataset at all — rather than
flagging any token not in the per-instance SHAP dict (which would generate false positives
for features with zero SHAP contribution on that instance).

**LLM judge** — a second-pass evaluation using an LLM as judge, enabled by setting
`use_llm_judge: true` in config or passing `--llm-judge` to `run_evaluation.py`. The judge
receives the SHAP values and narrative and returns a structured YES/NO verdict for each of
the five hallucination types. Rule-based and judge verdicts are merged with logical OR:
a narrative is flagged if either pass raises an alarm.

**Important:** for chain-of-thought narratives, `run_evaluation.py` automatically extracts
only the text after the `Narrative:` heading before running either evaluation pass. This
prevents the step-by-step reasoning text from being misread as a factual claim.

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

Templates live in `config/prompts/` and use Jinja2 syntax (`{{ variable }}`).
Two variables are available in every template:


| Variable           | Content                                                              |
| ------------------ | -------------------------------------------------------------------- |
| `{{ dataset }}`    | Dataset name (e.g. `"adult"`)                                        |
| `{{ shap_table }}` | SHAP table string, sorted by |SHAP| descending, one feature per line |



| File                   | Strategy         | What it does                                                                                                                                                 |
| ---------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `zero_shot.txt`        | Zero-shot        | Instructs the model directly; lists the three faithfulness requirements (direction, rank, magnitude)                                                         |
| `few_shot.txt`         | Few-shot         | Provides two worked examples (one per dataset) before asking for the target narrative                                                                        |
| `chain_of_thought.txt` | Chain-of-thought | Asks the model to explicitly rank features, note their directions, and identify negligible effects before writing the narrative under a `Narrative:` heading |


---

## Running the tests

```bash
pytest tests/ -v
```

`test_evaluator.py` contains 20 handcrafted test cases — both clean narratives that should
not be flagged, and narratives with deliberate errors targeting each hallucination type.
Cases test edge conditions such as direction words that appear in an ambiguous context,
variant feature names (e.g. `education num` instead of `education_num`), and features
with small vs large relative SHAP magnitude.

`test_llm_client.py` mocks all four provider SDKs to verify dispatch logic, parameter
passing, and response parsing without making any real API calls.

---

## Design principles

- **Config over code** — every tunable parameter lives in `config/default.yaml`. No value
that could plausibly change between runs is hardcoded in `src/`.
- **One run = one config snapshot** — the full config is serialised as JSON into the `runs`
table at generation time, so results from any run can always be reproduced from the DB alone.
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


| Phase | Description                                                                          | Status   |
| ----- | ------------------------------------------------------------------------------------ | -------- |
| 1     | Foundation — config, data loader, SQLite schema                                      | Complete |
| 2     | Generation — LLM client, Jinja2 renderer, narrative generator, CLI                   | Complete |
| 3     | Evaluation — rule-based detector, CoT stripping, LLM judge, CSV export               | Complete |
| 4     | Visualisation — dataset overview, SHAP distributions, hallucination charts, heatmaps | Complete |
| 5     | Export + tests                                                                       | Complete |


