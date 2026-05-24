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

1. Takes the Adult Income dataset from the [OpenXAI benchmark](https://github.com/AI4LIFE-GROUP/OpenXAI),
   paired with a pretrained ML model (logistic regression), per-instance SHAP values, and the
   model's own predicted probability and class label.
2. Generates 300 natural-language narratives by crossing two prompt strategies
   (Martens direct, chain-of-thought) × 150 instances using Llama 3 70B via Hugging Face.
3. Evaluates each narrative with an LLM extraction model and compares structured claims to
   ground-truth SHAP values (four-type hallucination taxonomy).
4. Optionally runs a **robustness check**: five high-temperature extractions per narrative to
   measure extraction-model agreement and flag low-reliability cases before interpreting
   hallucination flags.

---

## Hallucination taxonomy


| Type                     | What it means                                                                                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sign inversion**      | The narrative states the wrong direction of effect — e.g., says a feature pushed the prediction *up* when SHAP shows it pushed it *down*            |
| **Rank swap**           | The set of top-*k* features in the extraction (by narrative `rank`, default *k* = 3) does not match the set of top-*k* features by \|SHAP\| (order within the set is ignored) |
| **Feature fabrication** | The narrative mentions a feature that does not exist in the input at all                                                                            |
| **Omission**            | One of the top-k features by \|SHAP\| is not mentioned anywhere in the narrative                                                                      |

Relative effect-size wording (e.g. calling a large \|SHAP\| contribution "minor") is not a separate flag; top-*k* set membership is captured by **rank swap** and direction by **sign inversion** (see `src/evaluation/compare_to_shap.py`).


---

## Experimental design


| Dimension             | Values                                                                                          |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| Dataset               | Adult Income (OpenXAI benchmark); German Credit archived under `archive/german_credit/`        |
| Generation model      | Llama 3 70B (`meta-llama/Meta-Llama-3-70B-Instruct`) via Hugging Face Inference Providers       |
| Extraction model      | Mistral 7B (`mistralai/Mistral-7B-Instruct-v0.3`) via Hugging Face Inference Endpoint            |
| Prompt strategies     | `martens` (Martens et al. 2024 direct narrative), `chain_of_thought` (rank/sign steps + `Narrative:` section) |
| Instances per dataset | 150                                                                                             |
| Total narratives      | 300 (150 × 2 strategies)                                                                        |


**Martens (`narrative.j2`):** follows Martens et al. (2024), *Tell Me a Story! Narrative-Driven XAI
with Large Language Models* — task framing, predicted probability and class, SHAP table sorted
from most positive to most negative, and a fluent story focused on the most influential features.

**Chain-of-thought (`chain_of_thought.j2`):** same context and SHAP table, then structured
reasoning steps (rank top features, note negligible effects, verify top feature) before a final
`Narrative:` section. Evaluation strips the reasoning steps and scores only the narrative prose.
CoT uses `max_tokens: 1024` (strategy override); Martens uses the model default `2048`.


---

## Project structure

```
xAI-research/
├── config/
│   ├── default.yaml                  # Master config — change values here, not in code
│   └── prompts/
│       ├── narrative.j2              # Martens-style direct narrative (strategy: martens)
│       ├── chain_of_thought.j2       # CoT narrative (strategy: chain_of_thought)
│       └── extract.j2                # Evaluation extraction prompt
├── data/
│   ├── raw/                          # UCI original CSV (download_openxai_adult.py --task raw)
│   ├── adult/                        # OpenXAI cached train/test splits
│   └── processed/                    # CSVs with features + shap_* + pred_proba/pred_label
├── docs/
│   ├── adult_income_dataset.md
│   └── openxai_reference.md
├── outputs/
│   ├── generation/<run_id>/          # narratives.csv, narratives.jsonl, run_metadata.yaml (gitignored)
│   ├── evaluations/<run_id>/         # evaluations.csv, robustness.jsonl, eval_metadata.yaml
│   ├── human_labels/<run_id>/        # labels.jsonl, agreement_report.csv (human validation)
│   ├── figures/                      # datasets/ and <run_id>/ subfolders (gitignored)
│   └── xai_metrics/                  # OpenXAI PGI/PGU metrics (optional download script)
├── paper/
│   └── in_progress work/Paper_1.md
├── archive/german_credit/            # Archived dataset prep (not in main pipeline)
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── dataset_metadata.py           # Human-readable labels for categorical features in SHAP table
│   ├── generation/
│   │   ├── llm_client.py             # Hugging Face Inference client
│   │   ├── prompt_renderer.py
│   │   ├── narrative_text.py
│   │   ├── generator.py
│   │   └── exporters.py
│   ├── human_labels/
│   │   └── schema.py                 # Human label validation and conversion
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── compare_to_shap.py        # Four-type hallucination flags
│   │   ├── compare_extractions.py    # Human vs Mistral agreement metrics
│   │   ├── extraction_parser.py
│   │   ├── extraction_prompt_renderer.py
│   │   ├── exporters.py
│   │   ├── robustness.py
│   │   └── robustness_runner.py
│   ├── pipeline.py                   # Orchestrates generation → evaluation → robustness
│   ├── prompts/
│   │   └── jinja_env.py              # Shared Jinja2 Environment factory
│   ├── storage/
│   │   ├── narratives_store.py
│   │   ├── evaluations_store.py
│   │   └── record_io.py              # Generic CSV/JSONL writers
│   └── visualisation/
│       ├── dataset_overview.py
│       ├── shap_distributions.py
│       ├── hallucination_rates.py
│       ├── hallucination_analysis.py # Per-feature breakdown tables from eval notes
│       ├── heatmaps.py
│       ├── robustness_plots.py       # Reliability and low-reliability figures
│       └── export.py
├── notebooks/
│   ├── 00_data_preparation.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_narrative_inspection.ipynb
│   └── 03_results_visualisation.ipynb
├── scripts/
│   ├── prepare_data.py               # OpenXAI LR model + SHAP → data/processed/adult.csv
│   ├── download_openxai_adult.py     # Optional: raw UCI, ANN SHAP, OpenXAI metrics
│   ├── run_pipeline.py               # Full pipeline: generation → evaluation → robustness
│   ├── run_generation.py
│   ├── run_evaluation.py
│   ├── run_robustness.py
│   ├── export_results.py
│   ├── summarise_results.py          # Paper stats from an evaluation run directory
│   ├── human_extraction_ui.py        # Gradio UI for human extraction labels
│   └── compare_human_to_mistral.py   # Compare human labels vs cached Mistral extractions
├── tests/
│   ├── test_llm_client.py
│   ├── test_evaluator.py
│   ├── test_robustness.py
│   ├── test_data_loader.py
│   ├── test_generator.py
│   ├── test_pipeline.py
│   ├── test_narrative_text.py
│   ├── test_human_label_schema.py
│   ├── test_compare_extractions.py
│   └── test_visualisation.py
├── .env.example
└── requirements.txt
```

**Git ignore:** `outputs/generation/*` and `outputs/figures/*` are ignored; `outputs/evaluations/` may be committed.

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

Includes `jinja2` (prompt templates), `huggingface_hub`, and other runtime dependencies.
OpenXAI is installed separately (see step 3) and is only required for data-prep scripts.

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and set your Hugging Face token:

```
HF_TOKEN=hf_...
```

Accept the gated Llama 3 license at
[meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
before running generation.

Deploy [mistralai/Mistral-7B-Instruct-v0.3](https://ui.endpoints.huggingface.co/new?repository=mistralai%2FMistral-7B-Instruct-v0.3)
as a Hugging Face Inference Endpoint for extraction, then set the endpoint URL in
`config/default.yaml` (`models[].base_url` for `mistral-7b`) or as `HF_MISTRAL_ENDPOINT_URL` in `.env`.

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
# Prepare Adult Income (default)
python scripts/prepare_data.py

# Explicit dataset flag (Adult only)
python scripts/prepare_data.py --dataset adult

# Use the neural-network model instead of logistic regression
python scripts/prepare_data.py --model ann

# Quick smoke-test: only prepare 20 instances and print a validation summary
python scripts/prepare_data.py --n 20 --validate
```

**What the script does:**

1. Downloads the OpenXAI dataset via `ReturnLoaders` (cached under `data/adult/` on first run).
2. Loads the corresponding pretrained model from OpenXAI (`lr` by default).
3. Computes the model's predicted probability of class 1 and predicted class label per instance.
4. Runs the OpenXAI SHAP explainer over each instance.
5. Applies Adult-specific post-processing: drops `fnlwgt` (the uninformative survey
   weight column) and its SHAP twin, and substitutes raw feature values from the cached CSV.
6. Saves a CSV to `data/processed/adult.csv`.

German Credit preparation logic is archived under `archive/german_credit/`.

### Optional: extended Adult downloads

`scripts/download_openxai_adult.py` is not required for the main pipeline but supports:

```bash
python scripts/download_openxai_adult.py --task raw      # UCI CSV → data/raw/adult_original.csv
python scripts/download_openxai_adult.py --task ann      # ANN SHAP → data/processed/adult_ann.csv
python scripts/download_openxai_adult.py --task metrics  # OpenXAI metrics → outputs/xai_metrics/
python scripts/download_openxai_adult.py --task all
```

**Output CSV layout:**


| Column type           | Example names                                                                 |
| --------------------- | ----------------------------------------------------------------------------- |
| Raw feature values    | `age`, `capital_gain`, `hours_per_week`, `sex_Male`, `workclass_Private`, ... |
| Ground-truth label    | `label`                                                                       |
| Model prediction      | `pred_proba` (P(class=1)), `pred_label` (argmax class)                         |
| SHAP values           | `shap_age`, `shap_sex_Male`, `shap_marital_status_Non_Married`, ...           |

The processed Adult CSV has **11 features** (OpenXAI one-hot schema): four numeric
(`age`, `capital_gain`, `capital_loss`, `hours_per_week`) and seven binary indicators
(`sex_Male`, `workclass_Private`, `marital_status_Non_Married`, `occupation_Other`,
`relationship_Non_Husband`, `race_White`, `native_country_US`). The survey-weight column
`fnlwgt` is dropped. Categorical values in the SHAP table shown to the LLM are decoded
via `src/dataset_metadata.py` (e.g. `sex_Male: 1 [Male]`).


The `shap_` prefix must match `shap_col_prefix` in `config/default.yaml` (default: `shap_`).
The `pred_proba` and `pred_label` columns are required by the narrative prompt; if they are
missing the renderer will raise a `KeyError` asking you to re-run `prepare_data.py`.

SHAP computation on the full Adult Income test split (~9,000 instances) takes several minutes.
Run this step directly in a terminal, not as a background task.

---

## Running the full pipeline

### One command — generation, evaluation, and robustness

Runs all three stages in sequence for a single `run_id`. Evaluation reads
`outputs/generation/<run_id>/narratives.csv`; robustness merges into
`outputs/evaluations/<run_id>/` when evaluation has completed.

```bash
# Full pipeline (config/default.yaml)
python scripts/run_pipeline.py

# Dry-run: print prompts for all stages; no API calls or disk writes
python scripts/run_pipeline.py --dry-run

# Smoke test (3 instances, limits evaluation and robustness too)
python scripts/run_pipeline.py --model llama3-70b --dataset adult --n 3

# Calibrate robustness on 10% subsample after full generation + evaluation
python scripts/run_pipeline.py --subsample 0.1

# Resume evaluation + robustness on an existing generation run
python scripts/run_pipeline.py --run-id <run_id> --skip-generation
```

Skip flags: `--skip-generation` (requires `--run-id`), `--skip-evaluation`,
`--skip-robustness`. Individual stage scripts below remain available for
partial runs.

**Credentials:** generation needs `HF_TOKEN`; evaluation and robustness need
`HF_MISTRAL_ENDPOINT_URL` (or `models[].base_url` for the extraction model).

---

### Step 1 — Generate narratives

```bash
# Full run (reads all models and datasets from config/default.yaml)
python scripts/run_generation.py

# Dry-run: prints prompts to stdout; makes no API calls and writes nothing to disk
python scripts/run_generation.py --dry-run

# Restrict to one model, one dataset, one strategy, 5 randomly sampled instances
python scripts/run_generation.py --model llama3-70b --dataset adult --strategy martens --n 5

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

**Instance selection:** for each dataset, `load_dataset` draws `n_instances` rows
(without replacement) from the processed CSV using `run.seed` from config (default 42).
The `instance_id` stored in each narrative is the original row index in that CSV, so
inspection notebooks can join back to `data/processed/<dataset>.csv`. The same seed
always yields the same subset; change `run.seed` or re-run `prepare_data.py` to draw a
different set. Pass `--n` to override the sample size for quick tests.

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

### Step 3 — Evaluate narratives

```bash
# Faithfulness evaluation (Mistral 7B extraction at temperature 0.0; requires HF_MISTRAL_ENDPOINT_URL)
python scripts/run_evaluation.py --run-id <run_id>

# Dry-run: print extraction prompts only
python scripts/run_evaluation.py --run-id <run_id> --dry-run

# First N narratives only
python scripts/run_evaluation.py --run-id <run_id> --n 10
```

Outputs under `outputs/evaluations/<run_id>/`:

| File | Role |
| ---- | ---- |
| `evaluations.csv` | One row per narrative with hallucination flags |
| `evaluations.jsonl` | Same records as JSON (nested `notes`, `extraction_json`) |
| `eval_metadata.yaml` | Config snapshot and run summary |

### Step 4 — Extraction robustness (optional)

Runs the extraction model **five times** at high temperature (`0.9`) on the same narrative
and scores agreement on per-feature `sign` and `value`, plus narrative-level top-*k* **set**
agreement on importance (same rule as rank-swap evaluation). Inspired by semantic-uncertainty
work (Kuhn et al., 2023); sampling follows self-consistency (Wei et al., 2023).

```bash
# Calibrate on 10% subsample first (recommended)
python scripts/run_robustness.py --run-id <run_id> --subsample 0.1

# Full run (5× API calls per narrative — run after evaluation)
python scripts/run_robustness.py --run-id <run_id>

# Smoke test
python scripts/run_robustness.py --run-id <run_id> --n 3 --dry-run
```

Writes `robustness.jsonl` and merges a `robustness` block into `evaluations.jsonl` when
evaluation has already been run. Each record includes:

```json
"robustness": {
  "n_successful_runs": 5,
  "per_feature": {
    "age": {"sign_agreement": 1.0, "value_agreement": 1.0}
  },
  "top_k_set_agreement": 0.8,
  "narrative_reliability_score": 0.85,
  "flagged_low_reliability": false,
  "extraction_unreliable": false
}
```

Narratives with fewer than three successful parses are marked `extraction_unreliable`.
Scores below `0.8` (configurable) are `flagged_low_reliability`. Report hallucination rates
separately for high- and low-reliability extractions:

```python
from src.storage.evaluations_store import load_evaluations_csv, eval_run_dir
from src.evaluation.robustness_runner import reliability_summary

evals_df = load_evaluations_csv(eval_run_dir(cfg.evaluation.export_dir, run_id) / "evaluations.csv")
print(reliability_summary(evals_df))
```

Tune settings under `evaluation.robustness` in `config/default.yaml`.

### Step 5 — Evaluation figures and notebooks

```bash
python scripts/export_results.py --run-id <run_id> --eval-figures
```

### Step 6 — Visualise (notebooks)

- `notebooks/02_narrative_inspection.ipynb` — browse narratives from `narratives.csv`
- `notebooks/01_data_exploration.ipynb` — feature distributions and SHAP plots
- `notebooks/03_results_visualisation.ipynb` — evaluation figures, robustness plots, and hallucination breakdown tables

### Utilities — paper statistics

After evaluation (and optionally robustness), summarise rates and sample sizes for the thesis:

```bash
python scripts/summarise_results.py outputs/evaluations/<run_id>
```

Writes summary tables to stdout (used by `paper/in_progress work/Results.md`).

### Human extraction validation

Label narratives by hand (sign, rank, and unknown features only) and compare against cached Mistral extractions:

```bash
# Launch Gradio labeling UI (reads narratives.csv; SHAP hidden)
python scripts/human_extraction_ui.py --run-id <run_id> --annotator your_name

# After labeling, compare human vs Mistral (uses evaluations.csv; no API calls)
python scripts/compare_human_to_mistral.py \
  --run-id <run_id> \
  --eval-dir outputs/evaluations/<run_id> \
  --top-k 3
```

Labels are stored in `outputs/human_labels/<run_id>/labels.jsonl`. The comparison script writes `agreement_report.csv` in the same folder with per-narrative sign/rank/top-k agreement metrics.

---

## Configuration reference

Everything is controlled by `config/default.yaml`. No parameters are hardcoded in `src/`.

```yaml
run:
  name: "pilot_run"
  seed: 42

datasets:
  - name: "adult"
    path: "data/processed/adult.csv"
    shap_col_prefix: "shap_"
    n_instances: 150
    task_description: "predict whether a person's annual income exceeds $50,000, based on demographic and employment data"
    positive_class_label: "income above $50,000"
    negative_class_label: "income at or below $50,000"

models:
  - id: "llama3-70b"
    provider: "huggingface"
    model_name: "meta-llama/Meta-Llama-3-70B-Instruct"
    max_tokens: 2048
    temperature: 0.0
    generation: true
    # inference_provider: "novita"  # optional: override auto-routing

  - id: "mistral-7b"
    provider: "huggingface"
    model_name: "mistralai/Mistral-7B-Instruct-v0.3"
    max_tokens: 1024
    temperature: 0.0
    generation: false
    # base_url: "https://..."  # or set HF_MISTRAL_ENDPOINT_URL in .env

prompt:
  strategies:
    - id: martens
      template: "config/prompts/narrative.j2"
    - id: chain_of_thought
      template: "config/prompts/chain_of_thought.j2"
      max_tokens: 1024

storage:
  generation_dir: "outputs/generation/"

evaluation:
  extraction_model_id: "mistral-7b"
  template: "config/prompts/extract.j2"
  top_k_features: 3
  export_dir: "outputs/evaluations/"
  robustness:
    n_runs: 5
    temperature: 0.9
    min_successful_runs: 3
    reliability_threshold: 0.8
    subsample_fraction: 0.25    # calibration subsample (see also Pydantic defaults below)
    balanced_subsample: true   # equal Martens / chain_of_thought when subsampling
    require_successful_eval: true  # only narratives with a successful extraction
    max_workers: 5

visualisation:
  figure_dir: "outputs/figures/"
  format: "png"
  dpi: 150
```

| Setting | Meaning |
| ------- | ------- |
| `run.seed` | `load_dataset()` sampling; robustness subsample RNG |
| `datasets[].n_instances` | Instances per dataset per generation run (`--n` overrides) |
| `models[].generation` | `false` → excluded from `run_generation`; used for extraction |
| `models[].base_url` | HF Inference Endpoint; extraction models fall back to `HF_MISTRAL_ENDPOINT_URL` |
| `prompt.strategies[].max_tokens` | Per-strategy override (CoT → 1024; Martens uses model 2048) |
| `evaluation.top_k_features` | Omission check: top-k by \|SHAP\| must appear in extraction |
| `evaluation.robustness.*` | Multi-sample extraction reliability (`run_robustness.py`) |

**Config layers:** `config/default.yaml` is the source of truth for runs. If a key is omitted from YAML, Pydantic defaults in `src/config.py` apply (e.g. `subsample_fraction: 0.1`, `balanced_subsample: true`). CLI flags such as `--subsample` on `run_pipeline.py` / `run_robustness.py` override YAML for that invocation only.

---

## Source module reference

### `src/config.py`

Parses `config/default.yaml` into a fully typed `AppConfig` object (Pydantic v2).

```python
from src.config import load_config

cfg = load_config()                           # reads config/default.yaml
cfg = load_config("config/custom.yaml")       # custom path

model   = cfg.get_model("llama3-70b")         # → ModelConfig
dataset = cfg.get_dataset("adult")            # → DatasetConfig
```

---

### `src/data_loader.py`

Loads a processed CSV and provides helpers for two-part generation prompts: an
**instance profile** (feature values) and a **SHAP contributions** table (attributions only).
By default, `load_dataset` returns a reproducible random sample of `n_instances` rows
(`random_state=run.seed`). Pass `sample=False` to load the full CSV (e.g. for figures).

```python
from src.data_loader import (
    load_dataset,
    format_instance_snapshot,
    format_shap_table,
    top_k_shap_features,
    get_shap_columns,
    get_feature_columns,
)

df   = load_dataset(cfg.get_dataset("adult"), seed=cfg.run.seed)
row  = df.iloc[0]   # first row of the sampled subset; index is the CSV row number

print(format_instance_snapshot(row, prefix="shap_", dataset_name="adult"))
#   Age (years): 55
#   Hours worked per week: 51
#   Sex: Male
#   ...

print(format_shap_table(row, prefix="shap_"))
#   marital_status_Non_Married: +0.0618
#   relationship_Non_Husband: +0.0433
#   age: -0.0177

top3 = top_k_shap_features(row, prefix="shap_", k=3)
# → [("marital_status_Non_Married", 0.062), ...]  # ranked by |SHAP|

shap_cols = get_shap_columns(df, prefix="shap_")
feat_cols = get_feature_columns(df, prefix="shap_")
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

Single `generate(prompt, model_cfg)` interface via Hugging Face (`huggingface_hub.InferenceClient`).
Retry policy: up to 5 attempts with exponential back-off (`tenacity`).

```python
from src.generation import LLMClient

client = LLMClient()
text = client.generate(
    prompt="Explain this prediction.",
    model_cfg=cfg.get_model("llama3-70b"),
)
```


| `provider` value | SDK               | Required env var                          |
| ---------------- | ----------------- | ----------------------------------------- |
| `huggingface`    | `huggingface_hub` | `HF_TOKEN` (generation + Providers API)   |
|                  |                   | `HF_MISTRAL_ENDPOINT_URL` (extraction only) |


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
    filter_model="llama3-70b",    # optional: restrict to one model id
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

#### `hallucination_analysis.py` — per-feature breakdown tables

Parses evaluation `notes` and exports CSV tables (sign inversion, rank swap, omission, fabrication counts by feature) under `outputs/figures/<run_id>/analysis/`.

#### `robustness_plots.py` — extraction reliability figures

Reliability score distributions, rates by prompt strategy, and hallucination rates split by high/low reliability groups (requires robustness block in evaluations).

#### `export.py` — save figures to disk

```python
from src.visualisation.export import (
    export_dataset_figures,
    export_all_figures,
    export_evaluation_figures_complete,
)

# Dataset-level figures (no evaluation results needed)
saved = export_dataset_figures(
    df, dataset_name="adult", shap_cols=shap_cols, feature_cols=feat_cols, cfg=cfg
)
# Saves to outputs/figures/datasets/adult/

# Full evaluation export (CLI --eval-figures): rates, heatmaps, robustness, analysis tables
saved = export_evaluation_figures_complete(evals_df, cfg, run_id)

# Lower-level: hallucination rate bar charts only
saved = export_all_figures(evals_df, cfg, run_id)
```

`scripts/export_results.py --eval-figures` calls `export_evaluation_figures_complete`.

---

## Prompt templates

Three Jinja2 templates are used (`cfg.prompt.strategies` + `evaluation.template`):

| Strategy / role | File | Role |
| ----------------- | ---- | ---- |
| `martens` | `config/prompts/narrative.j2` | Martens et al. (2024) direct narrative |
| `chain_of_thought` | `config/prompts/chain_of_thought.j2` | Structured reasoning + `Narrative:` prose |
| extraction | `config/prompts/extract.j2` | Parse narrative claims to JSON (evaluation only) |

Narrative strategies share per-dataset task description and class labels from `config/default.yaml`.
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

`test_llm_client.py` mocks the Hugging Face client to verify parameter passing and
response parsing without real API calls. Eight test modules cover generation, evaluation,
robustness, data loading, pipeline orchestration, narrative text stripping, and visualisation.

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
| 1b    | Data prep refresh — drop `fnlwgt`, add `pred_proba` / `pred_label`; German Credit archived   | Complete                |
| 2     | Generation — LLM client, Jinja2 renderer, martens + chain-of-thought prompts, CLI              | Complete                |
| 2b    | Output formats — canonical `narratives.csv`, JSONL stream, `run_metadata.yaml`              | Complete                |
| 3     | Evaluation — LLM extraction + four-type SHAP comparison                                       | Complete                |
| 3b    | Robustness — multi-sample extraction agreement (optional calibration subsample)            | Complete                |
| 4     | Visualisation — dataset overview, SHAP distributions, hallucination charts, robustness plots, analysis tables | Available after eval   |
| 5     | Export + tests                                                                              | Complete                |


