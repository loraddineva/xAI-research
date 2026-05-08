# CLAUDE.md — XAI Hallucination Detection Project

## Goal

This project investigates whether LLMs can faithfully translate pre-computed SHAP values into natural-language narratives, and whether failures can be detected and classified automatically. It generates ~3,600 narratives across three LLMs (Claude Opus, Llama 3 70B, Mistral 7B), three prompt strategies (zero-shot, few-shot, chain-of-thought), and two tabular datasets (Adult Income, German Credit) drawn from the OpenXAI benchmark. Each narrative is evaluated against its ground-truth SHAP values using a five-type hallucination taxonomy: sign inversion, rank swap, feature fabrication, magnitude distortion, and omission. This is Paper 1 of a three-paper PhD thesis on the nature, causes, and human consequences of faithfulness failures in LLM-generated XAI.

---

## Tech Stack

- **Language:** Python 3.11+
- **Notebooks:** Jupyter (exploration, visualisation)
- **Scripts:** Plain Python (pipeline execution, evaluation)
- **Storage:** SQLite (structured results, queryable) + CSV/JSON (per-run exports, reproducibility)
- **Config:** YAML files via `pydantic-settings` — one file controls everything
- **LLM APIs:** Anthropic, Mistral, Together AI (Llama 3 70B); local (Ollama/vLLM) is a separate optional task
- **Key libraries:** `anthropic`, `mistralai`, `together`, `shap`, `pandas`, `matplotlib`, `seaborn`, `sqlite3`, `pydantic`, `tenacity` (retries), `tqdm`

---

## Project Structure

```
xai-hallucination/
├── CLAUDE.md
├── README.md
├── config/
│   ├── default.yaml          # Master config — edit this to change a run
│   └── prompts/
│       ├── zero_shot.txt
│       ├── few_shot.txt
│       └── chain_of_thought.txt
├── data/
│   ├── raw/                  # OpenXAI datasets as downloaded
│   └── processed/            # Cleaned dataframes with SHAP values attached
├── outputs/
│   ├── narratives/           # JSON files, one per run
│   ├── evaluations/          # CSV exports of hallucination labels
│   └── figures/              # Plots from visualisation module
├── src/
│   ├── config.py             # Pydantic settings — loads default.yaml
│   ├── data_loader.py        # Loads OpenXAI datasets + SHAP values
│   ├── llm_client.py         # Unified client — swaps provider via config
│   ├── narrative_generator.py # Calls LLM, saves raw output
│   ├── evaluator.py          # Hallucination detection logic (rule-based + LLM judge)
│   ├── db.py                 # SQLite read/write helpers
│   └── visualisation/
│       ├── __init__.py
│       ├── hallucination_rates.py   # Bar charts by type / model / prompt
│       ├── heatmaps.py              # Model × prompt strategy heatmaps
│       └── export.py                # Save figures to outputs/figures/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_narrative_inspection.ipynb
│   └── 03_results_visualisation.ipynb
├── scripts/
│   ├── run_generation.py     # Entry point: generate all narratives for a config
│   ├── run_evaluation.py     # Entry point: evaluate an existing narrative batch
│   └── export_results.py     # Dump DB → CSV for a given run_id
├── tests/
│   ├── test_evaluator.py
│   └── test_llm_client.py
├── .env.example
└── requirements.txt
```

---

## Configuration

**Everything is controlled by `config/default.yaml`.** Change values there rather than in code. Scripts read this file via `src/config.py` on startup.

```yaml
# config/default.yaml

run:
  name: "pilot_run"           # Stored in DB; used to group results
  seed: 42

datasets:
  - name: "adult"
    path: "data/processed/adult.csv"
    shap_col_prefix: "shap_"  # Columns named shap_age, shap_income, etc.
    n_instances: 100
  - name: "german_credit"
    path: "data/processed/german_credit.csv"
    shap_col_prefix: "shap_"
    n_instances: 100

models:
  - id: "claude-opus"
    provider: "anthropic"
    model_name: "claude-opus-4-6"
    max_tokens: 512
    temperature: 0.0
  - id: "llama3-70b"
    provider: "together"
    model_name: "meta-llama/Llama-3-70b-chat-hf"
    max_tokens: 512
    temperature: 0.0
  - id: "mistral-7b"
    provider: "mistral"
    model_name: "mistral-small-latest"
    max_tokens: 512
    temperature: 0.0
  # Local model (uncomment when local deployment is set up)
  # - id: "llama3-local"
  #   provider: "ollama"
  #   model_name: "llama3:70b"
  #   base_url: "http://localhost:11434"

prompts:
  strategies:
    - zero_shot
    - few_shot
    - chain_of_thought
  template_dir: "config/prompts/"

evaluation:
  top_k_features: 3            # How many top SHAP features to check rank/omission against
  magnitude_threshold: 0.5     # Relative threshold for magnitude distortion detection
  use_llm_judge: false         # Set true to run a second-pass LLM judge on borderline cases
  llm_judge_model: "claude-opus-4-6"

storage:
  db_path: "outputs/results.db"
  export_dir: "outputs/evaluations/"
  narrative_dir: "outputs/narratives/"

visualisation:
  figure_dir: "outputs/figures/"
  format: "png"                 # png or pdf
  dpi: 150
```

API keys go in `.env` (never committed):

```
ANTHROPIC_API_KEY=...
TOGETHER_API_KEY=...
MISTRAL_API_KEY=...
```

---

## Initial Plan

### Phase 1 — Foundation
1. Set up repo structure and `requirements.txt`
2. Implement `src/config.py` — Pydantic model that parses `default.yaml`
3. Implement `src/data_loader.py` — load OpenXAI CSVs, attach SHAP columns, return a clean dataframe per dataset
4. Implement `src/db.py` — create schema, write/read helpers for runs, narratives, evaluations

### Phase 2 — Generation
5. Implement `src/llm_client.py` — single `generate(prompt, model_cfg)` interface that dispatches to Anthropic / Together / Mistral / Ollama based on `provider` field in config
6. Write prompt templates in `config/prompts/`
7. Implement `src/narrative_generator.py` — iterates dataset × model × prompt, calls client, saves JSON to `outputs/narratives/` and writes to DB
8. Implement `scripts/run_generation.py` — CLI entry point, accepts `--config` and optional `--dry-run`

### Phase 3 — Evaluation
9. Implement `src/evaluator.py` — rule-based checks for each of the five hallucination types against ground-truth SHAP values
10. Add optional LLM-judge pass for ambiguous cases (controlled by `use_llm_judge` in config)
11. Implement `scripts/run_evaluation.py` — loads narratives by `run_id`, runs evaluator, writes labels to DB and CSV

### Phase 4 — Visualisation
12. Implement `src/visualisation/hallucination_rates.py` — bar charts of hallucination type frequency by model and by prompt strategy
13. Implement `src/visualisation/heatmaps.py` — model × prompt strategy heatmap of overall hallucination rate per dataset
14. Implement `src/visualisation/export.py` — save all figures from a run to `outputs/figures/`
15. Wire visualisation into `notebooks/03_results_visualisation.ipynb`

### Phase 5 — Export and tests
16. Implement `scripts/export_results.py` — dump a run from DB to CSV for sharing / archiving
17. Write `tests/test_evaluator.py` with handcrafted cases for each hallucination type
18. Write `tests/test_llm_client.py` with mocked API responses

---

## Database Schema (SQLite)

```sql
-- One row per experimental run (a full config execution)
CREATE TABLE runs (
    run_id      TEXT PRIMARY KEY,
    run_name    TEXT,
    config_json TEXT,          -- Full config snapshot for reproducibility
    created_at  TEXT
);

-- One row per generated narrative
CREATE TABLE narratives (
    narrative_id  TEXT PRIMARY KEY,
    run_id        TEXT REFERENCES runs(run_id),
    dataset       TEXT,
    instance_id   INTEGER,
    model_id      TEXT,
    prompt_strategy TEXT,
    narrative_text  TEXT,
    created_at    TEXT
);

-- One row per evaluated narrative (one-to-one with narratives)
CREATE TABLE evaluations (
    eval_id          TEXT PRIMARY KEY,
    narrative_id     TEXT REFERENCES narratives(narrative_id),
    sign_inversion   INTEGER,   -- Boolean (0/1)
    rank_swap        INTEGER,
    feature_fabrication INTEGER,
    magnitude_distortion INTEGER,
    omission         INTEGER,
    any_hallucination INTEGER,  -- Derived: 1 if any of above is 1
    notes            TEXT,      -- Optional: which feature caused the flag
    evaluated_at     TEXT
);
```

---

## Hallucination Types (reference)

| Type | What it means | How it is detected |
|---|---|---|
| **Sign inversion** | Narrative states wrong direction of effect | Compare stated direction against sign of SHAP value |
| **Rank swap** | Non-top feature described as most important | Compare stated importance rank against SHAP rank |
| **Feature fabrication** | Narrative mentions a feature not in the input | Check narrative tokens against feature name list |
| **Magnitude distortion** | Large effect called minor or vice versa | Compare relative SHAP magnitude against stated magnitude words |
| **Omission** | Top-ranked SHAP feature not mentioned at all | Check that top-k SHAP features appear in narrative |

---

## Conventions

- **Config over code** — if a parameter could plausibly change between runs (model, threshold, n_instances), it lives in `default.yaml`, not in source
- **One run = one config snapshot** — every run stores the full config as JSON in the `runs` table so results are always reproducible
- **Scripts are thin** — scripts in `scripts/` are entry points only; all logic lives in `src/`
- **No hardcoded paths** — all paths resolve through the config object
- **Notebooks for exploration, scripts for execution** — do not run the full pipeline from a notebook
- **Local model support is a separate task** — the `ollama` provider branch in `llm_client.py` can be stubbed initially and completed independently without touching anything else
- **Visualisation is a standalone module** — `src/visualisation/` has no dependencies on generation or evaluation logic; it reads only from the DB and produces figures
- **Prefer explicit over clever** — this is research code; clarity and auditability matter more than elegance

---

## Running the Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in API keys
cp .env.example .env

# 3. Generate narratives (reads config/default.yaml)
python scripts/run_generation.py

# 4. Evaluate the run just created
python scripts/run_evaluation.py --run-id <run_id>

# 5. Export results to CSV
python scripts/export_results.py --run-id <run_id>

# 6. Produce figures
python scripts/export_results.py --run-id <run_id> --figures
```

To test a single model and dataset before a full run:

```bash
python scripts/run_generation.py --dry-run --model claude-opus --dataset adult --n 5
```
