# xAI Hallucination Detection

Investigates whether LLMs faithfully translate pre-computed SHAP values into natural-language explanations, and whether failures can be detected and classified automatically.

This is Paper 1 of a three-paper PhD thesis on faithfulness failures in LLM-generated XAI.

---

## What it does

The pipeline generates ~3,600 narratives by crossing:

- **3 LLMs** — Claude Opus, Llama 3 70B, Mistral 7B
- **3 prompt strategies** — zero-shot, few-shot, chain-of-thought
- **2 datasets** — Adult Income, German Credit (OpenXAI benchmark)
- **100 instances per dataset**

Each narrative is then evaluated against its ground-truth SHAP values for five hallucination types:

| Type | Description |
|---|---|
| Sign inversion | Narrative states the wrong direction of a feature's effect |
| Rank swap | A non-top feature is described as most important |
| Feature fabrication | Narrative mentions a feature not present in the SHAP input |
| Magnitude distortion | Large effect called minor or vice versa |
| Omission | A top-ranked SHAP feature is not mentioned at all |

---

## Project structure

```
xai-hallucination/
├── config/
│   ├── default.yaml          # Master config — change values here, not in code
│   └── prompts/
│       ├── zero_shot.txt
│       ├── few_shot.txt
│       └── chain_of_thought.txt
├── data/
│   ├── raw/                  # OpenXAI datasets as downloaded
│   └── processed/            # CSVs with SHAP columns attached (shap_<feature>)
├── outputs/
│   ├── narratives/           # JSON exports, one file per run
│   ├── evaluations/          # CSV exports of hallucination labels
│   └── figures/              # Plots produced by the visualisation module
├── src/
│   ├── config.py             # Pydantic AppConfig — loads default.yaml
│   ├── data_loader.py        # Loads CSVs, formats SHAP tables for prompts
│   ├── db.py                 # SQLite schema + read/write helpers
│   ├── llm_client.py         # (Phase 2) Unified LLM client
│   ├── narrative_generator.py# (Phase 2) Calls LLM, saves output
│   ├── evaluator.py          # (Phase 3) Hallucination detection logic
│   └── visualisation/        # (Phase 4) Charts and heatmaps
├── scripts/
│   ├── run_generation.py     # (Phase 2) Entry point: generate narratives
│   ├── run_evaluation.py     # (Phase 3) Entry point: evaluate a batch
│   └── export_results.py     # (Phase 5) Dump DB → CSV
├── tests/
│   ├── test_evaluator.py     # (Phase 5)
│   └── test_llm_client.py    # (Phase 5)
├── .env.example
└── requirements.txt
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in API keys
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY, TOGETHER_API_KEY, MISTRAL_API_KEY
```

Place processed CSVs (with `shap_<feature>` columns) in `data/processed/` before running generation.

---

## Configuration

Everything is controlled by `config/default.yaml`. Key sections:

```yaml
run:
  name: "pilot_run"   # groups results in the DB
  seed: 42

datasets:             # which CSVs to load and how many rows
models:               # which LLMs to call (provider + model name)
prompts:              # which strategies to use
evaluation:
  top_k_features: 3           # features checked for rank/omission
  magnitude_threshold: 0.5    # relative threshold for magnitude distortion
  use_llm_judge: false        # enable second-pass LLM judge on borderline cases
storage:
  db_path: "outputs/results.db"
```

---

## Running the pipeline

```bash
# Generate narratives (reads config/default.yaml)
python scripts/run_generation.py

# Dry-run: single model + dataset, 5 instances
python scripts/run_generation.py --dry-run --model claude-opus --dataset adult --n 5

# Evaluate the run just created
python scripts/run_evaluation.py --run-id <run_id>

# Export results to CSV
python scripts/export_results.py --run-id <run_id>
```

---

## Source modules (Phase 1 — implemented)

### `src/config.py`
Loads `config/default.yaml` into a typed `AppConfig` object using pydantic-settings.

```python
from src.config import load_config
cfg = load_config()                        # reads config/default.yaml
template = cfg.load_prompt_template("zero_shot")
model_cfg = cfg.get_model("claude-opus")
```

### `src/data_loader.py`
Loads a processed CSV, validates SHAP columns, and provides formatting helpers.

```python
from src.data_loader import load_dataset, format_shap_table, top_k_shap_features

df = load_dataset(cfg.get_dataset("adult"))   # returns first n_instances rows
row = df.iloc[0]
print(format_shap_table(row, prefix="shap_")) # ready to inject into a prompt
top3 = top_k_shap_features(row, "shap_", k=3)# [(feature, shap_val), ...]
```

### `src/db.py`
SQLite helpers for all three tables (`runs`, `narratives`, `evaluations`).

```python
from src.db import init_db, db_connection, insert_run, insert_narrative

init_db("outputs/results.db")              # create schema (idempotent)

with db_connection("outputs/results.db") as conn:
    insert_run(conn, run_id="abc", run_name="pilot", config_json=cfg_dict, created_at="...")
    insert_narrative(conn, narrative_id="n1", run_id="abc", ...)
```

---

## Database schema

```sql
CREATE TABLE runs (
    run_id      TEXT PRIMARY KEY,
    run_name    TEXT,
    config_json TEXT,   -- full config snapshot for reproducibility
    created_at  TEXT
);

CREATE TABLE narratives (
    narrative_id    TEXT PRIMARY KEY,
    run_id          TEXT REFERENCES runs(run_id),
    dataset         TEXT,
    instance_id     INTEGER,
    model_id        TEXT,
    prompt_strategy TEXT,
    narrative_text  TEXT,
    created_at      TEXT
);

CREATE TABLE evaluations (
    eval_id              TEXT PRIMARY KEY,
    narrative_id         TEXT REFERENCES narratives(narrative_id),
    sign_inversion       INTEGER,
    rank_swap            INTEGER,
    feature_fabrication  INTEGER,
    magnitude_distortion INTEGER,
    omission             INTEGER,
    any_hallucination    INTEGER,
    notes                TEXT,
    evaluated_at         TEXT
);
```

---

## Build status

| Phase | Description | Status |
|---|---|---|
| 1 | Foundation — config, data loader, DB | Complete |
| 2 | Generation — LLM client, narrative generator, run script | Pending |
| 3 | Evaluation — rule-based hallucination detector, LLM judge | Pending |
| 4 | Visualisation — bar charts, heatmaps, figure export | Pending |
| 5 | Export + tests | Pending |
