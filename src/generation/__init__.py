"""
src/generation
Narrative-generation subpackage.

Public surface:
    LLMClient       — provider-agnostic text generation.
    PromptRenderer  — Jinja2 renderer for martens and chain-of-thought prompts.
    run_generation  — CLI-facing entry point; writes CSV + JSONL per run.
"""

from src.generation.exporters import (
    CSV_COLUMNS,
    NarrativeRecord,
    append_csv_row,
    append_jsonl,
    write_csv,
)
from src.generation.generator import run_generation
from src.generation.llm_client import LLMClient
from src.generation.prompt_renderer import PromptRenderer

__all__ = [
    "CSV_COLUMNS",
    "LLMClient",
    "NarrativeRecord",
    "PromptRenderer",
    "append_csv_row",
    "append_jsonl",
    "run_generation",
    "write_csv",
]
