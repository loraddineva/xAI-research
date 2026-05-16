"""
src/generation
Narrative-generation subpackage.

Public surface:
    LLMClient                — provider-agnostic text generation.
    PromptRenderer           — Jinja2 renderer for the single Martens-style prompt.
    run_generation           — CLI-facing entry point that produces narratives
                               and writes them to outputs/generation/<run_id>/.
    write_jsonl, write_run_json, write_xlsx — output writers.
"""

from src.generation.exporters import write_jsonl, write_run_json, write_xlsx
from src.generation.generator import run_generation
from src.generation.llm_client import LLMClient
from src.generation.prompt_renderer import PromptRenderer

__all__ = [
    "LLMClient",
    "PromptRenderer",
    "run_generation",
    "write_jsonl",
    "write_run_json",
    "write_xlsx",
]
