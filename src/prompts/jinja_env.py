"""
src/prompts/jinja_env.py
Shared Jinja2 environment factory for prompt templates.
"""

from __future__ import annotations

from typing import Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def make_jinja_env(template_dirs: Sequence[str]) -> Environment:
    return Environment(
        loader=FileSystemLoader(sorted(template_dirs)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
