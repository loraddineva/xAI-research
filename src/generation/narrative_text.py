"""
src/generation/narrative_text.py
Extract the narrative prose segment used for faithfulness evaluation.
"""

from __future__ import annotations

_NARRATIVE_HEADINGS = ("Narrative:", "NARRATIVE:")


def narrative_text_for_evaluation(raw: str, prompt_strategy: str) -> str:
    """
    Return the text passed to the extraction model.

    For chain-of-thought runs, only the content under the final ``Narrative:``
    heading is evaluated; reasoning steps are stripped.
    """
    text = raw.strip()
    if prompt_strategy != "chain_of_thought":
        return text

    for marker in _NARRATIVE_HEADINGS:
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx + len(marker) :].strip()

    return text
