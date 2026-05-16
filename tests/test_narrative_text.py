"""Tests for CoT narrative stripping before evaluation."""

from src.generation.narrative_text import narrative_text_for_evaluation


class TestNarrativeTextForEvaluation:
    def test_martens_returns_full_text(self):
        raw = "Age increased income. Education helped."
        assert narrative_text_for_evaluation(raw, "martens") == raw

    def test_chain_of_thought_strips_reasoning(self):
        raw = (
            "Step 1 — Rank: age | +0.5 | toward positive\n"
            "Step 2 — Negligible: fnlwgt\n"
            "Narrative:\n"
            "Age pushed the prediction toward income above $50,000."
        )
        assert narrative_text_for_evaluation(raw, "chain_of_thought") == (
            "Age pushed the prediction toward income above $50,000."
        )

    def test_chain_of_thought_uses_last_narrative_heading(self):
        raw = (
            "Step 1 mentions Narrative: as an example.\n"
            "Narrative:\n"
            "Final prose here."
        )
        assert narrative_text_for_evaluation(raw, "chain_of_thought") == (
            "Final prose here."
        )

    def test_chain_of_thought_fallback_without_heading(self):
        raw = "Only steps, no narrative heading."
        assert narrative_text_for_evaluation(raw, "chain_of_thought") == raw
