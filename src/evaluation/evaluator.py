"""
src/evaluation/evaluator.py
Rule-based hallucination detection for LLM-generated SHAP narratives.

Five hallucination types (from the taxonomy in CLAUDE.md):

    1. sign_inversion      — Narrative states the wrong direction of a feature's effect.
    2. rank_swap           — A non-top feature is described as the most important.
    3. feature_fabrication — Narrative mentions a feature not present in the SHAP input.
    4. magnitude_distortion— Large effect described as minor, or small effect as major.
    5. omission            — A top-k SHAP feature is not mentioned at all.

Public API
----------
    evaluate_narrative(narrative, shap_values, cfg) -> EvaluationResult
    llm_judge(narrative, shap_values, model_cfg)    -> EvaluationResult  (optional second pass)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.config import EvaluationConfig, ModelConfig


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    sign_inversion: bool = False
    rank_swap: bool = False
    feature_fabrication: bool = False
    magnitude_distortion: bool = False
    omission: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def any_hallucination(self) -> bool:
        return any([
            self.sign_inversion,
            self.rank_swap,
            self.feature_fabrication,
            self.magnitude_distortion,
            self.omission,
        ])

    def notes_str(self) -> str:
        return "; ".join(self.notes) if self.notes else ""

    def to_dict(self) -> dict:
        return {
            "sign_inversion": int(self.sign_inversion),
            "rank_swap": int(self.rank_swap),
            "feature_fabrication": int(self.feature_fabrication),
            "magnitude_distortion": int(self.magnitude_distortion),
            "omission": int(self.omission),
            "any_hallucination": int(self.any_hallucination),
            "notes": self.notes_str(),
        }


# ---------------------------------------------------------------------------
# Word lists for direction and magnitude detection
# ---------------------------------------------------------------------------

_POSITIVE_DIRECTION_WORDS = [
    "increase", "increases", "increased", "higher", "raises", "raised", "boost",
    "boosted", "positive", "positively", "pushes up", "pushed up", "elevate",
    "elevated", "greater", "stronger", "more likely", "contributed positively",
    "contributed to a higher", "supports", "promote", "promotes", "amplif",
]

_NEGATIVE_DIRECTION_WORDS = [
    "decrease", "decreases", "decreased", "lower", "reduces", "reduced", "diminish",
    "diminishes", "negative", "negatively", "pushes down", "pushed down", "penaliz",
    "penalty", "less likely", "contributed negatively", "contributed to a lower",
    "lowers", "suppresses", "weaken",
]

_LARGE_MAGNITUDE_WORDS = [
    "major", "significant", "substantial", "large", "strong", "dominant", "powerful",
    "considerable", "notable", "primary", "main", "chief", "strongest", "largest",
    "biggest", "greatest", "most important", "most influential", "highest",
]

_SMALL_MAGNITUDE_WORDS = [
    "minor", "slight", "small", "negligible", "minimal", "little", "marginal",
    "modest", "weak", "weakest", "smallest", "least", "trivial", "negligibly",
    "slightly", "minimally",
]

_IMPORTANCE_SUPERLATIVES = [
    "most important", "most influential", "most significant", "primary factor",
    "main factor", "dominant factor", "strongest factor", "biggest factor",
    "largest contribution", "biggest contribution", "primary driver",
    "main driver", "top factor", "key driver", "principal",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _feature_variants(feature_name: str) -> List[str]:
    """
    Return search strings for a feature name.
    e.g. "education_num" → ["education num", "education_num", "educationnum"]
    """
    base = feature_name.lower()
    spaced = base.replace("_", " ").replace("-", " ")
    nospace = base.replace("_", "").replace("-", "")
    variants = list({base, spaced, nospace})
    return [v for v in variants if v]


def _feature_in_text(feature_name: str, text: str) -> bool:
    """Return True if any variant of feature_name appears in text."""
    norm = _normalise(text)
    return any(v in norm for v in _feature_variants(feature_name))


def _sentence_context(feature_name: str, text: str) -> str:
    """
    Return the sentence(s) from *text* that contain *feature_name*.

    Scoping direction and magnitude checks to the containing sentence prevents
    the context window from bleeding into adjacent sentences and picking up
    direction/magnitude words that belong to a different feature's description.

    Falls back to the full normalised text if no matching sentence is found.
    """
    norm = _normalise(text)
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", norm.strip())
    matches = [
        s for s in sentences
        if any(v in s for v in _feature_variants(feature_name))
    ]
    return " ".join(matches) if matches else norm


def _contains_any(words: List[str], text: str) -> bool:
    return any(w in text for w in words)


def _shap_items_sorted(shap_values: Dict[str, float]) -> List[Tuple[str, float]]:
    """Return (feature, shap_val) pairs sorted by |shap_val| descending."""
    return sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)


# ---------------------------------------------------------------------------
# Check 1 — Sign inversion
# ---------------------------------------------------------------------------

def _check_sign_inversion(
    narrative: str,
    shap_values: Dict[str, float],
) -> Tuple[bool, List[str]]:
    """
    For each feature mentioned in the narrative, look at the context around it
    and check whether the stated direction conflicts with the SHAP sign.

    Returns (flagged, notes).
    """
    flagged = False
    notes: List[str] = []

    for feature, shap_val in shap_values.items():
        if not _feature_in_text(feature, narrative):
            continue  # Feature not mentioned — omission check handles this

        context = _sentence_context(feature, narrative)
        has_positive = _contains_any(_POSITIVE_DIRECTION_WORDS, context)
        has_negative = _contains_any(_NEGATIVE_DIRECTION_WORDS, context)

        if shap_val > 0 and has_negative and not has_positive:
            flagged = True
            notes.append(
                f"sign_inversion: '{feature}' has positive SHAP ({shap_val:+.4f}) "
                f"but narrative context suggests negative effect"
            )
        elif shap_val < 0 and has_positive and not has_negative:
            flagged = True
            notes.append(
                f"sign_inversion: '{feature}' has negative SHAP ({shap_val:+.4f}) "
                f"but narrative context suggests positive effect"
            )

    return flagged, notes


# ---------------------------------------------------------------------------
# Check 2 — Rank swap
# ---------------------------------------------------------------------------

def _check_rank_swap(
    narrative: str,
    shap_values: Dict[str, float],
) -> Tuple[bool, List[str]]:
    """
    Detect if the narrative describes a non-top feature using superlatives
    that should only apply to the top-ranked feature.

    Returns (flagged, notes).
    """
    if not shap_values:
        return False, []

    sorted_items = _shap_items_sorted(shap_values)
    top_feature = sorted_items[0][0]
    norm = _normalise(narrative)

    flagged = False
    notes: List[str] = []

    for superlative in _IMPORTANCE_SUPERLATIVES:
        if superlative not in norm:
            continue
        # Find which feature is closest to this superlative in the text
        idx_sup = norm.find(superlative)
        closest_feature = None
        closest_dist = float("inf")

        for feature, _ in shap_values.items():
            for variant in _feature_variants(feature):
                idx_feat = norm.find(variant)
                if idx_feat == -1:
                    continue
                dist = abs(idx_feat - idx_sup)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_feature = feature

        if closest_feature and closest_feature != top_feature and closest_dist < 200:
            flagged = True
            notes.append(
                f"rank_swap: superlative '{superlative}' appears near "
                f"'{closest_feature}' but true top feature is '{top_feature}' "
                f"(|SHAP|={abs(shap_values[top_feature]):.4f})"
            )

    return flagged, notes


# ---------------------------------------------------------------------------
# Check 3 — Feature fabrication
# ---------------------------------------------------------------------------

def _check_feature_fabrication(
    narrative: str,
    shap_values: Dict[str, float],
    all_dataset_features: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Check whether the narrative explicitly names features that are not in the
    SHAP input for this instance.

    *all_dataset_features* should be the full list of feature names in the
    dataset (including those with zero SHAP contribution for this instance).
    When provided, fabrication is flagged only for names outside this broader
    set — i.e., features that do not exist in the dataset at all.

    When not provided, fabrication is flagged for names outside the per-instance
    SHAP dict (which may include false positives for zero-contribution features).

    Returns (flagged, notes).
    """
    valid_features = set(all_dataset_features or shap_values.keys())
    norm = _normalise(narrative)
    flagged = False
    notes: List[str] = []

    # Build a regex that matches exact feature name occurrences (all variants)
    for feature in list(shap_values.keys()) + list(valid_features):
        pass  # just validate the known set

    # Check the known SHAP features appear correctly — already done in other checks.
    # Here we look for multi-word tokens in the narrative that look like feature
    # names (underscore-joined) but are NOT in the valid set.
    candidate_tokens = re.findall(r"\b[a-z][a-z0-9_]{2,}\b", norm)
    for token in set(candidate_tokens):
        # Only flag if the token itself (or its space-separated form) is very close
        # to a feature-name style string but matches nothing valid
        if "_" not in token:
            continue  # plain words are not suspicious
        spaced = token.replace("_", " ")
        if not any(token == f or spaced == f.replace("_", " ") for f in valid_features):
            flagged = True
            notes.append(
                f"feature_fabrication: token '{token}' in narrative "
                f"looks like a feature name but is not in the dataset"
            )

    return flagged, notes


# ---------------------------------------------------------------------------
# Check 4 — Magnitude distortion
# ---------------------------------------------------------------------------

def _check_magnitude_distortion(
    narrative: str,
    shap_values: Dict[str, float],
    magnitude_threshold: float = 0.5,
) -> Tuple[bool, List[str]]:
    """
    Flag cases where:
      - A feature with |SHAP| > threshold * max|SHAP| is described as minor/small.
      - A feature with |SHAP| <= threshold * max|SHAP| is described as major/dominant.

    Returns (flagged, notes).
    """
    if not shap_values:
        return False, []

    max_abs = max(abs(v) for v in shap_values.values())
    if max_abs == 0:
        return False, []

    flagged = False
    notes: List[str] = []

    for feature, shap_val in shap_values.items():
        if not _feature_in_text(feature, narrative):
            continue

        relative = abs(shap_val) / max_abs
        context = _sentence_context(feature, narrative)

        is_large = relative > magnitude_threshold
        is_small = relative <= magnitude_threshold

        if is_large and _contains_any(_SMALL_MAGNITUDE_WORDS, context):
            flagged = True
            notes.append(
                f"magnitude_distortion: '{feature}' has large relative SHAP "
                f"({relative:.0%} of max) but narrative suggests small magnitude"
            )
        elif is_small and _contains_any(_LARGE_MAGNITUDE_WORDS, context):
            flagged = True
            notes.append(
                f"magnitude_distortion: '{feature}' has small relative SHAP "
                f"({relative:.0%} of max) but narrative suggests large magnitude"
            )

    return flagged, notes


# ---------------------------------------------------------------------------
# Check 5 — Omission
# ---------------------------------------------------------------------------

def _check_omission(
    narrative: str,
    shap_values: Dict[str, float],
    top_k: int = 3,
) -> Tuple[bool, List[str]]:
    """
    Check that each of the top-k features (by |SHAP|) is mentioned somewhere
    in the narrative.

    Returns (flagged, notes).
    """
    sorted_items = _shap_items_sorted(shap_values)
    top_features = [f for f, _ in sorted_items[:top_k]]
    flagged = False
    notes: List[str] = []

    for feature in top_features:
        if not _feature_in_text(feature, narrative):
            flagged = True
            notes.append(
                f"omission: top feature '{feature}' "
                f"(|SHAP|={abs(shap_values[feature]):.4f}) not mentioned in narrative"
            )

    return flagged, notes


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_narrative(
    narrative: str,
    shap_values: Dict[str, float],
    cfg: EvaluationConfig,
    all_dataset_features: Optional[List[str]] = None,
) -> EvaluationResult:
    """
    Run all five rule-based hallucination checks on a single narrative.

    Args:
        narrative:            The LLM-generated explanation text.
        shap_values:          Dict mapping feature name → SHAP value for this instance.
        cfg:                  EvaluationConfig (top_k_features, magnitude_threshold).
        all_dataset_features: Full feature list of the dataset (for fabrication check).

    Returns:
        An EvaluationResult with per-type boolean flags and notes.
    """
    result = EvaluationResult()

    si_flag, si_notes = _check_sign_inversion(narrative, shap_values)
    result.sign_inversion = si_flag
    result.notes.extend(si_notes)

    rs_flag, rs_notes = _check_rank_swap(narrative, shap_values)
    result.rank_swap = rs_flag
    result.notes.extend(rs_notes)

    ff_flag, ff_notes = _check_feature_fabrication(narrative, shap_values, all_dataset_features)
    result.feature_fabrication = ff_flag
    result.notes.extend(ff_notes)

    md_flag, md_notes = _check_magnitude_distortion(
        narrative, shap_values, cfg.magnitude_threshold
    )
    result.magnitude_distortion = md_flag
    result.notes.extend(md_notes)

    om_flag, om_notes = _check_omission(narrative, shap_values, cfg.top_k_features)
    result.omission = om_flag
    result.notes.extend(om_notes)

    return result


# ---------------------------------------------------------------------------
# Optional LLM judge
# ---------------------------------------------------------------------------

def llm_judge(
    narrative: str,
    shap_values: Dict[str, float],
    model_cfg: ModelConfig,
) -> EvaluationResult:
    """
    Second-pass evaluation using an LLM as judge.
    Only called when cfg.use_llm_judge is True.

    The judge is given the SHAP values and the narrative and asked to identify
    any of the five hallucination types. Its response is parsed into an
    EvaluationResult.

    Args:
        narrative:  The narrative to judge.
        shap_values: Ground-truth SHAP values.
        model_cfg:   The model config for the judge LLM.

    Returns:
        An EvaluationResult populated from the judge's response.
    """
    from src.generation.llm_client import LLMClient

    shap_lines = "\n".join(
        f"  {feat}: {val:+.4f}"
        for feat, val in sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    prompt = f"""You are a hallucination detection expert evaluating an AI-generated explanation of SHAP feature importance values.

Ground-truth SHAP values (positive = pushes prediction up, negative = pushes prediction down):
{shap_lines}

Narrative to evaluate:
\"\"\"{narrative}\"\"\"

For each of the following hallucination types, answer YES or NO and briefly explain:

1. SIGN_INVERSION: Does the narrative state the wrong direction of effect for any feature?
2. RANK_SWAP: Does the narrative describe a non-top feature as the most important?
3. FEATURE_FABRICATION: Does the narrative mention any feature not in the SHAP values above?
4. MAGNITUDE_DISTORTION: Does the narrative wrongly describe a large effect as minor, or a small effect as major?
5. OMISSION: Does the narrative fail to mention any of the top-3 most important features?

Respond in this exact format:
SIGN_INVERSION: YES/NO — <brief reason>
RANK_SWAP: YES/NO — <brief reason>
FEATURE_FABRICATION: YES/NO — <brief reason>
MAGNITUDE_DISTORTION: YES/NO — <brief reason>
OMISSION: YES/NO — <brief reason>
"""

    client = LLMClient()
    response = client.generate(prompt, model_cfg)

    result = EvaluationResult()
    result.notes.append("[llm_judge]")

    for line in response.splitlines():
        line = line.strip()
        if line.startswith("SIGN_INVERSION:"):
            result.sign_inversion = "YES" in line.split("—")[0].upper()
            result.notes.append(line)
        elif line.startswith("RANK_SWAP:"):
            result.rank_swap = "YES" in line.split("—")[0].upper()
            result.notes.append(line)
        elif line.startswith("FEATURE_FABRICATION:"):
            result.feature_fabrication = "YES" in line.split("—")[0].upper()
            result.notes.append(line)
        elif line.startswith("MAGNITUDE_DISTORTION:"):
            result.magnitude_distortion = "YES" in line.split("—")[0].upper()
            result.notes.append(line)
        elif line.startswith("OMISSION:"):
            result.omission = "YES" in line.split("—")[0].upper()
            result.notes.append(line)

    return result
