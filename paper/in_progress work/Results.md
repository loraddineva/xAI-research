# Section 5 — Results

> **Run:** `pilot_run_20260518T135815_bdad28` · Adult Income · Llama 3 70B (generation) · Mistral 7B Instruct v0.3 (extraction) · seed 42
> **Status:** Full results — ready to integrate into Paper_1.md §5

---

## 5.1 Evaluation Overview

The pipeline generated 300 narratives (150 instances × 2 prompt strategies). The extraction model returned a parse error for 28 narratives (9.3%), all caused by HTTP 503 errors on the Mistral inference endpoint. The analysis below is restricted to the 272 narratives with valid extractions: approximately 130 Martens and 142 chain-of-thought narratives. Parse failures are distributed across both strategies and are treated as missing data rather than hallucination events.

---

## 5.2 Overall Hallucination Prevalence

Three in four valid narratives exhibit at least one hallucination type: 75.0% (95% CI: 69.5–79.8%). Rank swap is by far the most prevalent failure, present in 68.8% of narratives (95% CI: 63.0–74.0%). Sign inversion affects 30.9% (95% CI: 25.7–36.6%). Omission is less common at 17.3% (95% CI: 13.3–22.2%). Feature fabrication — the hallucination type in which the narrative mentions a feature not in the dataset — occurs in 0.0% of narratives (95% CI: 0.0–1.4%); the model never invents feature names.

**Table 5.1.** Hallucination type rate by prompt strategy (Adult Income; Llama 3 70B; *n* = 272 valid extractions).

| Type | Martens (*n* ≈ 130) | Chain-of-thought (*n* ≈ 142) | Overall (*n* = 272) |
|---|---|---|---|
| Sign inversion | 60.0% (51.4–68.0%) | 4.2% (2.0–8.9%) | 30.9% (25.7–36.6%) |
| Rank swap | 80.8% (73.2–86.6%) | 57.7% (49.5–65.6%) | 68.8% (63.0–74.0%) |
| Feature fabrication | 0.0% (0.0–2.9%) | 0.0% (0.0–2.6%) | 0.0% (0.0–1.4%) |
| Omission | 15.4% (10.2–22.6%) | 19.0% (13.4–26.3%) | 17.3% (13.3–22.2%) |
| **Any hallucination** | **93.1% (87.4–96.3%)** | **58.5% (50.2–66.2%)** | **75.0% (69.5–79.8%)** |

*95% Wilson confidence intervals in parentheses.*

---

## 5.3 Prompt Strategy Effects

Chain-of-thought prompting substantially reduces overall hallucination relative to the Martens baseline: 58.5% versus 93.1% (χ²(1) = 43.40, *p* < .001). The reduction is driven almost entirely by sign inversion. The Martens strategy produces sign inversions in 60.0% of its narratives; chain-of-thought reduces this to 4.2% (χ²(1) = 98.91, *p* < .001). Rank swap also drops significantly — from 80.8% to 57.7% (χ²(1) = 16.74, *p* < .001) — though it remains the dominant failure mode under both strategies.

Two hallucination types do not differ significantly between strategies. Feature fabrication is zero under both. Omission rates are nearly identical: 15.4% for Martens versus 19.0% for chain-of-thought (χ²(1) = 0.63, *n.s.*). Chain-of-thought's explicit ranking step eliminates directional errors but does not help the model select the correct top-*k* features or mention all SHAP-important features.

**Table 5.2.** Chi-squared tests comparing hallucination rates between prompt strategies (df = 1).

| Hallucination type | χ² | Significance |
|---|---|---|
| Sign inversion | 98.91 | *p* < .001 |
| Rank swap | 16.74 | *p* < .001 |
| Feature fabrication | 0.00 | *n.s.* |
| Omission | 0.63 | *n.s.* |
| Any hallucination | 43.40 | *p* < .001 |

---

## 5.4 Co-occurrence of Hallucination Types

Rank swap rarely occurs alone. The most common pattern is rank swap without any other type (29.8% of all valid narratives), followed by rank swap co-occurring with sign inversion (22.4%) and rank swap co-occurring with omission (13.6%). Sign inversion without rank swap is rare (5.5%), and omission alone is negligible (0.7%). No narrative exhibits feature fabrication alongside any other type. The co-occurrence structure suggests rank swap and sign inversion have a shared cause — likely the model assigning wrong attributions to the wrong features, which simultaneously inverts direction and displaces the true top features.

**Table 5.3.** Co-occurrence of hallucination types across all valid narratives (*n* = 272).

| Combination | Count | % of valid |
|---|---|---|
| Rank swap only | 81 | 29.8% |
| Rank swap + sign inversion | 61 | 22.4% |
| Omission + rank swap | 37 | 13.6% |
| Sign inversion only | 15 | 5.5% |
| Omission + rank swap + sign inversion | 8 | 2.9% |
| Omission only | 2 | 0.7% |

---

## 5.5 Feature-Level Mention Patterns

Across all 272 valid narratives, `hours_per_week` (242 mentions) and `age` (227) appear in almost every narrative. Binary indicators are mentioned less consistently: `occupation_Other` (152), `sex_Male` (161), and `marital_status_Non_Married` (164) appear in roughly half to two-thirds of narratives, while `capital_gain` (53) and `capital_loss` (60) appear in fewer than a quarter. The rarity of capital gain and capital loss mentions despite their often-large SHAP magnitudes may contribute to the omission rate. No narrative references a feature name outside the valid feature list, confirming the zero feature fabrication rate.

**Table 5.4.** Feature mention frequency across all valid narratives (*n* = 272).

| Feature | Mentions | % of narratives |
|---|---|---|
| `hours_per_week` | 242 | 89.0% |
| `age` | 227 | 83.5% |
| `marital_status_Non_Married` | 164 | 60.3% |
| `sex_Male` | 161 | 59.2% |
| `occupation_Other` | 152 | 55.9% |
| `relationship_Non_Husband` | 126 | 46.3% |
| `native_country_US` | 73 | 26.8% |
| `workclass_Private` | 72 | 26.5% |
| `capital_loss` | 60 | 22.1% |
| `race_White` | 60 | 22.1% |
| `capital_gain` | 53 | 19.5% |

---

## 5.6 Extraction Robustness

The robustness check was run on a subsample of 68 narratives (five repeated extractions at temperature 0.9 per narrative). One narrative failed the minimum-run threshold (fewer than three successful parses) and was excluded from robustness scoring. No narrative scored below the 0.8 reliability threshold.

The mean narrative reliability score across the 67 scoreable narratives is 0.946 (SD = 0.049), indicating that the extraction model's sign and feature-membership decisions are largely consistent across repeated runs. Chain-of-thought narratives score higher than Martens narratives on both metrics.

**Table 5.5.** Extraction robustness metrics by prompt strategy (subsample of 68 narratives; 1 excluded as unreliable).

| Metric | Martens | Chain-of-thought | Overall |
|---|---|---|---|
| Narrative reliability score | 0.921 (SD = 0.043) | 0.970 (SD = 0.044) | 0.946 (SD = 0.049) |
| Top-*k* set agreement | 0.525 (SD = 0.226) | 0.903 (SD = 0.157) | 0.717 (SD = 0.271) |

The gap in top-*k* set agreement is striking: for Martens narratives, repeated extractions agree on the top-3 feature set only 52.5% of the time on average, compared to 90.3% for chain-of-thought. This means that rank swap flags on Martens narratives are likely conservative — the extraction model itself is uncertain which features the narrative treats as most important, so some flagged rank swaps may reflect extraction noise rather than narrative error. Hallucination rates for the Martens strategy should be read with this caveat.

**Table 5.6.** Per-feature mean sign agreement and value agreement across robustness runs (*n* = 67 scoreable narratives).

| Feature | Mean sign agreement | *N* narratives | Mean value agreement | *N* narratives |
|---|---|---|---|---|
| `age` | 1.000 | 55 | 0.992 | 55 |
| `capital_gain` | 1.000 | 18 | 0.907 | 10 |
| `capital_loss` | 0.973 | 20 | 0.758 | 8 |
| `hours_per_week` | 0.986 | 61 | 0.978 | 59 |
| `marital_status_Non_Married` | 0.956 | 43 | 0.736 | 12 |
| `native_country_US` | 0.984 | 16 | 0.688 | 8 |
| `occupation_Other` | 1.000 | 31 | n/a | 0 |
| `race_White` | 0.983 | 15 | 0.710 | 7 |
| `relationship_Non_Husband` | 0.935 | 39 | 0.721 | 11 |
| `sex_Male` | 1.000 | 41 | 0.732 | 11 |
| `workclass_Private` | 1.000 | 16 | 0.775 | 4 |

Sign agreement is high across all features (minimum 0.935 for `relationship_Non_Husband`). Value agreement is lower for binary indicators — the extraction model frequently fails to recover whether the narrative stated the feature value explicitly — but this does not affect hallucination scoring, which depends on sign, not value.

---

## 5.7 Summary of Main Findings

Three findings stand out.

First, the overall hallucination rate is high. Three quarters of narratives generated by Llama 3 70B under at least one prompt strategy contain a detectable faithfulness error. This rate is not driven by a small number of pathological cases; it reflects systematic error patterns across the evaluation set.

Second, sign inversion is nearly eliminated by chain-of-thought prompting — dropping from 60.0% to 4.2% — but rank swap remains common (57.7%) even after the chain-of-thought's explicit ranking step. The model selects the right direction for individual features when forced to reason through them, but still assembles the wrong set of top features, or misidentifies which features belong in the most important group.

Third, feature fabrication is zero. The model does not invent feature names. Its failures are in how it assigns and orders attributions among the features it does discuss — not in conjuring features that do not exist.

---

*Script:* `scripts/summarise_results.py` · *Data:* `outputs/evaluations/pilot_run_20260518T135815_bdad28/`
