# Automated Detection of Hallucination in LLM-Generated SHAP Narratives

> **Status:** Results complete; full prose draft revised against peer-review brief — Paper 1 of PhD thesis
> **Target venue:** TBD (e.g. *AI & Society*, *Information Processing & Management*, *AAAI*, *ACL Findings*)
> **Word budget:** ~2,000–2,500 words (excluding references and appendices)

---

## Title (working)

*Automated Detection of Hallucination in LLM-Generated SHAP Narratives*

---

## Abstract

LLMs are increasingly used to translate SHAP feature-attribution values into natural-language explanations for tabular model predictions, yet the faithfulness of these narratives to the underlying attribution values has not been systematically measured. This paper defines and measures four classes of faithfulness failure (sign inversion, rank swap, feature fabrication, and omission) using an automated extraction-and-comparison pipeline applied to 300 narratives that cross two prompt strategies (direct and chain-of-thought) with 150 instances from the Adult Income benchmark. Seventy-five per cent of valid narratives contain at least one faithfulness failure (95% CI: 69.5–79.8%), with rank swap dominant at 68.8%. Chain-of-thought prompting nearly eliminates sign inversion (60.0% → 4.2%) but leaves rank swap largely unaddressed (80.8% → 57.7%); feature fabrication is absent under both strategies. These results provide the first systematic prevalence estimates for hallucination in LLM-generated SHAP narratives, together with a reusable open-source evaluation framework.

---

## 1. Introduction

Machine learning models are now deployed in settings where predictions must be explained to the people they affect. Credit applicants, patients, and people subject to automated employment or benefits decisions have a reasonable interest in knowing which factors drove the model's output. Shapley Additive Explanations, or SHAP (Lundberg & Lee, 2017), assign each input feature a signed numeric contribution that reflects how much it pushed the prediction toward or away from the positive class. Raw attribution tables are not easy for non-expert users to interpret [CITATION NEEDED], and large language models are now used to convert them into natural-language narratives [CITATION NEEDED].

The narratives work. Martens et al. (2024) showed that users who received LLM-generated SHAP narratives for income classification predictions answered comprehension questions significantly more accurately than users who received the attribution table alone, and more than 90% of participants rated the narratives as convincing. A narrative that reads as credible and improves comprehension is one users are likely to trust. If that narrative misrepresents the underlying attribution values, the persuasiveness becomes the hazard rather than the benefit. An inaccurate account that reads as plausible is more likely to be accepted, acted upon, and never questioned than one that reads as confused or incomplete [CITATION NEEDED: persuasiveness-versus-accuracy evidence, e.g. Bansal et al. 2021]. The more appealing the narrative form, the more consequential its inaccuracy.

Whether LLM-generated SHAP narratives are faithful to the attribution values they draw from has not been measured. Ichmoukamedov et al. (2024) proposed an automated evaluation pipeline and identified hallucination cases through manual analysis of a subsample, but did not report systematic rates and did not vary prompt design.

This paper tests whether Llama 3 70B generates faithful SHAP narratives. We compare two prompt strategies, direct and chain-of-thought, across 150 instances from the Adult Income benchmark (OpenXAI, Agarwal et al., 2022), producing 300 narratives in total. We apply and operationalise four classes of faithfulness failure, grounded in the NLG hallucination literature, as a detection pipeline that compares extracted narrative claims to ground-truth SHAP values. Three quarters of valid narratives contain at least one faithfulness failure. Rank swap is the dominant type, present in 68.8% of valid narratives and in 57.7% even under chain-of-thought prompting.

The paper contributes (a) an operationalised four-class taxonomy of faithfulness failure for narrative XAI; (b) the first systematic prevalence estimates, with confidence intervals, across a full evaluation set; (c) prompt strategy treated as a controlled experimental variable rather than a design choice; and (d) an open-source extraction-and-comparison pipeline. Section 2 reviews related work on hallucination in LLMs, narrative XAI, and faithfulness measurement. Section 3 describes the dataset, models, and prompt strategies. Section 4 defines the hallucination classes and detection procedure. Section 5 presents the results. Section 6 discusses the findings, states limitations, and concludes.

---

## 2. Related Work

### 2.1 Hallucination in LLMs

Hallucinations in LLMs broadly divide into two categories. Factual hallucination describes outputs that contradict world knowledge. Faithfulness hallucination describes outputs that contradict a provided grounding source (Ji et al., 2023; Maynez et al., 2020). This paper concerns the latter exclusively. The SHAP table is the complete grounding source, every claim the narrative makes is checkable against it, and world knowledge is irrelevant.

Agarwal, Tanneru and Lakkaraju (2024) identify a structural tension in LLM explanation generation. RLHF training optimises for outputs that appear coherent to human evaluators, not for outputs that correspond to a grounding source. A plausible explanation and a faithful one are not the same thing, and in high-stakes settings the divergence has direct consequences. Turpin et al. (2023) make a complementary point from a different direction. Chain-of-thought explanations can be systematically unfaithful to a model's actual decision process. The reasoning is logically coherent but rationalises the output post-hoc. In SHAP narration there is no LLM decision process to rationalise. The faithfulness question is whether the narrative correctly relays a numeric table produced by a separate classical model.

### 2.2 Narrative XAI: from SHAP values to prose

SHAP (Lundberg & Lee, 2017) is the standard post-hoc attribution method for tabular predictions, with established applications in credit scoring, clinical decision support, and algorithmic auditing [CITATION NEEDED]. Its output is a signed numeric score for each feature, precise but technically dense. Studies of XAI comprehension by lay users consistently find that raw attribution tables are difficult to interpret without statistical training [CITATION NEEDED]. Natural-language generation offers a communication layer that converts attribution outputs into prose [CITATION NEEDED].

Martens et al. (2024) introduced XAIstories, LLM-generated narratives grounded in SHAP values or counterfactuals for tabular classification predictions. In a user study, over 90% of participants rated the narratives as convincing, and users who received narratives answered comprehension questions significantly more accurately than users who received raw SHAP tables [CITATION NEEDED]. The study did not measure whether the narratives accurately represented the SHAP values they were generated from. Ichmoukamedov, Hinns and Martens (2024) proposed automated metrics for XAI narrative evaluation using a two-LLM pipeline. A generation model produces a narrative from a SHAP table, and an extraction model parses the narrative into structured feature claims, which are compared against the SHAP input. Manual analysis on a subsample identified hallucination cases but rates were not reported systematically, and prompt strategy was not treated as an experimental variable.

### 2.3 Measuring faithfulness in LLM explanations

Two recent studies measure SHAP-adjacent faithfulness in LLM-generated explanations, but in a setup that differs from this paper in a consequential way. AlMarri et al. (2025) evaluated four open-source LLMs as zero-shot classifiers on financial tabular data, computed SHAP values over the LLMs' own predictions, and compared those SHAP values to the LLMs' self-reported feature impacts. Sign agreement between self-explanation and SHAP reached only 50 to 60% even for the most important features. Kendall's τ alignment with LightGBM SHAP was near zero [CITATION NEEDED: page or section reference in AlMarri et al. 2025]. Matton et al. (2025) defined causal concept faithfulness as the alignment between the concepts an LLM explanation implies are influential and those that causally affect model outputs, measured via counterfactual inputs and a Bayesian hierarchical model. GPT-3.5 explanations systematically over-cited behaviour-related concepts and omitted identity-related ones regardless of their causal relevance. In both studies, the LLM is the model being explained and also the source of the explanation. This paper asks whether LLMs faithfully relay a numeric grounding source produced by a separate classical model. The ground truth is fully external, the generation model has no stake in the explanation, and every directional claim is verifiable without causal inference.

Whether chain-of-thought prompting reduces faithfulness failures in SHAP narratives is an open empirical question. Wei et al. (2022) showed that chain-of-thought improves structured reasoning on multi-step tasks. Turpin et al. (2023) showed it can simultaneously surface reasoning that is coherent but unfaithful to the model's actual computation. In SHAP narration, neither outcome can be assumed. We test chain-of-thought as a treatment variable, not as a solution.

---

## 3. Experimental Design

### 3.1 Dataset

The Adult Income dataset (UCI; Kohavi, 1996; Dua & Graff, 2019 [VERIFY YEAR]) is a binary classification benchmark predicting whether a person's annual income exceeds $50,000 based on demographic and employment records. We use the OpenXAI version (Agarwal et al., 2022), which provides a standardised preprocessed feature set, a pretrained logistic regression model, and per-instance SHAP values computed via `SHAPExplainerC`. After preprocessing, each instance is represented by eleven features: four continuous (`age`, `capital_gain`, `capital_loss`, `hours_per_week`) and seven binary indicators encoding categorical attributes (`sex_Male`, `workclass_Private`, `marital_status_Non_Married`, `occupation_Other`, `relationship_Non_Husband`, `race_White`, `native_country_US`). The survey-weight column `fnlwgt` is excluded. Class distribution in the 150-instance sample: [TO FILL — proportion of income >$50K vs. ≤$50K].

We draw 150 test instances as a simple random sample without replacement from the full OpenXAI processed test split ([TO FILL — exact row count], approximately 9,000 rows), using a fixed random seed of 42. Each instance's identifier is its row index in the processed CSV, allowing raw feature values and SHAP attributions to be traced for any narrative. In the prompt, features are presented in signed SHAP order (most positive to most negative), with binary indicators shown as an integer value followed by a human-readable label (e.g., `sex_Male: 1 [Male]`), using a fixed metadata registry; label mapping is consistent across all 300 narratives.

An initial design included German Credit (Statlog / OpenXAI) for cross-dataset comparison. That dataset and its preparation pipeline are archived in the repository (`archive/german_credit/`) but are not part of the reported study; results in this paper are specific to Adult Income and the logistic regression model provided by OpenXAI.

### 3.2 LLMs

**Narrative generation** uses `meta-llama/Meta-Llama-3-70B-Instruct` (Meta AI, 2024), accessed via `huggingface_hub.InferenceClient` through Hugging Face Inference Providers. At 70B parameters, this model is not available on the HF serverless tier for all users; requests may be routed to third-party inference providers (e.g., Novita). Temperature is set to 0.0 to maximise output determinism; LLM providers do not guarantee bit-identical outputs at temperature 0 across calls, so a small residual variance in generation cannot be ruled out. The direct strategy uses `max_tokens = 2048` to match the generation budget in Martens et al. (2024); chain-of-thought uses `max_tokens = 1024` for the narrative portion produced after the reasoning steps.

**Extraction** uses `mistralai/Mistral-7B-Instruct-v0.3` (Jiang et al., 2023), deployed as a dedicated Hugging Face Inference Endpoint, with temperature 0.0 and `max_tokens = 1024` to accommodate structured JSON output. Using a model distinct from the generation model avoids circular evaluation: the same model that produced the narrative does not assess its own claims. Mistral 7B is substantially smaller than Llama 3 70B; the capacity asymmetry is a deployment-cost choice and its implications are discussed in §6.2.

| Model                    | Role       | Provider               | Parameters | Temperature |
| ------------------------ | ---------- | ---------------------- | ---------- | ----------- |
| Llama 3 70B Instruct     | Generation | HF Inference Providers | 70B        | 0.0         |
| Mistral 7B Instruct v0.3 | Extraction | HF Inference Endpoint  | 7B         | 0.0         |

### 3.3 Prompt strategies

Two strategies are crossed with the single generation model.

**Direct.** The first prompt strategy closely follows the narrative template introduced by Martens et al. (2024), hereafter "direct." This choice is deliberate: Martens et al. is the study most directly concerned with SHAP-to-narrative generation, it has been applied to real user populations, and replicating its prompt design allows our faithfulness findings to be mapped back to a method that already has empirical evidence of adoption. Minor adaptations accommodate the Adult Income feature schema; the task description, instruction structure, and register are unchanged. The model receives the prediction task, predicted probability and class, a SHAP table sorted from most positive to most negative, and a direct instruction to write a fluent story focused on the most influential positive and negative features. Direct runs use `max_tokens = 2048`.

**Chain-of-thought.** The same task context and SHAP table are provided, but the model must first (1) rank the top five features by |SHAP|, (2) list negligible features, (3) verify the top feature and its sign, then (4) write the narrative under a `Narrative:` heading. The evaluation pipeline strips Steps 1–3 and passes only the text under `Narrative:` to the extraction model. The explicit ranking and sign-verification steps are designed to reduce directional and set-membership errors by forcing attention onto the numeric SHAP values before prose generation begins. Chain-of-thought runs use `max_tokens = 1024`.

Full prompt templates are reproduced in Appendix A.

### 3.4 Experimental factorial

The experiment crosses one generation model with two prompt strategies and 150 instances, producing 300 narratives. Each instance is processed under both strategies, so prompt strategy is a within-instance manipulation: each Adult Income instance contributes one narrative per strategy generated from the same underlying SHAP table. This paired structure is the basis for the McNemar test reported in §5.3.

---

## 4. Hallucination Taxonomy and Detection

### 4.1 Taxonomy overview

The four failure types map onto a structural distinction. Sign inversion and rank swap are attribution errors, in which the narrative misrepresents a feature present in the source by assigning it the wrong direction or by displacing it from the set of most important features. Feature fabrication is an invention error, in which the narrative mentions content with no counterpart in the grounding source. Omission is a coverage error, in which the narrative fails to include source content that is important. This structure divides the types by whether the error is one of commission (attributing incorrectly or inventing) or of non-coverage (failing to relay what the source shows). Types are mutually non-exclusive and a single narrative may exhibit more than one. Relative effect size is not treated as a separate type. Detecting whether a narrative's implied magnitude is proportional to the SHAP value would require a graded scale rather than a binary flag and falls outside the scope of this paper.

| Type                | Definition                                                                                                     | Detection summary                                       |
| ------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Sign inversion      | Narrative states the wrong directional effect of a feature (positive vs. negative class)                       | Extracted `sign` vs. sign of SHAP value                 |
| Rank swap           | The set of top-*k* extracted features (by narrative `rank`) does not match the top-*k* SHAP features by \|SHAP\| | Compare top-*k* sets, ordering within set ignored       |
| Feature fabrication | Narrative mentions a feature not present in the dataset's feature set                                          | Non-empty `unknown_features` list in extraction JSON    |
| Omission            | At least one top-*k* SHAP feature (*k* = 3 by default) is absent from the narrative                            | Top-*k* SHAP features absent from extraction `features` |

### 4.2 Detection procedure

For each narrative, the extraction model (Mistral 7B Instruct v0.3) receives the narrative text, the full feature list for the dataset, and the prediction task description. It returns a JSON object in which each feature mentioned in the narrative is recorded with an importance rank (0 = most important), a direction (`sign`: 1 or −1), the feature value if explicitly stated, and a one-sentence causal assumption. A separate field, `unknown_features`, lists any feature-like names that do not match the valid feature set. A rule-based, deterministic comparison step checks each extracted field against ground-truth SHAP values and sets four binary flags. A narrative is marked `any_hallucination = 1` if at least one flag is set. The full JSON schema and the comparison-rule pseudocode are reproduced in Appendix C. Parse failures and ambiguous extractions are logged for manual review.

The omission rule is parameterised by *k* (default *k* = 3), set in `config/default.yaml`. Rank swap uses a set-based criterion rather than a strict ordering. The flag is raised when the set of *k* features assigned the lowest rank values by the extraction model differs from the set of *k* features with the largest |SHAP| values. Internal ordering within that set is ignored. This design distinguishes set-membership errors, in which the model emphasises the wrong features altogether, from ordering noise, in which the narrative discusses the correct top features in a different sequence than SHAP magnitude order. Human-in-the-loop validation against a gold-standard label set is planned as immediate future work; see §6.2.

### 4.3 Extraction robustness check

Hallucination flags depend on the extraction model parsing the narrative correctly. To quantify extraction reliability independently of narrative faithfulness, we adapt the semantic uncertainty approach of Kuhn et al. (2023): for each narrative, the extraction prompt is submitted five times at temperature 0.9, following the high-variance sampling protocol used in that paper to surface latent disagreement between equally plausible parses. Each run returns the same JSON schema as the main evaluation. For every feature mentioned in a majority of runs, we compute sign agreement as the proportion of runs that return the plurality sign (e.g., a 3/2 split yields 0.6). Top-*k* set agreement scores the fraction of runs whose top-*k* feature set matches the plurality set, ignoring internal ordering. The narrative reliability score pools the per-feature sign-agreement values and the top-*k* set-agreement value into a single arithmetic mean:

$$
\text{reliability} = \mathrm{mean}\bigl(\{a_{\text{sign},f} : f \in F_{\text{majority}}\} \cup \{a_{\text{topk}}\}\bigr)
$$

Narratives scoring below 0.8 are flagged as low-reliability; those with fewer than three successful parses are marked extraction-unreliable.

In the main analysis, hallucination rates are reported for the full valid set; low-reliability narratives (score < 0.8) are flagged and their flags interpreted with caution, since extraction noise rather than narrative error may be responsible. The check multiplies extraction API cost by five; for cost-constrained settings, calibrating the reliability distribution on a 10% subsample is sufficient to estimate how much of the corpus is robustly extractable.

---

## 5. Results

### 5.1 Evaluation overview

The pipeline generated 300 narratives (150 instances × 2 prompt strategies). The extraction model returned a parse error for 28 (9.3%), all caused by HTTP 503 timeouts on the Mistral inference endpoint; these are treated as missing data rather than evidence of narrative failure. The analysis below is restricted to the 272 narratives with valid extractions (130 direct, 142 chain-of-thought). The 12-narrative difference in valid counts between strategies reflects the unequal distribution of API failures across the two batches, not a systematic bias in one strategy's extractability.

As a sensitivity check, we re-computed the any-hallucination rate under two extreme imputations: if all 28 parse failures are coded as hallucinated, the rate rises from 75.0% to 77.3% (232/300); if all are coded as faithful, the rate falls to 68.0% (204/300). The headline conclusion — that a clear majority of narratives contain at least one faithfulness failure — is robust to either treatment of the missing data.

### 5.2 Overall hallucination prevalence

Three quarters of valid narratives exhibit at least one hallucination type: 75.0% (95% CI: 69.5–79.8%; Wilson interval, used for all proportions below). Rank swap is the dominant failure, present in 68.8% of narratives (95% CI: 63.0–74.0%). Sign inversion affects 30.9% (95% CI: 25.7–36.6%). Omission occurs in 17.3% (95% CI: 13.3–22.2%). Feature fabrication is absent under both strategies: 0.0% (95% CI: 0.0–1.4%).

**Table 5.1.** Hallucination type rate by prompt strategy (Adult Income; Llama 3 70B; *n* = 272 valid extractions). Rows are ordered by overall prevalence. 95% Wilson confidence intervals in parentheses.

| Type                  | Direct (*n* = 130)     | Chain-of-thought (*n* = 142) | Overall (*n* = 272)    |
| --------------------- | ---------------------- | ---------------------------- | ---------------------- |
| Rank swap             | 80.8% (73.2–86.6%)     | 57.7% (49.5–65.6%)           | 68.8% (63.0–74.0%)     |
| Sign inversion        | 60.0% (51.4–68.0%)     | 4.2% (2.0–8.9%)              | 30.9% (25.7–36.6%)     |
| Omission              | 15.4% (10.2–22.6%)     | 19.0% (13.4–26.3%)           | 17.3% (13.3–22.2%)     |
| Feature fabrication   | 0.0% (0.0–2.9%)        | 0.0% (0.0–2.6%)              | 0.0% (0.0–1.4%)        |
| **Any hallucination** | **93.1% (87.4–96.3%)** | **58.5% (50.2–66.2%)**       | **75.0% (69.5–79.8%)** |

### 5.3 Prompt strategy effects

Because each Adult Income instance is processed under both strategies from the same SHAP table, the two columns of Table 5.1 are paired observations rather than independent samples. Strategy effects are therefore tested with McNemar's exact test on the 122 instances for which extraction succeeded under both strategies. Chain-of-thought reduces overall hallucination substantially: 40 instances are hallucinated under direct but not under chain-of-thought, while only 3 are hallucinated under chain-of-thought but not under direct (exact two-sided *p* < .001; McNemar χ²(1) = 30.14). The reduction is driven almost entirely by sign inversion: 67 instances flip from hallucinated to faithful under chain-of-thought and none flip in the other direction (exact *p* < .001; χ²(1) = 65.01). Rank swap also falls — 33 instances improve under chain-of-thought versus 10 that worsen (exact *p* < .001; χ²(1) = 11.26) — but it remains the dominant failure mode under both strategies, with an absolute reduction of 23.1 percentage points (80.8% → 57.7%). Two hallucination types show no significant difference between strategies: feature fabrication is zero under both, and omission rates are nearly identical at 15.4% versus 19.0% (15 instances worsen under chain-of-thought, 21 improve; exact *p* = .405; χ²(1) = 0.69). The omission result is consistent with the discussion in §6.1: chain-of-thought's explicit ranking step does not translate into complete coverage of top-SHAP features in the final narrative.

**Table 5.2.** McNemar exact tests comparing paired hallucination outcomes between prompt strategies (paired *n* = 122 instances with valid extractions under both strategies). `b` = direct flagged, chain-of-thought not; `c` = chain-of-thought flagged, direct not. Risk difference is direct − chain-of-thought on the unpaired Table 5.1 rates.

| Hallucination type  | b  | c  | χ²(1) | Exact two-sided *p* | Risk difference (pp) |
| ------------------- | -- | -- | ----- | ------------------- | -------------------- |
| Sign inversion      | 67 |  0 | 65.01 | < .001              | −55.8                |
| Rank swap           | 33 | 10 | 11.26 | < .001              | −23.1                |
| Feature fabrication |  0 |  0 |    —  |          —          |   0.0                |
| Omission            | 15 | 21 |  0.69 |   .405              |  +3.6                |
| Any hallucination   | 40 |  3 | 30.14 | < .001              | −34.6                |

### 5.4 Co-occurrence of hallucination types

Rank swap rarely occurs alone. It co-occurs with sign inversion in 22.4% of all valid narratives and with omission in a further 13.6%. Sign inversion without rank swap accounts for only 5.5%. The 204 hallucinated narratives in Table 5.3 reconcile to the n_any_hallucination total from the evaluation metadata (81 + 61 + 37 + 15 + 8 + 2 = 204), with the remaining 68 of 272 valid narratives free of any flagged failure. The mechanistic interpretation of this co-occurrence pattern is taken up in §6.1.

**Table 5.3.** Co-occurrence of hallucination types (*n* = 272 valid narratives).

| Combination                           | Count | % of valid |
| ------------------------------------- | ----- | ---------- |
| No hallucination                      | 68    | 25.0%      |
| Rank swap only                        | 81    | 29.8%      |
| Rank swap + sign inversion            | 61    | 22.4%      |
| Omission + rank swap                  | 37    | 13.6%      |
| Sign inversion only                   | 15    |  5.5%      |
| Omission + rank swap + sign inversion |  8    |  2.9%      |
| Omission only                         |  2    |  0.7%      |

### 5.5 Feature-level mention patterns

`hours_per_week` (242 mentions, 89.0% of narratives) and `age` (227, 83.5%) appear in almost every narrative regardless of strategy. Capital-related features are substantially underrepresented: `capital_gain` appears in only 19.5% of narratives and `capital_loss` in 22.1%, despite their often-large SHAP magnitudes (mean |SHAP| per feature is reported in Appendix B). Narratives that skip capital features in favour of features with clear demographic interpretations (sex, race, marital status) necessarily misidentify the top-*k* drivers — a direct contribution to rank swap and omission rates. No narrative references a feature name outside the valid feature list, confirming the zero fabrication rate.

### 5.6 Extraction robustness

The robustness check was run on a 68-narrative subsample drawn at random with seed 42 from the 272 valid extractions, stratified by prompt strategy (five repeated extractions per narrative at temperature 0.9). One narrative yielded fewer than three successful parses and is excluded from reliability scoring. No narrative scored below the 0.8 reliability threshold. The mean narrative reliability score across the 67 scoreable narratives is 0.946 (SD = 0.049), indicating that the extraction model's sign assignments and feature-membership decisions are largely consistent across repeated runs.

Chain-of-thought narratives are extracted more consistently than direct narratives. The gap is sharpest for top-*k* set agreement: direct achieves a mean of 0.525 (SD = 0.226) versus 0.903 (SD = 0.157) for chain-of-thought. For direct narratives, repeated independent extractions agree on which three features the narrative treats as most important only about half the time, compared to nine times in ten for chain-of-thought. Rank swap flags on direct narratives are therefore a lower bound on true rank swap prevalence: ambiguous narratives are harder to extract consistently and so are more likely to generate false negatives (missed rank swaps) than false positives (spurious flags), because an unstable extraction that returns different top-*k* sets across runs may converge on a set that happens to match the SHAP top-*k*. Per-feature sign agreement is high for all features (minimum 0.935 for `relationship_Non_Husband`), so sign inversion flags are reliable under both strategies.

**Table 5.4.** Extraction robustness by prompt strategy (67 scoreable narratives: *n* = 33 direct, *n* = 34 chain-of-thought).

| Metric                      | Direct (*n* = 33)  | Chain-of-thought (*n* = 34) | Overall (*n* = 67) |
| --------------------------- | ------------------ | --------------------------- | ------------------ |
| Narrative reliability score | 0.921 (SD = 0.043) | 0.970 (SD = 0.044)          | 0.946 (SD = 0.049) |
| Top-*k* set agreement       | 0.525 (SD = 0.226) | 0.903 (SD = 0.157)          | 0.717 (SD = 0.271) |

---

## 6. Discussion and Conclusion

### 6.1 Interpretation

The results separate qualitatively different failure modes; we discuss each in turn.

#### Sign inversion

Sign inversion is a local, within-feature error: the model discusses the right feature but assigns it the wrong direction. Chain-of-thought's explicit verification step — requiring the model to state the top feature and verify its sign as a reasoning step before writing the narrative — reduces this error from 60.0% to 4.2%. The intervention targets a specific, verifiable piece of information at generation time, and it works.

#### Rank swap

Rank swap is a different kind of failure and is not resolved by chain-of-thought prompting. It concerns which features the model treats as the most influential drivers of the prediction, not how it characterises any individual feature. Chain-of-thought reduces rank swap from 80.8% to 57.7%, but 57.7% remains a majority of narratives. The feature-mention data offers a partial explanation: `capital_gain` and `capital_loss` appear in fewer than a quarter of narratives despite their often-large SHAP magnitudes, while `hours_per_week` and `age` appear in almost every narrative regardless of strategy. The model appears to weight features by their interpretive salience — age and hours worked are easy to incorporate into a plausible income narrative — rather than by their SHAP magnitude. This is a hypothesis advanced by this paper rather than a directly tested finding; isolating interpretive salience as a cause would require systematically varying which features carry large SHAP values while holding their narrative plausibility constant, and we leave this to follow-up work. Chain-of-thought's ranking step partially corrects the bias by forcing the model to acknowledge which features rank highest by SHAP before writing, but the final narrative still underrepresents capital features.

The co-occurrence pattern in §5.4 is consistent with the same mechanism. Rank swap co-occurs with sign inversion in 22.4% of narratives and with omission in 13.6%; sign inversion without rank swap is only 5.5%. The pattern is consistent with a shared generative mechanism: when the model assigns SHAP attributions to the wrong features, it simultaneously inverts direction for those features and displaces the true top-*k* drivers from the set it presents as most important. The mechanism is not directly tested here and remains a hypothesis to be verified against a human-coded subsample.

#### Feature fabrication

Feature fabrication is absent throughout: Llama 3 70B never introduces a feature name outside the valid set, even in the direct condition where it receives no structured reasoning scaffold. The feature list in the prompt appears to function as a constraint on vocabulary; no synonyms or paraphrases of feature names were detected in the `unknown_features` field. The model's failures are in attribution — assigning effects to the wrong features or in the wrong direction — not in invention.

#### Omission

Omission is strategy-invariant: 15.4% versus 19.0%, with no significant paired difference (McNemar exact *p* = .405). Chain-of-thought's explicit ranking step identifies the top-five features by |SHAP| but does not guarantee that all of them appear in the final narrative. The slight directional increase under chain-of-thought (and its smaller `max_tokens` budget; see §6.2) is consistent with token-length pressure as a contributing factor. Whether omission reflects a systematic bias against specific feature types — capital gains and losses are mentioned least, as §5.5 shows — or is a random generation artefact is a question for future work.

### 6.2 Limitations

**Gold-standard validation.** No human-coded subsample exists for the four hallucination classes. The robustness check (§4.3) measures internal consistency of the extraction model across repeated high-temperature runs, not its agreement with human judgement. Of the 204 narratives flagged as hallucinated, the fraction that a domain-expert annotator would also flag has not been measured. We treat the prevalence rates reported here as automated-pipeline estimates calibrated only by internal robustness, and we identify gold-standard validation — a dual-annotated 30–50 narrative subsample with Cohen's κ, against which the pipeline's precision and recall can be computed — as the immediate next step.

**Detection coverage.** Extraction errors propagate to hallucination flags: a feature mentioned but mis-parsed as a different feature would generate a false sign inversion or rank swap flag. The robustness check partially addresses this by identifying narratives where repeated high-temperature extractions disagree. The rank swap rates for direct narratives in particular should be read as lower bounds: top-*k* set agreement for that strategy is 0.525, so extraction uncertainty is high and some true swaps will be missed by the pipeline.

**Dataset and model scope.** Results are specific to Adult Income, Llama 3 70B for generation, and Mistral 7B Instruct v0.3 for extraction, both at temperature 0.0. Results may not generalise to other tabular domains (e.g., credit risk, clinical prediction), non-linear model classes (e.g., gradient boosting, neural networks), other LLM families, or higher generation temperatures. The logistic regression model from OpenXAI produces SHAP values of moderate variance across instances; a model with more extreme attribution distributions might produce different hallucination patterns.

**Extraction-model capacity.** Mistral 7B is an order of magnitude smaller than the Llama 3 70B generator. A more capable extraction model (e.g., GPT-4o, Claude, or Llama 3 70B itself in an extraction role) might recover feature claims that Mistral misses, which would shift the hallucination-rate estimates. The robustness subsample did not reveal a systematic class of features that Mistral failed to extract, but the capacity asymmetry is a deployment-cost trade-off rather than a methodologically neutral choice.

**Sample size.** 150 instances are sufficient to detect the large differences observed — sign inversion at 60.0% vs. 4.2% would be detectable with far fewer — but small per-type differences between strategies (e.g., omission at 15.4% vs. 19.0%) are likely underpowered. The 28 parse failures (9.3%) reduce effective sample sizes and introduce missing data that is not uniformly distributed across strategies; the §5.1 sensitivity check bounds the resulting uncertainty.

**Generation length.** Direct narratives allow up to 2,048 tokens; chain-of-thought narratives are capped at 1,024. The slight directional increase in omission rates under chain-of-thought (15.4% vs. 19.0%) is not significant, but a token budget that halves the available length cannot be ruled out as a contributing factor. Future work comparing strategies at matched token budgets would resolve this.

### 6.3 Conclusion

LLMs do not reliably report what SHAP values show. Across 272 valid evaluations of Llama 3 70B narratives on the Adult Income benchmark, 75.0% contain at least one faithfulness failure. Rank swap, emphasising the wrong features as the most important drivers, affects 68.8% of narratives overall and 57.7% even under chain-of-thought prompting. Feature fabrication is absent under both strategies. Chain-of-thought nearly eliminates directional errors (sign inversion at 4.2%), but the problem of feature selection fidelity is not resolved by prompting alone.

Faithfulness cannot be assumed. Users of LLM-generated SHAP narratives, in credit decisions, clinical support, or audit contexts, may be reading explanations that emphasise different features than the model actually weighted most highly. Automated verification of the kind described here is tractable and should accompany any deployment that uses LLMs to narrate feature attributions. The detection framework is released as an open-source resource [CITATION NEEDED: anonymised repository link for review; public URL for camera-ready].

To our knowledge, this paper provides the first systematic study of hallucination rates at scale across a full SHAP-narrative evaluation set, varying prompt strategy as an experimental variable and applying automated detection. Papers 2 and 3 of this thesis address the human consequences of each hallucination type and the conditions under which users detect or accept faithfulness failures.

---

## Appendix A — Prompt Templates

*(Reproduce both templates verbatim; cross-reference §3.3.)*

### A.1 Direct template (`config/prompts/narrative.j2`)

### A.2 Chain-of-thought template (`config/prompts/chain_of_thought.j2`)

---

## Appendix B — Dataset Characteristics

*(Tables: feature names, types, class distribution, and mean |SHAP| per feature for the 150-instance sample. Mean |SHAP| per feature is required to support the claim in §5.5 that capital features have often-large SHAP magnitudes.)*

---

## Appendix C — Evaluation Rule Pseudocode and JSON Schema

*(Extraction JSON schema and pseudocode for the four comparison rules; maps to `src/evaluation/compare_to_shap.py`.)*

---

## References

- Agarwal, C., Krishna, S., Saxena, E., Pawelczyk, M., Johnson, N., Puri, I., Zitnik, M., & Lakkaraju, H. (2022). OpenXAI: Towards a Transparent Evaluation of Model Explanations. *NeurIPS*.
- Agarwal, C., Tanneru, S. H., & Lakkaraju, H. (2024). Faithfulness vs. Plausibility: On the (Un)Reliability of Explanations from Large Language Models. *arXiv:2402.04614*.
- AlMarri, S., Ravaut, M., Juhasz, K., Marti, G., Al Ahbabi, H., & Elfadel, I. (2025). Measuring What LLMs Think They Do: SHAP Faithfulness and Deployability on Financial Tabular Classification. *AAAI 2026*. arXiv:2512.00163.
- Dua, D., & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences. [VERIFY YEAR]
- Ichmoukamedov, T., Hinns, J., & Martens, D. (2024). How good is my story? Towards quantitative metrics for evaluating LLM-generated XAI narratives. *arXiv:2412.10220*. [VERIFY SPELLING OF FIRST AUTHOR]
- Ji, Z., et al. (2023). Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*, 55(12).
- Jiang, A., et al. (2023). Mistral 7B. *arXiv:2310.06825*. [CITATION NEEDED — add full author list]
- Kohavi, R. (1996). Scaling up the accuracy of naive-Bayes classifiers: A decision-tree hybrid. *Proceedings of KDD'96*.
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic uncertainty: Linguistic invariances for uncertainty quantification in language models. *ICLR*. arXiv:2302.09664.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- Martens, D., Hinns, J., Dams, C., Vergouwen, M., & Evgeniou, T. (2024). Tell Me a Story! Narrative-Driven XAI with Large Language Models. *arXiv:2309.17057*.
- Matton, K., Ness, R. O., Guttag, J., & Kiciman, E. (2025). Walk the Talk? Measuring the Faithfulness of Large Language Model Explanations. *ICLR 2025*. arXiv:2504.14150.
- Maynez, J., et al. (2020). On Faithfulness and Factuality in Abstractive Summarization. *ACL*.
- Meta AI. (2024). The Llama 3 Herd of Models. *arXiv:2407.21783*. [CITATION NEEDED — verify arXiv ID and author list]
- Turpin, M., Michael, J., Perez, E., & Bowman, S. (2023). Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. *NeurIPS*, 36, 74952–74965.
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS*.

*(Outstanding citations to resolve before submission: SHAP-comprehension difficulty for lay users; LLM narration of SHAP tables; domain-deployment studies for SHAP in credit/clinical/audit contexts; NLG-for-XAI literature; persuasiveness-versus-accuracy evidence for §1; AlMarri et al. Kendall's τ page/section; interpretive-salience bias in LLM generation for §6.1.)*
