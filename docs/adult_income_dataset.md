# Adult Income Dataset — Reference Documentation

## Overview

The Adult Income dataset (also known as the "Census Income" dataset) is a widely used benchmark for binary classification in machine learning and algorithmic fairness research. It was extracted from the 1994 United States Census Bureau database by Barry Becker and donated to the UCI Machine Learning Repository. The prediction task is to determine whether a person's annual income exceeds $50,000.

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository (ID: 2) |
| Original extractor | Barry Becker, 1996 |
| Census year | 1994 |
| Task type | Binary classification |
| Total instances | 48,842 (train: 32,561 + test: 16,281) |
| Features | 14 |
| Missing values | Yes (~7% of records; marked as "?") |
| Class distribution | <=50K: ~76%, >50K: ~24% |

---

## Target Variable

| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| `income` | Binary | `<=50K` (0), `>50K` (1) | Annual income relative to $50,000 threshold. Derived from the Census field `gross_income`. |

---

## Feature Variables

### 1. `age`
- **Type:** Continuous (integer)
- **Range:** 17–90
- **Description:** Age of the individual in years.
- **Notes:** Strong predictor; older individuals tend to have higher incomes due to accumulated experience and career progression.

### 2. `workclass`
- **Type:** Categorical
- **Values:**

| Value | Description |
|-------|-------------|
| `Private` | Employee in a private company |
| `Self-emp-not-inc` | Self-employed, unincorporated business |
| `Self-emp-inc` | Self-employed, incorporated business |
| `Federal-gov` | Federal government employee |
| `Local-gov` | Local government employee |
| `State-gov` | State government employee |
| `Without-pay` | Working without pay (e.g., family business) |
| `Never-worked` | Has never worked |
| `?` | Missing value |

- **Notes:** ~5.6% of records have missing workclass. The most common class is `Private` (~69%).

### 3. `fnlwgt`
- **Type:** Continuous (integer)
- **Range:** 12,285–1,484,705
- **Description:** Final sampling weight. This is a Census-derived weight representing the number of people in the population that the census record is estimated to represent. Higher values indicate the record represents more people.
- **Notes:** Not a demographic predictor per se; reflects survey methodology. Often dropped in fairness analyses. OpenXAI retains it as a feature.

### 4. `education`
- **Type:** Categorical (ordered)
- **Values (approximate ordering by level):**

| Value | Description |
|-------|-------------|
| `Preschool` | Less than 1st grade |
| `1st-4th` | Primary school, grades 1–4 |
| `5th-6th` | Primary school, grades 5–6 |
| `7th-8th` | Middle school, grades 7–8 |
| `9th` | High school, 9th grade |
| `10th` | High school, 10th grade |
| `11th` | High school, 11th grade |
| `12th` | High school, 12th grade (no diploma) |
| `HS-grad` | High school graduate or GED |
| `Some-college` | Some college, no degree |
| `Assoc-voc` | Associate degree, vocational |
| `Assoc-acdm` | Associate degree, academic |
| `Bachelors` | Bachelor's degree |
| `Masters` | Master's degree |
| `Prof-school` | Professional school degree (e.g., MD, JD) |
| `Doctorate` | Doctoral degree |

- **Notes:** Collinear with `education_num` (both encode educational attainment; the numeric version is used in most models).

### 5. `education_num`
- **Type:** Ordinal (integer)
- **Range:** 1–16
- **Description:** Numeric encoding of educational attainment level. Maps directly to the `education` categorical variable.

| `education_num` | `education` label |
|-----------------|-------------------|
| 1 | Preschool |
| 2 | 1st-4th |
| 3 | 5th-6th |
| 4 | 7th-8th |
| 5 | 9th |
| 6 | 10th |
| 7 | 11th |
| 8 | 12th |
| 9 | HS-grad |
| 10 | Some-college |
| 11 | Assoc-voc |
| 12 | Assoc-acdm |
| 13 | Bachelors |
| 14 | Masters |
| 15 | Prof-school |
| 16 | Doctorate |

### 6. `marital_status`
- **Type:** Categorical
- **Values:**

| Value | Description |
|-------|-------------|
| `Married-civ-spouse` | Married, civilian spouse present |
| `Divorced` | Divorced |
| `Never-married` | Never married |
| `Separated` | Separated |
| `Widowed` | Widowed |
| `Married-spouse-absent` | Married, spouse absent (e.g., overseas) |
| `Married-AF-spouse` | Married, Armed Forces spouse |

- **Notes:** Strong predictor. `Married-civ-spouse` is strongly associated with higher income in this dataset, likely reflecting dual-income households and cohort effects.

### 7. `occupation`
- **Type:** Categorical
- **Values:**

| Value | Description |
|-------|-------------|
| `Tech-support` | Technical support |
| `Craft-repair` | Craft and repair trades |
| `Other-service` | Other service occupations |
| `Sales` | Sales |
| `Exec-managerial` | Executive and managerial |
| `Prof-specialty` | Professional specialty (e.g., doctor, lawyer, engineer) |
| `Handlers-cleaners` | Handlers and cleaners |
| `Machine-op-inspct` | Machine operators and inspectors |
| `Adm-clerical` | Administrative and clerical |
| `Farming-fishing` | Farming and fishing |
| `Transport-moving` | Transportation and moving |
| `Priv-house-serv` | Private household services |
| `Protective-serv` | Protective services (e.g., police, firefighter) |
| `Armed-Forces` | Armed Forces |
| `?` | Missing value |

- **Notes:** ~5.7% of records have missing occupation, usually coinciding with missing `workclass`.

### 8. `relationship`
- **Type:** Categorical
- **Values:**

| Value | Description |
|-------|-------------|
| `Wife` | Spouse is wife (respondent is husband) |
| `Own-child` | Respondent is a child of the head of household |
| `Husband` | Respondent is husband (head of household) |
| `Not-in-family` | Not in a family unit |
| `Other-relative` | Other relative |
| `Unmarried` | Unmarried |

- **Notes:** Partially redundant with `marital_status`. Encodes the respondent's role within their household structure.

### 9. `race`
- **Type:** Categorical
- **Values:**

| Value | Description |
|-------|-------------|
| `White` | White |
| `Asian-Pac-Islander` | Asian or Pacific Islander |
| `Amer-Indian-Eskimo` | American Indian or Eskimo |
| `Other` | Other race |
| `Black` | Black |

- **Notes:** A protected attribute in fairness literature. Substantial disparities in income prediction exist across racial groups in this dataset.

### 10. `sex`
- **Type:** Binary categorical
- **Values:** `Male`, `Female`
- **Notes:** A protected attribute. The dataset reflects the income gap of the 1994 labour market: male respondents are overrepresented in the `>50K` class.

### 11. `capital_gain`
- **Type:** Continuous (integer)
- **Range:** 0–99,999
- **Description:** Capital gains in US dollars for the past year (income from investment sources other than wages/salary).
- **Notes:** Highly skewed; ~92% of individuals report zero capital gains. Non-zero values are strong predictors of >50K income. Values above $3,000 are rare but highly discriminative.

### 12. `capital_loss`
- **Type:** Continuous (integer)
- **Range:** 0–4,356
- **Description:** Capital losses in US dollars for the past year.
- **Notes:** Similarly skewed; ~95% report zero capital loss. Non-zero values often co-occur with non-zero capital gains and correlate with higher income.

### 13. `hours_per_week`
- **Type:** Continuous (integer)
- **Range:** 1–99
- **Description:** Number of hours worked per week (self-reported).
- **Notes:** Mean ~40 hours. Part-time workers (<35 h/week) are less likely to earn >50K. Extreme values (>60 h) correlate with higher income in certain occupations.

### 14. `native_country`
- **Type:** Categorical
- **Values:** 41 unique countries/regions including `United-States`, `Mexico`, `Philippines`, `Germany`, `Canada`, `Puerto-Rico`, `El-Salvador`, `India`, `Cuba`, `England`, `Jamaica`, `South`, `China`, `Italy`, `Dominican-Republic`, `Vietnam`, `Guatemala`, `Japan`, `Poland`, `Columbia`, `Taiwan`, `Haiti`, `Iran`, `Portugal`, `Nicaragua`, `Peru`, `France`, `Greece`, `Ecuador`, `Ireland`, `Hong`, `Trinadad&Tobago`, `Cambodia`, `Thailand`, `Laos`, `Yugoslavia`, `Outlying-US(Guam-USVI-etc)`, `Hungary`, `Honduras`, `Scotland`, `Holand-Netherlands`, and `?` (missing).
- **Notes:** ~1.8% missing values. ~90% of records are `United-States`. **OpenXAI drops this feature during preprocessing** (see below).

---

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total records | 48,842 |
| Training split | 32,561 |
| Test split | 16,281 |
| Class <=50K | ~75.9% (train) |
| Class >50K | ~24.1% (train) |
| Missing values | ~3,620 records (~7.4%) |
| Numeric features | 6 (`age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`) |
| Categorical features | 8 (`workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`) |

---

## OpenXAI Preprocessing

When loaded via the OpenXAI framework, the dataset undergoes the following transformations:

1. **Missing value removal:** Records with "?" in `workclass`, `occupation`, or `native_country` are dropped.
2. **Feature dropping:** `native_country` is excluded. The processed dataset has **13 features** (not 14).
3. **Categorical encoding:** All categorical features are one-hot encoded or label-encoded, then converted to float tensors.
4. **Feature normalisation:** All features are scaled to [0, 1] using min-max normalisation applied over the training set. This is reflected in `data/processed/adult.csv` where feature values appear as floats between 0 and 1.
5. **Train/test split:** OpenXAI uses the original UCI train/test split (32,561 / 16,281).

The processed files in this project (`data/processed/adult.csv` and `data/processed/adult_ann.csv`) contain normalised feature values alongside SHAP attributions computed on those normalised values.

The raw original dataset with string categorical values is available at `data/raw/adult_original.csv`.

---

## Feature Encoding in OpenXAI (Processed Files)

Since values in `data/processed/adult.csv` are normalised floats, interpreting them requires knowing the encoding:

| Feature | Encoding | Notes |
|---------|----------|-------|
| `age` | (age - 17) / (90 - 17) | Min-max over observed range |
| `workclass` | Label-encoded 0–7, then normalised | Arbitrary ordinal |
| `fnlwgt` | Min-max normalised | Large range compressed to [0,1] |
| `education` | Label-encoded 0–15, then normalised | Follows `education_num` ordering |
| `education_num` | (edu_num - 1) / 15 | Exact ordinal scale |
| `marital_status` | Label-encoded 0–6, then normalised | Arbitrary ordinal |
| `occupation` | Label-encoded 0–13, then normalised | Arbitrary ordinal |
| `relationship` | Label-encoded 0–5, then normalised | Arbitrary ordinal |
| `race` | Label-encoded 0–4, then normalised | Arbitrary ordinal |
| `sex` | 0 = Female, 1 = Male | Binary |
| `capital_gain` | Min-max normalised | Highly skewed; 0 → 0.0, 99999 → 1.0 |
| `capital_loss` | Min-max normalised | Highly skewed |
| `hours_per_week` | (hours - 1) / 98 | Min-max over observed range |

---

## Citations

**Primary dataset citation:**

> Becker, B., & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

**Original paper and data preparation:**

> Kohavi, R. (1996). Scaling up the accuracy of Naive-Bayes classifiers: A decision-tree hybrid. *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)*, 202–207.

**OpenXAI benchmark (dataset used via this framework):**

> Agarwal, C., Krishna, S., Saxena, E., Pawelczyk, M., Johnson, N., Puri, I., Zitnik, M., & Lakkaraju, H. (2022). OpenXAI: Towards a transparent evaluation of model explanations. *Advances in Neural Information Processing Systems (NeurIPS 2022)*, 35. https://arxiv.org/abs/2206.11104

**UCI ML Repository:**

> Dua, D., & Graff, C. (2019). *UCI Machine Learning Repository*. University of California, Irvine, School of Information and Computer Sciences. https://archive.ics.uci.edu

---

## Relevance to This Project

In this project (Paper 1: LLM Faithfulness in XAI Narratives), the Adult Income dataset serves as one of two evaluation benchmarks. SHAP values are computed for each instance using OpenXAI's pre-trained models (LR and ANN). These ground-truth SHAP values are then:

1. Injected into prompts for three LLMs (Claude Opus, Llama 3 70B, Mistral 7B) to generate natural-language narratives.
2. Used to evaluate narratives against five hallucination types: sign inversion, rank swap, feature fabrication, magnitude distortion, and omission.

The 13 features available in the OpenXAI-processed version (excluding `native_country`) are the features referenced in all generated narratives and evaluations.
