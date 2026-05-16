# OpenXAI Framework — Reference Documentation

## What is OpenXAI?

OpenXAI is an open-source benchmark framework for the systematic, transparent evaluation of post hoc explanation methods. It provides a unified interface for loading datasets, pre-trained models, explanation algorithms, and evaluation metrics, enabling standardised comparisons across XAI methods.

| Property | Value |
|----------|-------|
| Full name | OpenXAI: Towards a Transparent Evaluation of Model Explanations |
| Published | NeurIPS 2022 — Datasets and Benchmarks Track |
| GitHub | https://github.com/AI4LIFE-GROUP/OpenXAI |
| Website | https://open-xai.github.io |
| Authors | Agarwal, Krishna, Saxena, Pawelczyk, Johnson, Puri, Zitnik, Lakkaraju (2022) |
| License | MIT |

OpenXAI is **not published on PyPI**. It must be installed from source:

```bash
git clone https://github.com/AI4LIFE-GROUP/OpenXAI.git
cd OpenXAI && pip install -e .
```

---

## Datasets

OpenXAI includes eight datasets: three real-world tabular datasets and five synthetic datasets.

| Dataset | Type | Instances | Features | Task |
|---------|------|-----------|----------|------|
| Adult Income | Real-world | 48,842 | 13* | Binary classification (income >50K) |
| German Credit | Real-world | 1,000 | 20 | Binary classification (credit risk) |
| COMPAS | Real-world | ~7,000 | ~10 | Binary classification (recidivism) |
| Syn1–Syn5 | Synthetic | Varies | Varies | Binary classification (with known GT explanations) |

*OpenXAI drops `native_country` from Adult Income, leaving 13 features.

**Important:** Synthetic datasets (Syn1–Syn5) have analytically-derived **ground-truth explanations**, enabling computation of the full set of faithfulness metrics (FA, RA, SA, SRA, PRA). Real-world datasets (Adult, German, COMPAS) do **not** have ground-truth explanations; only ground-truth-free metrics (PGI, PGU, RIS, RRS, ROS) can be computed for them.

**This project** uses Adult Income only. German Credit data and preparation scripts are archived under `archive/german_credit/` and are not loaded by the active pipeline.

---

## Adult Income in OpenXAI

### Python API — Loading Data

```python
from openxai.dataloader import ReturnLoaders

loader_train, loader_test = ReturnLoaders(
    data_name='adult',   # OpenXAI internal name for Adult Income
    download=True,       # Downloads if not cached
)

# Iterate batches (returns PyTorch tensors)
for X_batch, y_batch in loader_test:
    ...
```

### Python API — Loading Pre-trained Models

```python
from openxai import LoadModel

model = LoadModel(
    data_name='adult',
    ml_model='lr',       # 'lr' or 'ann'
    pretrained=True,
)
```

---

## Pre-trained Models

OpenXAI provides two pre-trained models for the Adult Income dataset.

### Logistic Regression (LR)

| Property | Value |
|----------|-------|
| Model type | Logistic Regression |
| Implementation | scikit-learn (wrapped as a PyTorch-compatible interface) |
| Features | 13 (normalised float inputs) |
| Output | Binary: 0 (<=50K) or 1 (>50K) |
| Objective | Log-loss minimisation with L2 regularisation |
| Use in this project | SHAP values in `data/processed/adult.csv` |

The LR model provides a linear, globally interpretable baseline. SHAP values for LR are exact (no approximation needed) via the Linear SHAP explainer.

### Artificial Neural Network (ANN)

| Property | Value |
|----------|-------|
| Model type | Feedforward Neural Network (Deep Neural Network) |
| Architecture | Input(13) → Dense(100, ReLU) → Dense(100, ReLU) → Dense(2, Softmax) |
| Hidden layers | 2 fully connected layers |
| Nodes per layer | 100 |
| Activation (hidden) | ReLU |
| Output activation | Softmax (2-class probability) |
| Optimiser | Adam |
| Implementation | PyTorch |
| Use in this project | SHAP values in `data/processed/adult_ann.csv` |

The ANN model provides a non-linear, higher-capacity alternative. SHAP values for ANN are approximate (Kernel SHAP or Deep SHAP).

---

## Explainers Supported

OpenXAI implements seven post hoc feature attribution methods accessible through a unified `Explainer` interface.

```python
from openxai import Explainer

explainer = Explainer(
    method='shap',   # see table below
    model=model,
    param_dict={},
)
shap_values = explainer.get_explanations(X_tensor, y_tensor)
# Returns: Tensor of shape (n_instances, n_features)
```

| Method Key | Full Name | Type | Description |
|------------|-----------|------|-------------|
| `shap` | SHAP (Kernel SHAP) | Local, model-agnostic | Shapley value-based attributions using a weighted linear regression over feature coalitions. Satisfies efficiency, symmetry, and dummy axioms. |
| `lime` | LIME | Local, model-agnostic | Fits a sparse linear model to a perturbed neighbourhood around each instance. |
| `vanilla_gradients` | Vanilla Gradients | Local, gradient-based | Gradient of the output class score with respect to input features: ∂f(x)/∂x. Requires differentiable model. |
| `grad_x_input` | Gradient × Input | Local, gradient-based | Element-wise product of gradient and input: (∂f(x)/∂x) ⊙ x. Scales attributions by feature magnitude. |
| `smoothgrad` | SmoothGrad | Local, gradient-based | Average of gradients computed on input x plus Gaussian noise samples. Reduces gradient noise. |
| `integrated_gradients` | Integrated Gradients | Local, gradient-based | Path integral of gradients from a baseline x' to x: (x−x')·∫₀¹ (∂f(x'+α(x−x'))/∂x) dα. Satisfies completeness and sensitivity axioms. |
| `random` | Random | Baseline | Assigns random attributions. Serves as a lower-bound reference. |

**This project uses `shap` (Kernel SHAP) as its explanation method** for both LR and ANN models. The SHAP values are the ground truth against which LLM narratives are evaluated.

---

## Evaluation Metrics

OpenXAI implements **22 quantitative metrics** across four categories. Below is the complete reference with definitions.

> **Convention:** Let E = generated explanation (vector of feature attributions), E* = ground-truth explanation (vector), K = top-K features, f = model prediction.

---

### Category 1: Faithfulness Metrics — With Ground Truth

These metrics compare explanations to analytically-derived ground-truth explanations. They are only applicable to **synthetic datasets** where the data-generating process is known. They **cannot** be computed for Adult Income.

#### FA — Feature Agreement

**Definition:** The fraction of top-K features in the generated explanation E that also appear in the top-K features of the ground-truth explanation E*.

$$\text{FA}(E, E^*, K) = \frac{|\text{top-K}(E) \cap \text{top-K}(E^*)|}{K}$$

**Range:** [0, 1]. Higher is better. FA = 1 means the same K features are identified by both explanations.

**Hallucination analogue:** Omission detection. If FA < 1, important features are being missed or replaced.

---

#### RA — Rank Agreement

**Definition:** The fraction of top-K features that appear in both top-K sets **and** share the same rank position.

$$\text{RA}(E, E^*, K) = \frac{|\{k \in \text{top-K}(E) \cap \text{top-K}(E^*) : \text{rank}_E(k) = \text{rank}_{E^*}(k)\}|}{K}$$

**Range:** [0, 1]. Stricter than FA; requires both feature presence and correct ordering.

**Hallucination analogue:** Rank swap detection. RA < FA implies features are present but in the wrong order.

---

#### SA — Sign Agreement

**Definition:** The fraction of top-K features shared between E and E* that have the same attribution sign (direction of effect).

$$\text{SA}(E, E^*, K) = \frac{|\{k \in \text{top-K}(E) \cap \text{top-K}(E^*) : \text{sign}(E_k) = \text{sign}(E^*_k)\}|}{|\text{top-K}(E) \cap \text{top-K}(E^*)|}$$

**Range:** [0, 1]. SA = 1 means every shared feature has the correct direction (positive/negative contribution).

**Hallucination analogue:** Sign inversion detection. SA < 1 means some features have inverted directions in the narrative.

---

#### SRA — Signed Rank Agreement

**Definition:** The fraction of top-K features that are in both top-K sets, share the same rank position, **and** have the same sign.

$$\text{SRA}(E, E^*, K) = \frac{|\{k : k \in \text{top-K}(E) \cap \text{top-K}(E^*),\ \text{rank}_E(k) = \text{rank}_{E^*}(k),\ \text{sign}(E_k) = \text{sign}(E^*_k)\}|}{K}$$

**Range:** [0, 1]. The strictest agreement metric combining rank and sign.

---

#### PRA — Pairwise Rank Agreement

**Definition:** The fraction of all feature pairs (i, j) where the relative ordering is consistent between E and E*.

$$\text{PRA}(E, E^*, K) = \frac{|\{(i,j) : i \neq j,\ (E_i > E_j) \iff (E^*_i > E^*_j)\}|}{\binom{|F|}{2}}$$

**Range:** [0, 1]. Measures global ranking consistency rather than just top-K agreement.

---

### Category 2: Faithfulness Metrics — Without Ground Truth

These metrics do not require known ground-truth explanations. They are applicable to **real-world datasets** including Adult Income.

#### PGI — Prediction Gap on Important Features

**Definition:** The average change in the model's predicted probability when the top-K features (as identified by explanation E) are perturbed (set to a reference value, e.g., mean or zero).

$$\text{PGI}(E, f, X, K) = \mathbb{E}_{x \in X}\left[|f(x) - f(\tilde{x}_{\text{important}}|\right]$$

where $\tilde{x}_{\text{important}}$ is x with the top-K features replaced by a reference value.

**Interpretation:** A high PGI indicates that the features the explanation identifies as important genuinely affect the model's output when perturbed — the explanation is faithful to the model. If PGI is low, the "important" features are not actually driving predictions.

**Range:** [0, 1] (probability difference). Higher is better.

**Hallucination analogue:** Magnitude distortion. If an LLM narrative overstates the importance of features with low PGI contribution, that is a magnitude distortion.

---

#### PGU — Prediction Gap on Unimportant Features

**Definition:** The average change in predicted probability when the bottom-K features (identified as unimportant by E) are perturbed.

$$\text{PGU}(E, f, X, K) = \mathbb{E}_{x \in X}\left[|f(x) - f(\tilde{x}_{\text{unimportant}}|\right]$$

**Interpretation:** A low PGU indicates that features deemed unimportant by the explanation genuinely have little effect on predictions — a desirable property. High PGU indicates the explanation is incorrectly identifying truly influential features as unimportant.

**Range:** [0, 1]. Lower is better (when paired with high PGI).

**Combined use:** Good explanations have **high PGI and low PGU**. The ratio PGI / (PGI + PGU) is sometimes used as a single faithfulness score.

---

### Category 3: Stability / Robustness Metrics

These metrics measure how consistent explanations are when inputs are slightly perturbed. A stable explanation should not change drastically for similar inputs.

#### RIS — Relative Input Stability

**Definition:** The maximum ratio of explanation change to input change across perturbed neighbours of x.

$$\text{RIS}(E, x) = \max_{x' \in \mathcal{N}(x)} \frac{\|E(x) - E(x')\|_2 / \|E(x)\|_2}{\|x - x'\|_2 / \|x\|_2}$$

where $\mathcal{N}(x)$ is a neighbourhood of perturbed inputs.

**Interpretation:** Lower RIS = more stable explanations. A high RIS means the explanation changes disproportionately more than the input — suggesting the explainer is sensitive to noise rather than true model behaviour.

**Range:** [0, ∞). Lower is better.

---

#### RRS — Relative Representation Stability

**Definition:** Similar to RIS but measured in the model's internal representation (hidden layer activations) space rather than input space.

$$\text{RRS}(E, x) = \max_{x' \in \mathcal{N}(x)} \frac{\|E(x) - E(x')\|_2 / \|E(x)\|_2}{\|h(x) - h(x')\|_2 / \|h(x)\|_2}$$

where h(x) is a model representation (e.g., penultimate layer output).

**Interpretation:** Measures whether explanations are stable relative to changes in the model's learned representation. More meaningful than RIS for understanding explanation robustness in terms of model internals.

**Range:** [0, ∞). Lower is better.

---

#### ROS — Relative Output Stability

**Definition:** Similar to RIS/RRS but normalised by change in the model's output probability.

$$\text{ROS}(E, x) = \max_{x' \in \mathcal{N}(x)} \frac{\|E(x) - E(x')\|_2 / \|E(x)\|_2}{|f(x) - f(x')| / |f(x)|}$$

**Interpretation:** Measures whether explanations change proportionally to model output changes. A low ROS means the explanation is more stable than the output — ideal behaviour. A high ROS means explanations change even when model outputs are similar.

**Range:** [0, ∞). Lower is better.

---

### Category 4: Fairness Metrics

OpenXAI includes 11 group-based fairness metrics that measure whether explanation quality is consistent across demographic subgroups (e.g., by race, sex). These compute the metrics above (FA, RA, SA, PGI, PGU, RIS, RRS, ROS) separately per subgroup and report disparity scores. Detailed definitions are in the OpenXAI paper (Appendix D).

---

## Summary Table

| Metric | Category | GT Required | Applicable to Adult | Optimise |
|--------|----------|-------------|---------------------|---------|
| FA | Faithfulness | Yes (synthetic only) | No | Maximise |
| RA | Faithfulness | Yes (synthetic only) | No | Maximise |
| SA | Faithfulness | Yes (synthetic only) | No | Maximise |
| SRA | Faithfulness | Yes (synthetic only) | No | Maximise |
| PRA | Faithfulness | Yes (synthetic only) | No | Maximise |
| PGI | Faithfulness | No | **Yes** | Maximise |
| PGU | Faithfulness | No | **Yes** | Minimise |
| RIS | Stability | No | **Yes** | Minimise |
| RRS | Stability | No | **Yes** | Minimise |
| ROS | Stability | No | **Yes** | Minimise |
| Fairness (×11) | Fairness | Varies | Partial | Minimise disparity |

Computed metrics for this project are saved in `outputs/xai_metrics/adult_openxai_metrics.csv`.

---

## Mapping to Project Hallucination Taxonomy

The hallucination types used in this project (see `CLAUDE.md`) have direct analogues in OpenXAI's metric framework:

| Hallucination Type | Detection Method (in this project) | OpenXAI Metric Analogue |
|-------------------|-------------------------------------|-------------------------|
| **Sign inversion** | Compare stated direction vs. SHAP sign | SA (Sign Agreement) — on synthetic data; manual sign check on real data |
| **Rank swap** | Compare stated rank vs. SHAP rank | RA (Rank Agreement) |
| **Feature fabrication** | Check narrative tokens vs. feature name list | No direct metric — structural check |
| **Magnitude distortion** | Compare relative SHAP magnitude vs. stated magnitude words | PGI/PGU — indirect; magnitude distortion inflates/deflates feature importance |
| **Omission** | Check top-K SHAP features appear in narrative | FA (Feature Agreement) |

**Key distinction:** OpenXAI metrics evaluate explanation *methods* (e.g., SHAP vs. Integrated Gradients). This project applies an analogous evaluation to *LLM narratives* of SHAP values — measuring whether the LLM faithfully represents the pre-computed SHAP output, not whether SHAP itself is faithful.

---

## Python API Reference (Key Functions)

```python
# Load data
from openxai.dataloader import ReturnLoaders, ReturnTrainTestX
loader_train, loader_test = ReturnLoaders(data_name='adult', download=True)

# Load model
from openxai import LoadModel
model_lr  = LoadModel(data_name='adult', ml_model='lr',  pretrained=True)
model_ann = LoadModel(data_name='adult', ml_model='ann', pretrained=True)

# Generate explanations
from openxai import Explainer
explainer = Explainer(method='shap', model=model_lr, param_dict={})
attributions = explainer.get_explanations(X_tensor, y_tensor)
# → Tensor shape: (n_instances, n_features)

# Evaluate explanations
from openxai import Evaluator
evaluator = Evaluator(
    inputs=X_tensor,
    labels=y_tensor,
    model=model_lr,
    explainer=explainer,
)
pgi = evaluator.evaluate(metric='PGI')
pgu = evaluator.evaluate(metric='PGU')
ris = evaluator.evaluate(metric='RIS')
```

---

## Citation

> Agarwal, C., Krishna, S., Saxena, E., Pawelczyk, M., Johnson, N., Puri, I., Zitnik, M., & Lakkaraju, H. (2022). OpenXAI: Towards a transparent evaluation of model explanations. *Advances in Neural Information Processing Systems (NeurIPS 2022)*, 35, 27384–27399. https://arxiv.org/abs/2206.11104

BibTeX:

```bibtex
@inproceedings{agarwal2022openxai,
  title     = {OpenXAI: Towards a Transparent Evaluation of Model Explanations},
  author    = {Agarwal, Chirag and Krishna, Satyapriya and Saxena, Eshika and
               Pawelczyk, Martin and Johnson, Nari and Puri, Ioana and
               Zitnik, Marinka and Lakkaraju, Himabindu},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {27384--27399},
  year      = {2022},
  url       = {https://arxiv.org/abs/2206.11104}
}
```
