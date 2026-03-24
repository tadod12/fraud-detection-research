# TabNet Architecture — Study & Build Guide

> Steps to understand and build the TabNet architecture from the paper:
> *"Prediction of Bank Transaction Fraud Using TabNet — An Adaptive Deep Learning Architecture"*
> (Prashanth et al., 2026, International Review of Economics & Finance)

---

## Phase 1: Literature & Understanding

> **Goal**: Build a solid theoretical foundation before coding anything.

- **Step 1 — Read the paper thoroughly**
  - Focus on the 3 pillars: **What** (claims), **Why** (evidence), **So What** (impact)
  - Map out: problem statement → methodology → results → conclusions

- **Step 2 — Read the original TabNet paper**
  - *"TabNet: Attentive Interpretable Tabular Learning"* — Sercan Ö. Arık & Tomas Pfister (Google Cloud AI, 2019)
  - Source: [arXiv:1908.07442](https://arxiv.org/abs/1908.07442)

- **Step 3 — Search related literature**
  - Find 3–5 related fraud detection papers using Semantic Scholar / arXiv
  - Compare methodologies: TabNet vs XGBoost vs LightGBM vs traditional DL
  - Save summaries to a `literature/` folder

- **Step 4 — Identify gaps & form hypotheses**
  - What did this paper test well? What's missing?
  - Possible hypotheses: *Would TabNet beat XGBoost on this dataset? Does SMOTE ratio matter?*

---

## Phase 2: Architecture Deep-Dive

> **Goal**: Understand every component of TabNet before implementing.

- **Step 5 — Study TabNet's core components**

  | Component | What It Does |
  |---|---|
  | **Sequential Attention** | Multi-step decision process — selects features at each step |
  | **Attentive Transformer** | Generates feature selection masks using Sparsemax |
  | **Feature Transformer** | Processes selected features via Gated Linear Unit (GLU) blocks |
  | **Ghost Batch Normalization** | BN on virtual mini-batches for better generalization |
  | **Sparsemax** | Produces sparse outputs (some features → 0), unlike Softmax |
  | **Instance-wise Feature Selection** | Different features selected per input sample |

- **Step 6 — Trace a forward pass on paper**
  - Draw the data flow: Input → BN → Step 1 (Attend → Transform → Output) → Step 2 → ... → Aggregate → Prediction
  - Understand how prior scales prevent redundant feature reuse

- **Step 7 — Choose implementation approach**
  - **Option A** (Recommended for learning): Build from scratch in PyTorch
  - **Option B** (Faster): Use `pytorch-tabnet` library (`pip install pytorch-tabnet`)

---

## Phase 3: Data Preparation

> **Goal**: Reproduce the paper's dataset and preprocessing pipeline.

- **Step 8 — Download the dataset**
  - Source: Kaggle — Indian bank transactions dataset
  - Examine: number of samples, features, fraud ratio

- **Step 9 — Exploratory Data Analysis (EDA)**
  - Analyze transaction amounts, temporal patterns, feature correlations
  - Visualize class distribution (fraud vs. non-fraud)
  - Identify anomalous patterns that distinguish fraud

- **Step 10 — Preprocessing pipeline**
  - Handle missing values and encode categorical features
  - Normalize numerical features
  - Apply **SMOTE** to balance classes
  - Split data for 3-fold cross-validation

---

## Phase 4: Model Training & Experimentation

> **Goal**: Reproduce and extend the paper's results.

- **Step 11 — Train TabNet**
  - Key hyperparameters to set:
    - `n_d` / `n_a` (decision/attention embedding dimensions)
    - `n_steps` (number of sequential attention steps)
    - `gamma` (coefficient for feature reuse in attention)
    - `lambda_sparse` (sparsity regularization)
    - Learning rate, batch size, epochs
  - Target: ROC-AUC ≈ **0.9739**, Accuracy ≈ **97.39%**

- **Step 12 — Train baseline models**

  | Model | Purpose |
  |---|---|
  | DNN | Standard deep learning baseline |
  | LSTM | Sequential model baseline |
  | GRU | Sequential model baseline |
  | CNN1D | Convolutional baseline |
  | XGBoost *(bonus)* | Strong tabular ML baseline |
  | LightGBM *(bonus)* | Strong tabular ML baseline |

- **Step 13 — Evaluate & compare**
  - Metrics: ROC-AUC, Accuracy, Precision, Recall, F1-Score
  - Generate: confusion matrices, ROC curves, classification reports
  - Use 3-fold cross-validation for robust estimates

---

## Phase 5: Analysis & Interpretation

> **Goal**: Understand *why* TabNet works and others don't.

- **Step 14 — Analyze TabNet's attention masks**
  - Extract feature importance from trained TabNet
  - Visualize which features drive fraud predictions (local + global interpretability)

- **Step 15 — Diagnose baseline failures**
  - Why did LSTM/GRU get ~0.51 AUC? → Architecture-data mismatch (no temporal sequence)
  - Why did DNN/CNN1D fail? → Overfitting, classification collapse
  - Key insight: **Align architecture to data structure**

- **Step 16 — Test additional hypotheses** *(optional)*
  - Does SMOTE ratio affect results?
  - What happens without SMOTE?
  - Does TabNet still win on other fraud datasets?
  - How does TabNet compare to XGBoost/LightGBM?

---

## Phase 6: Documentation & Write-Up

> **Goal**: Consolidate findings into a publishable or presentable format.

- **Step 17 — Document your findings**
  - Write a clear narrative: problem → method → results → interpretation
  - Include all visualizations: ROC curves, feature importance heatmaps, confusion matrices

- **Step 18 — Verify all citations**
  - Use Semantic Scholar API — never write BibTeX from memory
  - Mark any unverified references as `[CITATION NEEDED]`

- **Step 19 — Generate a final report or presentation**
  - Summarize key takeaways
  - Highlight what you learned and potential next steps

---

## Quick Start Code Snippet

```python
# Install
# pip install pytorch-tabnet scikit-learn imbalanced-learn pandas

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# 1. Load data
df = pd.read_csv("bank_transactions.csv")
X = df.drop("is_fraud", axis=1).values
y = df["is_fraud"].values

# 2. SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 3. Cross-validation
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
auc_scores = []

for train_idx, val_idx in skf.split(X_res, y_res):
    X_train, X_val = X_res[train_idx], X_res[val_idx]
    y_train, y_val = y_res[train_idx], y_res[val_idx]

    clf = TabNetClassifier(
        n_d=32, n_a=32,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        verbose=1
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc"],
        max_epochs=100,
        patience=15,
        batch_size=1024
    )

    preds = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    auc_scores.append(auc)
    print(f"Fold AUC: {auc:.4f}")

print(f"\nMean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
```

---

## References

| Resource | Link |
|---|---|
| Original TabNet Paper | [arXiv:1908.07442](https://arxiv.org/abs/1908.07442) |
| pytorch-tabnet Library | [GitHub](https://github.com/dreamquark-ai/tabnet) |
| SMOTE Documentation | [imbalanced-learn](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) |
| Kaggle Dataset | Search: "Bank Transaction Fraud India" |

---

*Guide derived from skills: `0-autoresearch-skill`, `01-model-architecture`, `20-ml-paper-writing`, `21-research-ideation`*
