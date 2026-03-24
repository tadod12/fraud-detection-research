# Prediction of Bank Transaction Fraud Using TabNet — An Adaptive Deep Learning Architecture

## Paper Details

| Field | Details |
|---|---|
| **Authors** | B.S. Prashanth, Manoj Kumar, Ariful Hoque, Nasser Al Muraqab, Immanuel Azaad Moonesar, Udo Christian Braendle, Ananth Rao |
| **Journal** | *International Review of Economics & Finance* |
| **Published** | January 2026 |
| **Affiliation** | Murdoch University (among others) |

---

## 1. Abstract

This paper investigates the use of **TabNet**, an adaptive deep learning architecture, for predicting fraudulent bank transactions. As online banking grows, so does the threat of fraudulent operations — necessitating interpretable, scalable, and high-performing fraud detection systems. The study evaluates TabNet alongside four other deep learning models (DNN, GRU, LSTM, CNN1D) on a Kaggle dataset of real Indian bank transactions. Using SMOTE for class balancing and 3-fold cross-validation for evaluation, TabNet significantly outperformed all competitors, achieving a **ROC-AUC of 0.9739** and an **accuracy of 97.39%**. The paper contributes to the literature on explainable AI (XAI) in financial decision-making and operational fraud detection.

---

## 2. Introduction & Motivation

### Problem Context
- **Rising online banking fraud**: With the growing adoption of digital banking, fraudulent transactions have become a critical challenge for financial institutions worldwide.
- **Limitations of traditional approaches**: Conventional fraud detection methods (rule-based systems, traditional ML) struggle with evolving fraud patterns, high-dimensional data, and class imbalance.
- **Need for interpretability**: Regulatory compliance and operational risk management require models that can not only detect fraud accurately but also explain *why* a transaction was flagged — a major gap in black-box deep learning models.

### Research Objectives
1. Evaluate the effectiveness of **TabNet** as an adaptive, interpretable deep learning framework for bank transaction fraud detection.
2. Compare TabNet against other deep learning architectures (DNN, GRU, LSTM, CNN1D) to determine the best-suited architecture for tabular financial data.
3. Enhance **operational risk management** by improving transaction anomaly detection accuracy while ensuring **regulatory compliance** through transparent, explainable models.

---

## 3. Literature Review / Related Work

The paper reviews prior work in several key areas:

- **Traditional Machine Learning for Fraud Detection**: Prior studies have applied logistic regression, random forests, support vector machines (SVM), and gradient boosting methods (XGBoost, LightGBM) for fraud detection. While effective, these methods often require extensive manual feature engineering.
- **Deep Learning for Fraud Detection**: DNN, LSTM, GRU, and CNN-based models have been explored for fraud detection. However, many of these architectures were originally designed for sequential/image data and may not be optimal for structured tabular data.
- **Class Imbalance Handling**: Techniques such as SMOTE, random oversampling, and undersampling have been widely used to address the severe class imbalance inherent in fraud detection datasets (where fraudulent transactions represent a tiny minority).
- **Explainable AI (XAI)**: Growing emphasis on model interpretability in finance, particularly for regulatory compliance. TabNet's attention-based feature selection aligns with this need.
- **TabNet (Arık & Pfister, 2019)**: The original TabNet paper by Google Cloud AI introduced an architecture specifically designed for tabular data, combining the interpretability of tree-based methods with the learning power of deep neural networks.

---

## 4. Methodology

### 4.1 Dataset

| Property | Details |
|---|---|
| **Source** | Kaggle |
| **Type** | Real bank transactions from India |
| **Task** | Binary classification (Fraud vs. Non-Fraud) |
| **Class Distribution** | Highly imbalanced (fraud cases are a small minority) |

### 4.2 Exploratory Data Analysis (EDA)

- Thorough EDA was performed to identify **fraud patterns across different timeframes and behaviors**.
- Analysis of transaction amounts, frequency distributions, temporal patterns, and correlations between features.
- Identification of anomalous patterns distinguishing fraudulent from legitimate transactions.

### 4.3 Data Preprocessing

1. **Feature Engineering**: Construction of relevant features and anomaly detection indicators to help models identify fraudulent transactions.
2. **SMOTE (Synthetic Minority Over-sampling Technique)**: Applied to balance the highly skewed class distribution, generating synthetic samples of the minority (fraud) class to prevent model bias toward the majority class.
3. **Normalization/Encoding**: Standard preprocessing of numerical and categorical features for input into deep learning models.

### 4.4 Models Evaluated

Five deep learning architectures were compared:

| Model | Description |
|---|---|
| **TabNet** | Adaptive deep learning architecture with attention-based sparse feature selection, specifically designed for tabular data |
| **DNN** | Deep Neural Network — standard fully-connected feedforward architecture |
| **GRU** | Gated Recurrent Unit — recurrent architecture designed for sequential data |
| **LSTM** | Long Short-Term Memory — recurrent architecture with memory gating for sequences |
| **CNN1D** | 1-Dimensional Convolutional Neural Network — convolutional architecture adapted for 1D feature vectors |

### 4.5 TabNet Architecture — Key Components

TabNet, originally proposed by **Sercan Ö. Arık and Tomas Pfister (Google Cloud AI, 2019)**, is a deep learning model specifically tailored for tabular data. Its key innovations include:

#### Sequential Attention Mechanism
- TabNet processes data through **multiple sequential decision steps**.
- At each step, a **sequential attention mechanism** selects a subset of the most relevant features for processing.
- This mimics the interpretable decision-making of tree-based models while leveraging the representation power of neural networks.

#### Instance-Wise Sparse Feature Selection
- Unlike global feature selection, TabNet performs **instance-wise feature selection** — for each individual input, it dynamically determines which features are most relevant.
- Uses **Sparsemax** normalization (instead of Softmax) to produce sparse attention masks, effectively zeroing out irrelevant features.
- This results in improved interpretability and efficient learning.

#### Attentive Transformer
- Generates the **feature selection masks** at each decision step.
- Comprises a fully connected layer, Batch Normalization, prior scales layer (tracking previously used features), and Sparsemax activation.
- Ensures that features are not redundantly selected across steps.

#### Feature Transformer
- Processes the selected features at each step.
- Consists of multiple **Gated Linear Unit (GLU)** blocks.
- Outputs both a decision contribution for the current step and processed information for the next attentive transformer.

#### Ghost Batch Normalization (GBN)
- A variant of Batch Normalization designed for large-batch training.
- Computes normalization statistics on smaller "virtual mini-batches" (ghost batches) rather than the entire batch.
- Improves generalization and training stability.

#### Interpretability
- **Local interpretability**: Feature masks reveal which features influenced predictions for individual instances.
- **Global interpretability**: Aggregated feature importance across the dataset quantifies overall feature contributions.

### 4.6 Evaluation Framework

- **3-Fold Cross-Validation**: All models were evaluated using a rigorous 3-fold cross-validation framework to ensure robust and unbiased performance estimation.
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.

---

## 5. Experimental Results

### 5.1 Overall Performance Comparison

| Model | ROC-AUC | Accuracy |
|---|---|---|
| **TabNet** | **0.9739** | **97.39%** |
| LSTM | 0.5118 | ~51% |
| GRU | 0.5170 | ~52% |
| CNN1D | ~0.50 | ~50% |
| DNN | ~0.50 | ~50% |

### 5.2 Key Findings from Results

- **TabNet dramatically outperformed** all other deep learning architectures across all evaluation metrics.
- **LSTM and GRU** yielded ROC-AUC scores barely above 0.5 (random guessing), indicating they failed to learn meaningful patterns from the tabular data.
- **CNN1D and DNN** performed even worse, with ROC-AUC values close to 0.5 — essentially equivalent to random classification.
- TabNet's sparse feature selection led to:
  - **Fewer false positives** (legitimate transactions incorrectly flagged as fraud)
  - **Fewer false negatives** (fraudulent transactions missed by the model)
- Confusion matrices were presented for all models, clearly demonstrating TabNet's superior classification performance.
- Precision, Recall, and F1-Score all favored TabNet significantly.

### 5.3 Why Other Models Failed

The poor performance of DNN, GRU, LSTM, and CNN1D is attributed to:

1. **Architecture-Data Mismatch**: These models were not designed for tabular data. LSTM and GRU expect temporal sequences; CNN1D expects spatial/local patterns — neither of which are present in structured banking transaction features.
2. **Overfitting**: Without the built-in regularization and sparse selection of TabNet, these models tended to overfit on the training data.
3. **Classification Collapse**: Some models collapsed to predicting only the majority class, resulting in near-random ROC-AUC scores.
4. **Inability to Derive Meaningful Representations**: The architectures could not effectively extract discriminative features from raw tabular data without explicit sequence or spatial structure.

---

## 6. Discussion

### Key Insights

1. **Importance of Architecture-Data Alignment**: The results strongly underscore that choosing the right architecture for the data type is critical. TabNet, purpose-built for tabular data, vastly outperforms general-purpose deep learning models on structured banking data.

2. **Interpretability as a Competitive Advantage**: TabNet's attention-based feature masks provide built-in explainability — a crucial requirement for:
   - **Regulatory compliance** (e.g., explaining why a transaction was flagged)
   - **Operational trust** (fraud analysts can validate model decisions)
   - **Model debugging** (identifying potential biases or data issues)

3. **Sparse Feature Selection Benefits**:
   - Focuses model capacity on the most relevant features for each transaction
   - Reduces noise from irrelevant features
   - Improves generalization to unseen data

4. **Practical Implications for Banks**:
   - TabNet offers a production-viable solution that balances accuracy, interpretability, and scalability
   - Reduced false positives mean fewer legitimate customers are inconvenienced
   - Reduced false negatives mean more fraud is caught, directly reducing financial losses

5. **Contribution to Explainable AI (XAI) in Finance**: The study provides evidence that interpretable deep learning can match or exceed the performance of black-box models for financial applications, advancing the XAI literature in financial decision-making.

---

## 7. Conclusion

### Main Takeaways

1. **TabNet is the superior architecture** for bank transaction fraud detection on tabular data, achieving a ROC-AUC of 0.9739 and accuracy of 97.39% — far surpassing DNN, GRU, LSTM, and CNN1D.
2. **Deep learning architectures designed for sequential or spatial data are not suitable** for tabular fraud detection tasks, as demonstrated by the near-random performance of LSTM, GRU, CNN1D, and DNN.
3. **SMOTE** is an effective technique for handling class imbalance in fraud detection datasets.
4. **TabNet's interpretability** through sparse attention masks makes it particularly well-suited for regulated financial environments where model transparency is mandatory.
5. The study provides **critical insights for operational fraud detection systems** and contributes to the growing body of literature on explainable AI in financial services.

### Future Work Suggestions

- Evaluation on larger, more diverse datasets from multiple geographic regions
- Integration of TabNet into real-time fraud detection pipelines
- Exploration of ensemble approaches combining TabNet with other models
- Investigation of TabNet's performance with additional feature engineering techniques
- Application of TabNet to other financial risk assessment tasks beyond fraud detection
- Further study of TabNet's robustness against adversarial attacks and concept drift

---

## 8. Summary of Figures & Tables

| Figure/Table | Description |
|---|---|
| **EDA Visualizations** | Distributions of transaction amounts, temporal patterns of fraud vs. legitimate transactions, and correlation heatmaps |
| **SMOTE Visualization** | Before-and-after class distribution showing the effect of synthetic oversampling |
| **Model Architecture Diagrams** | Visual representation of TabNet's sequential attention mechanism and feature transformer blocks |
| **ROC Curves** | Comparison of ROC curves for all 5 models, showing TabNet's clear superiority |
| **Confusion Matrices** | Classification results for each model, highlighting TabNet's balanced performance across both classes |
| **Performance Metrics Table** | Comprehensive comparison of Accuracy, Precision, Recall, F1-Score, and ROC-AUC across all models |
| **Feature Importance Masks** | TabNet's attention masks showing which features contributed most to fraud detection decisions |

---

## 9. Critical Assessment

### Strengths
- ✅ Novel application of TabNet to bank fraud detection with strong empirical results
- ✅ Comprehensive comparison against multiple deep learning baselines
- ✅ Emphasis on interpretability aligns with regulatory needs in finance
- ✅ Use of real-world data (actual Indian bank transactions) enhances practical relevance
- ✅ Rigorous evaluation with 3-fold cross-validation

### Potential Limitations
- ⚠️ Single dataset from one geographic region (India) — generalizability to other markets is uncertain
- ⚠️ No comparison with traditional ML models (e.g., XGBoost, Random Forest, LightGBM) which are known strong performers on tabular data
- ⚠️ Specific hyperparameter details for all models are not extensively documented
- ⚠️ The extremely poor performance of DNN/LSTM/GRU/CNN1D may suggest suboptimal tuning rather than inherent architectural limitations
- ⚠️ No evaluation of real-time inference latency or deployment considerations
