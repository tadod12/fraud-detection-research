# Experiment: TabNet Fraud Detection

## Hypothesis
**H1**: TabNet outperforms general-purpose DL models (DNN, LSTM, GRU, CNN1D) on tabular fraud data due to its attention-based sparse feature selection.

## What
- Train TabNet on the Kaggle Indian bank transactions dataset
- Apply SMOTE for class balancing
- Evaluate with 3-fold stratified cross-validation

## Why
- Reproduce the results from Prashanth et al. (2026)
- Validate the claimed ROC-AUC of 0.9739

## Prediction
TabNet will achieve ROC-AUC ≥ 0.95, significantly above DNN/LSTM/GRU/CNN1D.

## Setup
- Framework: PyTorch (pytorch-tabnet)
- Metrics: ROC-AUC, Accuracy, Precision, Recall, F1-Score
- Validation: 3-fold stratified CV

## Status
`not started`
