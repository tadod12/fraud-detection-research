# Findings — TabNet Bank Fraud Detection

## Current Understanding

TabNet is a deep learning architecture purpose-built for tabular data (Arık & Pfister, 2019). It uses sequential attention with Sparsemax to perform instance-wise sparse feature selection, providing both high performance and built-in interpretability. Prashanth et al. (2026) demonstrated its superiority over DNN, LSTM, GRU, and CNN1D on an Indian bank transaction fraud dataset (ROC-AUC: 0.9739).

## Patterns and Insights

- **Architecture-data alignment matters**: Models designed for sequential (LSTM/GRU) or spatial (CNN) data fail on tabular data, often collapsing to majority-class prediction
- **TabNet's key advantages**: sparse attention, built-in regularization, interpretability through feature masks
- **SMOTE**: Used for class balancing — effectiveness not yet independently validated

## Lessons and Constraints

*(To be updated as experiments proceed)*

## Open Questions

1. How does TabNet compare to XGBoost/LightGBM on the same dataset?
2. What is the effect of SMOTE ratio on performance?
3. Which features does TabNet identify as most important for fraud detection?
4. Would TabNet generalize to fraud datasets from other regions?
