# Experiment: Baselines Comparison

## Hypothesis
**H4**: TabNet can match or exceed traditional ML models (XGBoost, LightGBM) on the same fraud dataset.

## What
- Train XGBoost, LightGBM, Random Forest, DNN, LSTM, GRU, CNN1D on the same dataset
- Compare against TabNet under identical conditions (same SMOTE, same CV folds)

## Why
- The original paper only compared TabNet against weak DL baselines
- XGBoost/LightGBM are known strong performers on tabular data — a fair comparison is needed

## Prediction
TabNet will match or narrowly beat XGBoost/LightGBM, while significantly outperforming DNN/LSTM/GRU/CNN1D.

## Setup
- Libraries: scikit-learn, xgboost, lightgbm, pytorch-tabnet
- Same SMOTE preprocessing for all models
- Same 3-fold stratified CV splits (fixed random seed)

## Status
`not started`
