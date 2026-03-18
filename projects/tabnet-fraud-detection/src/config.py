"""
Centralized configuration for TabNet fraud detection paper reproduction.

Paper: "Prediction of Bank Transaction Fraud Using TabNet —
        An Adaptive Deep Learning Architecture" (Prashanth et al., 2026)
"""

import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SRC_DIR = PROJECT_ROOT / "src"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Random Seed ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Cross-Validation ───────────────────────────────────────────────────────
N_FOLDS = 3

# ─── SMOTE ───────────────────────────────────────────────────────────────────
SMOTE_PARAMS = {
    "sampling_strategy": "auto",  # balance minority to majority
    "random_state": RANDOM_SEED,
    "k_neighbors": 5,
}

# ─── TabNet Hyperparameters (from paper) ─────────────────────────────────────
TABNET_PARAMS = {
    "n_d": 32,                  # Decision embedding dimension
    "n_a": 32,                  # Attention embedding dimension
    "n_steps": 5,               # Number of sequential attention steps
    "gamma": 1.5,               # Coefficient for feature reuse in attention
    "lambda_sparse": 1e-4,      # Sparsity regularization coefficient
    "momentum": 0.3,            # Momentum for Ghost Batch Normalization
    "n_shared": 2,              # Number of shared GLU layers
    "n_independent": 2,         # Number of independent GLU layers
    "mask_type": "sparsemax",   # Feature selection mask type
}

TABNET_FIT_PARAMS = {
    "max_epochs": 100,
    "patience": 15,
    "batch_size": 1024,
    "virtual_batch_size": 128,  # Ghost batch normalization batch size
    "eval_metric": ["auc"],
}

TABNET_OPTIMIZER_PARAMS = {
    "lr": 2e-2,
}

TABNET_SCHEDULER_PARAMS = {
    "step_size": 10,
    "gamma": 0.9,
}

# ─── DNN Baseline ────────────────────────────────────────────────────────────
DNN_PARAMS = {
    "hidden_layers": [256, 128, 64],
    "dropout": 0.3,
    "lr": 1e-3,
    "batch_size": 1024,
    "max_epochs": 100,
    "patience": 15,
}

# ─── LSTM Baseline ───────────────────────────────────────────────────────────
LSTM_PARAMS = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "lr": 1e-3,
    "batch_size": 1024,
    "max_epochs": 100,
    "patience": 15,
}

# ─── GRU Baseline ────────────────────────────────────────────────────────────
GRU_PARAMS = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "lr": 1e-3,
    "batch_size": 1024,
    "max_epochs": 100,
    "patience": 15,
}

# ─── CNN1D Baseline ──────────────────────────────────────────────────────────
CNN1D_PARAMS = {
    "filters": [64, 128],
    "kernel_size": 3,
    "dropout": 0.3,
    "lr": 1e-3,
    "batch_size": 1024,
    "max_epochs": 100,
    "patience": 15,
}

# ─── Models to Run ───────────────────────────────────────────────────────────
ALL_MODELS = ["TabNet", "DNN", "LSTM", "GRU", "CNN1D"]
