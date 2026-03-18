"""
Preprocessing pipeline for the bank transaction fraud dataset.

Implements:
  - Missing value handling
  - Categorical encoding (LabelEncoder)
  - Numerical scaling (StandardScaler)
  - SMOTE oversampling
  - 3-fold Stratified Cross-Validation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

from config import RANDOM_SEED, N_FOLDS, SMOTE_PARAMS


def identify_columns(df: pd.DataFrame, target_col: str):
    """Identify numerical and categorical columns (excluding target)."""
    feature_cols = [c for c in df.columns if c != target_col]

    categorical_cols = []
    numerical_cols = []

    for col in feature_cols:
        if df[col].dtype == "object" or df[col].nunique() < 20:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    return numerical_cols, categorical_cols


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values: fill numerical with median, categorical with mode."""
    df = df.copy()

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def encode_categoricals(
    df: pd.DataFrame, categorical_cols: list
) -> tuple[pd.DataFrame, dict]:
    """Encode categorical features using LabelEncoder."""
    df = df.copy()
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def scale_numericals(
    X_train: np.ndarray,
    X_val: np.ndarray,
    numerical_indices: list,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale numerical features using StandardScaler (fit on train only)."""
    scaler = StandardScaler()

    if len(numerical_indices) > 0:
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_train[:, numerical_indices] = scaler.fit_transform(
            X_train[:, numerical_indices]
        )
        X_val[:, numerical_indices] = scaler.transform(
            X_val[:, numerical_indices]
        )

    return X_train, X_val, scaler


def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """Apply SMOTE oversampling to balance classes."""
    smote = SMOTE(**SMOTE_PARAMS)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(
        f"[INFO] SMOTE: {len(y_train)} → {len(y_resampled)} samples "
        f"(fraud: {y_train.sum()} → {y_resampled.sum()})"
    )

    return X_resampled, y_resampled


def preprocess_dataframe(
    df: pd.DataFrame, target_col: str
) -> tuple[np.ndarray, np.ndarray, list, list, list]:
    """
    Full preprocessing pipeline:
      1. Handle missing values
      2. Encode categoricals
      3. Return X, y arrays and column info

    Returns:
        X, y, feature_names, numerical_indices, categorical_indices
    """
    # Handle missing values
    df = handle_missing_values(df)

    # Identify column types
    numerical_cols, categorical_cols = identify_columns(df, target_col)
    print(f"[INFO] Numerical columns ({len(numerical_cols)}): {numerical_cols[:5]}...")
    print(f"[INFO] Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}...")

    # Encode categoricals
    df, encoders = encode_categoricals(df, categorical_cols)

    # Separate features and target
    feature_cols = numerical_cols + categorical_cols
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.int64)

    # Track indices
    numerical_indices = list(range(len(numerical_cols)))
    categorical_indices = list(
        range(len(numerical_cols), len(numerical_cols) + len(categorical_cols))
    )

    return X, y, feature_cols, numerical_indices, categorical_indices


def get_cv_splits(X, y):
    """Generate 3-fold stratified cross-validation splits."""
    skf = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    return list(skf.split(X, y))


if __name__ == "__main__":
    from data_loader import load_dataset, profile_dataset

    df = load_dataset()
    profile = profile_dataset(df)
    target = profile["target_column"]

    X, y, features, num_idx, cat_idx = preprocess_dataframe(df, target)
    print(f"\nPreprocessed: X.shape={X.shape}, y.shape={y.shape}")
    print(f"Fraud ratio: {y.mean():.4f}")

    splits = get_cv_splits(X, y)
    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"Fold {i+1}: train={len(train_idx)}, val={len(val_idx)}")
