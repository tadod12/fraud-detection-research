"""
Training orchestrator for all models with 3-fold stratified cross-validation.

Manages the full training loop for TabNet and baseline models.
"""

import numpy as np
import time

from preprocessing import (
    apply_smote,
    scale_numericals,
    get_cv_splits,
)
from models.tabnet_model import create_tabnet_model, train_tabnet, predict_tabnet
from models.baseline_models import (
    train_baseline_model,
    predict_baseline_model,
)
from evaluate import compute_metrics
from config import ALL_MODELS, N_FOLDS


def train_and_evaluate_tabnet(
    X: np.ndarray,
    y: np.ndarray,
    numerical_indices: list,
    cv_splits: list,
) -> dict:
    """Train TabNet with cross-validation and collect results."""
    print("\n" + "=" * 60)
    print("TRAINING: TabNet")
    print("=" * 60)

    all_fold_metrics = []
    all_val_preds = []
    all_val_targets = []
    feature_importances = None

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        start_time = time.time()

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale numerical features
        X_train, X_val, _ = scale_numericals(X_train, X_val, numerical_indices)

        # Apply SMOTE on training data only
        X_train_res, y_train_res = apply_smote(X_train, y_train)

        # Create & train model
        model = create_tabnet_model()
        model = train_tabnet(model, X_train_res, y_train_res, X_val, y_val)

        # Predict
        y_pred_proba = predict_tabnet(model, X_val)

        # Metrics
        metrics = compute_metrics(y_val, y_pred_proba)
        metrics["fold"] = fold_idx + 1
        metrics["time_seconds"] = time.time() - start_time
        all_fold_metrics.append(metrics)

        all_val_preds.extend(y_pred_proba)
        all_val_targets.extend(y_val)

        # Save feature importances from last fold
        feature_importances = model.feature_importances_

        print(
            f"  Fold {fold_idx + 1} | AUC: {metrics['roc_auc']:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | Time: {metrics['time_seconds']:.1f}s"
        )

    return {
        "model_name": "TabNet",
        "fold_metrics": all_fold_metrics,
        "all_preds": np.array(all_val_preds),
        "all_targets": np.array(all_val_targets),
        "feature_importances": feature_importances,
    }


def train_and_evaluate_baseline(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    numerical_indices: list,
    cv_splits: list,
) -> dict:
    """Train a baseline model with cross-validation and collect results."""
    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    all_fold_metrics = []
    all_val_preds = []
    all_val_targets = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        start_time = time.time()

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale numerical features
        X_train, X_val, _ = scale_numericals(X_train, X_val, numerical_indices)

        # Apply SMOTE on training data only
        X_train_res, y_train_res = apply_smote(X_train, y_train)

        # Train model
        model, history = train_baseline_model(
            model_name, X_train_res, y_train_res, X_val, y_val
        )

        # Predict
        y_pred_proba = predict_baseline_model(model, X_val)

        # Metrics
        metrics = compute_metrics(y_val, y_pred_proba)
        metrics["fold"] = fold_idx + 1
        metrics["time_seconds"] = time.time() - start_time
        all_fold_metrics.append(metrics)

        all_val_preds.extend(y_pred_proba)
        all_val_targets.extend(y_val)

        print(
            f"  Fold {fold_idx + 1} | AUC: {metrics['roc_auc']:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | Time: {metrics['time_seconds']:.1f}s"
        )

    return {
        "model_name": model_name,
        "fold_metrics": all_fold_metrics,
        "all_preds": np.array(all_val_preds),
        "all_targets": np.array(all_val_targets),
    }


def train_all_models(
    X: np.ndarray,
    y: np.ndarray,
    numerical_indices: list,
    models_to_run: list = None,
) -> list[dict]:
    """Train all models and return aggregated results."""
    if models_to_run is None:
        models_to_run = ALL_MODELS

    cv_splits = get_cv_splits(X, y)
    results = []

    for model_name in models_to_run:
        if model_name == "TabNet":
            result = train_and_evaluate_tabnet(
                X, y, numerical_indices, cv_splits
            )
        else:
            result = train_and_evaluate_baseline(
                model_name, X, y, numerical_indices, cv_splits
            )
        results.append(result)

    return results
