"""
Evaluation metrics for fraud detection models.

Computes ROC-AUC, accuracy, precision, recall, F1, and generates
confusion matrices and classification reports.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from config import N_FOLDS


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute all evaluation metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def get_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray):
    """Compute ROC curve data points."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    return fpr, tpr, thresholds


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Aggregate metrics across CV folds (mean ± std)."""
    metric_names = ["roc_auc", "accuracy", "precision", "recall", "f1"]
    aggregated = {}

    for metric in metric_names:
        values = [fm[metric] for fm in fold_metrics]
        aggregated[f"{metric}_mean"] = np.mean(values)
        aggregated[f"{metric}_std"] = np.std(values)

    return aggregated


def create_results_table(all_results: list[dict]) -> pd.DataFrame:
    """Create a comparison results table across all models."""
    rows = []

    for result in all_results:
        agg = aggregate_fold_metrics(result["fold_metrics"])
        row = {
            "Model": result["model_name"],
            "ROC-AUC": f"{agg['roc_auc_mean']:.4f} ± {agg['roc_auc_std']:.4f}",
            "Accuracy": f"{agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}",
            "Precision": f"{agg['precision_mean']:.4f} ± {agg['precision_std']:.4f}",
            "Recall": f"{agg['recall_mean']:.4f} ± {agg['recall_std']:.4f}",
            "F1-Score": f"{agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def print_results_table(all_results: list[dict]):
    """Print a formatted results comparison table."""
    df = create_results_table(all_results)

    print("\n" + "=" * 90)
    print("RESULTS COMPARISON")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    return df


def print_classification_reports(all_results: list[dict]):
    """Print classification reports for each model."""
    for result in all_results:
        print(f"\n{'─' * 60}")
        print(f"Classification Report: {result['model_name']}")
        print(f"{'─' * 60}")

        y_true = result["all_targets"]
        y_pred = (result["all_preds"] >= 0.5).astype(int)

        print(classification_report(
            y_true, y_pred,
            target_names=["Non-Fraud", "Fraud"],
            digits=4,
        ))
