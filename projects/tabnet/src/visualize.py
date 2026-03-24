"""
Visualization module for fraud detection results.

Generates:
  - ROC curves (all models overlaid)
  - Confusion matrix heatmaps
  - TabNet feature importance bar charts
  - Results comparison bar chart
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from evaluate import get_roc_curve, aggregate_fold_metrics
from config import EXPERIMENTS_DIR


# Plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
SAVE_DIR = EXPERIMENTS_DIR / "plots"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def plot_roc_curves(all_results: list[dict], save_path: Path = None):
    """Plot overlaid ROC curves for all models."""
    if save_path is None:
        save_path = SAVE_DIR / "roc_curves.png"

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(all_results)))

    for result, color in zip(all_results, colors):
        y_true = result["all_targets"]
        y_pred = result["all_preds"]
        fpr, tpr, _ = get_roc_curve(y_true, y_pred)
        agg = aggregate_fold_metrics(result["fold_metrics"])
        auc_mean = agg["roc_auc_mean"]
        auc_std = agg["roc_auc_std"]

        ax.plot(
            fpr, tpr,
            label=f"{result['model_name']} (AUC = {auc_mean:.4f} ± {auc_std:.4f})",
            color=color,
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves — Model Comparison", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] ROC curves saved to {save_path}")


def plot_confusion_matrices(all_results: list[dict], save_path: Path = None):
    """Plot confusion matrices for all models in a grid."""
    if save_path is None:
        save_path = SAVE_DIR / "confusion_matrices.png"

    n_models = len(all_results)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, result in enumerate(all_results):
        y_true = result["all_targets"]
        y_pred = (result["all_preds"] >= 0.5).astype(int)
        cm = np.array([[
            np.sum((y_true == 0) & (y_pred == 0)),
            np.sum((y_true == 0) & (y_pred == 1)),
        ], [
            np.sum((y_true == 1) & (y_pred == 0)),
            np.sum((y_true == 1) & (y_pred == 1)),
        ]])

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Fraud", "Fraud"],
            yticklabels=["Non-Fraud", "Fraud"],
            ax=axes[idx],
        )
        axes[idx].set_title(result["model_name"], fontsize=13, fontweight="bold")
        axes[idx].set_ylabel("Actual")
        axes[idx].set_xlabel("Predicted")

    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrices saved to {save_path}")


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list,
    top_n: int = 20,
    save_path: Path = None,
):
    """Plot TabNet feature importance bar chart."""
    if save_path is None:
        save_path = SAVE_DIR / "feature_importance.png"

    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))

    bars = ax.barh(range(len(top_features)), top_importances, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Attention Weight)", fontsize=13)
    ax.set_title(
        f"TabNet — Top {top_n} Feature Importances",
        fontsize=15, fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Feature importance plot saved to {save_path}")


def plot_results_comparison(all_results: list[dict], save_path: Path = None):
    """Plot a bar chart comparing metrics across models."""
    if save_path is None:
        save_path = SAVE_DIR / "results_comparison.png"

    metrics_to_plot = ["roc_auc", "accuracy", "precision", "recall", "f1"]
    metric_labels = ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = [r["model_name"] for r in all_results]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(metric_labels))
    width = 0.15
    n_models = len(model_names)
    offsets = np.linspace(
        -(n_models - 1) * width / 2,
        (n_models - 1) * width / 2,
        n_models,
    )

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, result in enumerate(all_results):
        agg = aggregate_fold_metrics(result["fold_metrics"])
        means = [agg[f"{m}_mean"] for m in metrics_to_plot]
        stds = [agg[f"{m}_std"] for m in metrics_to_plot]

        ax.bar(
            x + offsets[i], means, width,
            yerr=stds,
            label=result["model_name"],
            color=colors[i],
            capsize=3,
        )

    ax.set_xlabel("Metric", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Comparison — All Metrics", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Results comparison plot saved to {save_path}")


def generate_all_plots(all_results: list[dict], feature_names: list = None):
    """Generate all visualization plots."""
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_roc_curves(all_results)
    plot_confusion_matrices(all_results)
    plot_results_comparison(all_results)

    # Feature importance (TabNet only)
    tabnet_result = next(
        (r for r in all_results if r["model_name"] == "TabNet"), None
    )
    if tabnet_result and "feature_importances" in tabnet_result and feature_names:
        plot_feature_importance(
            tabnet_result["feature_importances"],
            feature_names,
        )

    print(f"\n[INFO] All plots saved to {SAVE_DIR}")
