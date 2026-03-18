"""
TabNet model wrapper for fraud detection.

Uses the pytorch-tabnet library with hyperparameters from the paper:
  n_d=32, n_a=32, n_steps=5, gamma=1.5, lambda_sparse=1e-4
"""

import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

from config import (
    TABNET_PARAMS,
    TABNET_FIT_PARAMS,
    TABNET_OPTIMIZER_PARAMS,
    TABNET_SCHEDULER_PARAMS,
    RANDOM_SEED,
)


def create_tabnet_model() -> TabNetClassifier:
    """Create a TabNetClassifier with paper hyperparameters."""
    model = TabNetClassifier(
        **TABNET_PARAMS,
        optimizer_params=TABNET_OPTIMIZER_PARAMS,
        scheduler_fn=None,  # We'll use scheduler_params with StepLR
        scheduler_params=TABNET_SCHEDULER_PARAMS,
        seed=RANDOM_SEED,
        verbose=1,
    )
    return model


def train_tabnet(
    model: TabNetClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> TabNetClassifier:
    """Train the TabNet model."""
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=TABNET_FIT_PARAMS["eval_metric"],
        max_epochs=TABNET_FIT_PARAMS["max_epochs"],
        patience=TABNET_FIT_PARAMS["patience"],
        batch_size=TABNET_FIT_PARAMS["batch_size"],
        virtual_batch_size=TABNET_FIT_PARAMS["virtual_batch_size"],
    )
    return model


def predict_tabnet(model: TabNetClassifier, X: np.ndarray) -> np.ndarray:
    """Get prediction probabilities from TabNet."""
    return model.predict_proba(X)[:, 1]


def get_feature_importances(model: TabNetClassifier) -> np.ndarray:
    """Extract global feature importances from the trained TabNet model."""
    return model.feature_importances_
