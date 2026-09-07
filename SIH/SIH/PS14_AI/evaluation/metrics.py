"""
Detection Performance Evaluation — PS14 AI Cyber Defence Platform.

NOTE: These metrics are ONLY meaningful when the models have been trained
and tested on the actual operational telemetry dataset. Do not interpret
results from training on example or synthetic data as production-quality estimates.
"""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_detection_performance(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute binary classification metrics from continuous attack probabilities.

    Args:
        y_true:    True binary integer labels (0=benign, 1=attack).
        y_probs:   Continuous attack probability predictions in [0.0, 1.0].
        threshold: Decision boundary for binary classification.

    Returns:
        Dict containing: accuracy, precision, recall, f1_score, roc_auc,
        log_loss, confusion_matrix, total_samples, predicted_attacks.
    """
    y_pred = (y_probs >= threshold).astype(int)

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    cm    = confusion_matrix(y_true, y_pred).tolist()

    n_classes = len(np.unique(y_true))
    if n_classes > 1:
        roc_auc = roc_auc_score(y_true, y_probs)
        ll      = log_loss(y_true, y_probs)
    else:
        logger.warning(
            "[Evaluation] Only one class in y_true. "
            "ROC-AUC and log_loss are undefined."
        )
        roc_auc = float("nan")
        ll      = float("nan")

    return {
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1_score":         round(f1,   4),
        "roc_auc":          round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A",
        "log_loss":         round(ll,    4) if not np.isnan(ll)      else "N/A",
        "confusion_matrix": cm,
        "total_samples":    int(len(y_true)),
        "predicted_attacks": int(y_pred.sum()),
        "threshold":        threshold,
    }
