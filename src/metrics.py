from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def compute_binary_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
    uncertain_low: float = 0.4,
    uncertain_high: float = 0.6,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    uncertainty_scores = 1.0 - (2.0 * np.abs(y_prob - 0.5))
    uncertain_mask = (y_prob > uncertain_low) & (y_prob < uncertain_high)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "uncertain_rate": float(uncertain_mask.mean() * 100.0),
        "avg_uncertainty": float(uncertainty_scores.mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_multiclass_binary_metrics(y_true, y_pred, y_score) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score).astype(float)
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = float("nan")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
