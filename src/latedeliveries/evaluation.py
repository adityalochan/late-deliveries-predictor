import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

VOUCHER_COST = 4
MISSED_LATE_COST = 15

THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)


def order_cost(y_true, flagged):
    """Calculate the total business cost of delivery decisions."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(flagged, dtype=bool)

    if y_true.shape != flagged.shape:
        raise ValueError("y_true and flagged must have the same shape")

    vouchers = np.sum(flagged)
    missed_late = np.sum((y_true == 1) & (~flagged))

    return int(VOUCHER_COST * vouchers + MISSED_LATE_COST * missed_late)


def choose_threshold(y_true, probabilities):
    """Choose the lowest-cost threshold using training predictions."""
    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)

    if y_true.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must have the same shape")

    results = []

    for threshold in THRESHOLDS:
        flagged = probabilities >= threshold
        cost = order_cost(y_true, flagged)

        results.append(
            {
                "threshold": float(threshold),
                "cost": cost,
            }
        )

    best = min(results, key=lambda row: row["cost"])

    return best["threshold"], results


def evaluate_predictions(y_true, probabilities, threshold):
    """Calculate classification metrics and business costs."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    flagged = probabilities >= threshold

    return {
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, flagged, labels=[0, 1]).tolist(),
        "accuracy": float(accuracy_score(y_true, flagged)),
        "precision": float(precision_score(y_true, flagged, zero_division=0)),
        "recall": float(recall_score(y_true, flagged, zero_division=0)),
        "f1": float(f1_score(y_true, flagged, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "cost_flag_nobody": order_cost(y_true, np.zeros(len(y_true), dtype=bool)),
        "cost_flag_everybody": order_cost(y_true, np.ones(len(y_true), dtype=bool)),
        "cost_model": order_cost(y_true, flagged),
    }
