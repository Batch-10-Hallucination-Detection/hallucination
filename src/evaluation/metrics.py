"""
Evaluation Metrics Module (FR-11).
Calculates claim-level Precision, Recall, F1, Accuracy, AUROC, Confusion Matrix,
and optimal decision threshold tuning on dev splits.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)


def compute_binary_metrics(
    y_true: List[int],
    y_scores: List[float],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes claim-level binary metrics for hallucination detection (class 1 = hallucinated).
    """
    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_scores)
    y_pred_arr = (y_scores_arr >= threshold).astype(int)

    # Handle single-class edge cases gracefully
    num_positives = int(np.sum(y_true_arr == 1))
    num_negatives = int(np.sum(y_true_arr == 0))

    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))

    try:
        if len(set(y_true_arr)) > 1:
            auroc = float(roc_auc_score(y_true_arr, y_scores_arr))
        else:
            auroc = 0.5
    except Exception:
        auroc = 0.5

    # Confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "auroc": round(auroc, 4),
        "threshold": round(threshold, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "num_hallucinated": num_positives,
        "num_supported": num_negatives,
        "total_claims": len(y_true)
    }


def find_optimal_threshold(
    y_true: List[int],
    y_scores: List[float],
    min_thresh: float = 0.1,
    max_thresh: float = 0.9,
    step: float = 0.05
) -> Tuple[float, float]:
    """
    Tunes threshold on the dev split to maximize F1 score.
    Returns (best_threshold, best_f1).
    """
    best_thresh = 0.5
    best_f1 = -1.0

    thresholds = np.arange(min_thresh, max_thresh + 0.001, step)
    for t in thresholds:
        metrics = compute_binary_metrics(y_true, y_scores, threshold=float(t))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thresh = float(t)

    return best_thresh, best_f1

