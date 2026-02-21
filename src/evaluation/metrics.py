from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

def fbeta_from_counts(tp: int, fp: int, fn: int, beta: float) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta2 = beta * beta
    denom = beta2 * precision + recall
    return float((1 + beta2) * precision * recall / denom) if denom > 0 else 0.0

def fbeta_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float, beta: float) -> float:
    y_pred = (y_score >= thr).astype(int)
    cc = confusion_counts(y_true, y_pred)
    return fbeta_from_counts(cc["tp"], cc["fp"], cc["fn"], beta=beta)