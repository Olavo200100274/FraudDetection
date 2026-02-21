from __future__ import annotations
import numpy as np
from sklearn.metrics import precision_recall_curve

def tune_threshold_max_fbeta(y_true: np.ndarray, y_score: np.ndarray, beta: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    if thresholds.size == 0:
        return 0.5

    precision_t = precision[:-1]
    recall_t = recall[:-1]

    beta2 = beta * beta
    denom = beta2 * precision_t + recall_t

    fbeta = np.zeros_like(denom, dtype=float)
    mask = denom > 0
    fbeta[mask] = (1.0 + beta2) * precision_t[mask] * recall_t[mask] / denom[mask]

    best_idx = int(np.argmax(fbeta))
    return float(thresholds[best_idx])