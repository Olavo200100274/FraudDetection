import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
    fbeta_score,
    confusion_matrix,
)

def pr_auc(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))

def threshold_sweep(y_true, y_score, grid_points=200):
    thresholds = np.linspace(0.0, 1.0, grid_points)

    rows = []
    best_f1 = (-1.0, None)
    best_f2 = (-1.0, None)

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        rows.append({"threshold": float(t), "f1": float(f1), "f2": float(f2)})

        if f1 > best_f1[0]:
            best_f1 = (f1, t)
        if f2 > best_f2[0]:
            best_f2 = (f2, t)

    sweep_df = pd.DataFrame(rows)
    return sweep_df, {"tau_f1": float(best_f1[1]), "best_f1": float(best_f1[0]),
                      "tau_f2": float(best_f2[1]), "best_f2": float(best_f2[0])}

def evaluate_at_threshold(y_true, y_score, threshold: float):
    y_pred = (y_score >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return float(f1), float(f2), cm

def pr_curve_df(y_true, y_score):
    p, r, thr = precision_recall_curve(y_true, y_score)
    # thr has length n-1; keep p/r only (common for plotting)
    return pd.DataFrame({"precision": p, "recall": r})
