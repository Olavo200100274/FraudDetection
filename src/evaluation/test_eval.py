from __future__ import annotations
from typing import Dict, Any
import numpy as np
import time
from sklearn.metrics import precision_recall_curve

from src.evaluation.metrics import pr_auc, confusion_counts, precision_recall_f1, fbeta_from_counts

def _to_unit_interval(scores: np.ndarray, ref_scores: np.ndarray | None = None) -> np.ndarray:
    ref = ref_scores if ref_scores is not None else scores
    mn = float(np.min(ref))
    mx = float(np.max(ref))
    return (scores - mn) / (mx - mn + 1e-12)

def evaluate_on_test(model, X_train, y_train, X_test, y_test, thr_f1: float, thr_f2: float) -> Dict[str, Any]:
    """
    Assumes model is already fitted on full training data (or OCSVM fitted on legit-only).
    Returns test metrics + PR curve points + inference timing.
    """
    t_inf = time.perf_counter()

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        # decision_function path (e.g., OCSVM)
        # scale test scores using TRAIN decision scores only (leakage-free)
        ref_scores = model.decision_function(X_train)
        y_score_raw = model.decision_function(X_test)
        y_score = _to_unit_interval(y_score_raw, ref_scores=ref_scores)

    infer_s = float(time.perf_counter() - t_inf)

    y_true = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test, dtype=int)
    y_true = y_true.astype(int)

    pr = pr_auc(y_true, y_score)

    # F1 @ thr_f1
    y_pred_f1 = (y_score >= thr_f1).astype(int)
    cc_f1 = confusion_counts(y_true, y_pred_f1)
    prf1 = precision_recall_f1(y_true, y_pred_f1)

    # F2 @ thr_f2
    y_pred_f2 = (y_score >= thr_f2).astype(int)
    cc_f2 = confusion_counts(y_true, y_pred_f2)
    f2 = fbeta_from_counts(cc_f2["tp"], cc_f2["fp"], cc_f2["fn"], beta=2.0)

    # PR curve for plotting
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    return {
        "pr_auc": float(pr),
        "thr_f1": float(thr_f1),
        "thr_f2": float(thr_f2),
        "precision_at_f1_thr": float(prf1["precision"]),
        "recall_at_f1_thr": float(prf1["recall"]),
        "f1": float(prf1["f1"]),
        "tp_f1": int(cc_f1["tp"]),
        "fp_f1": int(cc_f1["fp"]),
        "fn_f1": int(cc_f1["fn"]),
        "tn_f1": int(cc_f1["tn"]),
        "f2": float(f2),
        "tp_f2": int(cc_f2["tp"]),
        "fp_f2": int(cc_f2["fp"]),
        "fn_f2": int(cc_f2["fn"]),
        "tn_f2": int(cc_f2["tn"]),
        "inference_seconds": float(infer_s),
        "pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        }
    }