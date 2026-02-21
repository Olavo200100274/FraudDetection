from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from src.evaluation.metrics import pr_auc, fbeta_at_threshold
from src.evaluation.thresholding import tune_threshold_max_fbeta
from src.utils.logging import start_timer

def _to_unit_interval(scores: np.ndarray, ref_scores: np.ndarray | None = None) -> np.ndarray:
    """
    Min-max scale to [0,1]. If ref_scores is provided, scaling params come from ref_scores (train-only).
    """
    ref = ref_scores if ref_scores is not None else scores
    mn = float(np.min(ref))
    mx = float(np.max(ref))
    return (scores - mn) / (mx - mn + 1e-12)

@dataclass
class CVFoldResult:
    pr_auc: float
    thr_f1: float
    thr_f2: float
    f1: float
    f2: float
    fit_seconds: float

def run_stratified_cv(
    X, y,
    pipeline: Pipeline,
    n_splits: int,
    seed: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      cv_summary: mean/std metrics + list of fold thresholds
      timing: total CV seconds
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_results: List[CVFoldResult] = []
    t_total = start_timer()

    for _, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        model = clone(pipeline)

        t_fit = start_timer()
        model.fit(X_tr, y_tr)
        fit_s = t_fit.stop()

        # Scores on validation
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_va)[:, 1]
        else:
            # decision_function -> scaled using TRAIN decision scores if possible
            # For leakage-free scaling: compute ref on train subset only
            ref_scores = model.decision_function(X_tr)
            y_score_raw = model.decision_function(X_va)
            y_score = _to_unit_interval(y_score_raw, ref_scores=ref_scores)

        y_true = y_va.to_numpy()
        pr = pr_auc(y_true, y_score)

        thr_f1 = tune_threshold_max_fbeta(y_true, y_score, beta=1.0)
        thr_f2 = tune_threshold_max_fbeta(y_true, y_score, beta=2.0)

        f1 = fbeta_at_threshold(y_true, y_score, thr_f1, beta=1.0)
        f2 = fbeta_at_threshold(y_true, y_score, thr_f2, beta=2.0)

        fold_results.append(CVFoldResult(pr, thr_f1, thr_f2, f1, f2, fit_s))

    total_s = t_total.stop()

    pr_list = np.array([r.pr_auc for r in fold_results], dtype=float)
    f1_list = np.array([r.f1 for r in fold_results], dtype=float)
    f2_list = np.array([r.f2 for r in fold_results], dtype=float)
    thr_f1_list = np.array([r.thr_f1 for r in fold_results], dtype=float)
    thr_f2_list = np.array([r.thr_f2 for r in fold_results], dtype=float)
    fit_list = np.array([r.fit_seconds for r in fold_results], dtype=float)

    cv_summary = {
        "pr_auc_mean": float(pr_list.mean()),
        "pr_auc_std": float(pr_list.std(ddof=1)),
        "f1_mean": float(f1_list.mean()),
        "f1_std": float(f1_list.std(ddof=1)),
        "f2_mean": float(f2_list.mean()),
        "f2_std": float(f2_list.std(ddof=1)),
        "thr_f1_folds": thr_f1_list.tolist(),
        "thr_f2_folds": thr_f2_list.tolist(),
        "fit_seconds_mean": float(fit_list.mean()),
        "fit_seconds_std": float(fit_list.std(ddof=1)),
    }

    timing = {"cv_total_seconds": float(total_s), "n_splits": int(n_splits)}
    return cv_summary, timing