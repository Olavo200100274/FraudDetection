import joblib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────

def save_run(
    model,
    metrics_cv,
    metrics_test,
    y_test,
    y_test_scores,
    config,
    model_name,
    strategy="none",
    dataset="ulb_2013",
    bootstrap_ci=None,
):
    """
    Persist a complete run with all artefacts.

    Directory layout
    ----------------
    results/<dataset>/<model_name>/<strategy>/run_<timestamp>/
        config.json            – full reproducibility config
        model.joblib           – serialised pipeline (preprocess + model)
        metrics_cv.json        – cross-validation metrics (per fold + aggregated)
        metrics_test.json      – holdout test metrics + optional bootstrap CI
        confusion_matrix.png   – confusion-matrix heatmap
        pr_curve.png           – precision-recall curve
        pr_curve_data.json     – raw PR-curve points for LaTeX pgfplots
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        str(_PROJECT_ROOT), "results", dataset, model_name, strategy, f"run_{timestamp}"
    )
    os.makedirs(run_dir, exist_ok=True)

    # 1. Config (reproducibility)
    full_config = {
        **config,
        "timestamp": timestamp,
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    _save_json(full_config, os.path.join(run_dir, "config.json"))

    # 2. Model
    joblib.dump(model, os.path.join(run_dir, "model.joblib"))

    # 3. Cross-validation metrics
    _save_json(metrics_cv, os.path.join(run_dir, "metrics_cv.json"))

    # 4. Test metrics (+ bootstrap CI if provided)
    test_out = dict(metrics_test)
    if bootstrap_ci is not None:
        test_out["bootstrap_ci"] = bootstrap_ci
    _save_json(test_out, os.path.join(run_dir, "metrics_test.json"))

    # 5. Confusion matrix plot
    threshold = metrics_test["threshold"]
    y_pred = (np.asarray(y_test_scores) >= threshold).astype(int)
    _save_confusion_matrix(y_test, y_pred, model_name, strategy, run_dir)

    # 6. PR curve plot
    _save_pr_curve(y_test, y_test_scores, model_name, strategy, run_dir)

    # 7. PR curve raw data
    _save_pr_data(y_test, y_test_scores, run_dir)

    # 8. Raw predictions (for threshold study, DeLong tests, etc.)
    np.save(os.path.join(run_dir, "y_test.npy"), np.asarray(y_test))
    np.save(os.path.join(run_dir, "y_test_scores.npy"), np.asarray(y_test_scores))

    print(f"  Run saved → {run_dir}")
    return run_dir


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────

def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(_make_serializable(obj), f, indent=2)


def _make_serializable(obj):
    """Recursively convert numpy types to Python-native types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_make_serializable(v) for v in obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _save_confusion_matrix(y_test, y_pred, model_name, strategy, run_dir):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"{model_name} ({strategy}) — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def _save_pr_curve(y_test, y_scores, model_name, strategy, run_dir):
    precisions, recalls, _ = precision_recall_curve(y_test, y_scores)
    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} ({strategy}) — Precision-Recall Curve")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "pr_curve.png"), dpi=150)
    plt.close()


def _save_pr_data(y_test, y_scores, run_dir):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    data = {
        "precisions": precisions.tolist(),
        "recalls": recalls.tolist(),
        "thresholds": thresholds.tolist(),
    }
    with open(os.path.join(run_dir, "pr_curve_data.json"), "w") as f:
        json.dump(data, f)