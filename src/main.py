"""
Fraud Detection Pipeline — Leakage-Free Evaluation Protocol
============================================================
Phase 3 of the thesis: baseline tuned models (strategy = None)
on the ULB Credit Card 2013 dataset.

For each supervised model:
  1. Hyperparameter tuning  — GridSearchCV, scoring = PR-AUC (average_precision)
  2. Threshold selection    — 5-fold CV, maximise F2 per fold, τ = median
  3. Final training         — best params on full training partition
  4. Test evaluation        — holdout test with median τ
  5. Persist                — config + model + metrics_cv + metrics_test + plots

For OCSVM (anomaly-detection baseline):
  - Trained only on class-0 (legitimate) samples
  - Scoring = negated decision_function
  - No hyperparameter grid (fixed params)
  - Same threshold selection + evaluation protocol
"""

import argparse
import hashlib
import os
import time

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data import load_ulb_data
from preprocess import get_preprocessor
from models.logreg import get_pipeline_and_params as get_logreg
from models.rf import get_pipeline_and_params as get_rf
from models.lgbm import get_pipeline_and_params as get_lgbm
from models.catboost import get_pipeline_and_params as get_catboost
from models.ocsvm import get_pipeline_and_params as get_ocsvm
from evaluation.metrics import (
    find_threshold_maximizing_f2,
    compute_all_metrics,
    bootstrap_ci,
)
from save_load import save_run


# ── Constants ─────────────────────────────────────────────────────────────
DATASET_NAME = "ulb_2013"
DATASET_FILE = "datasets/creditcard_2013.csv"
SPLIT_SEED = 42
CV_SPLITS = 5
BOOTSTRAP_ITERATIONS = 1000

# Inner CV — same folds for tuning AND threshold selection
CV = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SPLIT_SEED)

SUPERVISED_MODELS = {
    "logreg": get_logreg,
    "rf": get_rf,
    "lgbm": get_lgbm,
    "catboost": get_catboost,
}


# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fraud Detection — Leakage-Free Evaluation Pipeline"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["logreg", "rf", "lgbm", "catboost", "ocsvm", "all"],
        required=True,
        help="Models to run: logreg | rf | lgbm | catboost | ocsvm | all",
    )
    return parser.parse_args()


# ── Dataset fingerprint (reproducibility) ─────────────────────────────────
def _file_hash(path, algorithm="sha256"):
    """Return hex digest of a file for reproducibility logging."""
    h = hashlib.new(algorithm)
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "file_not_found"


# ── supervised models ─────────────────────────────────────────────────────
def run_supervised(model_name, X_train, X_test, y_train, y_test,
                   preprocessor, dataset_hash):
    """Full leakage-free protocol for a supervised classifier (strategy=None)."""

    get_fn = SUPERVISED_MODELS[model_name]
    pipeline, param_grid = get_fn(preprocessor)

    print(f"\n{'=' * 60}")
    print(f"  {model_name.upper()} — Strategy: None")
    print(f"{'=' * 60}")

    # ── 1. Hyperparameter tuning (GridSearchCV, scoring = PR-AUC) ─────
    print(f"\n[1/4] Hyperparameter tuning ({model_name}) ...")
    t0 = time.time()

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=CV,
        scoring="average_precision",   # PR-AUC
        verbose=1,
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    tuning_time = time.time() - t0
    best_params = grid_search.best_params_
    print(f"  Best params : {best_params}")
    print(f"  CV PR-AUC   : {grid_search.best_score_:.4f}")
    print(f"  Tuning time : {tuning_time:.1f}s")

    # ── 2. Threshold selection (5-fold CV, maximise F2 → median τ) ────
    #    Also collect per-fold validation metrics for metrics_cv.json
    print(f"\n[2/4] Threshold selection ({model_name}) ...")
    fold_results = []

    for i, (train_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        fold_model = clone(grid_search.best_estimator_)
        fold_model.fit(X_fold_train, y_fold_train)

        y_val_scores = fold_model.predict_proba(X_fold_val)[:, 1]
        tau, _ = find_threshold_maximizing_f2(y_fold_val, y_val_scores)
        fold_metrics = compute_all_metrics(y_fold_val, y_val_scores, tau)
        fold_metrics["fold"] = i + 1

        fold_results.append(fold_metrics)
        print(f"  Fold {i + 1}: τ={tau:.6f}  PR-AUC={fold_metrics['PR-AUC']:.4f}"
              f"  F1={fold_metrics['F1']:.4f}  F2={fold_metrics['F2']:.4f}")

    # Aggregate CV metrics
    cv_keys = ["PR-AUC", "F1", "F2", "brier_score", "precision_at_k", "recall_at_k"]
    cv_summary = {}
    for k in cv_keys:
        vals = [fr[k] for fr in fold_results]
        cv_summary[f"{k}_mean"] = round(float(np.mean(vals)), 6)
        cv_summary[f"{k}_std"] = round(float(np.std(vals)), 6)

    thresholds = [fr["threshold"] for fr in fold_results]
    tau_final = float(np.median(thresholds))
    cv_summary["threshold_median"] = round(tau_final, 6)
    cv_summary["threshold_per_fold"] = [round(float(t), 6) for t in thresholds]

    metrics_cv = {
        "per_fold": fold_results,
        "aggregated": cv_summary,
    }

    print(f"  ── CV Aggregated ──")
    print(f"  PR-AUC : {cv_summary['PR-AUC_mean']:.4f} ± {cv_summary['PR-AUC_std']:.4f}")
    print(f"  F1     : {cv_summary['F1_mean']:.4f} ± {cv_summary['F1_std']:.4f}")
    print(f"  F2     : {cv_summary['F2_mean']:.4f} ± {cv_summary['F2_std']:.4f}")
    print(f"  Median threshold: τ = {tau_final:.6f}")

    # ── 3. Final training on full training partition ──────────────────
    print(f"\n[3/4] Training final model ({model_name}) ...")
    t0 = time.time()
    final_model = clone(grid_search.best_estimator_)
    final_model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # ── 4. Evaluation on holdout test set ─────────────────────────────
    print(f"\n[4/4] Evaluating on holdout test ({model_name}) ...")
    t0 = time.time()
    y_test_scores = final_model.predict_proba(X_test)[:, 1]
    infer_time = time.time() - t0

    metrics_test = compute_all_metrics(y_test, y_test_scores, tau_final)

    # Bootstrap CI on test
    print(f"  Computing bootstrap CI ({BOOTSTRAP_ITERATIONS} iterations) ...")
    ci = bootstrap_ci(y_test, y_test_scores, tau_final,
                      n_bootstrap=BOOTSTRAP_ITERATIONS)

    _print_results(metrics_test, ci)

    # ── Config (full reproducibility record) ──────────────────────────
    config = {
        "dataset": DATASET_NAME,
        "dataset_file": DATASET_FILE,
        "dataset_hash_sha256": dataset_hash,
        "split_seed": SPLIT_SEED,
        "split_ratio": "80/20 stratified",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_fraud": int(y_train.sum()),
        "test_fraud": int(y_test.sum()),
        "model": model_name,
        "strategy": "none",
        "cv_folds": CV_SPLITS,
        "scoring": "average_precision (PR-AUC)",
        "threshold_rule": "maximise F2 on validation, take median across folds",
        "best_params": best_params,
        "tuning_time_s": round(tuning_time, 2),
        "train_time_s": round(train_time, 2),
        "infer_time_s": round(infer_time, 4),
    }

    # ── Save everything ──────────────────────────────────────────────
    save_run(
        model=final_model,
        metrics_cv=metrics_cv,
        metrics_test=metrics_test,
        y_test=np.asarray(y_test),
        y_test_scores=y_test_scores,
        config=config,
        model_name=model_name,
        strategy="none",
        dataset=DATASET_NAME,
        bootstrap_ci=ci,
    )

    return final_model, metrics_test


# ── OCSVM (anomaly-detection baseline) ────────────────────────────────────
def run_ocsvm(X_train, X_test, y_train, y_test, preprocessor, dataset_hash):
    """
    Leakage-free protocol for One-Class SVM.

    * Trained only on class-0 (legitimate) samples.
    * Scoring = negated decision_function (higher → more anomalous).
    * No hyperparameter grid (fixed config).
    * Same threshold selection + evaluation protocol.
    """
    print(f"\n{'=' * 60}")
    print(f"  OCSVM — Anomaly Detection Baseline")
    print(f"{'=' * 60}")

    pipeline, _ = get_ocsvm(preprocessor)
    ocsvm_params = {"kernel": "rbf", "nu": 0.01, "gamma": "scale"}

    # ── 1. Threshold selection (5-fold CV, max F2 → median τ) ─────────
    print("\n[1/3] Threshold selection (ocsvm) ...")
    fold_results = []

    for i, (train_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        X_fold_train_0 = X_fold_train[y_fold_train == 0]

        fold_model = clone(pipeline)
        fold_model.fit(X_fold_train_0)

        # negate: higher score → more anomalous → more likely fraud
        y_val_scores = -fold_model.decision_function(X_fold_val)
        tau, _ = find_threshold_maximizing_f2(y_fold_val, y_val_scores)
        fold_metrics = compute_all_metrics(y_fold_val, y_val_scores, tau)
        fold_metrics["fold"] = i + 1

        fold_results.append(fold_metrics)
        print(f"  Fold {i + 1}: τ={tau:.6f}  PR-AUC={fold_metrics['PR-AUC']:.4f}"
              f"  F1={fold_metrics['F1']:.4f}  F2={fold_metrics['F2']:.4f}")

    cv_keys = ["PR-AUC", "F1", "F2", "brier_score", "precision_at_k", "recall_at_k"]
    cv_summary = {}
    for k in cv_keys:
        vals = [fr[k] for fr in fold_results]
        cv_summary[f"{k}_mean"] = round(float(np.mean(vals)), 6)
        cv_summary[f"{k}_std"] = round(float(np.std(vals)), 6)

    thresholds = [fr["threshold"] for fr in fold_results]
    tau_final = float(np.median(thresholds))
    cv_summary["threshold_median"] = round(tau_final, 6)
    cv_summary["threshold_per_fold"] = [round(float(t), 6) for t in thresholds]

    metrics_cv = {
        "per_fold": fold_results,
        "aggregated": cv_summary,
    }

    print(f"  ── CV Aggregated ──")
    print(f"  PR-AUC : {cv_summary['PR-AUC_mean']:.4f} ± {cv_summary['PR-AUC_std']:.4f}")
    print(f"  F1     : {cv_summary['F1_mean']:.4f} ± {cv_summary['F1_std']:.4f}")
    print(f"  F2     : {cv_summary['F2_mean']:.4f} ± {cv_summary['F2_std']:.4f}")
    print(f"  Median threshold: τ = {tau_final:.6f}")

    # ── 2. Final training (class 0 only) ──────────────────────────────
    print("\n[2/3] Training final model (ocsvm) ...")
    t0 = time.time()
    X_train_0 = X_train[y_train == 0]
    final_model = clone(pipeline)
    final_model.fit(X_train_0)
    train_time = time.time() - t0
    print(f"  Training time : {train_time:.1f}s")
    print(f"  Trained on    : {len(X_train_0):,} legitimate samples")

    # ── 3. Evaluation on holdout test set ─────────────────────────────
    print("\n[3/3] Evaluating on holdout test (ocsvm) ...")
    t0 = time.time()
    y_test_scores = -final_model.decision_function(X_test)
    infer_time = time.time() - t0

    metrics_test = compute_all_metrics(y_test, y_test_scores, tau_final)

    print(f"  Computing bootstrap CI ({BOOTSTRAP_ITERATIONS} iterations) ...")
    ci = bootstrap_ci(y_test, y_test_scores, tau_final,
                      n_bootstrap=BOOTSTRAP_ITERATIONS)

    _print_results(metrics_test, ci)

    config = {
        "dataset": DATASET_NAME,
        "dataset_file": DATASET_FILE,
        "dataset_hash_sha256": dataset_hash,
        "split_seed": SPLIT_SEED,
        "split_ratio": "80/20 stratified",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_fraud": int(y_train.sum()),
        "test_fraud": int(y_test.sum()),
        "model": "ocsvm",
        "strategy": "n/a",
        "cv_folds": CV_SPLITS,
        "threshold_rule": "maximise F2 on validation, take median across folds",
        "best_params": ocsvm_params,
        "train_time_s": round(train_time, 2),
        "infer_time_s": round(infer_time, 4),
        "note": "Trained only on class-0 (legitimate) samples. Scores = negated decision_function.",
    }

    save_run(
        model=final_model,
        metrics_cv=metrics_cv,
        metrics_test=metrics_test,
        y_test=np.asarray(y_test),
        y_test_scores=y_test_scores,
        config=config,
        model_name="ocsvm",
        strategy="none",
        dataset=DATASET_NAME,
        bootstrap_ci=ci,
    )

    return final_model, metrics_test


# ── helpers ───────────────────────────────────────────────────────────────
def _print_results(m, ci=None):
    print("\n  ── Test Results ──")
    print(f"  PR-AUC       : {m['PR-AUC']:.4f}", end="")
    if ci:
        print(f"  [95% CI: {ci['PR-AUC_ci'][0]:.4f} – {ci['PR-AUC_ci'][1]:.4f}]")
    else:
        print()
    print(f"  F1           : {m['F1']:.4f}")
    print(f"  F2           : {m['F2']:.4f}", end="")
    if ci:
        print(f"  [95% CI: {ci['F2_ci'][0]:.4f} – {ci['F2_ci'][1]:.4f}]")
    else:
        print()
    print(f"  Brier score  : {m['brier_score']:.4f}")
    print(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")
    print(f"  Alert rate   : {m['alert_rate']:.4%}")
    print(f"  FP/TP        : {m['FP/TP']:.2f}")
    print(f"  Prec@k       : {m['precision_at_k']:.4f}  (k={m['k_used']})")
    print(f"  Recall@k     : {m['recall_at_k']:.4f}  (k={m['k_used']})")
    print(f"  Threshold    : {m['threshold']:.6f}")


# ── entry point ───────────────────────────────────────────────────────────
def main():
    args = parse_args()
    models = args.models

    if "all" in models:
        models = ["logreg", "rf", "lgbm", "catboost", "ocsvm"]

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading ULB 2013 dataset ...")
    X_train, X_test, y_train, y_test = load_ulb_data()
    print(f"  Train : {len(X_train):,} samples  ({y_train.sum()} fraud)")
    print(f"  Test  : {len(X_test):,} samples   ({y_test.sum()} fraud)")

    dataset_hash = _file_hash(DATASET_FILE)
    print(f"  SHA-256 : {dataset_hash[:16]}...")

    # ── Preprocessor ──────────────────────────────────────────────────
    preprocessor = get_preprocessor(X_train)

    # ── Run requested models ──────────────────────────────────────────
    results = {}
    for name in models:
        if name == "ocsvm":
            _, metrics = run_ocsvm(
                X_train, X_test, y_train, y_test, preprocessor, dataset_hash
            )
        else:
            _, metrics = run_supervised(
                name, X_train, X_test, y_train, y_test, preprocessor, dataset_hash
            )
        results[name] = metrics

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print(f"  SUMMARY — ULB 2013 (Strategy: None)")
    print(f"{'=' * 100}")
    header = (
        f"{'Model':<10} {'PR-AUC':>8} {'F1':>8} {'F2':>8} {'Brier':>8} "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>7} "
        f"{'Alert%':>8} {'FP/TP':>7} {'P@k':>6} {'R@k':>6}"
    )
    print(header)
    print("-" * 100)
    for name in models:
        m = results[name]
        print(
            f"{name:<10} {m['PR-AUC']:>8.4f} {m['F1']:>8.4f} {m['F2']:>8.4f} "
            f"{m['brier_score']:>8.4f} "
            f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>7} "
            f"{m['alert_rate']:>7.4%} {m['FP/TP']:>7.2f} "
            f"{m['precision_at_k']:>6.3f} {m['recall_at_k']:>6.3f}"
        )
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()