"""
FT-Transformer Training Script — Fraud Detection Pipeline
==========================================================
Standalone script for training the FT-Transformer model.
Separate from main.py because:
  - Holdout validation (not 5-fold CV) — DL training is too expensive for CV
  - PyTorch training loop with epoch-based early stopping
  - GPU training with CUDA tensors

Follows the same 4-phase protocol as main.py:
  strategy=none (baseline):
    [1/4] Hyperparameter tuning — Optuna TPE, 50 trials, holdout val, PR-AUC
    [2/4] Threshold selection   — max-F2 on validation set
    [3/4] Final training        — best HP on full DEV set (train+val)
    [4/4] Test evaluation       — holdout test with τ from step 2

  strategy=<resampling|weights>:
    [1/3] Load baseline HP      — no re-tuning
    [2/3] Train + threshold     — with resampling/weighted loss, early stop on val
    [3/3] Test evaluation       — holdout test

Usage:
    cd src/
    python main_transformer.py --dataset ulb --n_trials 50
    python main_transformer.py --dataset baf_base --strategy all
    python main_transformer.py --dataset ulb --sample 0.01 --n_trials 3  # smoke test
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from torch.utils.data import DataLoader

optuna.logging.set_verbosity(optuna.logging.WARNING)

from data import load_dataset, get_dataset_info, DATASET_REGISTRY
from evaluation.metrics import (
    find_threshold_maximizing_f2,
    compute_all_metrics,
    bootstrap_ci,
)
from save_load import save_run
from strategies.balancing import (
    STRATEGY_NAMES,
    RESAMPLING_STRATEGIES,
    get_sampler,
)
from models.fttransformer import (
    FTTransformer,
    TabularDataset,
    train_one_epoch,
    evaluate,
    build_model,
    suggest_hyperparams,
)


# ── Constants ─────────────────────────────────────────────────────────────
SPLIT_SEED = 42
VAL_FRACTION = 0.2          # 20% of DEV for validation
MAX_EPOCHS = 200
PATIENCE = 15               # early stopping patience (epochs)
EVAL_BATCH_SIZE = 2048
BOOTSTRAP_ITERATIONS = 1000
NUM_WORKERS = 0             # DataLoader workers (0 = main process, safest on Windows)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = _PROJECT_ROOT / "results"


# ── Reproducibility ──────────────────────────────────────────────────────
def _seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Preprocessing ─────────────────────────────────────────────────────────
def preprocess_for_transformer(X_train_df, X_other_df):
    """
    Separate numeric and categorical preprocessing.

    - Numeric: SimpleImputer(mean) → StandardScaler  (fitted on X_train)
    - Categorical: OrdinalEncoder → integer indices   (fitted on X_train)

    Returns
    -------
    X_num_train, X_cat_train, X_num_other, X_cat_other,
    cat_cardinalities, d_numerical, num_pipeline, cat_encoder, num_cols, cat_cols
    """
    num_cols = X_train_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = X_train_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numeric branch
    num_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    X_num_train = np.asarray(num_pipe.fit_transform(X_train_df[num_cols]), dtype=np.float32)
    X_num_other = np.asarray(num_pipe.transform(X_other_df[num_cols]), dtype=np.float32)

    # Categorical branch
    X_cat_train = None
    X_cat_other = None
    cat_cardinalities = []
    cat_encoder = None

    if cat_cols:
        cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
            dtype=np.int64,
        )
        X_cat_train = np.asarray(cat_encoder.fit_transform(X_train_df[cat_cols]), dtype=np.int64)
        X_cat_other = np.asarray(cat_encoder.transform(X_other_df[cat_cols]), dtype=np.int64)
        cat_cardinalities = [len(c) for c in cat_encoder.categories_]

    return (
        X_num_train, X_cat_train,
        X_num_other, X_cat_other,
        cat_cardinalities, len(num_cols),
        num_pipe, cat_encoder, num_cols, cat_cols,
    )


# ── Helpers ───────────────────────────────────────────────────────────────

def _file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "file_not_found"


def _find_latest_run(dataset_label, model_name, strategy="none"):
    base = RESULTS_ROOT / dataset_label / model_name / strategy
    if not base.exists():
        return None
    runs = sorted(base.iterdir())
    return runs[-1] if runs else None


def _load_baseline_params(dataset_name):
    """Load best hyperparameters from baseline (strategy=none) run."""
    run_dir = _find_latest_run(dataset_name, "fttransformer", "none")
    if run_dir is None:
        raise FileNotFoundError(
            f"No baseline run found for fttransformer on {dataset_name}. "
            "Run baseline first: python main_transformer.py --dataset ... --n_trials 50"
        )
    config_path = run_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    return config["best_params"], config.get("best_epoch", MAX_EPOCHS)


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _print_results(m, ci=None):
    print("\n  -- Test Results --")
    print(f"  PR-AUC       : {m['PR-AUC']:.4f}", end="")
    if ci:
        print(f"  [95% CI: {ci['PR-AUC_ci'][0]:.4f} - {ci['PR-AUC_ci'][1]:.4f}]")
    else:
        print()
    print(f"  ROC-AUC      : {m['ROC-AUC']:.4f}", end="")
    if ci:
        print(f"  [95% CI: {ci['ROC-AUC_ci'][0]:.4f} - {ci['ROC-AUC_ci'][1]:.4f}]")
    else:
        print()
    print(f"  F1           : {m['F1']:.4f}")
    print(f"  F2           : {m['F2']:.4f}", end="")
    if ci:
        print(f"  [95% CI: {ci['F2_ci'][0]:.4f} - {ci['F2_ci'][1]:.4f}]")
    else:
        print()
    print(f"  Brier score  : {m['brier_score']:.4f}")
    print(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")
    print(f"  Alert rate   : {m['alert_rate']:.4%}")
    print(f"  FP/TP        : {m['FP/TP']:.2f}")
    print(f"  Threshold    : {m['threshold']:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Training core
# ═══════════════════════════════════════════════════════════════════════════

def _train_model(model, train_loader, val_loader, hp, device,
                 criterion=None, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 trial=None):
    """
    Train an FT-Transformer with early stopping.

    Returns
    -------
    best_state_dict : dict
    best_val_prauc : float
    best_epoch : int
    val_scores : np.ndarray (scores from best epoch)
    val_true : np.ndarray
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-7,
    )

    best_val_prauc = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    best_val_true = None
    best_val_scores = None

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        y_true_val, y_scores_val = evaluate(model, val_loader, device)
        val_prauc = float(average_precision_score(y_true_val, y_scores_val))

        # Report to Optuna for pruning
        if trial is not None:
            trial.report(val_prauc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
            best_val_true = y_true_val
            best_val_scores = y_scores_val
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_state, best_val_prauc, best_epoch, best_val_scores, best_val_true


# ═══════════════════════════════════════════════════════════════════════════
#  Baseline (strategy=none)
# ═══════════════════════════════════════════════════════════════════════════

def run_baseline(X_train, X_test, y_train, y_test,
                 dataset_name, dataset_file, dataset_hash,
                 n_trials=50):
    """Full protocol for FT-Transformer baseline (strategy=none)."""
    _seed_everything(SPLIT_SEED)
    device = _get_device()

    print(f"\n{'=' * 60}")
    print(f"  FT-TRANSFORMER -- Strategy: None")
    print(f"{'=' * 60}")
    print(f"  Device: {device}")

    # ── Split DEV into train (80%) + val (20%) ─────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_FRACTION, stratify=y_train, random_state=SPLIT_SEED,
    )
    print(f"  DEV split: train={len(X_tr):,}, val={len(X_val):,}")

    # ── Preprocess ─────────────────────────────────────────────────
    (X_num_tr, X_cat_tr, X_num_val, X_cat_val,
     cat_cards, d_num, num_pipe, cat_enc, num_cols, cat_cols) = \
        preprocess_for_transformer(X_tr, X_val)

    val_loader = DataLoader(
        TabularDataset(X_num_val, X_cat_val, y_val.values),
        batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    # ── [1/4] Optuna HPO ───────────────────────────────────────────
    print(f"\n[1/4] Hyperparameter tuning ({n_trials} trials) ...")
    t0 = time.time()

    def objective(trial):
        hp = suggest_hyperparams(trial)
        trial.set_user_attr("hp", hp)

        model = build_model(hp, d_num, cat_cards).to(device)

        train_loader = DataLoader(
            TabularDataset(X_num_tr, X_cat_tr, y_tr.values),
            batch_size=hp["batch_size"], shuffle=True, num_workers=NUM_WORKERS,
        )

        _, best_prauc, best_ep, _, _ = _train_model(
            model, train_loader, val_loader, hp, device, trial=trial,
        )
        trial.set_user_attr("best_epoch", best_ep)
        return best_prauc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SPLIT_SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10,
        ),
    )
    study.optimize(objective, n_trials=n_trials)

    best_hp = study.best_trial.user_attrs["hp"]
    best_epoch = study.best_trial.user_attrs.get("best_epoch", MAX_EPOCHS)
    tuning_time = time.time() - t0
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    print(f"  Best val PR-AUC: {study.best_value:.4f}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Pruned: {n_pruned}/{n_trials}")
    print(f"  Tuning time: {tuning_time:.1f}s")
    print(f"  Best HP: { {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best_hp.items()} }")

    # ── [2/4] Threshold selection on val set ───────────────────────
    print("\n[2/4] Threshold selection (max-F2 on val set) ...")

    _seed_everything(SPLIT_SEED)
    model = build_model(best_hp, d_num, cat_cards).to(device)
    train_loader = DataLoader(
        TabularDataset(X_num_tr, X_cat_tr, y_tr.values),
        batch_size=best_hp["batch_size"], shuffle=True, num_workers=NUM_WORKERS,
    )
    best_state, _, _, val_scores, val_true = _train_model(
        model, train_loader, val_loader, best_hp, device,
    )
    model.load_state_dict(best_state)

    tau_final, best_f2_val = find_threshold_maximizing_f2(val_true, val_scores)
    print(f"  Threshold (max-F2): {tau_final:.6f}  (val F2={best_f2_val:.4f})")

    # ── [3/4] Final training on full DEV (train + val) ─────────────
    print("\n[3/4] Final training on full DEV set ...")
    t0_train = time.time()

    # Re-preprocess with all DEV data
    (X_num_dev, X_cat_dev, X_num_test, X_cat_test,
     cat_cards_final, d_num_final, num_pipe_final, cat_enc_final,
     num_cols_final, cat_cols_final) = \
        preprocess_for_transformer(X_train, X_test)

    _seed_everything(SPLIT_SEED)
    final_model = build_model(best_hp, d_num_final, cat_cards_final).to(device)
    dev_loader = DataLoader(
        TabularDataset(X_num_dev, X_cat_dev, y_train.values),
        batch_size=best_hp["batch_size"], shuffle=True, num_workers=NUM_WORKERS,
    )

    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_hp["learning_rate"],
        weight_decay=best_hp["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=best_epoch, eta_min=1e-7,
    )
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(best_epoch):
        train_one_epoch(final_model, dev_loader, optimizer, criterion, device)
        scheduler.step()

    train_time = time.time() - t0_train
    print(f"  Trained for {best_epoch} epochs in {train_time:.1f}s")

    # ── [4/4] Test evaluation ──────────────────────────────────────
    print("\n[4/4] Test evaluation ...")
    t0_infer = time.time()

    test_loader = DataLoader(
        TabularDataset(X_num_test, X_cat_test, y_test.values),
        batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    y_true_test, y_test_scores = evaluate(final_model, test_loader, device)
    infer_time = time.time() - t0_infer

    metrics_test = compute_all_metrics(y_test, y_test_scores, tau_final)
    ci = bootstrap_ci(y_test, y_test_scores, tau_final)

    _print_results(metrics_test, ci)

    # ── Validation metrics (for metrics_cv.json compatibility) ─────
    val_metrics = compute_all_metrics(val_true, val_scores, tau_final)
    metrics_cv = {
        "validation": val_metrics,
        "note": "Single holdout validation split (not 5-fold CV)",
    }

    # ── Save checkpoint + artefacts ────────────────────────────────
    checkpoint = {
        "model_state_dict": final_model.cpu().state_dict(),
        "hyperparams": best_hp,
        "d_numerical": d_num_final,
        "cat_cardinalities": cat_cards_final,
        "num_cols": num_cols_final,
        "cat_cols": cat_cols_final,
        "best_epoch": best_epoch,
        "threshold": tau_final,
    }

    config = {
        "dataset": dataset_name,
        "dataset_file": dataset_file,
        "dataset_hash_sha256": dataset_hash,
        "split_seed": SPLIT_SEED,
        "split_ratio": "80/20 stratified",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_fraud": int(y_train.sum()),
        "test_fraud": int(y_test.sum()),
        "model": "fttransformer",
        "strategy": "none",
        "validation_split": f"{1 - VAL_FRACTION:.0%}/{VAL_FRACTION:.0%} within DEV",
        "scoring": "average_precision (PR-AUC)",
        "tuning": "Optuna (TPE sampler + MedianPruner)",
        "n_trials": n_trials,
        "n_pruned": n_pruned,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "best_epoch": best_epoch,
        "threshold_rule": "maximise F2 on holdout validation set",
        "best_params": best_hp,
        "tuning_time_s": round(tuning_time, 2),
        "train_time_s": round(train_time, 2),
        "infer_time_s": round(infer_time, 4),
    }

    # Save using existing save_run (model=checkpoint for torch)
    run_dir = save_run(
        model=checkpoint,
        metrics_cv=metrics_cv,
        metrics_test=metrics_test,
        y_test=np.asarray(y_test),
        y_test_scores=y_test_scores,
        config=config,
        model_name="fttransformer",
        strategy="none",
        dataset=dataset_name,
        bootstrap_ci=ci,
        model_type="torch",
    )

    return final_model, metrics_test, run_dir


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy runs (resampling / class weights)
# ═══════════════════════════════════════════════════════════════════════════

def run_strategy(strategy, X_train, X_test, y_train, y_test,
                 dataset_name, dataset_file, dataset_hash):
    """FT-Transformer with an imbalance strategy (no re-tuning)."""
    _seed_everything(SPLIT_SEED)
    device = _get_device()

    print(f"\n{'=' * 60}")
    print(f"  FT-TRANSFORMER -- Strategy: {strategy}")
    print(f"{'=' * 60}")

    # Load baseline HP
    best_hp, best_epoch_baseline = _load_baseline_params(dataset_name)
    print(f"  Loaded baseline HP (best_epoch={best_epoch_baseline})")

    # ── Split DEV into train (80%) + val (20%) ─────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_FRACTION, stratify=y_train, random_state=SPLIT_SEED,
    )

    # ── Preprocess ─────────────────────────────────────────────────
    (X_num_tr, X_cat_tr, X_num_val, X_cat_val,
     cat_cards, d_num, num_pipe, cat_enc, num_cols, cat_cols) = \
        preprocess_for_transformer(X_tr, X_val)

    # ── Apply resampling strategy ──────────────────────────────────
    use_weights = (strategy == "weights")
    criterion = nn.BCEWithLogitsLoss()

    if strategy in RESAMPLING_STRATEGIES:
        print(f"  Applying {strategy} resampling ...")
        # Concatenate num + cat, resample, split back
        if X_cat_tr is not None:
            X_combined = np.hstack([X_num_tr, X_cat_tr])
        else:
            X_combined = X_num_tr

        sampler = get_sampler(strategy, random_state=SPLIT_SEED)
        X_res, y_res = sampler.fit_resample(X_combined, y_tr.values)

        X_num_tr = X_res[:, :d_num].astype(np.float32)
        if X_cat_tr is not None:
            X_cat_tr = X_res[:, d_num:].astype(np.int64)

        y_tr_values = y_res.astype(np.float32)
        print(f"  Resampled: {len(y_res):,} samples "
              f"({int(y_res.sum()):,} fraud, {y_res.mean():.2%})")
    elif use_weights:
        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        y_tr_values = y_tr.values.astype(np.float32)
        print(f"  Using weighted loss: pos_weight={n_neg / n_pos:.2f}")
    else:
        y_tr_values = y_tr.values.astype(np.float32)

    # ── Train with early stopping on val ───────────────────────────
    print("\n[1/3] Training with early stopping ...")
    t0 = time.time()

    model = build_model(best_hp, d_num, cat_cards).to(device)
    train_loader = DataLoader(
        TabularDataset(X_num_tr, X_cat_tr, y_tr_values),
        batch_size=best_hp["batch_size"], shuffle=True, num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        TabularDataset(X_num_val, X_cat_val, y_val.values),
        batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    best_state, best_prauc, best_epoch, val_scores, val_true = _train_model(
        model, train_loader, val_loader, best_hp, device,
        criterion=criterion,
    )
    model.load_state_dict(best_state)

    # ── [2/3] Threshold selection ──────────────────────────────────
    print("\n[2/3] Threshold selection (max-F2 on val set) ...")
    tau_final, best_f2_val = find_threshold_maximizing_f2(val_true, val_scores)
    print(f"  Threshold: {tau_final:.6f}  (val F2={best_f2_val:.4f})")

    # ── Final training on full DEV ─────────────────────────────────
    print("\n[3/3] Final training on full DEV + test evaluation ...")

    (X_num_dev, X_cat_dev, X_num_test, X_cat_test,
     cat_cards_f, d_num_f, num_pipe_f, cat_enc_f,
     num_cols_f, cat_cols_f) = \
        preprocess_for_transformer(X_train, X_test)

    # Apply resampling to full DEV if needed
    y_dev_values = y_train.values.astype(np.float32)
    final_criterion = nn.BCEWithLogitsLoss()

    if strategy in RESAMPLING_STRATEGIES:
        if X_cat_dev is not None:
            X_combined_dev = np.hstack([X_num_dev, X_cat_dev])
        else:
            X_combined_dev = X_num_dev
        sampler = get_sampler(strategy, random_state=SPLIT_SEED)
        X_res_dev, y_res_dev = sampler.fit_resample(X_combined_dev, y_train.values)
        X_num_dev = X_res_dev[:, :d_num_f].astype(np.float32)
        if X_cat_dev is not None:
            X_cat_dev = X_res_dev[:, d_num_f:].astype(np.int64)
        y_dev_values = y_res_dev.astype(np.float32)
    elif use_weights:
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        pos_w = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
        final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    _seed_everything(SPLIT_SEED)
    final_model = build_model(best_hp, d_num_f, cat_cards_f).to(device)
    dev_loader = DataLoader(
        TabularDataset(X_num_dev, X_cat_dev, y_dev_values),
        batch_size=best_hp["batch_size"], shuffle=True, num_workers=NUM_WORKERS,
    )

    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_hp["learning_rate"],
        weight_decay=best_hp["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=best_epoch, eta_min=1e-7,
    )
    for epoch in range(best_epoch):
        train_one_epoch(final_model, dev_loader, optimizer, final_criterion, device)
        scheduler.step()

    train_time = time.time() - t0

    # ── Test evaluation ────────────────────────────────────────────
    t0_infer = time.time()
    test_loader = DataLoader(
        TabularDataset(X_num_test, X_cat_test, y_test.values),
        batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    y_true_test, y_test_scores = evaluate(final_model, test_loader, device)
    infer_time = time.time() - t0_infer

    metrics_test = compute_all_metrics(y_test, y_test_scores, tau_final)
    ci = bootstrap_ci(y_test, y_test_scores, tau_final)

    _print_results(metrics_test, ci)

    val_metrics = compute_all_metrics(val_true, val_scores, tau_final)
    metrics_cv = {
        "validation": val_metrics,
        "note": f"Single holdout validation, strategy={strategy}",
    }

    checkpoint = {
        "model_state_dict": final_model.cpu().state_dict(),
        "hyperparams": best_hp,
        "d_numerical": d_num_f,
        "cat_cardinalities": cat_cards_f,
        "num_cols": num_cols_f,
        "cat_cols": cat_cols_f,
        "best_epoch": best_epoch,
        "threshold": tau_final,
    }

    config = {
        "dataset": dataset_name,
        "dataset_file": dataset_file,
        "dataset_hash_sha256": dataset_hash,
        "split_seed": SPLIT_SEED,
        "split_ratio": "80/20 stratified",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_fraud": int(y_train.sum()),
        "test_fraud": int(y_test.sum()),
        "model": "fttransformer",
        "strategy": strategy,
        "validation_split": f"{1 - VAL_FRACTION:.0%}/{VAL_FRACTION:.0%} within DEV",
        "threshold_rule": "maximise F2 on holdout validation set",
        "best_params": best_hp,
        "best_params_source": "loaded from baseline run (no re-tuning)",
        "best_epoch": best_epoch,
        "train_time_s": round(train_time, 2),
        "infer_time_s": round(infer_time, 4),
    }

    run_dir = save_run(
        model=checkpoint,
        metrics_cv=metrics_cv,
        metrics_test=metrics_test,
        y_test=np.asarray(y_test),
        y_test_scores=y_test_scores,
        config=config,
        model_name="fttransformer",
        strategy=strategy,
        dataset=dataset_name,
        bootstrap_ci=ci,
        model_type="torch",
    )

    return final_model, metrics_test


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="FT-Transformer — Fraud Detection Pipeline"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset: " + " | ".join(DATASET_REGISTRY.keys()),
    )
    parser.add_argument(
        "--strategy", type=str, nargs="+", default=["none"],
        choices=STRATEGY_NAMES + ["all"],
        help="Imbalance strategy (default: none = baseline)",
    )
    parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Optuna trials for baseline tuning (default: 50)",
    )
    parser.add_argument(
        "--sample", type=float, default=None,
        help="Stratified subsample fraction for smoke tests (e.g. 0.01)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve strategies
    strategies = args.strategy
    if "all" in strategies:
        strategies = [s for s in STRATEGY_NAMES if s != "none"]

    # Load data
    dataset_file, dataset_name = get_dataset_info(args.dataset)
    print(f"Loading {dataset_name} dataset ...")
    X_train, X_test, y_train, y_test = load_dataset(args.dataset, sample=args.sample)
    print(f"  Train : {len(X_train):,} samples ({int(y_train.sum())} fraud)")
    print(f"  Test  : {len(X_test):,} samples  ({int(y_test.sum())} fraud)")

    dataset_hash = _file_hash(dataset_file)
    device = _get_device()
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Run baseline first if "none" in strategies
    if "none" in strategies:
        run_baseline(
            X_train, X_test, y_train, y_test,
            dataset_name, dataset_file, dataset_hash,
            n_trials=args.n_trials,
        )
        strategies = [s for s in strategies if s != "none"]

    # Run other strategies
    for strategy in strategies:
        run_strategy(
            strategy, X_train, X_test, y_train, y_test,
            dataset_name, dataset_file, dataset_hash,
        )

    print(f"\n{'=' * 60}")
    print("  FT-Transformer runs complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
