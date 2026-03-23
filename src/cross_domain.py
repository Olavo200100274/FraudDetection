"""
Cross-Domain Generalization Study (BAF Suite)
==============================================
Evaluates how well models trained on BAF Base transfer to Variants I–V
WITHOUT retraining or re-tuning.

Protocol:
  1. Load the saved BAF Base model (model.joblib) and its threshold (median τ)
  2. For each Variant I–V:
     a. Load the variant dataset, apply the same 80/20 split (seed=42)
     b. Use only the variant's TEST set (20%)
     c. Transform with the Base preprocessor (no re-fitting)
     d. Score with the Base model
     e. Apply the Base threshold τ
     f. Compute all metrics (PR-AUC, ROC-AUC, F1, F2, CM, bootstrap CI)
  3. Save results to cross_domain.json alongside the Base model artefacts

Usage:
    cd src/
    python cross_domain.py                    # all models
    python cross_domain.py --models lgbm      # single model
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from data import load_dataset, get_dataset_info
from evaluation.metrics import compute_all_metrics, bootstrap_ci


# ── Constants ────────────────────────────────────────────────────────────
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
SOURCE_DATASET = "baf_base"               # registry key for training data
SOURCE_LABEL = "baf_base"                 # results directory name

# Target variants: registry key → (human label, results dir label)
TARGET_VARIANTS = {
    "baf_var1": ("Variant I",   "baf_var1"),
    "baf_var2": ("Variant II",  "baf_var2"),
    "baf_var3": ("Variant III", "baf_var3"),
    "baf_var4": ("Variant IV",  "baf_var4"),
    "baf_var5": ("Variant V",   "baf_var5"),
}

MODEL_ORDER = ["logreg", "rf", "lgbm", "catboost", "fttransformer", "ocsvm"]


# ── Helpers ──────────────────────────────────────────────────────────────

def find_latest_run(dataset_label, model_name, strategy="none"):
    base = RESULTS_ROOT / dataset_label / model_name / strategy
    if not base.exists():
        return None
    runs = sorted(base.iterdir())
    return runs[-1] if runs else None


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    """Save dict as JSON, converting numpy types."""
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [convert(v) for v in o]
        return o

    with open(path, "w") as f:
        json.dump(convert(obj), f, indent=2)


def score_model(model, X, is_ocsvm=False):
    """Get anomaly scores from a trained model."""
    if is_ocsvm:
        return -model.decision_function(X)
    else:
        return model.predict_proba(X)[:, 1]


def score_torch_model(checkpoint, X_df, device=None):
    """Score a DataFrame using a saved FT-Transformer checkpoint."""
    import torch
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SkPipeline
    from models.fttransformer import FTTransformer, TabularDataset, evaluate, build_model
    from torch.utils.data import DataLoader

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hp = checkpoint["hyperparams"]
    d_num = checkpoint["d_numerical"]
    cat_cards = checkpoint["cat_cardinalities"]
    num_cols = checkpoint["num_cols"]
    cat_cols = checkpoint["cat_cols"]

    # Preprocess: fit on BAF Base train, transform variant test
    # We need to load Base train data for fitting the preprocessor
    from data import load_dataset
    X_base_train, _, _, _ = load_dataset("baf_base")

    # Numeric
    num_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    X_num = num_pipe.fit_transform(X_base_train[num_cols])  # fit on Base
    X_num_test = num_pipe.transform(X_df[num_cols]).astype(np.float32)  # transform variant

    # Categorical
    X_cat_test = None
    if cat_cols:
        cat_enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
        )
        cat_enc.fit(X_base_train[cat_cols])  # fit on Base
        X_cat_test = cat_enc.transform(X_df[cat_cols])

    # Build and load model
    model = build_model(hp, d_num, cat_cards).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loader = DataLoader(
        TabularDataset(X_num_test, X_cat_test),
        batch_size=2048, shuffle=False,
    )
    _, y_scores = evaluate(model, loader, device)
    return y_scores


# ── Main evaluation ─────────────────────────────────────────────────────

def evaluate_cross_domain(model_name, run_dir):
    """
    Evaluate one Base model on all 5 BAF Variants.

    Returns dict with per-variant metrics.
    """
    is_ocsvm = (model_name == "ocsvm")
    is_torch = (model_name == "fttransformer")

    print(f"\n{'═' * 60}")
    print(f"  {model_name.upper()} — Cross-Domain Evaluation")
    print(f"{'═' * 60}")

    # ── Load Base model + config ─────────────────────────────────
    if is_torch:
        import torch
        checkpoint = torch.load(run_dir / "model.pt", map_location="cpu",
                                weights_only=False)
        model = None  # scores computed via score_torch_model
    else:
        checkpoint = None
        model = joblib.load(run_dir / "model.joblib")

    base_metrics = load_json(run_dir / "metrics_test.json")
    base_threshold = base_metrics["threshold"]
    print(f"  Source     : BAF Base")
    print(f"  Run dir    : {run_dir}")
    print(f"  Threshold τ: {base_threshold:.6f}")
    print(f"  Base PR-AUC: {base_metrics['PR-AUC']:.4f}")
    print(f"  Base F2    : {base_metrics['F2']:.4f}")

    # ── Evaluate on each variant ─────────────────────────────────
    results = {}

    for var_key, (var_label, var_dir_label) in TARGET_VARIANTS.items():
        print(f"\n  ── {var_label} ──")
        t0 = time.time()

        # Load variant data (uses same split seed=42)
        _, X_var_test, _, y_var_test = load_dataset(var_key)
        print(f"    Test set : {len(X_var_test):,} samples, "
              f"{int(y_var_test.sum()):,} fraud ({y_var_test.mean():.2%})")

        # Score with Base model (no re-fitting)
        if is_torch:
            y_scores = score_torch_model(checkpoint, X_var_test)
        else:
            y_scores = score_model(model, X_var_test, is_ocsvm=is_ocsvm)

        # Compute metrics using Base threshold
        metrics = compute_all_metrics(y_var_test, y_scores, base_threshold)

        # Bootstrap CI
        ci = bootstrap_ci(y_var_test, y_scores, base_threshold)
        metrics["bootstrap_ci"] = ci

        elapsed = time.time() - t0
        print(f"    PR-AUC   : {metrics['PR-AUC']:.4f}")
        print(f"    ROC-AUC  : {metrics['ROC-AUC']:.4f}")
        print(f"    F2       : {metrics['F2']:.4f}  (τ={base_threshold:.6f})")
        print(f"    Time     : {elapsed:.1f}s")

        results[var_key] = {
            "variant_label": var_label,
            "metrics": metrics,
            "threshold_used": base_threshold,
            "test_size": len(X_var_test),
            "test_fraud": int(y_var_test.sum()),
            "eval_time_s": round(elapsed, 2),
        }

    # ── Add Base reference metrics ───────────────────────────────
    results["baf_base"] = {
        "variant_label": "Base (reference)",
        "metrics": base_metrics,
        "threshold_used": base_threshold,
    }

    return results


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-domain evaluation: BAF Base → Variants I–V"
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        choices=MODEL_ORDER + ["all"],
        default=["all"],
        help="Models to evaluate (default: all)",
    )
    args = parser.parse_args()

    models = MODEL_ORDER if "all" in args.models else args.models

    print("=" * 60)
    print("  Cross-Domain Generalization Study (BAF Suite)")
    print("=" * 60)

    for model_name in models:
        run_dir = find_latest_run(SOURCE_LABEL, model_name)
        if run_dir is None:
            print(f"\n  [SKIP] {model_name} — no baseline run found in "
                  f"results/{SOURCE_LABEL}/{model_name}/none/")
            continue

        results = evaluate_cross_domain(model_name, run_dir)

        # Save alongside the Base model
        out_path = run_dir / "cross_domain.json"
        save_json(results, out_path)
        print(f"\n  → Saved: {out_path}")

    print("\n" + "=" * 60)
    print("  Cross-domain evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
