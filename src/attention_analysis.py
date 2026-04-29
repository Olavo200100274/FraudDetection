"""
Attention Map Analysis for FT-Transformer
==========================================
Extracts and visualises attention weights from the trained FT-Transformer
to diagnose how the [CLS] token distributes attention across numerical
vs categorical features.

No retraining required — loads the saved checkpoint and runs inference.

Usage:
    cd src/
    python attention_analysis.py --dataset baf_base
    python attention_analysis.py --dataset ulb
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline

from data import load_dataset, get_dataset_info, DATASET_REGISTRY
from models.fttransformer import FTTransformer, TabularDataset, build_model

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = _PROJECT_ROOT / "results"
SPLIT_SEED = 42
VAL_FRACTION = 0.2
N_CASES = 3  # number of FP/FN cases to visualise


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def find_run_dir(dataset_name, strategy="none"):
    """Find the latest FT-Transformer run directory."""
    base = RESULTS_ROOT / dataset_name / "fttransformer" / strategy
    if not base.exists():
        raise FileNotFoundError(f"No runs found at {base}")
    runs = sorted(base.iterdir())
    if not runs:
        raise FileNotFoundError(f"No runs found at {base}")
    return runs[-1]


def load_checkpoint(run_dir, device):
    """Load the FT-Transformer checkpoint and rebuild the model."""
    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)

    hp = ckpt["hyperparams"]
    model = build_model(
        hp,
        d_numerical=ckpt["d_numerical"],
        cat_cardinalities=ckpt["cat_cardinalities"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, ckpt


def preprocess_data(X_train_df, X_other_df, num_cols, cat_cols):
    """Reproduce the same preprocessing as main_transformer.py."""
    num_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    X_num_train = np.asarray(
        num_pipe.fit_transform(X_train_df[num_cols]), dtype=np.float32
    )
    X_num_other = np.asarray(
        num_pipe.transform(X_other_df[num_cols]), dtype=np.float32
    )

    X_cat_train = None
    X_cat_other = None
    cat_cardinalities = []

    if cat_cols:
        cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64,
        )
        X_cat_train = np.asarray(
            cat_encoder.fit_transform(X_train_df[cat_cols]), dtype=np.int64
        )
        X_cat_other = np.asarray(
            cat_encoder.transform(X_other_df[cat_cols]), dtype=np.int64
        )
        cat_cardinalities = [len(c) for c in cat_encoder.categories_]

    return X_num_train, X_cat_train, X_num_other, X_cat_other, cat_cardinalities


def classify_predictions(y_true, y_scores, threshold):
    """Classify each sample into TP/FP/FN/TN."""
    y_pred = (y_scores >= threshold).astype(int)
    labels = np.full(len(y_true), "", dtype=object)
    labels[(y_true == 1) & (y_pred == 1)] = "TP"
    labels[(y_true == 0) & (y_pred == 1)] = "FP"
    labels[(y_true == 1) & (y_pred == 0)] = "FN"
    labels[(y_true == 0) & (y_pred == 0)] = "TN"
    return labels, y_pred


def select_hard_cases(labels, y_scores, threshold, n=N_CASES):
    """Select the most borderline FP and FN cases."""
    cases = {}
    distance_to_threshold = np.abs(y_scores - threshold)

    for label in ["FN", "FP"]:
        mask = labels == label
        if mask.sum() == 0:
            print(f"  Warning: no {label} cases found.")
            continue
        indices = np.where(mask)[0]
        # Sort by distance to threshold (most borderline first)
        sorted_idx = indices[np.argsort(distance_to_threshold[indices])]
        cases[label] = sorted_idx[:n]

    return cases


# ─────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────

def extract_attention(model, x_num, x_cat, device):
    """
    Extract attention weights for given samples.

    Returns
    -------
    attn_weights : list of np.ndarray
        One (n_samples, n_tokens, n_tokens) array per layer.
    """
    x_num_t = torch.tensor(x_num, dtype=torch.float32).to(device)
    x_cat_t = (
        torch.tensor(x_cat, dtype=torch.long).to(device) if x_cat is not None else None
    )

    logits, attn_layers = model.forward_with_attention(x_num_t, x_cat_t)
    return [w.cpu().numpy() for w in attn_layers], torch.sigmoid(logits).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────

def plot_cls_attention_bar(
    cls_weights, feature_names, num_cols, cat_cols, title, save_path,
    score=None, true_label=None,
):
    """
    Bar chart of [CLS] attention weights across features.

    Parameters
    ----------
    cls_weights : (n_features,) array — attention from CLS to each feature
    feature_names : list of str — [CLS] + numerical + categorical
    """
    # Skip the CLS→CLS weight (index 0); features start at index 1
    feat_weights = cls_weights[1:]  # exclude CLS self-attention
    feat_names = feature_names[1:]
    n_num = len(num_cols)
    n_cat = len(cat_cols)

    # Colours: blue for numerical, orange for categorical
    colors = ["#4472C4"] * n_num + ["#ED7D31"] * n_cat

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(feat_names))
    bars = ax.bar(x, feat_weights, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Attention Weight", fontsize=11)
    ax.set_xlabel("Feature", fontsize=11)

    # Add CLS self-attention annotation
    cls_self = cls_weights[0]
    subtitle = f"CLS→CLS weight: {cls_self:.4f}"
    if score is not None:
        subtitle += f" | P(fraud): {score:.4f}"
    if true_label is not None:
        subtitle += f" | True label: {'Fraud' if true_label == 1 else 'Legit'}"

    ax.set_title(f"{title}\n{subtitle}", fontsize=12)

    # Add mean lines for num vs cat
    mean_num = feat_weights[:n_num].mean()
    mean_cat = feat_weights[n_num:].mean()
    ax.axhline(mean_num, color="#4472C4", linestyle="--", alpha=0.7, linewidth=1.2,
               label=f"Mean numerical: {mean_num:.4f}")
    ax.axhline(mean_cat, color="#ED7D31", linestyle="--", alpha=0.7, linewidth=1.2,
               label=f"Mean categorical: {mean_cat:.4f}")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4472C4", label=f"Numerical ({n_num}) — mean: {mean_num:.4f}"),
        Patch(facecolor="#ED7D31", label=f"Categorical ({n_cat}) — mean: {mean_cat:.4f}"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_xlim(-0.5, len(feat_names) - 0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_aggregate_attention(
    all_cls_weights, feature_names, num_cols, cat_cols, save_path,
):
    """
    Aggregate bar chart: mean [CLS] attention over all test samples.
    """
    mean_weights = all_cls_weights.mean(axis=0)
    std_weights = all_cls_weights.std(axis=0)

    feat_weights = mean_weights[1:]
    feat_std = std_weights[1:]
    feat_names = feature_names[1:]
    n_num = len(num_cols)
    n_cat = len(cat_cols)

    colors = ["#4472C4"] * n_num + ["#ED7D31"] * n_cat

    # Sort by weight for readability
    sort_idx = np.argsort(feat_weights)[::-1]
    feat_weights = feat_weights[sort_idx]
    feat_std = feat_std[sort_idx]
    feat_names = [feat_names[i] for i in sort_idx]
    colors = [colors[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(feat_names))
    ax.bar(x, feat_weights, yerr=feat_std, color=colors,
           edgecolor="white", linewidth=0.5, capsize=2, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Mean Attention Weight", fontsize=11)
    ax.set_xlabel("Feature (sorted by weight)", fontsize=11)

    # Global means
    orig_weights = mean_weights[1:]
    mean_num = orig_weights[:n_num].mean()
    mean_cat = orig_weights[n_num:].mean()
    cls_self_mean = mean_weights[0]

    ax.set_title(
        f"Aggregate [CLS] Attention — All Test Samples (N={len(all_cls_weights):,})\n"
        f"CLS→CLS: {cls_self_mean:.4f} | "
        f"Mean num: {mean_num:.4f} | Mean cat: {mean_cat:.4f} | "
        f"Ratio cat/num: {mean_cat / mean_num:.2f}x",
        fontsize=11,
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4472C4", label=f"Numerical ({n_num})"),
        Patch(facecolor="#ED7D31", label=f"Categorical ({n_cat})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_xlim(-0.5, len(feat_names) - 0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_attention_heatmap(attn_weights, feature_names, title, save_path):
    """
    Heatmap of full attention matrix (all tokens → all tokens) for one sample.
    Uses the last layer's attention.
    """
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attn_weights, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=75, ha="right", fontsize=7)
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_xlabel("Key (attended to)", fontsize=10)
    ax.set_ylabel("Query (attending from)", fontsize=10)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Attention Map Analysis for FT-Transformer"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_REGISTRY.keys()),
    )
    parser.add_argument(
        "--batch_size", type=int, default=2048,
        help="Batch size for aggregate extraction (default: 2048)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load run artifacts ────────────────────────────────────────────
    run_dir = find_run_dir(args.dataset)
    print(f"Run dir: {run_dir}")

    model, ckpt = load_checkpoint(run_dir, device)
    num_cols = ckpt["num_cols"]
    cat_cols = ckpt["cat_cols"]
    threshold = ckpt["threshold"]
    feature_names = ["[CLS]"] + num_cols + cat_cols

    print(f"Model: d_token={ckpt['hyperparams']['d_token']}, "
          f"n_blocks={ckpt['hyperparams']['n_blocks']}, "
          f"n_heads={ckpt['hyperparams']['attention_n_heads']}")
    print(f"Features: {len(num_cols)} numerical + {len(cat_cols)} categorical")
    print(f"Threshold: {threshold:.6f}")

    # ── Load dataset and preprocess ───────────────────────────────────
    print(f"\nLoading {args.dataset} dataset ...")
    X_train, X_test, y_train, y_test = load_dataset(args.dataset)

    X_num_train, X_cat_train, X_num_test, X_cat_test, _ = preprocess_data(
        X_train, X_test, num_cols, cat_cols,
    )
    print(f"Test set: {len(X_num_test):,} samples")

    # ── Load saved scores for case selection ──────────────────────────
    y_test_arr = np.load(run_dir / "y_test.npy")
    y_scores_arr = np.load(run_dir / "y_test_scores.npy")

    labels, y_pred = classify_predictions(y_test_arr, y_scores_arr, threshold)
    print(f"\nConfusion matrix at threshold={threshold:.4f}:")
    for lbl in ["TP", "FP", "FN", "TN"]:
        print(f"  {lbl}: {(labels == lbl).sum():,}")

    # ── Output directory ──────────────────────────────────────────────
    out_dir = run_dir.parent.parent / "attention_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput dir: {out_dir}")

    # ── Select hard cases ─────────────────────────────────────────────
    hard_cases = select_hard_cases(labels, y_scores_arr, threshold)

    # ── Individual case visualisations ────────────────────────────────
    print("\n--- Individual Case Analysis ---")
    case_details = {}

    for case_type, indices in hard_cases.items():
        for rank, idx in enumerate(indices):
            x_num_i = X_num_test[idx : idx + 1]
            x_cat_i = X_cat_test[idx : idx + 1] if X_cat_test is not None else None

            attn_layers, scores = extract_attention(model, x_num_i, x_cat_i, device)

            # Use the last layer's attention
            last_layer_attn = attn_layers[-1][0]  # (n_tokens, n_tokens)
            cls_weights = last_layer_attn[0]  # CLS row → weights over all tokens

            # Sanity check
            assert abs(cls_weights.sum() - 1.0) < 1e-4, \
                f"CLS weights don't sum to 1: {cls_weights.sum()}"

            title = f"{case_type} Case #{rank + 1} (sample idx={idx})"
            save_name = f"attention_bar_{case_type}_{rank + 1}.png"

            plot_cls_attention_bar(
                cls_weights, feature_names, num_cols, cat_cols,
                title=title,
                save_path=out_dir / save_name,
                score=y_scores_arr[idx],
                true_label=y_test_arr[idx],
            )

            # Heatmap for one case per type
            if rank == 0:
                plot_attention_heatmap(
                    last_layer_attn, feature_names,
                    title=f"Full Attention Matrix — {case_type} Case #{rank + 1}",
                    save_path=out_dir / f"attention_heatmap_{case_type}.png",
                )

            case_details[f"{case_type}_{rank + 1}"] = {
                "sample_idx": int(idx),
                "score": float(y_scores_arr[idx]),
                "true_label": int(y_test_arr[idx]),
                "cls_self_attention": float(cls_weights[0]),
                "mean_numerical_attention": float(cls_weights[1 : 1 + len(num_cols)].mean()),
                "mean_categorical_attention": float(cls_weights[1 + len(num_cols) :].mean()),
            }

    # ── Aggregate analysis over full test set ─────────────────────────
    print("\n--- Aggregate Analysis (full test set) ---")
    all_cls_weights = []

    n_test = len(X_num_test)
    bs = args.batch_size
    for start in range(0, n_test, bs):
        end = min(start + bs, n_test)
        x_num_batch = X_num_test[start:end]
        x_cat_batch = X_cat_test[start:end] if X_cat_test is not None else None

        attn_layers, _ = extract_attention(model, x_num_batch, x_cat_batch, device)
        # Last layer, CLS row
        cls_w = attn_layers[-1][:, 0, :]  # (batch, n_tokens)
        all_cls_weights.append(cls_w)

        if (start // bs) % 10 == 0:
            print(f"  Processed {end:,}/{n_test:,} samples ...")

    all_cls_weights = np.concatenate(all_cls_weights, axis=0)  # (N, n_tokens)
    print(f"  Total: {all_cls_weights.shape[0]:,} samples")

    plot_aggregate_attention(
        all_cls_weights, feature_names, num_cols, cat_cols,
        save_path=out_dir / "attention_aggregate_cls.png",
    )

    # ── Compute summary statistics ────────────────────────────────────
    mean_w = all_cls_weights.mean(axis=0)
    cls_self = float(mean_w[0])
    num_mean = float(mean_w[1 : 1 + len(num_cols)].mean())
    cat_mean = float(mean_w[1 + len(num_cols) :].mean())

    # Entropy of attention distribution (higher = more uniform)
    eps = 1e-12
    entropy_per_sample = -(all_cls_weights * np.log(all_cls_weights + eps)).sum(axis=1)
    max_entropy = np.log(len(feature_names))  # uniform distribution entropy

    # Per-class analysis
    fraud_mask = y_test_arr == 1
    legit_mask = y_test_arr == 0

    summary = {
        "dataset": args.dataset,
        "n_test_samples": int(n_test),
        "n_numerical_features": len(num_cols),
        "n_categorical_features": len(cat_cols),
        "threshold": float(threshold),
        "attention_summary": {
            "cls_self_attention": cls_self,
            "mean_numerical_attention": num_mean,
            "mean_categorical_attention": cat_mean,
            "ratio_cat_over_num": cat_mean / num_mean if num_mean > 0 else float("inf"),
            "attention_entropy_mean": float(entropy_per_sample.mean()),
            "attention_entropy_std": float(entropy_per_sample.std()),
            "max_possible_entropy": float(max_entropy),
            "normalized_entropy": float(entropy_per_sample.mean() / max_entropy),
        },
        "per_class": {
            "fraud_cls_self": float(all_cls_weights[fraud_mask, 0].mean()) if fraud_mask.any() else None,
            "fraud_num_mean": float(all_cls_weights[fraud_mask, 1:1+len(num_cols)].mean()) if fraud_mask.any() else None,
            "fraud_cat_mean": float(all_cls_weights[fraud_mask, 1+len(num_cols):].mean()) if fraud_mask.any() else None,
            "legit_cls_self": float(all_cls_weights[legit_mask, 0].mean()) if legit_mask.any() else None,
            "legit_num_mean": float(all_cls_weights[legit_mask, 1:1+len(num_cols)].mean()) if legit_mask.any() else None,
            "legit_cat_mean": float(all_cls_weights[legit_mask, 1+len(num_cols):].mean()) if legit_mask.any() else None,
        },
        "per_feature_mean_attention": {
            name: float(mean_w[i]) for i, name in enumerate(feature_names)
        },
        "individual_cases": case_details,
    }

    # ── Diagnosis ─────────────────────────────────────────────────────
    ratio = cat_mean / num_mean if num_mean > 0 else float("inf")
    norm_entropy = entropy_per_sample.mean() / max_entropy

    diagnosis = []
    if cls_self > 0.3:
        diagnosis.append("SELF-OBSESSION: CLS token attends mostly to itself "
                         f"({cls_self:.1%}). Consider more blocks or lower LR.")
    if ratio > 2.0:
        diagnosis.append(f"CATEGORICAL TRAP: Categorical features get {ratio:.1f}x "
                         "more attention than numerical. Consider Periodic Embeddings.")
    if norm_entropy > 0.95:
        diagnosis.append("FOGGY VISION: Attention is nearly uniform "
                         f"(normalized entropy={norm_entropy:.3f}). "
                         "Model may struggle to focus on informative features.")
    # Check tunnel vision: any single feature > 30% of total non-CLS attention
    feat_w = mean_w[1:]
    max_feat_pct = feat_w.max() / feat_w.sum()
    if max_feat_pct > 0.30:
        top_feat = feature_names[1 + feat_w.argmax()]
        diagnosis.append(f"TUNNEL VISION: Feature '{top_feat}' receives "
                         f"{max_feat_pct:.1%} of total feature attention.")
    if not diagnosis:
        diagnosis.append("HEALTHY: Attention distribution appears balanced "
                         "across numerical and categorical features.")

    summary["diagnosis"] = diagnosis

    # Save summary
    summary_path = out_dir / "attention_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {summary_path}")

    # ── Print diagnosis ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  ATTENTION ANALYSIS DIAGNOSIS")
    print(f"{'=' * 60}")
    print(f"  CLS → CLS (self):      {cls_self:.4f} ({cls_self:.1%})")
    print(f"  Mean numerical attn:    {num_mean:.4f}")
    print(f"  Mean categorical attn:  {cat_mean:.4f}")
    print(f"  Ratio cat/num:          {ratio:.2f}x")
    print(f"  Normalized entropy:     {norm_entropy:.3f} (1.0 = perfectly uniform)")
    print()
    for d in diagnosis:
        print(f"  >> {d}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
