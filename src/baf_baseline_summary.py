"""Quick script to print BAF baseline metrics and generate PR curve overlay."""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

import numpy as np

models = ["logreg", "rf", "lgbm", "catboost", "ocsvm"]
labels = ["LR", "RF", "LGBM", "CatBoost", "OCSVM"]
base = "results/baf_base"

print("=== BAF Base Baseline — Full Metrics ===")
header = f"{'Model':<10} {'PR-AUC':>7} {'F1':>6} {'F2':>6} {'Prec':>6} {'Recall':>6} {'tau':>10} {'Brier':>7} {'Alert%':>7} {'FP/TP':>6} {'P@k':>5} {'R@k':>5} {'k':>5}"
print(header)
print("-" * len(header))

for m, lbl in zip(models, labels):
    none_dir = os.path.join(base, m, "none")
    run_dir = None
    for d in sorted(os.listdir(none_dir)):
        if d.startswith("run_"):
            run_dir = os.path.join(none_dir, d)
    with open(os.path.join(run_dir, "metrics_test.json")) as f:
        mt = json.load(f)
    tp, fp, fn = mt["TP"], mt["FP"], mt["FN"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(
        f"{lbl:<10} {mt['PR-AUC']:>7.4f} {mt['F1']:>6.4f} {mt['F2']:>6.4f} "
        f"{prec:>6.3f} {rec:>6.3f} {mt['threshold']:>10.6f} {mt['brier_score']:>7.4f} "
        f"{mt['alert_rate']*100:>6.2f}% {mt['FP/TP']:>6.2f} "
        f"{mt['precision_at_k']:>5.3f} {mt['recall_at_k']:>5.3f} {mt['k_used']:>5}"
    )

# === Generate PR Curve Overlay ===
print("\n=== Generating PR Curve Overlay ===")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = {
    "logreg": "#9467bd",
    "rf": "#2ca02c",
    "lgbm": "#1f77b4",
    "catboost": "#d62728",
    "ocsvm": "#7f7f7f",
}
model_labels = {
    "logreg": "LR",
    "rf": "RF",
    "lgbm": "LGBM",
    "catboost": "CatBoost",
    "ocsvm": "OCSVM",
}

fig, ax = plt.subplots(figsize=(8, 6))

for m in models:
    none_dir = os.path.join(base, m, "none")
    run_dir = None
    for d in sorted(os.listdir(none_dir)):
        if d.startswith("run_"):
            run_dir = os.path.join(none_dir, d)

    pr_path = os.path.join(run_dir, "pr_curve_data.json")
    mt_path = os.path.join(run_dir, "metrics_test.json")

    with open(pr_path) as f:
        pr = json.load(f)
    with open(mt_path) as f:
        mt = json.load(f)

    precision = pr["precisions"]
    recall = pr["recalls"]
    auc_val = mt["PR-AUC"]

    ax.plot(
        recall, precision,
        color=colors[m],
        linewidth=2,
        label=f"{model_labels[m]} (PR-AUC = {auc_val:.3f})",
    )

# Prevalence line
prevalence = 2206 / 200000  # test fraud rate
ax.axhline(y=prevalence, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
           label=f"Random ({prevalence:.2%})")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision–Recall Curves — BAF Base (Baseline)", fontsize=14)
ax.legend(loc="upper right", fontsize=10)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 0.5])  # BAF has low precision values
ax.grid(True, alpha=0.3)

figures_dir = os.path.join("results_thesis", "figures", "baf")
os.makedirs(figures_dir, exist_ok=True)
out_path = os.path.join(figures_dir, "pr_curves_baseline.pdf")
fig.savefig(out_path, bbox_inches="tight", dpi=300)
print(f"  Saved: {out_path}")
plt.close()
print("Done.")
