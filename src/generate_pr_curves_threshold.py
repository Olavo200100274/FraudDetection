"""
Generate Article-1 PR curves with threshold-rule markers.

For each dataset, draws the PR curve of every model (baseline, no resampling)
and overlays four markers per supervised model showing where each threshold
strategy (fixed 0.5, max-F1, max-F2, Precision>=0.5) operates on that curve.

OCSVM is drawn as a curve but receives no markers because its threshold
values live on a different scale (decision-function output, not probability).

Usage:
    cd src/
    python generate_pr_curves_threshold.py

Reads from:  ../results/<dataset>/<model>/none/<latest_run>/
Writes to:   ../Article 1/figs/{ulb,baf}_pr_curves_threshold.pdf
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "Article 1" / "figs"

MODEL_ORDER = ["logreg", "rf", "lgbm", "catboost", "fttransformer", "ocsvm"]
MODEL_LABELS = {
    "logreg": "LR", "rf": "RF", "lgbm": "LGBM",
    "catboost": "CatBoost", "fttransformer": "FT-Trans.", "ocsvm": "OCSVM",
}
MODEL_COLORS = {
    "logreg":  "#9467bd", "rf": "#2ca02c", "lgbm": "#1f77b4",
    "catboost": "#d62728", "fttransformer": "#ff7f0e", "ocsvm": "#7f7f7f",
}

# Threshold-rule markers: (key, label, marker_shape)
THRESHOLD_STRATEGIES = [
    ("fixed_05", r"Fixed $\tau=0.5$",       "o"),
    ("max_f1",   r"max-$F_1$",              "^"),
    ("max_f2",   r"max-$F_2$",              "s"),
    ("prec_ge_05", r"Precision $\geq$ 0.5", "D"),
]

SUPERVISED = ["logreg", "rf", "lgbm", "catboost", "fttransformer"]


def find_latest_run(dataset, model):
    base = RESULTS_DIR / dataset / model / "none"
    if not base.exists():
        return None
    runs = sorted(base.iterdir())
    return runs[-1] if runs else None


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def collect(dataset):
    data = {}
    for model in MODEL_ORDER:
        run_dir = find_latest_run(dataset, model)
        if run_dir is None:
            print(f"  [SKIP] {model} - no run for {dataset}")
            continue
        pr_path = run_dir / "pr_curve_data.json"
        metrics_path = run_dir / "metrics_test.json"
        ts_path = run_dir / "threshold_study.json"
        entry = {
            "pr_data": load_json(pr_path),
            "metrics": load_json(metrics_path),
        }
        if ts_path.exists():
            entry["ts"] = load_json(ts_path)
        data[model] = entry
    return data


def plot_dataset(data, dataset_label, fraud_rate, out_path, zoom_to_curves=True):
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    # -- PR curves --
    for model in MODEL_ORDER:
        if model not in data:
            continue
        pr = data[model]["pr_data"]
        recalls = np.array(pr["recalls"])
        precisions = np.array(pr["precisions"])
        prauc = data[model]["metrics"]["PR-AUC"]
        label = f"{MODEL_LABELS[model]} (PR-AUC = {prauc:.3f})"
        style = dict(linewidth=2.0, alpha=0.9)
        if model == "ocsvm":
            style["linestyle"] = "--"
            style["linewidth"] = 1.5
            style["alpha"] = 0.6
        ax.plot(recalls, precisions, color=MODEL_COLORS[model], label=label, **style)

    # -- Threshold markers (supervised only) --
    for model in SUPERVISED:
        if model not in data or "ts" not in data[model]:
            continue
        ts = data[model]["ts"]["test_results"]
        for key, _, mshape in THRESHOLD_STRATEGIES:
            if key not in ts:
                continue
            r = ts[key]["Recall"]
            p = ts[key]["Precision"]
            ax.scatter(r, p,
                       marker=mshape,
                       s=110,
                       facecolor=MODEL_COLORS[model],
                       edgecolor="black",
                       linewidth=1.0,
                       zorder=5)

    # Random baseline
    ax.axhline(y=fraud_rate, color="black", linestyle=":",
               linewidth=0.8, alpha=0.5,
               label=f"Random (prevalence = {fraud_rate:.4f})")

    # -- Axes & cosmetics --
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    if zoom_to_curves and fraud_rate < 0.05:
        ax.set_ylim([0.0, 1.0])
    else:
        ax.set_ylim([0.0, 1.05])

    # No in-figure title: the figure caption (in the LaTeX source) already identifies
    # the dataset. In-figure titles are discouraged in journal style.
    ax.grid(True, alpha=0.3)

    # Two-part legend: models (auto from ax) + threshold-rule markers (manual)
    model_legend = ax.legend(loc="upper right", fontsize=9, framealpha=0.95,
                             title="Models")
    ax.add_artist(model_legend)

    # Build threshold-rule legend
    from matplotlib.lines import Line2D
    thr_handles = [
        Line2D([0], [0],
               marker=mshape, linestyle="",
               markersize=8, markerfacecolor="lightgray",
               markeredgecolor="black", markeredgewidth=0.9,
               label=lbl)
        for _, lbl, mshape in THRESHOLD_STRATEGIES
    ]
    ax.legend(handles=thr_handles, loc="lower left",
              fontsize=9, framealpha=0.95, title="Threshold rule")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    print("ULB 2013:")
    ulb = collect("ulb_2013")
    plot_dataset(ulb, "ULB 2013", fraud_rate=0.00173,
                 out_path=OUT_DIR / "ulb_pr_curves_threshold.pdf")

    print("BAF Base:")
    baf = collect("baf_base")
    plot_dataset(baf, "BAF Base", fraud_rate=0.01102,
                 out_path=OUT_DIR / "baf_pr_curves_threshold.pdf")


if __name__ == "__main__":
    main()
