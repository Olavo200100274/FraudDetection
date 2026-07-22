"""
Generate thesis-ready tables (LaTeX) and figures (PDF) from actual results.

Usage:
    cd src/
    python generate_results.py

Reads from:  ../results/ulb_2013/<model>/none/<run>/
Writes to:   ../results_thesis/tables/{ulb,baf}/ and ../results_thesis/figures/{ulb,baf}/
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
TABLES_BASE = ROOT / "results_thesis" / "tables"
FIGURES_BASE = ROOT / "results_thesis" / "figures"

def _tables_dir(filename):
    d = TABLES_BASE / filename
    d.mkdir(parents=True, exist_ok=True)
    return d

def _figures_dir(filename):
    d = FIGURES_BASE / filename
    d.mkdir(parents=True, exist_ok=True)
    return d

def _emit(out: Path):
    """Print generated file path."""
    print(f"  → {out}")

# ── display ordering & names ─────────────────────────────────────────────
MODEL_ORDER = ["logreg", "rf", "lgbm", "catboost", "fttransformer", "ocsvm"]
MODEL_LABELS = {
    "logreg": "LR",
    "rf": "RF",
    "lgbm": "LGBM",
    "catboost": "CatBoost",
    "fttransformer": "FT-Trans.",
    "ocsvm": "OCSVM",
}
MODEL_COLORS = {
    "logreg":  "#9467bd",
    "rf":      "#2ca02c",
    "lgbm":    "#1f77b4",
    "catboost": "#d62728",
    "fttransformer": "#ff7f0e",
    "ocsvm":   "#7f7f7f",
}


# ── helpers ──────────────────────────────────────────────────────────────
def find_latest_run(dataset, model, strategy="none"):
    """Return Path to the latest run directory for a given model."""
    base = RESULTS_DIR / dataset / model / strategy
    if not base.exists():
        return None
    runs = sorted(base.iterdir())
    return runs[-1] if runs else None


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════
#  COLLECT DATA
# ══════════════════════════════════════════════════════════════════════════

def collect_baseline(dataset):
    """Collect metrics_test, config, and pr_curve_data for all models."""
    data = {}
    for model in MODEL_ORDER:
        run_dir = find_latest_run(dataset, model)
        if run_dir is None:
            print(f"  [SKIP] {model} — no results found for {dataset}")
            continue
        data[model] = {
            "metrics": load_json(run_dir / "metrics_test.json"),
            "config": load_json(run_dir / "config.json"),
            "pr_data": load_json(run_dir / "pr_curve_data.json"),
        }
    return data


def collect_threshold_study(dataset):
    """Collect threshold_study.json for all models that have it."""
    data = {}
    for model in MODEL_ORDER:
        run_dir = find_latest_run(dataset, model)
        if run_dir is None:
            continue
        ts_path = run_dir / "threshold_study.json"
        if ts_path.exists():
            data[model] = load_json(ts_path)
    return data


# ══════════════════════════════════════════════════════════════════════════
#  TABLE 1 — Baseline comparison
# ══════════════════════════════════════════════════════════════════════════

def generate_baseline_table(data, dataset_label, filename):
    """Generate LaTeX table: PR-AUC, F1, F2, Precision, Recall, τ, Brier."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Baseline comparison on "
        + dataset_label
        + r" (strategy\,=\,None, threshold\,=\,max-$F_2$).}"
    )
    lines.append(r"\label{tab:baseline_" + filename + "}")
    lines.append(r"\begin{tabular}{l c c c c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"Model & PR-AUC & ROC-AUC & $F_1$ & $F_2$ & Precision & Recall & $\tau$ & Brier \\"
    )
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in data:
            continue
        m = data[model]["metrics"]
        tp, fp, fn = m["TP"], m["FP"], m["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        tau = m["threshold"]
        if abs(tau) >= 10:
            tau_str = f"{tau:.2f}"
        else:
            tau_str = f"{tau:.4f}"
        brier_str = f"{m['brier_score']:.5f}"
        roc_auc = m.get("ROC-AUC", 0)

        label = MODEL_LABELS[model]
        # Pad label to 10 chars for alignment
        lines.append(
            f"{label:<10s} & {m['PR-AUC']:.4f} & {roc_auc:.4f} & {m['F1']:.4f} & {m['F2']:.4f} "
            f"& {prec:.4f} & {rec:.4f} & {tau_str} & {brier_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir(filename) / "baseline.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  TABLE 2 — Operational metrics
# ══════════════════════════════════════════════════════════════════════════

def generate_ops_table(data, dataset_label, filename):
    """Generate LaTeX table: Alert Rate, FP/TP, P@k, R@k, k."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Operational metrics on "
        + dataset_label
        + r" (strategy\,=\,None, threshold\,=\,max-$F_2$).}"
    )
    lines.append(r"\label{tab:ops_" + filename + "}")
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & Alert Rate & FP/TP & P@$k$ & R@$k$ & $k$ \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in data:
            continue
        m = data[model]["metrics"]
        label = MODEL_LABELS[model]
        alert_rate = m["alert_rate"]
        fp_tp = m["FP/TP"]
        fp_tp_str = f"{fp_tp:.4f}" if fp_tp < 100 else f"{fp_tp:.1f}"

        lines.append(
            f"{label:<10s} & {alert_rate:.5f} & {fp_tp_str} "
            f"& {m['precision_at_k']:.4f} & {m['recall_at_k']:.4f} "
            f"& {m['k_used']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir(filename) / "ops.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  TABLE 3 — Bootstrap CIs
# ══════════════════════════════════════════════════════════════════════════

def generate_ci_table(data, dataset_label, filename):
    """Generate LaTeX table: PR-AUC and ROC-AUC with 95% Bootstrap CI."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{95\% Bootstrap CI for PR-AUC and ROC-AUC on "
        + dataset_label
        + r" (strategy\,=\,None, 1\,000 iterations).}"
    )
    lines.append(r"\label{tab:ci_" + filename + "}")
    lines.append(r"\begin{tabular}{l c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & PR-AUC & 95\% CI & ROC-AUC & 95\% CI \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in data:
            continue
        m = data[model]["metrics"]
        ci_pr = m["bootstrap_ci"]["PR-AUC_ci"]
        ci_roc = m["bootstrap_ci"].get("ROC-AUC_ci", (0, 0))
        roc_auc = m.get("ROC-AUC", 0)
        label = MODEL_LABELS[model]
        lines.append(
            f"{label:<10s} & {m['PR-AUC']:.3f} "
            f"& [{ci_pr[0]:.3f}, {ci_pr[1]:.3f}] "
            f"& {roc_auc:.3f} "
            f"& [{ci_roc[0]:.3f}, {ci_roc[1]:.3f}] \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir(filename) / "ci.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  TABLE 4 — Computational cost
# ══════════════════════════════════════════════════════════════════════════

def generate_cost_table(data, dataset_label, filename):
    """Generate LaTeX table: tuning, training, and inference times."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Computational cost on "
        + dataset_label
        + r" (strategy\,=\,None). Tuning includes Optuna TPE (50 trials, 5-fold CV).}"
    )
    lines.append(r"\label{tab:cost_" + filename + "}")
    lines.append(r"\begin{tabular}{l r r r}")
    lines.append(r"\toprule")
    lines.append(r"Model & Tuning (s) & Train (s) & Inference (s) \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in data:
            continue
        c = data[model]["config"]
        label = MODEL_LABELS[model]
        tuning = c.get("tuning_time_s", 0)
        train = c.get("train_time_s", 0)
        infer = c.get("infer_time_s", 0)

        if tuning > 0:
            tuning_str = f"{tuning:,.0f}"
        else:
            tuning_str = "---"

        lines.append(
            f"{label:<10s} & {tuning_str} & {train:.1f} & {infer:.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir(filename) / "cost.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — PR Curves (all models overlaid)
# ══════════════════════════════════════════════════════════════════════════

def generate_pr_curve(data, dataset_label, filename, fraud_rate=0.0017):
    """Generate PR curve PDF with all models overlaid."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for model in MODEL_ORDER:
        if model not in data:
            continue
        pr = data[model]["pr_data"]
        precisions = np.array(pr["precisions"])
        recalls = np.array(pr["recalls"])
        prauc = data[model]["metrics"]["PR-AUC"]
        label = f"{MODEL_LABELS[model]} ({prauc:.3f})"

        style = {"linewidth": 2.0}
        if model == "ocsvm":
            style["linestyle"] = "--"
            style["linewidth"] = 1.5

        ax.plot(recalls, precisions,
                color=MODEL_COLORS[model],
                label=label, **style)

    # Random baseline
    ax.axhline(y=fraud_rate, color="black", linestyle=":",
               linewidth=0.8, alpha=0.6, label=f"Random ({fraud_rate:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(f"Precision–Recall Curves — {dataset_label} (strategy = None)",
                 fontsize=13)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    out = _figures_dir(filename) / "pr_curves_baseline.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Bar chart: PR-AUC comparison
# ══════════════════════════════════════════════════════════════════════════

def generate_prauc_bar(data, dataset_label, filename):
    """Generate horizontal bar chart of PR-AUC with CI error bars."""
    models = [m for m in MODEL_ORDER if m in data]
    labels = [MODEL_LABELS[m] for m in models]
    praucs = [data[m]["metrics"]["PR-AUC"] for m in models]
    colors = [MODEL_COLORS[m] for m in models]

    # CI error bars
    ci_low = [data[m]["metrics"]["bootstrap_ci"]["PR-AUC_ci"][0] for m in models]
    ci_high = [data[m]["metrics"]["bootstrap_ci"]["PR-AUC_ci"][1] for m in models]
    err_low = [p - lo for p, lo in zip(praucs, ci_low)]
    err_high = [hi - p for p, hi in zip(praucs, ci_high)]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    y_pos = np.arange(len(models))

    ax.barh(y_pos, praucs, color=colors, edgecolor="white", height=0.6,
            xerr=[err_low, err_high], capsize=4, error_kw={"linewidth": 1.2})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("PR-AUC", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_title(f"PR-AUC — {dataset_label} (strategy = None)", fontsize=13)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    # Value annotations
    for i, (v, hi) in enumerate(zip(praucs, ci_high)):
        ax.text(hi + 0.01, i, f"{v:.3f}", va="center", fontsize=10)

    out = _figures_dir(filename) / "prauc_bar_baseline.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Confusion matrix grid
# ══════════════════════════════════════════════════════════════════════════

def generate_confusion_grid(data, dataset_label, filename):
    """Generate a grid of confusion matrices for all models."""
    models = [m for m in MODEL_ORDER if m in data]
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0))
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        m = data[model]["metrics"]
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])

        ax.imshow(cm, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = "white" if val > cm.max() * 0.5 else "black"
                ax.text(j, i, f"{val:,}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Legit", "Fraud"], fontsize=8)
        ax.set_yticklabels(["Legit", "Fraud"], fontsize=8)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        if model == models[0]:
            ax.set_ylabel("Actual", fontsize=9)

    fig.suptitle(f"Confusion Matrices — {dataset_label} (strategy = None)",
                 fontsize=13, y=1.02)
    out = _figures_dir(filename) / "confusion_grid_baseline.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  THRESHOLD STUDY TABLES
# ══════════════════════════════════════════════════════════════════════════

STRATEGY_ORDER = ["fixed_05", "max_f1", "max_f2", "prec_ge_05"]
STRATEGY_COL_HEADERS = {
    "fixed_05":   r"Fixed (0.5)",
    "max_f1":     r"max-$F_1$",
    "max_f2":     r"max-$F_2$",
    "prec_ge_05": r"Prec\,$\geq$\,0.5",
}


def _generate_threshold_table(ts_data, metric_key, dataset_label, filename,
                                caption_metric, label_suffix, fmt=".3f"):
    """Generic helper: one threshold-study table for a given metric."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Threshold sensitivity on "
        + dataset_label
        + r"~--- " + caption_metric
        + r" by threshold strategy (strategy\,=\,None).}"
    )
    lines.append(r"\label{tab:threshold_" + filename + "_" + label_suffix + "}")
    lines.append(r"\begin{tabular}{l c c c c}")
    lines.append(r"\toprule")

    col_headers = " & ".join(STRATEGY_COL_HEADERS[s] for s in STRATEGY_ORDER)
    lines.append(r"Model & " + col_headers + r" \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in ts_data:
            continue
        label = MODEL_LABELS[model]
        tr = ts_data[model]["test_results"]
        vals = []
        for s in STRATEGY_ORDER:
            v = tr[s][metric_key]
            if metric_key == "alert_rate":
                vals.append(f"{v * 100:.2f}\\%")
            else:
                vals.append(f"{v:{fmt}}")
        lines.append(f"{label:<10s} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir(filename) / f"threshold_{label_suffix}.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_threshold_tables(ts_data, dataset_label, filename):
    """Generate all 4 threshold study LaTeX tables for a dataset."""
    _generate_threshold_table(
        ts_data, "F2", dataset_label, filename, "$F_2$", "f2"
    )
    _generate_threshold_table(
        ts_data, "F1", dataset_label, filename, "$F_1$", "f1"
    )
    _generate_threshold_table(
        ts_data, "Recall", dataset_label, filename, "Recall", "recall"
    )
    _generate_threshold_table(
        ts_data, "alert_rate", dataset_label, filename, r"Alert Rate (\%)", "alert"
    )


# ══════════════════════════════════════════════════════════════════════════
#  FACTORIAL TABLES  —  Model × Imbalance Strategy
# ══════════════════════════════════════════════════════════════════════════

BALANCE_ORDER = ["none", "rus", "ros", "smote", "smote_tomek", "smoteenn", "weights"]
BALANCE_LABELS = {
    "none":       "None",
    "rus":        "RUS",
    "ros":        "ROS",
    "smote":      "SMOTE",
    "smote_tomek": "SM+T",
    "smoteenn":   "SMOTEENN",
    "weights":    "Weights",
}
# Models eligible for factorial (OCSVM excluded — anomaly detection paradigm)
FACTORIAL_MODELS = ["logreg", "rf", "lgbm", "catboost", "fttransformer"]


def collect_factorial(dataset):
    """
    Collect metrics_test.json for every (model, strategy) combination.

    Returns dict[model][strategy] = metrics_test dict, or None if missing.
    """
    data = {}
    for model in FACTORIAL_MODELS:
        data[model] = {}
        for strat in BALANCE_ORDER:
            run_dir = find_latest_run(dataset, model, strategy=strat)
            if run_dir is None:
                data[model][strat] = None
                continue
            mt_path = run_dir / "metrics_test.json"
            if mt_path.exists():
                data[model][strat] = load_json(mt_path)
            else:
                data[model][strat] = None
    return data


def generate_factorial_table(fdata, metric_key, dataset_label, filename,
                             caption_metric, label_suffix, fmt=".3f"):
    """One factorial table: rows = models, columns = strategies."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{" + caption_metric + r" by model $\times$ strategy on "
        + dataset_label + r".}"
    )
    lines.append(r"\label{tab:factorial_" + filename + "_" + label_suffix + "}")
    lines.append(r"\begin{tabular}{l" + " c" * len(BALANCE_ORDER) + "}")
    lines.append(r"\toprule")

    col_headers = " & ".join(BALANCE_LABELS[s] for s in BALANCE_ORDER)
    lines.append(r"Model & " + col_headers + r" \\")
    lines.append(r"\midrule")

    for model in FACTORIAL_MODELS:
        label = MODEL_LABELS[model]
        vals = []
        # Find baseline value for bolding the best
        baseline_val = None
        if fdata[model]["none"] is not None:
            baseline_val = fdata[model]["none"].get(metric_key)

        row_vals = []
        for strat in BALANCE_ORDER:
            mt = fdata[model].get(strat)
            if mt is None:
                row_vals.append(("---", None))
            else:
                v = mt[metric_key]
                row_vals.append((f"{v:{fmt}}", v))

        # Find the best value in this row
        numeric_vals = [rv[1] for rv in row_vals if rv[1] is not None]
        best_val = max(numeric_vals) if numeric_vals else None

        formatted = []
        for text, v in row_vals:
            if v is not None and best_val is not None and abs(v - best_val) < 1e-6:
                formatted.append(r"\textbf{" + text + "}")
            else:
                formatted.append(text)

        lines.append(f"{label:<10s} & " + " & ".join(formatted) + r" \\")

    # FT-Trans. placeholder row
    lines.append(
        r"% FT-Trans. & --- & --- & --- & --- & --- & --- & --- \\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir(filename) / f"factorial_{label_suffix}.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_factorial_tables(fdata, dataset_label, filename):
    """Generate PR-AUC, ROC-AUC, F2 and F1 factorial tables."""
    generate_factorial_table(
        fdata, "PR-AUC", dataset_label, filename, "PR-AUC", "prauc"
    )
    generate_factorial_table(
        fdata, "ROC-AUC", dataset_label, filename, "ROC-AUC", "rocauc"
    )
    generate_factorial_table(
        fdata, "F2", dataset_label, filename, "$F_2$", "f2"
    )
    generate_factorial_table(
        fdata, "F1", dataset_label, filename, "$F_1$", "f1"
    )


def generate_transformer_robustness_table(fdata, dataset_label, filename):
    """Compare FT-Transformer and CatBoost across all seven strategies.

    The final row is the population standard deviation (``ddof=0``) across the
    complete set of designed strategy conditions, computed from the unrounded
    metrics stored in the run artefacts.
    """
    models = ("fttransformer", "catboost")
    if any(fdata[model].get(strategy) is None
           for model in models for strategy in BALANCE_ORDER):
        print(f"  Incomplete {dataset_label} robustness grid — skipping.")
        return

    row_labels = {
        "none": "None",
        "rus": "RUS",
        "ros": "ROS",
        "smote": "SMOTE",
        "smote_tomek": "SM+Tomek",
        "smoteenn": "SMOTEENN",
        "weights": "Weights",
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (r"\caption{FT-Transformer and CatBoost performance across imbalance "
         rf"strategies on {dataset_label}. The final row is the population "
         r"standard deviation across the seven complete strategy values, "
         r"computed from unrounded metrics.}"),
        rf"\label{{tab:transformer_robustness_{filename}}}",
        r"\begin{tabular}{l cc cc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{FT-Transformer} & \multicolumn{2}{c}{CatBoost} \\",
        r"Strategy & PR-AUC & $F_2$ & PR-AUC & $F_2$ \\",
        r"\midrule",
    ]

    for strategy in BALANCE_ORDER:
        ft_metrics = fdata["fttransformer"][strategy]
        cb_metrics = fdata["catboost"][strategy]
        lines.append(
            f"{row_labels[strategy]:<9s} & {ft_metrics['PR-AUC']:.3f} & "
            f"{ft_metrics['F2']:.3f} & {cb_metrics['PR-AUC']:.3f} & "
            f"{cb_metrics['F2']:.3f} " + r"\\"
        )

    def population_sd(model, metric):
        values = [fdata[model][strategy][metric] for strategy in BALANCE_ORDER]
        return float(np.std(values, ddof=0))

    lines.extend([
        r"\midrule",
        (r"\textit{Pop.\ std.\ dev.} & "
         f"\\textit{{{population_sd('fttransformer', 'PR-AUC'):.3f}}} & "
         f"\\textit{{{population_sd('fttransformer', 'F2'):.3f}}} & "
         f"\\textit{{{population_sd('catboost', 'PR-AUC'):.3f}}} & "
         f"\\textit{{{population_sd('catboost', 'F2'):.3f}}} " + r"\\"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out = _tables_dir(filename) / "transformer_robustness.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_factorial_heatmap(fdata, dataset_label, filename):
    """
    Generate a heatmap of ΔPR-AUC (strategy − baseline) for each model.
    Green = improvement, red = degradation.
    """
    import seaborn as sns

    # Non-baseline strategies only for the heatmap
    strats = [s for s in BALANCE_ORDER if s != "none"]
    strat_labels = [BALANCE_LABELS[s] for s in strats]
    model_labels = [MODEL_LABELS[m] for m in FACTORIAL_MODELS]

    matrix = np.full((len(FACTORIAL_MODELS), len(strats)), np.nan)
    for i, model in enumerate(FACTORIAL_MODELS):
        baseline = fdata[model].get("none")
        if baseline is None:
            continue
        base_val = baseline["PR-AUC"]
        for j, strat in enumerate(strats):
            mt = fdata[model].get(strat)
            if mt is not None:
                matrix[i, j] = mt["PR-AUC"] - base_val

    # Also build F2 heatmap
    matrix_f2 = np.full((len(FACTORIAL_MODELS), len(strats)), np.nan)
    for i, model in enumerate(FACTORIAL_MODELS):
        baseline = fdata[model].get("none")
        if baseline is None:
            continue
        base_val = baseline["F2"]
        for j, strat in enumerate(strats):
            mt = fdata[model].get(strat)
            if mt is not None:
                matrix_f2[i, j] = mt["F2"] - base_val

    for mat, metric_name, suffix in [
        (matrix, "PR-AUC", "prauc"),
        (matrix_f2, "$F_2$", "f2"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
        vmax = max(vmax, 0.01)  # avoid zero range

        sns.heatmap(
            mat, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
            vmin=-vmax, vmax=vmax,
            xticklabels=strat_labels, yticklabels=model_labels,
            ax=ax, linewidths=0.5, cbar_kws={"label": f"Δ{metric_name}"},
        )
        ax.set_title(
            f"Δ{metric_name} (strategy − None) — {dataset_label}",
            fontsize=13,
        )
        ax.set_ylabel("")

        out = _figures_dir(filename) / f"heatmap_{suffix}.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        _emit(out)


# ══════════════════════════════════════════════════════════════════════════
#  CROSS-DOMAIN GENERALIZATION  (BAF Base → Variants I–V)
# ══════════════════════════════════════════════════════════════════════════

VARIANT_ORDER = ["baf_base", "baf_var1", "baf_var2", "baf_var3", "baf_var4", "baf_var5"]
VARIANT_LABELS = {
    "baf_base": "Base",
    "baf_var1": "Var I",
    "baf_var2": "Var II",
    "baf_var3": "Var III",
    "baf_var4": "Var IV",
    "baf_var5": "Var V",
}


def collect_cross_domain():
    """Load cross_domain.json from each Base model's run directory."""
    data = {}
    for model in MODEL_ORDER:
        run_dir = find_latest_run("baf_base", model)
        if run_dir is None:
            continue
        cd_path = run_dir / "cross_domain.json"
        if cd_path.exists():
            data[model] = load_json(cd_path)
    return data


def generate_cross_domain_table(cd_data, metric_key, caption_metric,
                                 label_suffix, fmt=".4f"):
    """Cross-domain table: rows = models, columns = Base + Variants."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Cross-domain " + caption_metric
        + r": trained on BAF Base, evaluated on each Variant (no retraining).}"
    )
    lines.append(r"\label{tab:crossdomain_" + label_suffix + "}")
    lines.append(r"\begin{tabular}{l" + " c" * len(VARIANT_ORDER) + "}")
    lines.append(r"\toprule")

    col_headers = " & ".join(VARIANT_LABELS[v] for v in VARIANT_ORDER)
    lines.append(r"Model & " + col_headers + r" \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in cd_data:
            continue
        label = MODEL_LABELS[model]
        vals = []
        numeric = []
        for var in VARIANT_ORDER:
            if var in cd_data[model]:
                m = cd_data[model][var]["metrics"]
                v = m[metric_key]
                vals.append((f"{v:{fmt}}", v))
                numeric.append(v)
            else:
                vals.append(("---", None))

        best = max(numeric) if numeric else None
        formatted = []
        for text, v in vals:
            if v is not None and best is not None and abs(v - best) < 1e-6:
                formatted.append(r"\textbf{" + text + "}")
            else:
                formatted.append(text)

        lines.append(f"{label:<10s} & " + " & ".join(formatted) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir("baf") / f"crossdomain_{label_suffix}.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_cross_domain_tables(cd_data):
    """Generate PR-AUC and F2 cross-domain tables."""
    generate_cross_domain_table(cd_data, "PR-AUC", "PR-AUC", "prauc")
    generate_cross_domain_table(cd_data, "F2", "$F_2$", "f2")
    generate_cross_domain_table(cd_data, "ROC-AUC", "ROC-AUC", "rocauc")


# ══════════════════════════════════════════════════════════════════════════
#  BAF VARIANTS IN-DOMAIN  (train & test on same variant)
# ══════════════════════════════════════════════════════════════════════════

def collect_variants_indomain():
    """Collect baseline metrics for all BAF variants (in-domain runs)."""
    data = {}
    for var in VARIANT_ORDER:
        data[var] = {}
        for model in MODEL_ORDER:
            run_dir = find_latest_run(var, model)
            if run_dir is None:
                data[var][model] = None
                continue
            mt_path = run_dir / "metrics_test.json"
            if mt_path.exists():
                data[var][model] = load_json(mt_path)
            else:
                data[var][model] = None
    return data


def generate_variants_indomain_table(vi_data, metric_key, caption_metric,
                                      label_suffix, fmt=".4f"):
    """In-domain table: rows = models, columns = Base + Variants."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{In-domain " + caption_metric
        + r" across BAF datasets (each model trained and tested on the same dataset).}"
    )
    lines.append(r"\label{tab:indomain_" + label_suffix + "}")
    lines.append(r"\begin{tabular}{l" + " c" * len(VARIANT_ORDER) + "}")
    lines.append(r"\toprule")

    col_headers = " & ".join(VARIANT_LABELS[v] for v in VARIANT_ORDER)
    lines.append(r"Model & " + col_headers + r" \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        label = MODEL_LABELS[model]
        vals = []
        numeric = []
        for var in VARIANT_ORDER:
            mt = vi_data[var].get(model)
            if mt is not None:
                v = mt[metric_key]
                vals.append((f"{v:{fmt}}", v))
                numeric.append(v)
            else:
                vals.append(("---", None))

        best = max(numeric) if numeric else None
        formatted = []
        for text, v in vals:
            if v is not None and best is not None and abs(v - best) < 1e-6:
                formatted.append(r"\textbf{" + text + "}")
            else:
                formatted.append(text)

        lines.append(f"{label:<10s} & " + " & ".join(formatted) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir("baf") / f"indomain_{label_suffix}.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_variants_indomain_tables(vi_data):
    """Generate in-domain summary tables for PR-AUC and F2."""
    generate_variants_indomain_table(vi_data, "PR-AUC", "PR-AUC", "prauc")
    generate_variants_indomain_table(vi_data, "F2", "$F_2$", "f2")


# ══════════════════════════════════════════════════════════════════════════
#  CROSS vs IN-DOMAIN DELTA TABLE
# ══════════════════════════════════════════════════════════════════════════

def generate_delta_table(cd_data, vi_data, metric_key, caption_metric,
                          label_suffix, fmt="+.4f"):
    """Δ table (cross-domain − in-domain) for Variants I–V only."""
    var_keys = [v for v in VARIANT_ORDER if v != "baf_base"]

    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{$\Delta$" + caption_metric
        + r" (cross-domain $-$ in-domain) on BAF Variants. "
        + r"Negative values indicate degradation under distribution shift.}"
    )
    lines.append(r"\label{tab:delta_" + label_suffix + "}")
    lines.append(r"\begin{tabular}{l" + " c" * len(var_keys) + "}")
    lines.append(r"\toprule")

    col_headers = " & ".join(VARIANT_LABELS[v] for v in var_keys)
    lines.append(r"Model & " + col_headers + r" \\")
    lines.append(r"\midrule")

    for model in MODEL_ORDER:
        if model not in cd_data:
            continue
        label = MODEL_LABELS[model]
        vals = []
        for var in var_keys:
            cd_metrics = cd_data[model].get(var, {}).get("metrics")
            id_metrics = vi_data.get(var, {}).get(model)
            if cd_metrics is not None and id_metrics is not None:
                delta = cd_metrics[metric_key] - id_metrics[metric_key]
                vals.append(f"{delta:{fmt}}")
            else:
                vals.append("---")
        lines.append(f"{label:<10s} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir("baf") / f"delta_{label_suffix}.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_delta_tables(cd_data, vi_data):
    """Generate Δ tables for PR-AUC and F2."""
    generate_delta_table(cd_data, vi_data, "PR-AUC", "PR-AUC", "prauc")
    generate_delta_table(cd_data, vi_data, "F2", "$F_2$", "f2")


# ══════════════════════════════════════════════════════════════════════════
#  SHAP ANALYSIS TABLES & FIGURES
# ══════════════════════════════════════════════════════════════════════════

SHAP_MODELS = ["logreg", "lgbm", "catboost", "fttransformer"]
SHAP_LABELS = {
    "logreg": "LR",
    "lgbm": "LGBM",
    "catboost": "CatBoost",
    "fttransformer": "FT-Trans.",
}

VARIANT_LABEL_MAP = {
    "baf_base": "Base",
    "baf_var1": "Var.~I",
    "baf_var2": "Var.~II",
    "baf_var3": "Var.~III",
    "baf_var4": "Var.~IV",
    "baf_var5": "Var.~V",
}


def collect_shap_data():
    """Collect shap_global.json, shap_consistency.json, shap_variant_stability.json."""
    global_data = {}
    for model in SHAP_MODELS:
        run_dir = find_latest_run("baf_base", model)
        if run_dir is None:
            continue
        gpath = run_dir / "shap_global.json"
        if gpath.exists():
            with open(gpath) as f:
                global_data[model] = json.load(f)

    # Consistency matrix
    consistency = None
    lgbm_dir = find_latest_run("baf_base", "lgbm")
    if lgbm_dir:
        cpath = lgbm_dir / "shap_consistency.json"
        if cpath.exists():
            with open(cpath) as f:
                consistency = json.load(f)

    # Variant stability
    stability = None
    if lgbm_dir:
        spath = lgbm_dir / "shap_variant_stability.json"
        if spath.exists():
            with open(spath) as f:
                stability = json.load(f)

    # Local cases
    local_data = {}
    for model in SHAP_MODELS:
        run_dir = find_latest_run("baf_base", model)
        if run_dir is None:
            continue
        lpath = run_dir / "shap_local_cases.json"
        if lpath.exists():
            with open(lpath) as f:
                local_data[model] = json.load(f)

    return global_data, consistency, stability, local_data


def generate_shap_global_bar(global_data):
    """Horizontal grouped bar chart: top-15 features by mean |SHAP| for multiple models."""
    if not global_data:
        return

    # Use LGBM's feature order as reference
    ref_model = "lgbm" if "lgbm" in global_data else list(global_data.keys())[0]
    features = list(global_data[ref_model].keys())[:15]  # top-15

    models_to_plot = [m for m in SHAP_MODELS if m in global_data]
    n_models = len(models_to_plot)
    bar_height = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    for i, model in enumerate(models_to_plot):
        importances = [global_data[model].get(f, 0) for f in features]
        y_pos = np.arange(len(features)) + i * bar_height
        ax.barh(y_pos, importances, height=bar_height, label=SHAP_LABELS[model],
                color=colors[i % len(colors)], alpha=0.85)

    ax.set_yticks(np.arange(len(features)) + bar_height * (n_models - 1) / 2)
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Global Feature Importance (BAF Base, top-15)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = _figures_dir("baf") / "shap_global.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    _emit(out)


def generate_shap_beeswarm(global_data):
    """
    Generate a beeswarm-style bar chart showing feature importance with direction.

    Note: True beeswarm requires raw SHAP values + feature values.
    This generates a simplified importance bar chart as a fallback.
    Full beeswarm is generated directly by shap_analysis.py if shap_values.npy exists.
    """
    # The actual beeswarm is better generated directly with shap library
    # in shap_analysis.py. Here we check if it was already generated.
    lgbm_dir = find_latest_run("baf_base", "lgbm")
    if lgbm_dir is None:
        return

    shap_path = lgbm_dir / "shap_values.npy"
    if not shap_path.exists():
        return

    try:
        import shap as shap_lib
        shap_values = np.load(shap_path)

        # Load test data feature names from preprocessor
        import joblib
        pipeline = joblib.load(lgbm_dir / "model.joblib")
        preprocessor = pipeline.named_steps["preprocessor"]

        from data import load_dataset
        _, X_test, _, _ = load_dataset("baf_base")
        X_transformed = preprocessor.transform(X_test)
        feature_names = list(preprocessor.get_feature_names_out())

        if hasattr(X_transformed, "values"):
            X_transformed_np = X_transformed.values
        else:
            X_transformed_np = X_transformed

        # Create SHAP Explanation object
        explanation = shap_lib.Explanation(
            values=shap_values,
            data=X_transformed_np,
            feature_names=feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        shap_lib.plots.beeswarm(explanation, max_display=15, show=False)
        plt.title("SHAP Beeswarm — LGBM on BAF Base", fontsize=13)
        plt.tight_layout()

        out = _figures_dir("baf") / "shap_beeswarm.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        _emit(out)
    except Exception as e:
        print(f"  [WARN] Beeswarm plot failed: {e}")


def generate_shap_consistency_table(consistency):
    """Generate Jaccard cross-model consistency table."""
    if consistency is None:
        return

    models = consistency["models"]
    jm = consistency["jaccard_matrix"]

    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\caption{Jaccard similarity of top-10 SHAP features across model pairs (BAF Base).}",
        r"\label{tab:shap_consistency}",
    ]

    col_spec = "l " + " ".join(["c"] * len(models))
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = " & ".join(SHAP_LABELS.get(m, m) for m in models)
    lines.append(f"          & {header} \\\\")
    lines.append(r"\midrule")

    for m1 in models:
        label = SHAP_LABELS.get(m1, m1)
        vals = []
        for m2 in models:
            if m1 == m2:
                vals.append("---")
            else:
                v = jm[m1][m2]
                vals.append(f"{v:.2f}")
        lines.append(f"{label:<10s} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir("baf") / "shap_consistency.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_shap_stability_table(stability):
    """Generate Jaccard cross-variant stability table."""
    if stability is None:
        return

    var_keys = stability["variant_keys"]
    jm = stability["jaccard_matrix"]

    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\caption{Jaccard similarity of top-10 SHAP features for LGBM across BAF Variants.}",
        r"\label{tab:shap_stability}",
    ]

    col_spec = "l " + " ".join(["c"] * len(var_keys))
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = " & ".join(VARIANT_LABEL_MAP.get(v, v) for v in var_keys)
    lines.append(f"         & {header} \\\\")
    lines.append(r"\midrule")

    for v1 in var_keys:
        label = VARIANT_LABEL_MAP.get(v1, v1)
        vals = []
        for v2 in var_keys:
            if v1 == v2:
                vals.append("---")
            else:
                v = jm[v1][v2]
                vals.append(f"{v:.2f}")
        lines.append(f"{label:<10s} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = _tables_dir("baf") / "shap_stability.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    _emit(out)


def generate_shap_waterfall(local_data):
    """Generate 3-panel waterfall plot for TP, FP, FN (LGBM on BAF Base)."""
    model = "lgbm"
    if model not in local_data:
        return

    cases = local_data[model]
    case_types = ["TP", "FP", "FN"]
    available = [ct for ct in case_types if ct in cases]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 6))
    if len(available) == 1:
        axes = [axes]

    titles = {
        "TP": "True Positive\n(correctly flagged fraud)",
        "FP": "False Positive\n(false alarm)",
        "FN": "False Negative\n(missed fraud)",
    }

    for ax, ct in zip(axes, available):
        case = cases[ct]
        sv = case["shap_values"]

        # Sort by absolute value, take top-10
        sorted_feats = sorted(sv.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        features = [f[0] for f in sorted_feats][::-1]
        values = [f[1] for f in sorted_feats][::-1]

        colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]

        ax.barh(features, values, color=colors, height=0.6)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value", fontsize=10)
        ax.set_title(f"{titles[ct]}\n(score={case['score']:.4f})", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Local SHAP Explanations — LGBM on BAF Base", fontsize=13, y=1.02)
    plt.tight_layout()

    out = _figures_dir("baf") / "shap_waterfall_tp_fp_fn.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    _emit(out)


def generate_shap_dependence(global_data):
    """Generate dependence plots for top-3 features (LGBM on BAF Base)."""
    lgbm_dir = find_latest_run("baf_base", "lgbm")
    if lgbm_dir is None or "lgbm" not in global_data:
        return

    shap_path = lgbm_dir / "shap_values.npy"
    if not shap_path.exists():
        return

    try:
        import joblib
        shap_values = np.load(shap_path)
        pipeline = joblib.load(lgbm_dir / "model.joblib")
        preprocessor = pipeline.named_steps["preprocessor"]

        from data import load_dataset
        _, X_test, _, _ = load_dataset("baf_base")
        X_transformed = preprocessor.transform(X_test)
        feature_names = list(preprocessor.get_feature_names_out())

        if hasattr(X_transformed, "values"):
            X_transformed_np = X_transformed.values
        else:
            X_transformed_np = X_transformed

        # Top-3 features from global importance
        top3 = list(global_data["lgbm"].keys())[:3]

        # The global report groups one-hot columns back to their source
        # categorical features.  A grouped name therefore cannot be plotted
        # directly against a single transformed column.  Do not emit a
        # partially empty and potentially misleading figure when this occurs.
        missing_features = [name for name in top3 if name not in feature_names]
        if missing_features:
            print(
                "  [WARN] SHAP dependence plot skipped: grouped top-feature "
                f"names are absent from the transformed matrix: {missing_features}"
            )
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, feat_name in zip(axes, top3):
            idx = feature_names.index(feat_name)
            ax.scatter(X_transformed_np[:, idx], shap_values[:, idx],
                       alpha=0.05, s=3, c=shap_values[:, idx],
                       cmap="coolwarm", rasterized=True)
            ax.set_xlabel(feat_name, fontsize=11)
            ax.set_ylabel("SHAP value", fontsize=11)
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
            ax.grid(alpha=0.2)

        plt.suptitle("SHAP Dependence Plots — LGBM on BAF Base (top-3 features)",
                     fontsize=13)
        plt.tight_layout()

        out = _figures_dir("baf") / "shap_dependence.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        _emit(out)
    except Exception as e:
        print(f"  [WARN] Dependence plots failed: {e}")


def generate_all_shap():
    """Generate all SHAP tables and figures."""
    global_data, consistency, stability, local_data = collect_shap_data()

    if not global_data:
        print("\n  No SHAP results found — skipping.")
        return

    print(f"\n── BAF Base — SHAP Analysis ({len(global_data)} models) ──")
    generate_shap_global_bar(global_data)
    generate_shap_beeswarm(global_data)
    generate_shap_consistency_table(consistency)
    generate_shap_stability_table(stability)
    generate_shap_waterfall(local_data)
    generate_shap_dependence(global_data)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Generating thesis tables and figures from actual results")
    print("=" * 60)

    # ── ULB 2013 ──
    print("\n── ULB Credit Card 2013 ──")
    ulb = collect_baseline("ulb_2013")
    if ulb:
        print(f"  Found {len(ulb)} models: {', '.join(ulb.keys())}")
        generate_baseline_table(ulb, "ULB 2013", "ulb")
        generate_ops_table(ulb, "ULB 2013", "ulb")
        generate_ci_table(ulb, "ULB 2013", "ulb")
        generate_cost_table(ulb, "ULB 2013", "ulb")
        generate_pr_curve(ulb, "ULB 2013", "ulb", fraud_rate=0.001727)
        generate_prauc_bar(ulb, "ULB 2013", "ulb")
        generate_confusion_grid(ulb, "ULB 2013", "ulb")

    # ── ULB 2013 — Threshold Study ──
    ulb_ts = collect_threshold_study("ulb_2013")
    if ulb_ts:
        print(f"\n── ULB 2013 — Threshold Study ({len(ulb_ts)} models) ──")
        generate_threshold_tables(ulb_ts, "ULB 2013", "ulb")
    else:
        print("\n  No ULB threshold study results found — skipping.")

    # ── ULB 2013 — Factorial (Model × Strategy) ──
    ulb_fact = collect_factorial("ulb_2013")
    n_combos = sum(
        1
        for m in ulb_fact
        for s in ulb_fact[m]
        if s != "none" and ulb_fact[m][s] is not None
    )
    if n_combos > 0:
        print(f"\n── ULB 2013 — Factorial ({n_combos} combos) ──")
        generate_factorial_tables(ulb_fact, "ULB 2013", "ulb")
        generate_transformer_robustness_table(ulb_fact, "ULB 2013", "ulb")
        generate_factorial_heatmap(ulb_fact, "ULB 2013", "ulb")
    else:
        print("\n  No ULB factorial results found — skipping.")

    # ── BAF Base (if exists) ──
    print("\n── BAF Base ──")
    baf = collect_baseline("baf_base")
    if baf:
        print(f"  Found {len(baf)} models: {', '.join(baf.keys())}")
        generate_baseline_table(baf, "BAF Base", "baf")
        generate_ops_table(baf, "BAF Base", "baf")
        generate_ci_table(baf, "BAF Base", "baf")
        generate_cost_table(baf, "BAF Base", "baf")
        generate_pr_curve(baf, "BAF Base", "baf", fraud_rate=0.011)
        generate_prauc_bar(baf, "BAF Base", "baf")
        generate_confusion_grid(baf, "BAF Base", "baf")
    else:
        print("  No BAF results found yet — skipping.")

    # ── BAF Base — Threshold Study ──
    baf_ts = collect_threshold_study("baf_base")
    if baf_ts:
        print(f"\n── BAF Base — Threshold Study ({len(baf_ts)} models) ──")
        generate_threshold_tables(baf_ts, "BAF Base", "baf")
    else:
        print("\n  No BAF threshold study results found — skipping.")

    # ── BAF Base — Factorial (Model × Strategy) ──
    baf_fact = collect_factorial("baf_base")
    n_baf_combos = sum(
        1
        for m in baf_fact
        for s in baf_fact[m]
        if s != "none" and baf_fact[m][s] is not None
    )
    if n_baf_combos > 0:
        print(f"\n── BAF Base — Factorial ({n_baf_combos} combos) ──")
        generate_factorial_tables(baf_fact, "BAF Base", "baf")
        generate_transformer_robustness_table(baf_fact, "BAF Base", "baf")
        generate_factorial_heatmap(baf_fact, "BAF Base", "baf")
    else:
        print("\n  No BAF factorial results found — skipping.")

    # ── BAF Cross-Domain Generalization ──
    cd_data = collect_cross_domain()
    if cd_data:
        print(f"\n── BAF Cross-Domain ({len(cd_data)} models) ──")
        generate_cross_domain_tables(cd_data)
    else:
        print("\n  No BAF cross-domain results found — skipping.")

    # ── BAF Variants In-Domain ──
    vi_data = collect_variants_indomain()
    n_vi = sum(
        1 for v in vi_data for m in vi_data[v] if vi_data[v][m] is not None
    )
    # Subtract Base models (already reported as baselines)
    n_vi_variants = sum(
        1 for v in vi_data if v != "baf_base"
        for m in vi_data[v] if vi_data[v][m] is not None
    )
    if n_vi_variants > 0:
        print(f"\n── BAF Variants In-Domain ({n_vi_variants} runs) ──")
        generate_variants_indomain_tables(vi_data)
        # If we also have cross-domain data, generate delta tables
        if cd_data:
            print(f"\n── BAF Δ Cross-vs-In-Domain ──")
            generate_delta_tables(cd_data, vi_data)
    else:
        print("\n  No BAF variant in-domain results found — skipping delta tables.")

    # ── SHAP Analysis ──
    generate_all_shap()

    print("\n" + "=" * 60)
    print("  Done! Check results_thesis/tables/{ulb,baf}/ and results_thesis/figures/{ulb,baf}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
