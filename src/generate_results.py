"""
Generate thesis-ready tables (LaTeX) and figures (PDF) from actual results.

Usage:
    cd src/
    python generate_results.py

Reads from:  ../results/ulb_2013/<model>/none/<run>/
Writes to:   ../thesis/tables/{ulb,baf}/  and  ../thesis/figures/{ulb,baf}/
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
TABLES_BASE = ROOT / "thesis" / "tables"
FIGURES_BASE = ROOT / "thesis" / "figures"

# Overleaf project mirror — generated artifacts are copied here automatically
OVERLEAF_ROOT = ROOT / "2026.Thesis.MSc.Olavo"
OVERLEAF_TABLES = OVERLEAF_ROOT / "tables"
OVERLEAF_FIGURES = OVERLEAF_ROOT / "figures"

import shutil

def _mirror_to_overleaf(src: Path):
    """Copy a thesis/ artifact into the Overleaf project at the same relative path."""
    # Determine if it's under tables/ or figures/
    try:
        rel = src.relative_to(TABLES_BASE)
        dst = OVERLEAF_TABLES / rel
    except ValueError:
        try:
            rel = src.relative_to(FIGURES_BASE)
            dst = OVERLEAF_FIGURES / rel
        except ValueError:
            return  # not a tables/figures file
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def _tables_dir(filename):
    d = TABLES_BASE / filename
    d.mkdir(parents=True, exist_ok=True)
    return d

def _figures_dir(filename):
    d = FIGURES_BASE / filename
    d.mkdir(parents=True, exist_ok=True)
    return d

def _emit(out: Path):
    """Print path and mirror the file to the Overleaf project."""
    _mirror_to_overleaf(out)
    print(f"  → {out}")

# ── display ordering & names ─────────────────────────────────────────────
MODEL_ORDER = ["logreg", "rf", "lgbm", "catboost", "ocsvm"]
MODEL_LABELS = {
    "logreg": "LR",
    "rf": "RF",
    "lgbm": "LGBM",
    "catboost": "CatBoost",
    "ocsvm": "OCSVM",
}
MODEL_COLORS = {
    "logreg":  "#9467bd",
    "rf":      "#2ca02c",
    "lgbm":    "#1f77b4",
    "catboost": "#d62728",
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
    lines.append(r"\begin{table}[ht]")
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
        tau_str = f"{m['threshold']:.3f}" if model != "ocsvm" else f"{m['threshold']:.2f}"
        brier_str = f"{m['brier_score']:.4f}" if model != "ocsvm" else "---"
        roc_auc = m.get("ROC-AUC", 0)

        label = MODEL_LABELS[model]
        # Pad label to 10 chars for alignment
        lines.append(
            f"{label:<10s} & {m['PR-AUC']:.3f} & {roc_auc:.3f} & {m['F1']:.3f} & {m['F2']:.3f} "
            f"& {prec:.3f} & {rec:.3f} & {tau_str} & {brier_str} \\\\"
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
    lines.append(r"\begin{table}[ht]")
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
        alert_pct = m["alert_rate"] * 100
        fp_tp = m["FP/TP"]
        fp_tp_str = f"{fp_tp:.2f}" if fp_tp < 100 else f"{fp_tp:.1f}"

        lines.append(
            f"{label:<10s} & {alert_pct:.2f}\\% & {fp_tp_str} "
            f"& {m['precision_at_k']:.3f} & {m['recall_at_k']:.3f} "
            f"& {m['k_used']:,} \\\\"
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
    lines.append(r"\begin{table}[ht]")
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
    lines.append(r"\begin{table}[ht]")
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
    lines.append(r"\begin{table}[ht]")
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
FACTORIAL_MODELS = ["logreg", "rf", "lgbm", "catboost"]


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
    lines.append(r"\begin{table}[ht]")
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
    """Generate PR-AUC and F2 factorial tables."""
    generate_factorial_table(
        fdata, "PR-AUC", dataset_label, filename, "PR-AUC", "prauc"
    )
    generate_factorial_table(
        fdata, "F2", dataset_label, filename, "$F_2$", "f2"
    )
    generate_factorial_table(
        fdata, "F1", dataset_label, filename, "$F_1$", "f1"
    )


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
    n_combos = sum(1 for m in ulb_fact for s in ulb_fact[m] if ulb_fact[m][s] is not None)
    if n_combos > 0:
        print(f"\n── ULB 2013 — Factorial ({n_combos} combos) ──")
        generate_factorial_tables(ulb_fact, "ULB 2013", "ulb")
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
    n_baf_combos = sum(1 for m in baf_fact for s in baf_fact[m] if baf_fact[m][s] is not None)
    if n_baf_combos > 0:
        print(f"\n── BAF Base — Factorial ({n_baf_combos} combos) ──")
        generate_factorial_tables(baf_fact, "BAF Base", "baf")
        generate_factorial_heatmap(baf_fact, "BAF Base", "baf")
    else:
        print("\n  No BAF factorial results found — skipping.")

    print("\n" + "=" * 60)
    print("  Done! Check thesis/tables/{ulb,baf}/ and thesis/figures/{ulb,baf}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
