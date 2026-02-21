from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Load helpers
# -------------------------
def read_test_metrics(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "test_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    df = pd.read_csv(p)

    required = {
        "model", "pr_auc", "f2",
        "tp_f2", "fp_f2", "fn_f2", "tn_f2",
        "precision_at_f1_thr", "recall_at_f1_thr",
        "tp_f1", "fp_f1", "fn_f1", "tn_f1",
        "train_fit_seconds",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test_metrics.csv missing columns: {missing}")

    return df


def read_cv_metrics(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "cv_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    df = pd.read_csv(p)

    required = {"model", "pr_auc_mean", "pr_auc_std"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"cv_metrics.csv missing columns: {missing}")

    return df


def read_pr_curve(run_dir: Path, model: str) -> pd.DataFrame:
    p = run_dir / "plots" / f"pr_curve_{model}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing PR curve file for model '{model}': {p}")
    df = pd.read_csv(p)
    required = {"precision", "recall"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{p.name} missing columns: {missing}")
    return df.dropna(subset=["precision", "recall"])


# -------------------------
# (A) PR curves
# -------------------------
def plot_pr_curves(run_dir: Path, out_dir: Path, order: list[str] | None = None) -> Path:
    df_test = read_test_metrics(run_dir)

    if order is None:
        models = df_test.sort_values("pr_auc", ascending=False)["model"].tolist()
    else:
        models = order

    plt.figure()
    for m in models:
        df_curve = read_pr_curve(run_dir, m)
        plt.plot(df_curve["recall"], df_curve["precision"], label=m)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("ULB2013 Baseline — Precision-Recall Curves (Test)")
    plt.legend()
    plt.grid(True, linewidth=0.4, alpha=0.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()

    png_path = out_dir / "ulb2013_test_pr_curves.png"
    pdf_path = out_dir / "ulb2013_test_pr_curves.pdf"
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()
    return pdf_path


# -------------------------
# (B)(C) Bars: PR-AUC, F2
# -------------------------
def plot_bar_metric(run_dir: Path, out_dir: Path, metric: str, filename_stem: str, title: str) -> Path:
    df = read_test_metrics(run_dir).copy()
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(df.columns)}")
    df = df.sort_values(metric, ascending=False)

    plt.figure()
    plt.bar(df["model"], df[metric])
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.grid(True, axis="y", linewidth=0.4, alpha=0.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()

    png_path = out_dir / f"{filename_stem}.png"
    pdf_path = out_dir / f"{filename_stem}.pdf"
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()
    return pdf_path


# -------------------------
# (D) Confusion matrix for best model @ F2 threshold
# -------------------------
def plot_best_confusion_matrix_f2(run_dir: Path, out_dir: Path) -> Path:
    df = read_test_metrics(run_dir).copy()
    # escolhe o melhor por PR-AUC (podes mudar para F2 se quiseres)
    best = df.sort_values("pr_auc", ascending=False).iloc[0]

    model = str(best["model"])
    tp = int(best["tp_f2"])
    fp = int(best["fp_f2"])
    fn = int(best["fn_f2"])
    tn = int(best["tn_f2"])

    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=int)

    plt.figure()
    plt.imshow(cm)  # sem especificar cores
    plt.title(f"ULB2013 Baseline — Confusion Matrix (Test) @ F2 threshold — {model}")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    # anotar valores
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "ulb2013_test_confusion_matrix_best_f2.png"
    pdf_path = out_dir / "ulb2013_test_confusion_matrix_best_f2.pdf"
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()
    return pdf_path


# -------------------------
# (E) Precision/Recall trade-off (F1 vs F2) per model
# -------------------------
def _precision_recall_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


def plot_tradeoff_precision_recall(run_dir: Path, out_dir: Path) -> Path:
    df = read_test_metrics(run_dir).copy()
    df = df.sort_values("pr_auc", ascending=False)

    # F1 operating point
    p_f1 = df["precision_at_f1_thr"].astype(float).to_numpy()
    r_f1 = df["recall_at_f1_thr"].astype(float).to_numpy()

    # F2 operating point (compute from counts)
    p_f2 = []
    r_f2 = []
    for _, row in df.iterrows():
        precision, recall = _precision_recall_from_counts(
            int(row["tp_f2"]), int(row["fp_f2"]), int(row["fn_f2"])
        )
        p_f2.append(precision)
        r_f2.append(recall)
    p_f2 = np.array(p_f2, dtype=float)
    r_f2 = np.array(r_f2, dtype=float)

    x = np.arange(len(df))
    w = 0.22

    plt.figure()
    plt.bar(x - 1.5*w, p_f1, width=w, label="Precision @ F1")
    plt.bar(x - 0.5*w, r_f1, width=w, label="Recall @ F1")
    plt.bar(x + 0.5*w, p_f2, width=w, label="Precision @ F2")
    plt.bar(x + 1.5*w, r_f2, width=w, label="Recall @ F2")

    plt.xticks(x, df["model"].tolist())
    plt.ylim(0, 1.0)
    plt.xlabel("Model")
    plt.ylabel("Value")
    plt.title("ULB2013 Baseline — Precision/Recall Trade-off (Test) at F1 vs F2 thresholds")
    plt.legend()
    plt.grid(True, axis="y", linewidth=0.4, alpha=0.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()

    png_path = out_dir / "ulb2013_test_tradeoff_precision_recall.png"
    pdf_path = out_dir / "ulb2013_test_tradeoff_precision_recall.pdf"
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()
    return pdf_path


# -------------------------
# (F) Training time comparison
# -------------------------
def plot_training_time(run_dir: Path, out_dir: Path) -> Path:
    df = read_test_metrics(run_dir).copy()
    df = df.sort_values("train_fit_seconds", ascending=False)

    plt.figure()
    plt.bar(df["model"], df["train_fit_seconds"])
    plt.xlabel("Model")
    plt.ylabel("Train fit seconds")
    plt.title("ULB2013 Baseline — Training Time (Refit on full train split)")
    plt.grid(True, axis="y", linewidth=0.4, alpha=0.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()

    png_path = out_dir / "ulb2013_train_time_bar.png"
    pdf_path = out_dir / "ulb2013_train_time_bar.pdf"
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()
    return pdf_path


# -------------------------
# (G) CV stability plot: PR-AUC mean ± std
# -------------------------
def plot_cv_stability_pr_auc(run_dir: Path, out_dir: Path) -> Path:
    df = read_cv_metrics(run_dir).copy()
    df = df.sort_values("pr_auc_mean", ascending=False)

    x = np.arange(len(df))
    y = df["pr_auc_mean"].astype(float).to_numpy()
    yerr = df["pr_auc_std"].astype(float).to_numpy()

    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt="o")  # sem cores explícitas
    plt.xticks(x, df["model"].tolist())
    plt.ylim(0, 1.0)
    plt.xlabel("Model")
    plt.ylabel("PR-AUC (CV mean ± std)")
    plt.title("ULB2013 Baseline — CV Stability (PR-AUC)")
    plt.grid(True, axis="y", linewidth=0.4, alpha=0.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()

    png_path = out_dir / "ulb2013_cv_stability_pr_auc.png"
    pdf_path = out_dir / "ulb2013_cv_stability_pr_auc.pdf"
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()
    return pdf_path


# -------------------------
# Generate all
# -------------------------
def generate_all(run_dir: Path, out_dir: Path | None = None) -> dict[str, Path]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else (run_dir / "figures")

    paths = {}
    paths["pr_curves"] = plot_pr_curves(run_dir, out_dir)
    paths["pr_auc_bar"] = plot_bar_metric(
        run_dir, out_dir, metric="pr_auc",
        filename_stem="ulb2013_test_pr_auc_bar",
        title="ULB2013 Baseline — PR-AUC (Test)",
    )
    paths["f2_bar"] = plot_bar_metric(
        run_dir, out_dir, metric="f2",
        filename_stem="ulb2013_test_f2_bar",
        title="ULB2013 Baseline — F2 (Test)",
    )

    # Fortemente recomendados
    paths["best_confusion_matrix_f2"] = plot_best_confusion_matrix_f2(run_dir, out_dir)
    paths["tradeoff_precision_recall"] = plot_tradeoff_precision_recall(run_dir, out_dir)
    paths["training_time"] = plot_training_time(run_dir, out_dir)

    # Opcional forte
    paths["cv_stability_pr_auc"] = plot_cv_stability_pr_auc(run_dir, out_dir)

    return paths