# src/aggregate_d1_results.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

D1_RESULTS = Path("results") / "D1_ulb2013"
D1_FIGS = Path("figures") / "D1_ulb2013"

MODELS = ["LR", "RF", "LGBM", "CAT", "OCSVM"]  # nomes como tens nos ficheiros

def read_metrics():
    rows = []
    for m in MODELS:
        f = D1_RESULTS / f"metrics_{m}.csv"
        if not f.exists():
            print(f"[WARN] missing {f}")
            continue
        df = pd.read_csv(f)
        # espera-se uma linha com colunas: model, pr_auc, tau_f1, f1_at_tau_f1, tau_f2, f2_at_tau_f2
        rows.append(df.iloc[0].to_dict())
    out = pd.DataFrame(rows)

    # ordenar por F2 (ou PR-AUC) - escolhe o critério que queres destacar
    if "f2_at_tau_f2" in out.columns:
        out = out.sort_values("f2_at_tau_f2", ascending=False)
    return out

def read_confusion():
    rows = []
    for m in MODELS:
        f = D1_RESULTS / f"confusion_matrix_tauF2_{m}.csv"
        if not f.exists():
            continue
        cm = pd.read_csv(f, index_col=0)
        # formato esperado:
        #           pred_0 pred_1
        # actual_0    TN     FP
        # actual_1    FN     TP
        TN = int(cm.loc["actual_0", "pred_0"])
        FP = int(cm.loc["actual_0", "pred_1"])
        FN = int(cm.loc["actual_1", "pred_0"])
        TP = int(cm.loc["actual_1", "pred_1"])
        rows.append({"model": m, "TN": TN, "FP": FP, "FN": FN, "TP": TP})
    return pd.DataFrame(rows)

def plot_bar(summary: pd.DataFrame):
    D1_FIGS.mkdir(parents=True, exist_ok=True)

    # normalizar nomes
    x = summary["model"].tolist()
    pr = summary["pr_auc"].astype(float).tolist()
    f1 = summary["f1_at_tau_f1"].astype(float).tolist()
    f2 = summary["f2_at_tau_f2"].astype(float).tolist()

    idx = np.arange(len(x))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(idx - width, pr, width, label="PR-AUC")
    plt.bar(idx, f1, width, label="F1@τF1")
    plt.bar(idx + width, f2, width, label="F2@τF2")
    plt.xticks(idx, x)
    plt.ylim(0, 1.0)
    plt.title("D1 (ULB 2013) — Model Comparison")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(D1_FIGS / "summary_bar_metrics_D1.png", dpi=200)
    plt.close()

def plot_pr_curves():
    D1_FIGS.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    for m in MODELS:
        f = D1_RESULTS / f"pr_curve_{m}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        # espera colunas: precision, recall (ou recall, precision)
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        if "recall" not in df.columns or "precision" not in df.columns:
            print(f"[WARN] unexpected columns in {f}: {df.columns}")
            continue
        plt.plot(df["recall"], df["precision"], label=m)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("D1 (ULB 2013) — Precision-Recall Curves (Test)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(D1_FIGS / "summary_pr_curves_D1.png", dpi=200)
    plt.close()

def main():
    D1_RESULTS.mkdir(parents=True, exist_ok=True)
    D1_FIGS.mkdir(parents=True, exist_ok=True)

    summary = read_metrics()
    if summary.empty:
        print("[ERROR] No metrics files found.")
        return

    # junta confusion matrix para teres visão operacional
    cm = read_confusion()
    if not cm.empty:
        summary = summary.merge(cm, on="model", how="left")

    # guarda tabela final
    summary_path = D1_RESULTS / "summary_table_D1.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] wrote {summary_path}")

    # figuras comparativas
    plot_bar(summary)
    plot_pr_curves()
    print(f"[OK] wrote figures to {D1_FIGS}")

if __name__ == "__main__":
    main()
