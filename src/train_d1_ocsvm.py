import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.utils_timing import timed, fmt_seconds
from src.config import (
    SEED, DATA_PATH_D1, TEST_SIZE, VAL_SIZE,
    THRESH_GRID_POINTS, RESULTS_DIR_D1, FIGURES_DIR_D1
)
from src.data_d1 import load_d1
from src.preprocessing_d1 import build_preprocessor_d1
from src.models.ocsvm import build_model
from src.evaluation import pr_auc, threshold_sweep, evaluate_at_threshold, pr_curve_df


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    timings = []

    with timed("Prepare output dirs", timings):
        ensure_dir(RESULTS_DIR_D1)
        ensure_dir(FIGURES_DIR_D1)

    with timed("CSV loading + dedup/cleanup", timings):
        df = load_d1(DATA_PATH_D1)

    with timed("Prepare X/y", timings):
        y = df["Class"].astype(int)
        X = df.drop(columns=["Class"])

    with timed("Split (stratified)", timings):
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=VAL_SIZE, stratify=y_trainval, random_state=SEED
        )

    print(f"[Split] train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    print(f"[Prevalence] train={y_train.mean():.6f} val={y_val.mean():.6f} test={y_test.mean():.6f}")

    # IMPORTANT: Train OCSVM only on class 0 (legitimate)
    with timed("Filter train normals (class 0 only)", timings):
        mask0 = (y_train.values == 0)
        X_train0 = X_train.loc[mask0]
        y_train0 = y_train.loc[mask0]
        print(f"[OCSVM] train_normals={len(X_train0)} / train_total={len(X_train)}")

    # Preprocess: fit ONLY on training normals
    with timed("Preprocessing (fit on normals, transform val/test)", timings):
        pre = build_preprocessor_d1(log_amount=True)
        X_train0_p = pre.fit_transform(X_train0, y_train0)
        X_val_p = pre.transform(X_val)
        X_test_p = pre.transform(X_test)

    with timed("Train OCSVM (fit on normals)", timings):
        oc = build_model()
        oc.fit(X_train0_p)

    # Scores: decision_function -> higher means more normal.
    # We invert so that higher = more suspicious (fraud-like)
    with timed("Predictions + threshold sweep (validation)", timings):
        val_score = -oc.decision_function(X_val_p)
        test_score = -oc.decision_function(X_test_p)
        sweep_df, best = threshold_sweep(y_val.values, val_score, grid_points=THRESH_GRID_POINTS)

    with timed("Test evaluation (metrics + confusion matrix)", timings):
        pra = pr_auc(y_test.values, test_score)
        f1_test, _, _ = evaluate_at_threshold(y_test.values, test_score, best["tau_f1"])
        _, f2_test, cm_f2 = evaluate_at_threshold(y_test.values, test_score, best["tau_f2"])

        metrics = pd.DataFrame([{
            "model": "OCSVM",
            "pr_auc": pra,
            "tau_f1": best["tau_f1"],
            "f1_at_tau_f1": f1_test,
            "tau_f2": best["tau_f2"],
            "f2_at_tau_f2": f2_test,
        }])

        cm_df = pd.DataFrame(cm_f2, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])

    with timed("Build PR curve points (test)", timings):
        pr_df = pr_curve_df(y_test.values, test_score)

    with timed("Save CSV artifacts", timings):
        sweep_df.to_csv(os.path.join(RESULTS_DIR_D1, "threshold_sweep_OCSVM.csv"), index=False)
        metrics.to_csv(os.path.join(RESULTS_DIR_D1, "metrics_OCSVM.csv"), index=False)
        cm_df.to_csv(os.path.join(RESULTS_DIR_D1, "confusion_matrix_tauF2_OCSVM.csv"), index=True)
        pr_df.to_csv(os.path.join(RESULTS_DIR_D1, "pr_curve_OCSVM.csv"), index=False)

    with timed("Generate plots", timings):
        plt.figure()
        plt.plot(sweep_df["threshold"], sweep_df["f1"], label="F1")
        plt.plot(sweep_df["threshold"], sweep_df["f2"], label="F2")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("D1 (ULB 2013) — OCSVM Threshold Sweep (Validation)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR_D1, "threshold_sweep_OCSVM.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(pr_df["recall"], pr_df["precision"])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("D1 (ULB 2013) — OCSVM Precision-Recall Curve (Test)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR_D1, "pr_curve_OCSVM.png"), dpi=200)
        plt.close()

    with timed("Save timing report", timings):
        total = sum(t for _, t in timings)
        timing_df = pd.DataFrame(timings, columns=["stage", "seconds"])
        timing_df.loc[len(timing_df)] = ["TOTAL", total]
        timing_df.to_csv(os.path.join(RESULTS_DIR_D1, "timing_OCSVM.csv"), index=False)

    print("\n=== DONE (D1 + OCSVM) ===")
    print(metrics.to_string(index=False))
    print("\nConfusion matrix at tau_F2:")
    print(cm_df)

    print("\nTiming breakdown:")
    for name, t in timings:
        print(f"- {name}: {fmt_seconds(t)}")
    print(f"TOTAL: {fmt_seconds(sum(t for _, t in timings))}")


if __name__ == "__main__":
    main()
