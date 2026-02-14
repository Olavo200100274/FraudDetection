import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.utils_timing import timed, fmt_seconds

from src.config import (
    SEED, DATA_PATH_D1, TEST_SIZE, VAL_SIZE,
    THRESH_GRID_POINTS, RESULTS_DIR_D1, FIGURES_DIR_D1
)
from src.data_d1 import load_d1
from src.preprocessing_d1 import build_preprocessor_d1
from src.models.lgbm import build_model
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

    with timed("Build pipeline (preprocessor + model)", timings):
        # para árvores, scaling não é necessário, mas manter o mesmo preprocessor dá consistência.
        pre = build_preprocessor_d1(log_amount=True)
        clf = build_model(SEED)   # baseline: class_weight="balanced" dentro do modelo
        pipe = Pipeline([("pre", pre), ("clf", clf)])

    with timed("Train (fit)", timings):
        pipe.fit(X_train, y_train)

    with timed("Predictions + threshold sweep (validation)", timings):
        val_score = pipe.predict_proba(X_val)[:, 1]
        test_score = pipe.predict_proba(X_test)[:, 1]
        sweep_df, best = threshold_sweep(y_val.values, val_score, grid_points=THRESH_GRID_POINTS)

    with timed("Test evaluation (metrics + confusion matrix)", timings):
        pra = pr_auc(y_test.values, test_score)
        f1_test, _, _ = evaluate_at_threshold(y_test.values, test_score, best["tau_f1"])
        _, f2_test, cm_f2 = evaluate_at_threshold(y_test.values, test_score, best["tau_f2"])

        metrics = pd.DataFrame([{
            "model": "LGBM",
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
        sweep_df.to_csv(os.path.join(RESULTS_DIR_D1, "threshold_sweep_LGBM.csv"), index=False)
        metrics.to_csv(os.path.join(RESULTS_DIR_D1, "metrics_LGBM.csv"), index=False)
        cm_df.to_csv(os.path.join(RESULTS_DIR_D1, "confusion_matrix_tauF2_LGBM.csv"), index=True)
        pr_df.to_csv(os.path.join(RESULTS_DIR_D1, "pr_curve_LGBM.csv"), index=False)

    with timed("Generate plots", timings):
        plt.figure()
        plt.plot(sweep_df["threshold"], sweep_df["f1"], label="F1")
        plt.plot(sweep_df["threshold"], sweep_df["f2"], label="F2")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("D1 (ULB 2013) — LightGBM Threshold Sweep (Validation)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR_D1, "threshold_sweep_LGBM.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(pr_df["recall"], pr_df["precision"])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("D1 (ULB 2013) — LightGBM Precision-Recall Curve (Test)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR_D1, "pr_curve_LGBM.png"), dpi=200)
        plt.close()

    with timed("Save timing report", timings):
        total = sum(t for _, t in timings)
        timing_df = pd.DataFrame(timings, columns=["stage", "seconds"])
        timing_df.loc[len(timing_df)] = ["TOTAL", total]
        timing_df.to_csv(os.path.join(RESULTS_DIR_D1, "timing_LGBM.csv"), index=False)

    print("\n=== DONE (D1 + LightGBM) ===")
    print(metrics.to_string(index=False))
    print("\nConfusion matrix at tau_F2:")
    print(cm_df)

    print("\nTiming breakdown:")
    for name, t in timings:
        print(f"- {name}: {fmt_seconds(t)}")
    print(f"TOTAL: {fmt_seconds(sum(t for _, t in timings))}")


if __name__ == "__main__":
    main()
