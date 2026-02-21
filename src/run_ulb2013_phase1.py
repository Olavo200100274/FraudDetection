from __future__ import annotations
from pathlib import Path
import datetime as dt
import time
import yaml
import numpy as np
from sklearn.pipeline import Pipeline

from src.data.load_ulb2013 import load_ulb2013_csv
from src.data.splitters import SplitConfig, stratified_train_test_split
from src.preprocessing.build_preprocessor import (
    build_ulb_preprocessor_with_scaler,
    build_ulb_preprocessor_no_scaler,
)
from src.models.classic_factory import make_classic_model
from src.evaluation.cv import run_stratified_cv
from src.evaluation.test_eval import evaluate_on_test
from src.evaluation.reporting import (
    save_run_artifacts, save_model, save_cv_table, save_test_table, save_pr_curve
)
from src.utils.seed import set_global_seed
from src.utils.logging import build_system_meta, start_timer, write_json

def _aggregate_threshold(thrs: list[float], method: str = "median") -> float:
    arr = np.asarray(thrs, dtype=float)
    if method == "mean":
        return float(arr.mean())
    return float(np.median(arr))

def main(cfg_path: str = "configs/ulb2013.yaml") -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["split"]["seed"])
    set_global_seed(seed)

    X, y, data_meta = load_ulb2013_csv(cfg["dataset"]["path"], target_col=cfg["dataset"]["target_col"])
    numeric_cols = list(X.columns)

    split_cfg = SplitConfig(
        test_size=float(cfg["split"]["test_size"]),
        seed=seed,
        stratify=bool(cfg["split"]["stratify"])
    )
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, split_cfg)

    run_id = f"run_{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg["output"]["runs_dir"]) / "ulb2013" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Marker files to avoid confusion with interrupted runs
    write_json(run_dir / "run_status.json", {"status": "RUNNING"})

    models = ["lr", "rf", "lgbm", "catboost", "ocsvm"]

    cv_rows = []
    test_rows = []

    threshold_agg = cfg.get("thresholding", {}).get("aggregate", "median")  # mean|median
    n_splits = int(cfg["cv"]["n_splits"])
    cv_seed = int(cfg["cv"]["seed"])

    t_run = start_timer()

    for mname in models:
        model = make_classic_model(mname, seed=seed)

        # Scaling for LR + OCSVM; no scaling for tree models
        if mname in ["lr", "ocsvm"]:
            pre = build_ulb_preprocessor_with_scaler(numeric_cols)
        else:
            pre = build_ulb_preprocessor_no_scaler(numeric_cols)

        pipe = Pipeline([("pre", pre), ("model", model)])

        # CV on training split
        cv_summary, cv_timing = run_stratified_cv(
            X_train, y_train,
            pipeline=pipe,
            n_splits=n_splits,
            seed=cv_seed
        )

        thr_f1_final = _aggregate_threshold(cv_summary["thr_f1_folds"], method=threshold_agg)
        thr_f2_final = _aggregate_threshold(cv_summary["thr_f2_folds"], method=threshold_agg)

        cv_rows.append({
            "model": mname,
            "pr_auc_mean": cv_summary["pr_auc_mean"],
            "pr_auc_std": cv_summary["pr_auc_std"],
            "f1_mean": cv_summary["f1_mean"],
            "f1_std": cv_summary["f1_std"],
            "f2_mean": cv_summary["f2_mean"],
            "f2_std": cv_summary["f2_std"],
            "thr_f1_agg": thr_f1_final,
            "thr_f2_agg": thr_f2_final,
            "cv_total_seconds": cv_timing["cv_total_seconds"],
            "fit_seconds_mean": cv_summary["fit_seconds_mean"],
            "fit_seconds_std": cv_summary["fit_seconds_std"],
        })

        # Refit on full training data (leakage-free)
        t_fit = time.perf_counter()

        if mname == "ocsvm":
            # One-class: train only on legitimate
            X_train_legit = X_train[y_train == 0]
            pipe.fit(X_train_legit, y_train[y_train == 0])
            X_train_for_ref = X_train_legit
            y_train_for_ref = y_train[y_train == 0]
        else:
            pipe.fit(X_train, y_train)
            X_train_for_ref = X_train
            y_train_for_ref = y_train

        fit_s = float(time.perf_counter() - t_fit)

        # Test evaluation
        test_res = evaluate_on_test(
            model=pipe,
            X_train=X_train_for_ref,
            y_train=y_train_for_ref,
            X_test=X_test,
            y_test=y_test,
            thr_f1=thr_f1_final,
            thr_f2=thr_f2_final
        )

        test_rows.append({
            "model": mname,
            "pr_auc": test_res["pr_auc"],
            "f1": test_res["f1"],
            "f2": test_res["f2"],
            "thr_f1": test_res["thr_f1"],
            "thr_f2": test_res["thr_f2"],
            "precision_at_f1_thr": test_res["precision_at_f1_thr"],
            "recall_at_f1_thr": test_res["recall_at_f1_thr"],
            "tp_f1": test_res["tp_f1"],
            "fp_f1": test_res["fp_f1"],
            "fn_f1": test_res["fn_f1"],
            "tn_f1": test_res["tn_f1"],
            "tp_f2": test_res["tp_f2"],
            "fp_f2": test_res["fp_f2"],
            "fn_f2": test_res["fn_f2"],
            "tn_f2": test_res["tn_f2"],
            "train_fit_seconds": fit_s,
            "test_inference_seconds": test_res["inference_seconds"],
        })

        save_pr_curve(run_dir, mname, test_res["pr_curve"])
        save_model(run_dir, pipe, filename=f"{mname}.joblib")

    total_run_s = t_run.stop()

    meta = {
        "run_id": run_id,
        "dataset": data_meta,
        "protocol": {
            "split": cfg["split"],
            "cv": cfg["cv"],
            "thresholding": {**cfg.get("thresholding", {}), "aggregate": threshold_agg},
            "tuning": cfg["tuning"],
            "note": "Baseline run with leakage-free split; thresholds aggregated from CV folds; test evaluation performed once.",
        },
        "system": build_system_meta(),
        "total_run_seconds": total_run_s,
    }

    save_run_artifacts(run_dir, meta=meta)
    save_cv_table(run_dir, cv_rows)
    save_test_table(run_dir, test_rows)

    write_json(run_dir / "run_status.json", {"status": "SUCCESS"})
    print(f"[OK] Saved run to: {run_dir}")

if __name__ == "__main__":
    main()