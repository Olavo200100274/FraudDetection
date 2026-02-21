from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import joblib

from src.utils.logging import write_json

def save_run_artifacts(run_dir: Path, meta: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "meta.json", meta)

def save_model(run_dir: Path, model, filename: str) -> None:
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / "models" / filename)

def save_cv_table(run_dir: Path, rows: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "cv_metrics.csv", index=False)

def save_test_table(run_dir: Path, rows: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "test_metrics.csv", index=False)

def save_pr_curve(run_dir: Path, model_name: str, pr_curve: Dict[str, Any]) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "precision": pr_curve["precision"],
        "recall": pr_curve["recall"],
        # thresholds tem menos 1 elemento; guardamos NaN onde não existe
        "threshold": pr_curve["thresholds"] + [None],
    })
    df.to_csv(plots_dir / f"pr_curve_{model_name}.csv", index=False)