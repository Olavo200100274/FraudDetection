from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM

def make_classic_model(name: str, seed: int = 42):
    n = name.lower()
    if n in ["logreg", "logistic_regression", "lr"]:
        return LogisticRegression(
            max_iter=5000,
            n_jobs=None,
            class_weight=None,
            random_state=seed,
            solver="lbfgs"
        )
    if n in ["rf", "random_forest"]:
        return RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            class_weight=None
        )
    if n in ["ocsvm", "one_class_svm"]:
        # Anomaly detector: trained on legitimate only
        return OneClassSVM(
            kernel="rbf",
            nu=0.05,
            gamma="scale"
        )

    # Optional: LightGBM / CatBoost (import here to avoid hard dependency)
    if n in ["lgbm", "lightgbm"]:
        try:
            from lightgbm import LGBMClassifier
        except Exception as e:
            raise ImportError("LightGBM not installed. pip install lightgbm") from e
        return LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1
        )

    if n in ["catboost", "cb"]:
        try:
            from catboost import CatBoostClassifier
        except Exception as e:
            raise ImportError("CatBoost not installed. pip install catboost") from e
        return CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=False
        )

    raise ValueError(f"Unknown model name: {name}")