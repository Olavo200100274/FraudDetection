from __future__ import annotations
from dataclasses import dataclass
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

@dataclass(frozen=True)
class ULBPreprocessBundle:
    numeric_cols: List[str]

def build_ulb_preprocessor_with_scaler(cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

def build_ulb_preprocessor_no_scaler(cols: List[str]) -> ColumnTransformer:
    # Identity transform: passthrough numeric columns
    return ColumnTransformer(
        transformers=[("num", "passthrough", cols)],
        remainder="drop",
        verbose_feature_names_out=False
    )