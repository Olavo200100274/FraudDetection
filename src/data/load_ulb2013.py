from __future__ import annotations
import pandas as pd
from typing import Tuple, Dict, Any

def load_ulb2013_csv(path: str, target_col: str = "Class") -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    meta = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_features": int(X.shape[1]),
        "target_col": target_col,
        "fraud_count": int((y == 1).sum()),
        "legit_count": int((y == 0).sum()),
        "fraud_rate": float((y == 1).mean()),
        "missing_total": int(df.isna().sum().sum()),
        "columns": list(df.columns),
    }
    return X, y, meta