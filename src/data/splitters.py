from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split

@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    seed: int = 42
    stratify: bool = True

def stratified_train_test_split(
    X, y, cfg: SplitConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    strat = y if cfg.stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=strat
    )
    return X_train, X_test, y_train, y_test